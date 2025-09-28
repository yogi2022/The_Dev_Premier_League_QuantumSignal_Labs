"""
Advanced Backtesting Engine with Snowpark Integration
Implements sophisticated backtesting framework with risk management and performance analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum
import uuid

# Statistical and financial libraries
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class PositionStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"

@dataclass
class Trade:
    """Individual trade record"""
    id: str
    symbol: str
    signal_type: SignalType
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    entry_time: datetime
    exit_time: Optional[datetime]
    stop_loss: float
    take_profit: float
    commission: float
    status: PositionStatus
    pnl: float = 0.0
    return_pct: float = 0.0
    holding_period: int = 0  # in hours
    exit_reason: str = ""

@dataclass
class BacktestResults:
    """Comprehensive backtesting results"""
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    var_95: float
    var_99: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    trades: List[Trade]
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    performance_metrics: Dict[str, Any]

class AdvancedBacktester:
    """
    Advanced backtesting engine with sophisticated risk management and analytics
    """
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001, 
                 slippage: float = 0.0005, max_position_size: float = 0.1):
        self.initial_capital = initial_capital
        self.commission = commission  # Commission per trade (as percentage)
        self.slippage = slippage      # Slippage per trade (as percentage)
        self.max_position_size = max_position_size  # Maximum position size as % of capital
        
        # Backtesting state
        self.current_capital = initial_capital
        self.positions = {}  # symbol -> Trade
        self.closed_trades = []
        self.equity_history = []
        self.trade_log = []
        
        # Performance tracking
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        
    def run_backtest(self, price_data: pd.DataFrame, signals_data: pd.DataFrame, 
                    strategy_name: str = "AI_Signal_Strategy") -> BacktestResults:
        """
        Run comprehensive backtest with advanced analytics
        """
        logger.info(f"Starting backtest for strategy: {strategy_name}")
        
        # Prepare data
        price_data = price_data.sort_index()
        signals_data = signals_data.sort_values('created_at')
        
        # Reset backtesting state
        self._reset_backtest_state()
        
        # Process each timestamp
        for timestamp in price_data.index:
            current_prices = price_data.loc[timestamp]
            
            # Check for new signals at this timestamp
            current_signals = self._get_signals_for_timestamp(signals_data, timestamp)
            
            # Process signals
            for _, signal in current_signals.iterrows():
                self._process_signal(signal, current_prices, timestamp)
            
            # Update open positions
            self._update_positions(current_prices, timestamp)
            
            # Record equity
            self._record_equity(timestamp)
        
        # Close all remaining positions
        final_timestamp = price_data.index[-1]
        final_prices = price_data.loc[final_timestamp]
        self._close_all_positions(final_prices, final_timestamp, "BACKTEST_END")
        
        # Calculate final results
        results = self._calculate_backtest_results(strategy_name, price_data.index[0], 
                                                 price_data.index[-1])
        
        logger.info(f"Backtest completed. Total return: {results.total_return_pct:.2f}%")
        return results
    
    def _reset_backtest_state(self):
        """Reset backtesting state for new run"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_history = []
        self.trade_log = []
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
    
    def _get_signals_for_timestamp(self, signals_data: pd.DataFrame, timestamp: datetime) -> pd.DataFrame:
        """Get signals that should be executed at given timestamp"""
        # Assuming signals are generated slightly before execution
        signal_window_start = timestamp - timedelta(minutes=30)
        signal_window_end = timestamp + timedelta(minutes=30)
        
        mask = (
            (pd.to_datetime(signals_data['created_at']) >= signal_window_start) & 
            (pd.to_datetime(signals_data['created_at']) <= signal_window_end)
        )
        
        return signals_data[mask]
    
    def _process_signal(self, signal: pd.Series, current_prices: pd.Series, timestamp: datetime):
        """Process a trading signal"""
        symbol = signal['symbol']
        signal_type = SignalType(signal['signal_type'])
        
        # Get current price
        if 'close' in current_prices:
            current_price = current_prices['close']
        elif 'close_price' in current_prices:
            current_price = current_prices['close_price']
        else:
            logger.warning(f"No price data available for {symbol} at {timestamp}")
            return
        
        # Apply slippage
        if signal_type == SignalType.BUY:
            execution_price = current_price * (1 + self.slippage)
        elif signal_type == SignalType.SELL:
            execution_price = current_price * (1 - self.slippage)
        else:
            return  # HOLD signal, no action needed
        
        # Check if we already have a position
        if symbol in self.positions:
            existing_trade = self.positions[symbol]
            
            # Close existing position if signal is opposite
            if ((existing_trade.signal_type == SignalType.BUY and signal_type == SignalType.SELL) or
                (existing_trade.signal_type == SignalType.SELL and signal_type == SignalType.BUY)):
                self._close_position(symbol, execution_price, timestamp, "SIGNAL_REVERSAL")
        
        # Open new position
        if signal_type in [SignalType.BUY, SignalType.SELL]:
            self._open_position(signal, execution_price, timestamp)
    
    def _open_position(self, signal: pd.Series, execution_price: float, timestamp: datetime):
        """Open a new trading position"""
        symbol = signal['symbol']
        signal_type = SignalType(signal['signal_type'])
        
        # Calculate position size
        position_value = self.current_capital * min(
            signal.get('position_size', 0.1), 
            self.max_position_size
        )
        
        # Calculate quantity
        if signal_type == SignalType.BUY:
            quantity = position_value / execution_price
        else:  # SELL (short)
            quantity = -position_value / execution_price
        
        # Calculate commission
        commission_cost = abs(quantity * execution_price * self.commission)
        
        # Create trade
        trade = Trade(
            id=str(uuid.uuid4()),
            symbol=symbol,
            signal_type=signal_type,
            entry_price=execution_price,
            exit_price=None,
            quantity=quantity,
            entry_time=timestamp,
            exit_time=None,
            stop_loss=signal.get('stop_loss', execution_price * 0.95 if signal_type == SignalType.BUY else execution_price * 1.05),
            take_profit=signal.get('price_target', execution_price * 1.05 if signal_type == SignalType.BUY else execution_price * 0.95),
            commission=commission_cost,
            status=PositionStatus.OPEN
        )
        
        # Update capital
        self.current_capital -= commission_cost
        
        # Store position
        self.positions[symbol] = trade
        
        logger.debug(f"Opened {signal_type.value} position for {symbol} at {execution_price:.2f}")
    
    def _update_positions(self, current_prices: pd.Series, timestamp: datetime):
        """Update all open positions and check for stop loss/take profit"""
        positions_to_close = []
        
        for symbol, trade in self.positions.items():
            if trade.status != PositionStatus.OPEN:
                continue
            
            # Get current price
            if 'close' in current_prices:
                current_price = current_prices['close']
            elif 'close_price' in current_prices:
                current_price = current_prices['close_price']
            else:
                continue
            
            # Check stop loss and take profit
            should_close = False
            exit_reason = ""
            
            if trade.signal_type == SignalType.BUY:
                if current_price <= trade.stop_loss:
                    should_close = True
                    exit_reason = "STOP_LOSS"
                elif current_price >= trade.take_profit:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"
            else:  # SELL (short)
                if current_price >= trade.stop_loss:
                    should_close = True
                    exit_reason = "STOP_LOSS"
                elif current_price <= trade.take_profit:
                    should_close = True
                    exit_reason = "TAKE_PROFIT"
            
            # Check for expiration (positions held too long)
            holding_period = (timestamp - trade.entry_time).total_seconds() / 3600  # hours
            if holding_period > 168:  # 1 week maximum
                should_close = True
                exit_reason = "EXPIRATION"
            
            if should_close:
                positions_to_close.append((symbol, current_price, exit_reason))
        
        # Close positions that need to be closed
        for symbol, exit_price, exit_reason in positions_to_close:
            self._close_position(symbol, exit_price, timestamp, exit_reason)
    
    def _close_position(self, symbol: str, exit_price: float, timestamp: datetime, exit_reason: str):
        """Close a trading position"""
        if symbol not in self.positions:
            return
        
        trade = self.positions[symbol]
        
        # Apply slippage to exit price
        if trade.signal_type == SignalType.BUY:
            execution_price = exit_price * (1 - self.slippage)
        else:  # SELL (short)
            execution_price = exit_price * (1 + self.slippage)
        
        # Calculate P&L
        if trade.signal_type == SignalType.BUY:
            pnl = trade.quantity * (execution_price - trade.entry_price)
        else:  # SELL (short)
            pnl = abs(trade.quantity) * (trade.entry_price - execution_price)
        
        # Subtract commission
        exit_commission = abs(trade.quantity * execution_price * self.commission)
        pnl -= exit_commission
        
        # Calculate return percentage
        position_value = abs(trade.quantity * trade.entry_price)
        return_pct = (pnl / position_value) * 100 if position_value > 0 else 0.0
        
        # Update trade
        trade.exit_price = execution_price
        trade.exit_time = timestamp
        trade.pnl = pnl
        trade.return_pct = return_pct
        trade.holding_period = int((timestamp - trade.entry_time).total_seconds() / 3600)
        trade.status = PositionStatus.CLOSED
        trade.exit_reason = exit_reason
        trade.commission += exit_commission
        
        # Update capital
        self.current_capital += pnl
        
        # Update consecutive wins/losses
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        # Store closed trade
        self.closed_trades.append(trade)
        
        # Remove from active positions
        del self.positions[symbol]
        
        logger.debug(f"Closed {trade.signal_type.value} position for {symbol}. P&L: {pnl:.2f} ({return_pct:.2f}%)")
    
    def _close_all_positions(self, final_prices: pd.Series, timestamp: datetime, reason: str):
        """Close all remaining open positions"""
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            if 'close' in final_prices:
                final_price = final_prices['close']
            elif 'close_price' in final_prices:
                final_price = final_prices['close_price']
            else:
                # Use entry price as fallback
                final_price = self.positions[symbol].entry_price
            
            self._close_position(symbol, final_price, timestamp, reason)
    
    def _record_equity(self, timestamp: datetime):
        """Record current equity value"""
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for trade in self.positions.values():
            # This would require current price, simplified for now
            unrealized_pnl += 0  # TODO: Calculate based on current market price
        
        total_equity = self.current_capital + unrealized_pnl
        
        # Update peak and drawdown
        if total_equity > self.peak_capital:
            self.peak_capital = total_equity
        
        current_drawdown = (self.peak_capital - total_equity) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Record equity point
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'drawdown': current_drawdown,
            'cash': self.current_capital,
            'unrealized_pnl': unrealized_pnl
        })
    
    def _calculate_backtest_results(self, strategy_name: str, start_date: datetime, 
                                  end_date: datetime) -> BacktestResults:
        """Calculate comprehensive backtesting results"""
        
        # Basic metrics
        final_capital = self.current_capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        
        # Trade statistics
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for trade in self.closed_trades if trade.pnl > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # P&L statistics
        trade_returns = [trade.pnl for trade in self.closed_trades]
        winning_trades_pnl = [trade.pnl for trade in self.closed_trades if trade.pnl > 0]
        losing_trades_pnl = [trade.pnl for trade in self.closed_trades if trade.pnl <= 0]
        
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0.0
        avg_winning_trade = np.mean(winning_trades_pnl) if winning_trades_pnl else 0.0
        avg_losing_trade = np.mean(losing_trades_pnl) if losing_trades_pnl else 0.0
        
        largest_win = max(trade_returns) if trade_returns else 0.0
        largest_loss = min(trade_returns) if trade_returns else 0.0
        
        # Profit factor
        gross_profit = sum(winning_trades_pnl) if winning_trades_pnl else 0.0
        gross_loss = abs(sum(losing_trades_pnl)) if losing_trades_pnl else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk metrics
        equity_df = pd.DataFrame(self.equity_history)
        if not equity_df.empty:
            equity_df.set_index('timestamp', inplace=True)
            
            # Returns calculation
            equity_df['returns'] = equity_df['equity'].pct_change()
            daily_returns = equity_df['returns'].dropna()
            
            # Sharpe ratio (assuming 252 trading days)
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0.0
            
            # Calmar ratio
            calmar_ratio = (total_return_pct / 100) / max(self.max_drawdown, 0.01)
            
            # Volatility
            volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.0
            
            # Value at Risk (VaR)
            if len(daily_returns) > 1:
                var_95 = np.percentile(daily_returns, 5) * np.sqrt(252)
                var_99 = np.percentile(daily_returns, 1) * np.sqrt(252)
            else:
                var_95 = var_99 = 0.0
            
            # Monthly returns
            monthly_returns = self._calculate_monthly_returns(equity_df)
            
            # Drawdown analysis
            drawdown_curve = equity_df[['drawdown']].copy()
            max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown_curve)
            
        else:
            sharpe_ratio = calmar_ratio = volatility = var_95 = var_99 = 0.0
            monthly_returns = pd.DataFrame()
            drawdown_curve = pd.DataFrame()
            max_drawdown_duration = 0
        
        # Additional performance metrics
        performance_metrics = {
            'total_commission_paid': sum(trade.commission for trade in self.closed_trades),
            'avg_holding_period_hours': np.mean([trade.holding_period for trade in self.closed_trades]) if self.closed_trades else 0,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'largest_win_pct': (largest_win / self.initial_capital * 100) if largest_win > 0 else 0.0,
            'largest_loss_pct': (largest_loss / self.initial_capital * 100) if largest_loss < 0 else 0.0,
            'expectancy': avg_trade_return,
            'kelly_criterion': self._calculate_kelly_criterion() if trade_returns else 0.0
        }
        
        return BacktestResults(
            strategy_name=strategy_name,
            symbol="MULTI",  # For multi-symbol strategies
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=self.max_drawdown * 100,  # Convert to percentage
            max_drawdown_duration=max_drawdown_duration,
            volatility=volatility * 100,  # Convert to percentage
            var_95=var_95 * 100,
            var_99=var_99 * 100,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_trade_return=avg_trade_return,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            consecutive_wins=self.max_consecutive_wins,
            consecutive_losses=self.max_consecutive_losses,
            trades=self.closed_trades,
            equity_curve=equity_df if not equity_df.empty else pd.DataFrame(),
            drawdown_curve=drawdown_curve,
            monthly_returns=monthly_returns,
            performance_metrics=performance_metrics
        )
    
    def _calculate_monthly_returns(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns from equity curve"""
        if equity_df.empty:
            return pd.DataFrame()
        
        try:
            monthly_equity = equity_df['equity'].resample('M').last()
            monthly_returns = monthly_equity.pct_change().dropna()
            
            monthly_df = pd.DataFrame({
                'month': monthly_returns.index.strftime('%Y-%m'),
                'return_pct': monthly_returns.values * 100
            })
            
            return monthly_df
        except Exception as e:
            logger.warning(f"Error calculating monthly returns: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_max_drawdown_duration(self, drawdown_curve: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in days"""
        if drawdown_curve.empty:
            return 0
        
        try:
            # Find periods where drawdown > 0
            in_drawdown = drawdown_curve['drawdown'] > 0
            
            # Find the start and end of drawdown periods
            drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
            drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)
            
            max_duration = 0
            start_dates = drawdown_curve.index[drawdown_starts]
            end_dates = drawdown_curve.index[drawdown_ends]
            
            # Handle case where drawdown period extends to end of data
            if len(start_dates) > len(end_dates):
                end_dates = end_dates.append(pd.Index([drawdown_curve.index[-1]]))
            
            for start, end in zip(start_dates, end_dates):
                duration = (end - start).days
                max_duration = max(max_duration, duration)
            
            return max_duration
            
        except Exception as e:
            logger.warning(f"Error calculating max drawdown duration: {str(e)}")
            return 0
    
    def _calculate_kelly_criterion(self) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if not self.closed_trades:
            return 0.0
        
        wins = [trade.return_pct/100 for trade in self.closed_trades if trade.pnl > 0]
        losses = [abs(trade.return_pct/100) for trade in self.closed_trades if trade.pnl <= 0]
        
        if not wins or not losses:
            return 0.0
        
        win_prob = len(wins) / len(self.closed_trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 0.0
        
        kelly = win_prob - ((1 - win_prob) / (avg_win / avg_loss))
        return max(0.0, min(kelly, 1.0))  # Cap at 100%

# Signal quality analyzer
class SignalQualityAnalyzer:
    """
    Analyze and score the quality of trading signals
    """
    
    def __init__(self):
        self.signal_history = []
        self.performance_history = []
    
    def analyze_signal_quality(self, signals_df: pd.DataFrame, 
                             actual_returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze signal quality against actual market performance
        """
        analysis_results = {}
        
        if signals_df.empty or actual_returns_df.empty:
            return analysis_results
        
        try:
            # Merge signals with actual returns
            merged_data = self._merge_signals_with_returns(signals_df, actual_returns_df)
            
            if merged_data.empty:
                return analysis_results
            
            # Calculate signal accuracy metrics
            analysis_results['signal_accuracy'] = self._calculate_signal_accuracy(merged_data)
            analysis_results['directional_accuracy'] = self._calculate_directional_accuracy(merged_data)
            analysis_results['confidence_correlation'] = self._calculate_confidence_correlation(merged_data)
            analysis_results['signal_distribution'] = self._calculate_signal_distribution(signals_df)
            analysis_results['timing_analysis'] = self._calculate_timing_analysis(merged_data)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing signal quality: {str(e)}")
            return {}
    
    def _merge_signals_with_returns(self, signals_df: pd.DataFrame, 
                                  returns_df: pd.DataFrame) -> pd.DataFrame:
        """Merge signals with actual market returns"""
        # This is a simplified version - in practice, you'd need more sophisticated
        # matching based on signal timing and market data timestamps
        merged = signals_df.copy()
        
        # Add actual returns for the period following each signal
        # This would be implemented based on your specific data structure
        
        return merged
    
    def _calculate_signal_accuracy(self, merged_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various signal accuracy metrics"""
        return {
            'overall_accuracy': 0.0,  # Placeholder
            'buy_signal_accuracy': 0.0,
            'sell_signal_accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    def _calculate_directional_accuracy(self, merged_data: pd.DataFrame) -> float:
        """Calculate directional accuracy of signals"""
        # Placeholder implementation
        return 0.0
    
    def _calculate_confidence_correlation(self, merged_data: pd.DataFrame) -> float:
        """Calculate correlation between signal confidence and actual performance"""
        # Placeholder implementation
        return 0.0
    
    def _calculate_signal_distribution(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze signal distribution and characteristics"""
        if signals_df.empty:
            return {}
        
        return {
            'total_signals': len(signals_df),
            'buy_signals': len(signals_df[signals_df['signal_type'] == 'BUY']),
            'sell_signals': len(signals_df[signals_df['signal_type'] == 'SELL']),
            'hold_signals': len(signals_df[signals_df['signal_type'] == 'HOLD']),
            'avg_confidence': signals_df['confidence_score'].mean() if 'confidence_score' in signals_df else 0.0,
            'avg_signal_strength': signals_df['signal_strength'].mean() if 'signal_strength' in signals_df else 0.0
        }
    
    def _calculate_timing_analysis(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze signal timing effectiveness"""
        # Placeholder implementation
        return {
            'avg_signal_lead_time': 0.0,
            'optimal_holding_period': 0.0,
            'signal_decay_rate': 0.0
        }

# Example usage
if __name__ == "__main__":
    # Test backtesting framework
    print("Testing Advanced Backtesting Framework...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    price_data = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100
    }, index=dates)
    
    signals_data = pd.DataFrame({
        'symbol': ['AAPL'] * 10,
        'signal_type': ['BUY', 'SELL'] * 5,
        'signal_strength': np.random.uniform(0.5, 1.0, 10),
        'confidence_score': np.random.uniform(0.6, 0.9, 10),
        'position_size': [0.1] * 10,
        'stop_loss': price_data['close'].iloc[:10] * 0.95,
        'price_target': price_data['close'].iloc[:10] * 1.05,
        'created_at': dates[:10]
    })
    
    # Run backtest
    backtester = AdvancedBacktester(initial_capital=100000)
    results = backtester.run_backtest(price_data, signals_data)
    
    print(f"Total Return: {results.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2f}%")
    print(f"Win Rate: {results.win_rate:.2f}%")
    print(f"Total Trades: {results.total_trades}")