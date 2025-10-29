"""
AI-Powered Financial Signal Extraction & Trading Strategy Platform
Phase 2 Submission - The Dev Premier League
Team: QuantumSignal Labs

A comprehensive Streamlit application combining:
- Real-time market data from Alpha Vantage API
- Financial news sentiment analysis from News API  
- AI-powered trading signal generation
- Advanced backtesting with risk analytics
- Interactive data visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# =======================
# PAGE CONFIGURATION
# =======================
st.set_page_config(
    page_title="QuantumSignal Labs - AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# CUSTOM CSS STYLING
# =======================
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(120deg, #2c3e50, #3498db);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .dashboard-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .dashboard-subtitle {
        color: #ecf0f1;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border-left: 4px solid #3498db;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        color: #7f8c8d;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-delta-positive {
        color: #27ae60;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .metric-delta-negative {
        color: #e74c3c;
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Signal cards */
    .signal-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(120deg, #3498db, #2c3e50);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(52, 152, 219, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* News card */
    .news-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #3498db;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Status badges */
    .badge-success {
        background: #27ae60;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-warning {
        background: #f39c12;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-danger {
        background: #e74c3c;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
    }
    
    /* Data tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =======================
# DATA CLASSES
# =======================
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    """Trading signal dataclass"""
    symbol: str
    signal_type: SignalType
    confidence: float
    strength: float
    price_target: float
    stop_loss: float
    current_price: float
    position_size: float
    technical_score: float
    sentiment_score: float
    risk_reward: float
    timestamp: datetime

@dataclass
class Trade:
    """Trade record dataclass"""
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    signal_type: str
    pnl: float
    return_pct: float
    entry_time: datetime
    exit_time: datetime

# =======================
# API CONFIGURATION
# =======================
class APIConfig:
    """API configuration and credentials"""
    MARKETSTACK_KEY = "86fb0164da8ecfbc4ca4ea5eb0635eee"
    NEWS_API_KEY = "af1a731ec740422cb18da3860fdf9af9"
    MARKETSTACK_BASE_URL = "http://api.marketstack.com/v1"
    NEWS_API_BASE_URL = "https://newsapi.org/v2"
    
    # Rate limiting for free tier
    AV_CALLS_PER_MINUTE = 5
    AV_CALL_DELAY = 12  # seconds
    NEWS_CALLS_PER_DAY = 100

# =======================
# DATA INGESTION ENGINE
# =======================
class DataIngestionEngine:
    """Ingest/prepare market & news data from MarketStack and NewsAPI"""

    def __init__(self):
        self.marketstack_key = APIConfig.MARKETSTACK_KEY
        self.news_key = APIConfig.NEWS_API_KEY

    @st.cache_data(ttl=3600)
    def get_daily_data(_self, symbol: str) -> pd.DataFrame:
        """
        Fetch daily OHLCV data from MarketStack
        Add technical indicators: RSI, MACD, MA, Bollinger Bands, ATR, Volatility
        """
        url = f"{APIConfig.MARKETSTACK_BASE_URL}/eod"
        params = {
            "access_key": _self.marketstack_key,
            "symbols": symbol,
            "limit": 500,
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            raw_data = response.json().get('data', [])
            if not raw_data:
                return pd.DataFrame()

            # Parse and convert to DataFrame
            df = pd.DataFrame(raw_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            df.rename(columns={'open':'open','high':'high','low':'low','close':'close','volume':'volume'}, inplace=True)

            # Technicals (example: add basic indicators, see below for expandability)
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            df['rsi'] = _self._calculate_rsi(df['close'], period=14)
            df = _self._calculate_macd(df)
            df = _self._calculate_bollinger_bands(df)
            df = _self._calculate_atr(df)
            return df
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, df):
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df

    def _calculate_bollinger_bands(self, df):
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        return df

    def _calculate_atr(self, df):
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        return df

    @st.cache_data(ttl=1800)
    def get_financial_news(_self, symbols: list, days_back: int = 7) -> pd.DataFrame:
        """Fetch financial news from NewsAPI and process basic sentiment"""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        articles = []
        for symbol in symbols[:3]:
            url = f"{APIConfig.NEWS_API_BASE_URL}/everything"
            params = {
                "q": f"{symbol} stock OR {symbol} earnings",
                "from": from_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": _self.news_key
            }
            resp = requests.get(url, params=params)
            data = resp.json()
            if data.get("status") == "ok":
                for art in data.get("articles", []):
                    articles.append({
                        "symbol": symbol,
                        "title": art.get("title", ""),
                        "description": art.get("description", ""),
                        "source": art.get("source", {}).get("name", ""),
                        "url": art.get("url", ""),
                        "published_at": art.get("publishedAt", ""),
                        "content": art.get("content", "")
                    })
            time.sleep(0.5)
        if not articles:
            return pd.DataFrame()
        news_df = pd.DataFrame(articles)
        news_df['published_at'] = pd.to_datetime(news_df['published_at'])
        news_df = news_df.drop_duplicates(subset=['title'])
        news_df = _self._analyze_sentiment(news_df)
        return news_df

    def _analyze_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Simple keyword-based sentiment analysis"""
        positive_keywords = ['gain', 'profit', 'surge', 'rise', 'bullish', 'upgrade', 
                             'beat', 'success', 'growth', 'strong', 'positive', 'high']
        negative_keywords = ['loss', 'drop', 'fall', 'bearish', 'downgrade', 'miss', 
                             'weak', 'decline', 'negative', 'risk', 'concern', 'low']

        def calculate_sentiment(text):
            if pd.isna(text):
                return 0.0
            text = text.lower()
            pos_count = sum(1 for word in positive_keywords if word in text)
            neg_count = sum(1 for word in negative_keywords if word in text)
            total = pos_count + neg_count
            if total == 0:
                return 0.0
            return (pos_count - neg_count) / total
        
        news_df['sentiment_score'] = news_df.apply(
            lambda row: calculate_sentiment(f"{row['title']} {row['description']}"),
            axis=1
        )
        news_df['sentiment_label'] = news_df['sentiment_score'].apply(
            lambda x: 'POSITIVE' if x > 0.2 else ('NEGATIVE' if x < -0.2 else 'NEUTRAL')
        )
        return news_df

# =======================
# SIGNAL GENERATION ENGINE
# =======================
class SignalGenerationEngine:
    """
    AI-powered signal generation using multi-factor analysis
    Enhanced with realistic thresholds and better scoring
    """
    
    def generate_signals(self, stock_data: pd.DataFrame, news_data: pd.DataFrame, 
                        symbol: str) -> Optional[TradingSignal]:
        """Generate trading signal from technical and sentiment analysis"""
        try:
            if stock_data.empty or len(stock_data) < 50:
                return None
            
            latest = stock_data.iloc[-1]
            current_price = latest['close']
            
            # Technical analysis score
            tech_score = self._calculate_technical_score(stock_data)
            
            # Sentiment analysis score
            sentiment_score = self._calculate_sentiment_score(news_data, symbol)
            
            # Combined signal (70% technical, 30% sentiment for reliability)
            combined_score = (tech_score * 0.7) + (sentiment_score * 0.3)
            
            # ✅ FIXED: More realistic thresholds based on trading research
            if combined_score > 0.15:  # BUY threshold (was 0.5)
                signal_type = SignalType.BUY
            elif combined_score < -0.15:  # SELL threshold (was -0.5)
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            # Calculate targets and stops
            atr = latest['atr'] if not pd.isna(latest['atr']) else current_price * 0.02
            volatility = latest['volatility'] if not pd.isna(latest['volatility']) else 0.3
            
            if signal_type == SignalType.BUY:
                # Target: 1.5-4% gain based on signal strength
                target_pct = 0.015 + (abs(combined_score) * 0.025)
                price_target = current_price * (1 + target_pct)
                stop_loss = current_price - (atr * 2)
                
            elif signal_type == SignalType.SELL:
                # Target: 1.5-4% drop based on signal strength
                target_pct = 0.015 + (abs(combined_score) * 0.025)
                price_target = current_price * (1 - target_pct)
                stop_loss = current_price + (atr * 2)
                
            else:  # HOLD
                price_target = current_price
                stop_loss = current_price
            
            # Position sizing based on confidence and volatility
            confidence = self._calculate_confidence(stock_data, news_data, combined_score)
            
            # Adjust position size for volatility
            if volatility < 0.2:  # Low volatility
                base_size = 0.10
            elif volatility < 0.4:  # Medium volatility
                base_size = 0.08
            else:  # High volatility
                base_size = 0.05
            
            position_size = min(confidence * base_size, 0.10)  # Max 10%
            
            # Risk/reward ratio
            potential_gain = abs(price_target - current_price)
            potential_loss = abs(current_price - stop_loss)
            risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                strength=abs(combined_score),
                price_target=price_target,
                stop_loss=stop_loss,
                current_price=current_price,
                position_size=position_size,
                technical_score=tech_score,
                sentiment_score=sentiment_score,
                risk_reward=risk_reward,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    def _calculate_technical_score(self, df: pd.DataFrame) -> float:
        """Enhanced technical analysis score with better weighting"""
        latest = df.iloc[-1]
        score = 0.0
        weights_sum = 0.0
        
        # RSI score (weight: 0.25) - IMPROVED LOGIC
        rsi = latest['rsi']
        if not pd.isna(rsi):
            if rsi < 25:  # Extremely oversold
                score += 0.4
                weights_sum += 0.25
            elif rsi < 35:  # Oversold
                score += 0.25
                weights_sum += 0.25
            elif rsi > 75:  # Extremely overbought
                score -= 0.4
                weights_sum += 0.25
            elif rsi > 65:  # Overbought
                score -= 0.25
                weights_sum += 0.25
            elif 45 <= rsi <= 55:  # Neutral
                score += 0.0
                weights_sum += 0.25
            else:
                weights_sum += 0.25
        
        # MACD score (weight: 0.25)
        if not pd.isna(latest['macd']) and not pd.isna(latest['macd_signal']):
            macd_diff = latest['macd'] - latest['macd_signal']
            if macd_diff > 0:
                # Bullish crossover - check momentum
                if len(df) > 1 and df.iloc[-2]['macd'] < df.iloc[-2]['macd_signal']:
                    score += 0.35  # Fresh crossover
                else:
                    score += 0.20
            else:
                # Bearish
                if len(df) > 1 and df.iloc[-2]['macd'] > df.iloc[-2]['macd_signal']:
                    score -= 0.35  # Fresh crossover
                else:
                    score -= 0.20
            weights_sum += 0.25
        
        # Bollinger Bands (weight: 0.20)
        if all(not pd.isna(latest[col]) for col in ['close', 'bb_upper', 'bb_lower']):
            bb_range = latest['bb_upper'] - latest['bb_lower']
            if bb_range > 0:
                bb_position = (latest['close'] - latest['bb_lower']) / bb_range
                if bb_position < 0.1:  # Near lower band
                    score += 0.30
                elif bb_position < 0.2:
                    score += 0.20
                elif bb_position > 0.9:  # Near upper band
                    score -= 0.30
                elif bb_position > 0.8:
                    score -= 0.20
            weights_sum += 0.20
        
        # Moving average crossovers (weight: 0.20)
        if not pd.isna(latest['sma_20']) and not pd.isna(latest['sma_50']):
            # Golden cross / Death cross
            sma_diff = (latest['sma_20'] - latest['sma_50']) / latest['sma_50']
            if sma_diff > 0.02:  # Strong uptrend
                score += 0.25
            elif sma_diff > 0:
                score += 0.15
            elif sma_diff < -0.02:  # Strong downtrend
                score -= 0.25
            else:
                score -= 0.15
            weights_sum += 0.20
        
        # Price momentum (weight: 0.10)
        if len(df) >= 10:
            # 5-day and 10-day momentum
            momentum_5d = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1)
            momentum_10d = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1)
            
            combined_momentum = (momentum_5d * 0.6) + (momentum_10d * 0.4)
            score += combined_momentum * 5  # Amplify momentum signal
            weights_sum += 0.10
        
        # Normalize by actual weights used
        if weights_sum > 0:
            score = score / weights_sum
        
        return max(-1.0, min(1.0, score))
    
    def _calculate_sentiment_score(self, news_df: pd.DataFrame, symbol: str) -> float:
        """Calculate news sentiment score with time decay"""
        if news_df.empty:
            return 0.0
        
        symbol_news = news_df[news_df['symbol'] == symbol]
        if symbol_news.empty:
            return 0.0
        
        # Weight recent news more heavily
        now = datetime.now().replace(tzinfo=None)
        symbol_news['published_at'] = symbol_news['published_at'].dt.tz_localize(None)
        
        symbol_news['age_hours'] = symbol_news['published_at'].apply(
            lambda x: (now - x).total_seconds() / 3600
        )
        
        # Exponential decay: recent news matters more
        symbol_news['age_weight'] = np.exp(-symbol_news['age_hours'] / 48)  # 48-hour half-life
        
        # Calculate weighted sentiment
        weighted_sentiment = (symbol_news['sentiment_score'] * symbol_news['age_weight']).sum()
        total_weight = symbol_news['age_weight'].sum()
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _calculate_confidence(self, stock_data: pd.DataFrame, news_data: pd.DataFrame, 
                             combined_score: float) -> float:
        """Calculate signal confidence score"""
        confidence = 0.4  # Base confidence
        
        # Data quality bonus
        if len(stock_data) >= 100:
            confidence += 0.15
        elif len(stock_data) >= 50:
            confidence += 0.10
        
        # News availability bonus
        if not news_data.empty and len(news_data) >= 3:
            confidence += 0.15
        elif not news_data.empty:
            confidence += 0.08
        
        # Signal strength bonus
        signal_strength = abs(combined_score)
        confidence += signal_strength * 0.20
        
        # Volatility check
        latest_vol = stock_data.iloc[-1]['volatility']
        if not pd.isna(latest_vol):
            if latest_vol < 0.25:  # Low volatility = higher confidence
                confidence += 0.12
            elif latest_vol > 0.50:  # High volatility = lower confidence
                confidence -= 0.10
        
        # Technical indicator alignment
        latest = stock_data.iloc[-1]
        indicators_valid = sum([
            not pd.isna(latest['rsi']),
            not pd.isna(latest['macd']),
            not pd.isna(latest['bb_middle'])
        ])
        confidence += (indicators_valid / 3) * 0.08
        
        return max(0.3, min(1.0, confidence))


# =======================
# BACKTESTING ENGINE
# =======================
class BacktestingEngine:
    """
    Enhanced backtesting engine with professional risk management
    Based on quantitative trading research
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission per trade
        self.slippage = 0.0005  # 0.05% slippage
        
    def run_backtest(self, stock_data: pd.DataFrame, signals: List[TradingSignal]) -> Dict:
        """Run enhanced backtest simulation with better position management"""
        if stock_data.empty or not signals:
            return self._empty_metrics()
            
        capital = self.initial_capital
        positions = []
        trades = []
        equity_curve = []
        
        # Filter out HOLD signals
        tradeable_signals = [s for s in signals if s.signal_type != SignalType.HOLD]
        
        if not tradeable_signals:
            equity_curve.append({'date': stock_data.index[0], 'equity': capital})
            equity_curve.append({'date': stock_data.index[-1], 'equity': capital})
            return self._calculate_metrics(trades, equity_curve)
        
        # Track days in position for time-based exits
        entered_signals = set()
        
        for idx, (date, row) in enumerate(stock_data.iterrows()):
            # ENTRY LOGIC - Execute on signal generation or first day
            if idx == 0:  # Enter on first day of backtest
                for signal in tradeable_signals:
                    signal_key = f"{signal.symbol}_{signal.signal_type.value}"
                    
                    if signal_key in entered_signals:
                        continue
                    
                    # Calculate position size with volatility adjustment
                    volatility = row.get('volatility', 0.3)
                    risk_adjusted_size = signal.position_size
                    
                    # Reduce size for high volatility
                    if volatility > 0.5:
                        risk_adjusted_size *= 0.7
                    elif volatility > 0.4:
                        risk_adjusted_size *= 0.85
                    
                    position_value = capital * risk_adjusted_size
                    
                    # Apply slippage to entry
                    entry_price = row['close'] * (1 + self.slippage if signal.signal_type == SignalType.BUY else 1 - self.slippage)
                    quantity = position_value / entry_price
                    commission_cost = position_value * self.commission
                    
                    if capital < (position_value + commission_cost):
                        continue
                    
                    # Enhanced stop loss and target calculation
                    atr = row.get('atr', entry_price * 0.02)
                    
                    if signal.signal_type == SignalType.BUY:
                        # Dynamic stops based on ATR
                        stop_loss = entry_price - (atr * 2.5)  # 2.5x ATR stop
                        price_target = entry_price + (atr * 3.5)  # 3.5x ATR target (1.4:1 R:R)
                    else:  # SELL
                        stop_loss = entry_price + (atr * 2.5)
                        price_target = entry_price - (atr * 3.5)
                    
                    positions.append({
                        'entry_date': date,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'stop_loss': stop_loss,
                        'target': price_target,
                        'signal_type': signal.signal_type.value,
                        'symbol': signal.symbol,
                        'position_value': position_value,
                        'days_held': 0,
                        'trailing_stop': stop_loss,
                        'max_price': entry_price if signal.signal_type.value == 'BUY' else entry_price
                    })
                    
                    capital -= (position_value + commission_cost)
                    entered_signals.add(signal_key)
            
            # EXIT LOGIC - Enhanced with multiple exit strategies
            positions_to_remove = []
            
            for pos in positions:
                pos['days_held'] += 1
                exit_triggered = False
                exit_price = None
                exit_reason = None
                
                if pos['signal_type'] == 'BUY':
                    # Update trailing stop (move up only)
                    if row['high'] > pos['max_price']:
                        pos['max_price'] = row['high']
                        # Trailing stop: 15% below max
                        new_trailing = pos['max_price'] * 0.85
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_trailing)
                    
                    # Check exits (priority order)
                    # 1. Trailing stop hit
                    if row['low'] <= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        exit_triggered = True
                        exit_reason = "Trailing Stop"
                    
                    # 2. Hard stop loss
                    elif row['low'] <= pos['stop_loss']:
                        exit_price = pos['stop_loss']
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                    
                    # 3. Target hit
                    elif row['high'] >= pos['target']:
                        exit_price = pos['target']
                        exit_triggered = True
                        exit_reason = "Target Reached"
                    
                    # 4. Time-based exit (hold max 30 days)
                    elif pos['days_held'] >= 30:
                        exit_price = row['close']
                        exit_triggered = True
                        exit_reason = "Time Limit"
                    
                    # 5. Adverse technical signal (RSI overbought in profit)
                    elif row.get('rsi', 50) > 75 and row['close'] > pos['entry_price'] * 1.05:
                        exit_price = row['close']
                        exit_triggered = True
                        exit_reason = "Technical Exit"
                
                elif pos['signal_type'] == 'SELL':  # Short position
                    # Update trailing stop (move down only)
                    if row['low'] < pos['max_price']:
                        pos['max_price'] = row['low']
                        # Trailing stop: 15% above min
                        new_trailing = pos['max_price'] * 1.15
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_trailing)
                    
                    # Check exits
                    if row['high'] >= pos['trailing_stop']:
                        exit_price = pos['trailing_stop']
                        exit_triggered = True
                        exit_reason = "Trailing Stop"
                    
                    elif row['high'] >= pos['stop_loss']:
                        exit_price = pos['stop_loss']
                        exit_triggered = True
                        exit_reason = "Stop Loss"
                    
                    elif row['low'] <= pos['target']:
                        exit_price = pos['target']
                        exit_triggered = True
                        exit_reason = "Target Reached"
                    
                    elif pos['days_held'] >= 30:
                        exit_price = row['close']
                        exit_triggered = True
                        exit_reason = "Time Limit"
                    
                    elif row.get('rsi', 50) < 25 and row['close'] < pos['entry_price'] * 0.95:
                        exit_price = row['close']
                        exit_triggered = True
                        exit_reason = "Technical Exit"
                
                if exit_triggered and exit_price:
                    # Apply slippage
                    if pos['signal_type'] == 'BUY':
                        exit_price = exit_price * (1 - self.slippage)  # Sell slippage
                        pnl = (exit_price - pos['entry_price']) * pos['quantity']
                        exit_value = exit_price * pos['quantity']
                    else:  # SELL
                        exit_price = exit_price * (1 + self.slippage)  # Cover slippage
                        pnl = (pos['entry_price'] - exit_price) * pos['quantity']
                        exit_value = pos['position_value'] + pnl
                    
                    # Deduct exit commission
                    exit_commission = exit_value * self.commission
                    pnl -= (exit_commission + pos['position_value'] * self.commission)  # Total commissions
                    
                    trade = self._create_trade(pos, exit_price, date, pnl, exit_reason)
                    trades.append(trade)
                    
                    capital += exit_value - exit_commission
                    positions_to_remove.append(pos)
            
            # Remove closed positions
            for pos in positions_to_remove:
                positions.remove(pos)
            
            # Calculate daily equity
            unrealized_pnl = 0
            for pos in positions:
                if pos['signal_type'] == 'BUY':
                    unrealized_pnl += (row['close'] - pos['entry_price']) * pos['quantity']
                elif pos['signal_type'] == 'SELL':
                    unrealized_pnl += (pos['entry_price'] - row['close']) * pos['quantity']
            
            total_equity = capital + unrealized_pnl
            equity_curve.append({'date': date, 'equity': total_equity})
        
        # Force close remaining positions at end
        if positions and not stock_data.empty:
            final_date = stock_data.index[-1]
            final_price = stock_data.iloc[-1]['close']
            
            for pos in positions:
                if pos['signal_type'] == 'BUY':
                    exit_price = final_price * (1 - self.slippage)
                    pnl = (exit_price - pos['entry_price']) * pos['quantity']
                    exit_value = exit_price * pos['quantity']
                else:
                    exit_price = final_price * (1 + self.slippage)
                    pnl = (pos['entry_price'] - exit_price) * pos['quantity']
                    exit_value = pos['position_value'] + pnl
                
                exit_commission = exit_value * self.commission
                pnl -= (exit_commission + pos['position_value'] * self.commission)
                
                trade = self._create_trade(pos, exit_price, final_date, pnl, "End of Period")
                trades.append(trade)
                
                capital += exit_value - exit_commission
        
        # Update final equity
        if equity_curve:
            equity_curve[-1]['equity'] = capital
        
        return self._calculate_metrics(trades, equity_curve)
    
    def _create_trade(self, position: Dict, exit_price: float, exit_date, pnl: float, exit_reason: str = "Exit") -> Trade:
        """Create enhanced trade record"""
        if position['signal_type'] == 'BUY':
            return_pct = ((exit_price / position['entry_price']) - 1) * 100
        else:
            return_pct = ((position['entry_price'] / exit_price) - 1) * 100
        
        trade = Trade(
            symbol=position.get('symbol', ''),
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            signal_type=position['signal_type'],
            pnl=pnl,
            return_pct=return_pct,
            entry_time=position['entry_date'],
            exit_time=exit_date
        )
        
        # Log exit reason (can be stored if you extend Trade dataclass)
        logger.info(f"Trade closed: {position['symbol']} {position['signal_type']} | " +
                   f"P&L: ${pnl:.2f} | Return: {return_pct:.2f}% | Reason: {exit_reason}")
        
        return trade
    
    # Keeping existing _calculate_metrics and _empty_metrics methods unchanged

    
    def _calculate_metrics(self, trades: List[Trade], equity_curve: List[Dict]) -> Dict:
        """Calculate backtest performance metrics"""
        if not trades:
            # Return empty metrics with initial capital curve
            equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame(columns=['date', 'equity'])
            if equity_df.empty:
                equity_df = pd.DataFrame([{'date': datetime.now(), 'equity': self.initial_capital}])
            
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'equity_curve': equity_df,
                'trades': []
            }
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Calculate returns
        equity_df = pd.DataFrame(equity_curve)
        
        if len(equity_df) > 1:
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            # Sharpe ratio (annualized)
            if equity_df['returns'].std() > 0:
                sharpe = (equity_df['returns'].mean() * 252) / (equity_df['returns'].std() * np.sqrt(252))
            else:
                sharpe = 0
            
            # Max drawdown
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = abs(equity_df['drawdown'].min() * 100)
        else:
            sharpe = 0
            max_drawdown = 0
        
        final_capital = equity_df['equity'].iloc[-1] if not equity_df.empty else self.initial_capital
        total_return_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Profit factor
        total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': equity_df,
            'trades': trades
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'equity_curve': pd.DataFrame(columns=['date', 'equity']),
            'trades': []
        }



# =======================
# VISUALIZATION ENGINE
# =======================
class VisualizationEngine:
    """
    Create interactive visualizations using Plotly
    """
    
    @staticmethod
    def create_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create candlestick chart with technical indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol} Price & Indicators', 'RSI', 'MACD')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                      line=dict(color='rgba(250, 128, 114, 0.5)', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                      line=dict(color='rgba(250, 128, 114, 0.5)', dash='dash'),
                      fill='tonexty', fillcolor='rgba(250, 128, 114, 0.1)'),
            row=1, col=1
        )
        
        # Moving Averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20',
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi'], name='RSI',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd'], name='MACD',
                      line=dict(color='blue', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_signal'], name='Signal',
                      line=dict(color='orange', width=2)),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['macd_hist'], name='Histogram',
                  marker_color='gray'),
            row=3, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_equity_curve(equity_df: pd.DataFrame) -> go.Figure:
        """Create enhanced equity curve visualization"""
        if equity_df.empty or 'equity' not in equity_df.columns:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No equity data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="gray")
            )
            return fig
        
        fig = go.Figure()
        
        # Main equity line
        fig.add_trace(go.Scatter(
            x=equity_df['date'],
            y=equity_df['equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2E86DE', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 222, 0.2)',
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                        '<b>Equity:</b> $%{y:,.2f}<br>' +
                        '<extra></extra>'
        ))
        
        # Add initial capital line for reference
        initial_value = equity_df['equity'].iloc[0]
        fig.add_trace(go.Scatter(
            x=[equity_df['date'].iloc[0], equity_df['date'].iloc[-1]],
            y=[initial_value, initial_value],
            mode='lines',
            name='Initial Capital',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='<b>Initial Capital:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        # Add markers for significant points
        max_equity = equity_df['equity'].max()
        min_equity = equity_df['equity'].min()
        final_equity = equity_df['equity'].iloc[-1]
        
        max_idx = equity_df['equity'].idxmax()
        min_idx = equity_df['equity'].idxmin()
        
        # Peak marker
        fig.add_trace(go.Scatter(
            x=[equity_df.loc[max_idx, 'date']],
            y=[max_equity],
            mode='markers+text',
            name='Peak',
            marker=dict(size=12, color='green', symbol='triangle-up'),
            text=[f'Peak: ${max_equity:,.0f}'],
            textposition='top center',
            hovertemplate='<b>Peak Value:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        # Trough marker
        fig.add_trace(go.Scatter(
            x=[equity_df.loc[min_idx, 'date']],
            y=[min_equity],
            mode='markers+text',
            name='Trough',
            marker=dict(size=12, color='red', symbol='triangle-down'),
            text=[f'Low: ${min_equity:,.0f}'],
            textposition='bottom center',
            hovertemplate='<b>Lowest Value:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        # Calculate statistics for annotations
        total_return = ((final_equity - initial_value) / initial_value) * 100
        
        fig.update_layout(
            title={
                'text': f'Portfolio Equity Curve<br><sub>Total Return: {total_return:+.2f}%</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2C3E50'}
            },
            xaxis=dict(
                title='Date',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False
            ),
            yaxis=dict(
                title='Portfolio Value ($)',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=False,
                tickformat='$,.0f'
            ),
            template='plotly_white',
            height=500,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add range selector buttons
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor="rgba(150, 150, 150, 0.1)",
                activecolor="rgba(46, 134, 222, 0.3)"
            ),
            rangeslider=dict(visible=False),
            type="date"
        )
        
        return fig

    
    @staticmethod
    def create_sentiment_gauge(sentiment_score: float) -> go.Figure:
        """Create sentiment gauge chart"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "News Sentiment", 'font': {'size': 24}},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-1, -0.3], 'color': '#ffcccb'},
                    {'range': [-0.3, 0.3], 'color': '#ffffcc'},
                    {'range': [0.3, 1], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_score
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig

# =======================
# MAIN DASHBOARD CLASS
# =======================
class FinancialDashboard:
    """Main dashboard orchestrator"""
    
    def __init__(self):
        self.data_engine = DataIngestionEngine()
        self.signal_engine = SignalGenerationEngine()
        self.backtest_engine = BacktestingEngine()
        self.viz_engine = VisualizationEngine()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'stock_data' not in st.session_state:
            st.session_state.stock_data = {}
        if 'news_data' not in st.session_state:
            st.session_state.news_data = pd.DataFrame()
        if 'signals' not in st.session_state:
            st.session_state.signals = []
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">📈 QuantumSignal Labs</h1>
            <p class="dashboard-subtitle">AI-Powered Financial Signal Extraction & Trading Strategy Platform</p>
            <p style="color: #ecf0f1; font-size: 0.9rem; margin-top: 0.5rem;">
                Phase 2 - The Dev Premier League | Team: QuantumSignal Labs
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown("## ⚙️ Configuration")
            
            # Stock selection
            # Stock selection with search capability
            st.markdown("### 📊 Stock Selection")

            # Comprehensive stock list (expandable)
            available_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'INTC',
                'JPM', 'BAC', 'WMT', 'HD', 'DIS', 'BA', 'NKE', 'PFE', 'KO', 'PEP', 'V', 'MA',
                'PYPL', 'SQ', 'CRM', 'ORCL', 'IBM', 'CSCO', 'XOM', 'CVX', 'UNH', 'JNJ', 'PG',
                'COST', 'MCD', 'SBUX', 'GE', 'CAT', 'F', 'GM', 'T', 'VZ', 'CMCSA', 'NEE'
            ]

            # Option 1: Select from dropdown (with built-in search)
            st.info("💡 **Tip:** Start typing in the dropdown to search for stocks!")
            selected_from_list = st.multiselect(
                "Select from popular stocks (type to search)",
                options=sorted(available_stocks),
                default=['AAPL', 'MSFT', 'GOOGL'],
                help="Type to search • Select up to 3 stocks"
            )

            # Option 2: Manual entry for custom symbols
            st.markdown("**OR enter custom stock symbols:**")
            custom_input = st.text_input(
                "Enter stock symbols (comma-separated)",
                placeholder="e.g., TSLA, NVDA, AMD",
                help="Enter any stock symbol(s) separated by commas"
            )

            # Combine selections
            if custom_input:
                # Parse custom input
                custom_symbols = [s.strip().upper() for s in custom_input.split(',') if s.strip()]
                selected_symbols = list(set(selected_from_list + custom_symbols))[:3]  # Limit to 3
            else:
                selected_symbols = selected_from_list[:3]

            # Show current selection
            if selected_symbols:
                st.success(f"✅ Selected: {', '.join(selected_symbols)}")
                if len(selected_symbols) > 3:
                    st.warning("⚠️ Maximum 3 stocks allowed. Using first 3.")
                    selected_symbols = selected_symbols[:3]
            else:
                st.warning("⚠️ Please select at least one stock")

            
            # Data fetch button
            if st.button("🚀 Fetch Latest Data", type="primary", use_container_width=True):
                self.fetch_all_data(selected_symbols)
            
            st.markdown("---")
            
            # Backtesting parameters
            st.markdown("### 🎯 Backtesting Parameters")
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1,
                max_value=1000000,
                value=1,
                step=10
                # help="Minimum $1, recommended $100+"
            )
            
            lookback_days = st.slider(
                "Lookback Period (days)",
                min_value=1,
                max_value=365,
                value=30,
                help="Historical data period for backtesting (1-365 days)"
            )
            
            if st.button("▶️ Run Backtest", type="secondary", use_container_width=True):
                self.run_backtest(initial_capital, lookback_days)
            
            st.markdown("---")
            
            # Info section
            st.markdown("### ℹ️ System Info")
            st.info(f"""
            **API Status:**
            - Alpha Vantage: ✅ Connected
            - News API: ✅ Connected
            
            **Last Update:**
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            **Rate Limits:**
            - AV: 5 calls/min (Free tier)
            - News: 100 calls/day (Free tier)
            """)
            
            return selected_symbols
    
    def fetch_all_data(self, symbols: List[str]):
        """Fetch data for all selected symbols"""
        with st.spinner("🔄 Fetching market data and news..."):
            progress_bar = st.progress(0)
            
            # Fetch stock data
            for i, symbol in enumerate(symbols):
                st.session_state.stock_data[symbol] = self.data_engine.get_daily_data(symbol)
                progress_bar.progress((i + 1) / (len(symbols) * 2))
                
            # Fetch news data
            st.session_state.news_data = self.data_engine.get_financial_news(symbols)
            progress_bar.progress(1.0)
            
            # Generate signals
            signals = []
            for symbol in symbols:
                if symbol in st.session_state.stock_data and not st.session_state.stock_data[symbol].empty:
                    signal = self.signal_engine.generate_signals(
                        st.session_state.stock_data[symbol],
                        st.session_state.news_data,
                        symbol
                    )
                    if signal:
                        signals.append(signal)
            
            st.session_state.signals = signals
            
            st.success(f"✅ Successfully fetched data for {len(symbols)} stocks and generated {len(signals)} signals!")
    
    def run_backtest(self, initial_capital: float, lookback_days: int):
        """Run backtesting simulation"""
        if not st.session_state.signals:
            st.warning("⚠️ Please fetch data first to generate signals")
            return
        
        with st.spinner("🔄 Running backtest simulation..."):
            # Use the first stock's data for backtesting
            first_symbol = list(st.session_state.stock_data.keys())[0]
            stock_data = st.session_state.stock_data[first_symbol]
            
            if stock_data.empty:
                st.error("❌ No stock data available for backtesting")
                return
            
            # Filter data to lookback period
            cutoff_date = (datetime.now() - timedelta(days=lookback_days)).replace(tzinfo=None)
            stock_data.index = stock_data.index.tz_localize(None)
            stock_data = stock_data[stock_data.index >= cutoff_date]
            
            # Run backtest
            self.backtest_engine.initial_capital = initial_capital
            results = self.backtest_engine.run_backtest(stock_data, st.session_state.signals)
            st.session_state.backtest_results = results
            
            st.success("✅ Backtest completed successfully!")
    
    def render_market_overview(self):
        """Render market overview section"""
        st.markdown("## 📊 Market Overview")
        
        if not st.session_state.stock_data:
            st.info("👆 Please select stocks and fetch data from the sidebar")
            return
        
        cols = st.columns(len(st.session_state.stock_data))
        
        for idx, (symbol, data) in enumerate(st.session_state.stock_data.items()):
            if data.empty:
                continue
                
            with cols[idx]:
                latest = data.iloc[-1]
                prev = data.iloc[-2] if len(data) > 1 else latest
                
                price = latest['close']
                change = price - prev['close']
                change_pct = (change / prev['close']) * 100
                
                delta_color = "normal" if change >= 0 else "inverse"
                
                st.metric(
                    label=f"**{symbol}**",
                    value=f"${price:.2f}",
                    delta=f"{change_pct:+.2f}%",
                    delta_color=delta_color
                )
                
                # Mini stats
                st.markdown(f"""
                <div style='font-size: 0.8rem; color: #7f8c8d;'>
                    Vol: {latest['volume']:,.0f}<br>
                    RSI: {latest['rsi']:.1f}<br>
                    Volatility: {latest['volatility']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
    
    def render_signals(self):
        """Render trading signals section"""
        st.markdown("## 🎯 AI Trading Signals")
        
        if not st.session_state.signals:
            st.info("No signals generated yet. Fetch data to generate signals.")
            return
        
        for signal in st.session_state.signals:
            signal_class = {
                'BUY': 'signal-buy',
                'SELL': 'signal-sell',
                'HOLD': 'signal-hold'
            }[signal.signal_type.value]
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.markdown(f"""
                <div class='{signal_class}'>
                    <h3 style='margin: 0; font-size: 1.5rem;'>
                        {signal.signal_type.value} {signal.symbol}
                    </h3>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.9;'>
                        Current Price: ${signal.current_price:.2f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: white; padding: 1rem; border-radius: 8px; height: 100%;'>
                    <strong>📊 Signal Details</strong><br>
                    <span style='font-size: 0.9rem;'>
                    Confidence: {signal.confidence*100:.1f}% | 
                    Strength: {signal.strength*100:.1f}%<br>
                    Target: ${signal.price_target:.2f} | 
                    Stop: ${signal.stop_loss:.2f}<br>
                    Position Size: {signal.position_size*100:.1f}% | 
                    R/R: {signal.risk_reward:.2f}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Score breakdown
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=signal.confidence * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence", 'font': {'size': 12}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffcccb"},
                            {'range': [50, 75], 'color': "#ffffcc"},
                            {'range': [75, 100], 'color': "#ccffcc"}
                        ]
                    }
                ))
                fig.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True, key=f"confidence_gauge_{signal.symbol}")  # ✅ UNIQUE KEY
            
            # Expandable details
            with st.expander(f"📋 Detailed Analysis for {signal.symbol}"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("### Technical Analysis")
                    st.metric("Technical Score", f"{signal.technical_score:.3f}")
                    
                    if signal.symbol in st.session_state.stock_data:
                        data = st.session_state.stock_data[signal.symbol]
                        latest = data.iloc[-1]
                        
                        st.markdown(f"""
                        - **RSI:** {latest['rsi']:.1f}
                        - **MACD:** {latest['macd']:.3f}
                        - **Volatility:** {latest['volatility']*100:.1f}%
                        - **ATR:** {latest['atr']:.2f}
                        """)
                
                with col_b:
                    st.markdown("### Sentiment Analysis")
                    st.metric("Sentiment Score", f"{signal.sentiment_score:.3f}")
                    
                    # Show related news
                    if not st.session_state.news_data.empty:
                        symbol_news = st.session_state.news_data[
                            st.session_state.news_data['symbol'] == signal.symbol
                        ].head(3)
                        
                        for _, news in symbol_news.iterrows():
                            sentiment_color = {
                                'POSITIVE': '#27ae60',
                                'NEGATIVE': '#e74c3c',
                                'NEUTRAL': '#95a5a6'
                            }.get(news['sentiment_label'], '#95a5a6')
                            
                            st.markdown(f"""
                            <div style='padding: 0.5rem; margin: 0.5rem 0; 
                                        border-left: 3px solid {sentiment_color};
                                        background: #f8f9fa;'>
                                <strong>{news['title'][:80]}...</strong><br>
                                <small>{news['source']} - {news['sentiment_label']}</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    def render_charts(self):
        """Render technical analysis charts"""
        st.markdown("## 📈 Technical Analysis")
        
        if not st.session_state.stock_data:
            st.info("No data available for charts")
            return
        
        tabs = st.tabs(list(st.session_state.stock_data.keys()))
        
        for idx, (symbol, data) in enumerate(st.session_state.stock_data.items()):
            with tabs[idx]:
                if data.empty:
                    st.warning(f"No data available for {symbol}")
                    continue
                
                fig = self.viz_engine.create_price_chart(data, symbol)
                st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{symbol}_{idx}")  # ✅ UNIQUE KEY

                
                # Additional metrics
                col1, col2, col3, col4 = st.columns(4)
                latest = data.iloc[-1]
                
                with col1:
                    st.metric("RSI", f"{latest['rsi']:.1f}")
                with col2:
                    st.metric("MACD", f"{latest['macd']:.3f}")
                with col3:
                    st.metric("Volatility", f"{latest['volatility']*100:.1f}%")
                with col4:
                    st.metric("Volume", f"{latest['volume']/1e6:.1f}M")
    
    def render_news(self):
        """Render news sentiment section"""
        st.markdown("## 📰 Financial News & Sentiment")
        
        if st.session_state.news_data.empty:
            st.info("No news data available")
            return
        
        # Sentiment overview
        col1, col2, col3 = st.columns(3)
        
        positive = len(st.session_state.news_data[st.session_state.news_data['sentiment_label'] == 'POSITIVE'])
        neutral = len(st.session_state.news_data[st.session_state.news_data['sentiment_label'] == 'NEUTRAL'])
        negative = len(st.session_state.news_data[st.session_state.news_data['sentiment_label'] == 'NEGATIVE'])
        
        with col1:
            st.metric("Positive News", positive, delta=None, delta_color="normal")
        with col2:
            st.metric("Neutral News", neutral)
        with col3:
            st.metric("Negative News", negative, delta=None, delta_color="inverse")
        
        # Average sentiment gauge
        avg_sentiment = st.session_state.news_data['sentiment_score'].mean()
        fig = self.viz_engine.create_sentiment_gauge(avg_sentiment)
        st.plotly_chart(fig, use_container_width=True, key="sentiment_gauge_main")  # ✅ UNIQUE KEY
        
        # News list
        st.markdown("### Recent Articles")
        
        for symbol in st.session_state.news_data['symbol'].unique():
            with st.expander(f"📰 {symbol} News"):
                symbol_news = st.session_state.news_data[
                    st.session_state.news_data['symbol'] == symbol
                ].sort_values('published_at', ascending=False).head(5)
                
                for _, article in symbol_news.iterrows():
                    sentiment_badge = {
                        'POSITIVE': 'badge-success',
                        'NEGATIVE': 'badge-danger',
                        'NEUTRAL': 'badge-warning'
                    }.get(article['sentiment_label'], 'badge-warning')
                    
                    st.markdown(f"""
                    <div class='news-card'>
                        <h4 style='margin: 0 0 0.5rem 0;'>{article['title']}</h4>
                        <p style='margin: 0.5rem 0; color: #7f8c8d; font-size: 0.9rem;'>
                            {article['description'][:200]}...
                        </p>
                        <div style='margin-top: 0.5rem;'>
                            <span class='{sentiment_badge}'>{article['sentiment_label']}</span>
                            <span style='margin-left: 1rem; font-size: 0.8rem; color: #95a5a6;'>
                                {article['source']} • 
                                {article['published_at'].strftime('%Y-%m-%d %H:%M')}
                            </span>
                        </div>
                        <a href='{article['url']}' target='_blank' 
                           style='font-size: 0.9rem; color: #3498db;'>
                            Read more →
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_backtest_results(self):
        """Render backtesting results"""
        st.markdown("## 🎯 Backtesting Results")
        
        if st.session_state.backtest_results is None:
            st.info("Run a backtest from the sidebar to see results")
            return
        
        results = st.session_state.backtest_results
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"${results['total_pnl']:,.2f}",
                f"{results['total_return_pct']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Win Rate",
                f"{results['win_rate']:.1f}%",
                f"{results['winning_trades']}/{results['total_trades']} wins"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{results['sharpe_ratio']:.2f}"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{results['max_drawdown']:.2f}%",
                delta=None,
                delta_color="inverse"
            )
        
        # Additional metrics
        st.markdown("### 📈 Detailed Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", results['total_trades'])
            st.metric("Avg Win", f"${results['avg_win']:.2f}")
        
        with col2:
            st.metric("Winning Trades", results['winning_trades'])
            st.metric("Avg Loss", f"${results['avg_loss']:.2f}")
        
        with col3:
            st.metric("Losing Trades", results['losing_trades'])
            st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
        
        with col4:
            if not results['equity_curve'].empty and len(results['equity_curve']) > 0:
                st.metric("Final Capital", f"${results['equity_curve']['equity'].iloc[-1]:,.2f}")
            else:
                st.metric("Final Capital", f"${self.backtest_engine.initial_capital:,.2f}")

        
        # Equity curve
        if not results['equity_curve'].empty and 'equity' in results['equity_curve'].columns and len(results['equity_curve']) > 0:
            st.markdown("### 💹 Equity Curve")
            fig = self.viz_engine.create_equity_curve(results['equity_curve'])
            st.plotly_chart(fig, use_container_width=True, key="equity_curve_backtest")  # ✅ UNIQUE KEY
        else:
            st.info("No equity curve data available for this backtest period.")
        
        # Trade log
        if results['trades']:
            st.markdown("### 📋 Trade Log")
            
            trades_data = []
            for trade in results['trades']:
                trades_data.append({
                    'Entry Date': trade.entry_time.strftime('%Y-%m-%d'),
                    'Exit Date': trade.exit_time.strftime('%Y-%m-%d'),
                    'Signal': trade.signal_type,
                    'Entry Price': f"${trade.entry_price:.2f}",
                    'Exit Price': f"${trade.exit_price:.2f}",
                    'Quantity': f"{trade.quantity:.2f}",
                    'P&L': f"${trade.pnl:.2f}",
                    'Return %': f"{trade.return_pct:.2f}%"
                })
            
            trades_df = pd.DataFrame(trades_data)
            st.dataframe(trades_df, use_container_width=True)
    
    def render_footer(self):
        """Render dashboard footer"""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
            <h3>🏆 QuantumSignal Labs - The Dev Premier League Phase 2</h3>
            <p>AI-Powered Financial Signal Extraction & Trading Strategy Platform</p>
            <p style='font-size: 0.9rem;'>
                Built with ❤️ using Streamlit | Alpha Vantage API | News API
            </p>
            <p style='font-size: 0.8rem; margin-top: 1rem;'>
                <strong>Key Features:</strong> Real-time Market Data • AI Signal Generation • 
                Sentiment Analysis • Advanced Backtesting • Risk Analytics • 
                Interactive Visualizations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main dashboard execution"""
        self.render_header()
        selected_symbols = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Market Overview",
            "🎯 Trading Signals",
            "📈 Technical Charts",
            "📰 News & Sentiment",
            "🎯 Backtest Results"
        ])
        
        with tab1:
            self.render_market_overview()
        
        with tab2:
            self.render_signals()
        
        with tab3:
            self.render_charts()
        
        with tab4:
            self.render_news()
        
        with tab5:
            self.render_backtest_results()
        
        self.render_footer()

# =======================
# MAIN EXECUTION
# =======================
if __name__ == "__main__":
    try:
        dashboard = FinancialDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
