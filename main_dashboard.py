"""
Main Streamlit Dashboard for Financial Signal Extraction & Trading Strategy Validation
Advanced interactive dashboard with real-time data visualization and backtesting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Optional, Any
import logging

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our custom modules
from data_ingestion import DataIngestionEngine, NewsEnhancer
from snowflake_manager import SnowflakeManager, get_snowflake_config
from backtesting_engine import AdvancedBacktester, SignalQualityAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Financial Signal Extraction & Trading Strategy Validator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .success-metric {
        border-left-color: #2ca02c;
    }
    
    .warning-metric {
        border-left-color: #ff7f0e;
    }
    
    .danger-metric {
        border-left-color: #d62728;
    }
    
    .signal-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .buy-signal {
        border-left: 5px solid #2ca02c;
    }
    
    .sell-signal {
        border-left: 5px solid #d62728;
    }
    
    .hold-signal {
        border-left: 5px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

class FinancialDashboard:
    """
    Main dashboard class for financial signal extraction and trading validation
    """
    
    def __init__(self):
        self.initialize_components()
        self.initialize_session_state()
    
    def initialize_components(self):
        """Initialize data components"""
        try:
            # API keys
            self.av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            self.news_key = os.getenv('NEWS_API_KEY')
            
            if not self.av_key or not self.news_key:
                st.error("❌ API keys not found. Please check your .env file.")
                st.stop()
            
            # Initialize data ingestion
            self.data_ingester = DataIngestionEngine(self.av_key, self.news_key)
            self.news_enhancer = NewsEnhancer()
            
            # Initialize Snowflake manager
            try:
                sf_config = get_snowflake_config()
                if all(sf_config.values()):
                    self.sf_manager = SnowflakeManager(sf_config)
                    self.snowflake_available = True
                else:
                    self.snowflake_available = False
                    st.warning("⚠️ Snowflake configuration incomplete. Some features may be limited.")
            except Exception as e:
                self.snowflake_available = False
                st.warning(f"⚠️ Snowflake connection failed: {str(e)}")
            
            # Initialize backtesting engine
            self.backtester = AdvancedBacktester(initial_capital=100000)
            self.signal_analyzer = SignalQualityAnalyzer()
            
        except Exception as e:
            st.error(f"❌ Error initializing components: {str(e)}")
            st.stop()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'stock_data' not in st.session_state:
            st.session_state.stock_data = {}
        
        if 'news_data' not in st.session_state:
            st.session_state.news_data = pd.DataFrame()
        
        if 'signals_data' not in st.session_state:
            st.session_state.signals_data = pd.DataFrame()
        
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = None
        
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">🚀 AI-Powered Financial Signal Extraction & Trading Strategy Validator</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        **Innovation Features:**
        - 🤖 Real-time AI signal extraction using Snowflake Cortex
        - 📊 Multi-modal data fusion (market data + news sentiment)
        - 🔍 Explainable AI for signal attribution
        - 📈 Advanced backtesting with risk management
        - 🎯 Live trading simulation with performance analytics
        """)
        
        st.divider()
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.header("🎛️ Control Panel")
            
            # Symbol selection
            st.subheader("📊 Stock Selection")
            available_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM']
            
            selected_symbols = st.multiselect(
                "Select stocks to analyze:",
                available_symbols,
                default=st.session_state.selected_symbols,
                max_selections=5
            )
            
            if selected_symbols != st.session_state.selected_symbols:
                st.session_state.selected_symbols = selected_symbols
                st.rerun()
            
            st.divider()
            
            # Data refresh controls
            st.subheader("🔄 Data Controls")
            
            if st.button("🚀 Fetch Latest Data", type="primary"):
                self.fetch_all_data()
            
            auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
            
            if auto_refresh:
                st.info("⏰ Auto-refresh enabled")
                time.sleep(300)  # 5 minutes
                st.rerun()
            
            st.divider()
            
            # Backtesting parameters
            st.subheader("📈 Backtesting Settings")
            
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=10000,
                max_value=1000000,
                value=100000,
                step=10000
            )
            
            commission = st.slider(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01
            )
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=5,
                max_value=50,
                value=10,
                step=5
            )
            
            self.backtester.initial_capital = initial_capital
            self.backtester.commission = commission / 100
            self.backtester.max_position_size = max_position_size / 100
            
            st.divider()
            
            # System status
            st.subheader("🔧 System Status")
            
            status_items = [
                ("Data Ingestion", "🟢 Active"),
                ("Snowflake", "🟢 Connected" if self.snowflake_available else "🔴 Disconnected"),
                ("AI Signals", "🟢 Generating"),
                ("Backtesting", "🟢 Ready")
            ]
            
            for label, status in status_items:
                st.text(f"{label}: {status}")
    
    def fetch_all_data(self):
        """Fetch all data for selected symbols"""
        if not st.session_state.selected_symbols:
            st.warning("Please select at least one symbol")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_steps = len(st.session_state.selected_symbols) * 2  # Stock data + news for each symbol
            current_step = 0
            
            # Fetch stock data
            for symbol in st.session_state.selected_symbols:
                status_text.text(f"Fetching stock data for {symbol}...")
                
                # Get stock data
                stock_data = self.data_ingester.get_daily_data(symbol)
                if not stock_data.empty:
                    st.session_state.stock_data[symbol] = stock_data
                    
                    # Store in Snowflake if available
                    if self.snowflake_available:
                        try:
                            self.sf_manager.insert_stock_data(stock_data, symbol)
                        except Exception as e:
                            logger.warning(f"Failed to insert stock data for {symbol}: {str(e)}")
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            # Fetch news data
            status_text.text("Fetching financial news...")
            news_data = self.data_ingester.get_financial_news(st.session_state.selected_symbols, hours_back=24)
            
            if not news_data.empty:
                # Enhance news data
                news_data['impact_score'] = news_data.apply(
                    lambda row: self.news_enhancer.calculate_news_impact_score(row.to_dict()), 
                    axis=1
                )
                
                # Extract financial entities
                news_data['entities'] = news_data.apply(
                    lambda row: self.news_enhancer.extract_financial_entities(
                        f"{row.get('title', '')} {row.get('description', '')}"
                    ), axis=1
                )
                
                st.session_state.news_data = news_data
                
                # Store in Snowflake if available
                if self.snowflake_available:
                    try:
                        self.sf_manager.insert_news_data(news_data)
                    except Exception as e:
                        logger.warning(f"Failed to insert news data: {str(e)}")
            
            current_step += len(st.session_state.selected_symbols)
            progress_bar.progress(1.0)
            
            # Generate signals
            status_text.text("Generating AI trading signals...")
            self.generate_signals()
            
            st.session_state.last_update = datetime.now()
            
            progress_bar.empty()
            status_text.success("✅ Data fetch completed successfully!")
            
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"❌ Error fetching data: {str(e)}")
    
    def generate_signals(self):
        """Generate trading signals for all symbols"""
        if not st.session_state.selected_symbols:
            return
        
        all_signals = []
        
        for symbol in st.session_state.selected_symbols:
            if self.snowflake_available:
                try:
                    # Generate signals using Snowflake AI
                    signals = self.sf_manager.generate_trading_signals(symbol)
                    if not signals.empty:
                        all_signals.append(signals)
                except Exception as e:
                    logger.warning(f"Failed to generate signals for {symbol}: {str(e)}")
                    # Fallback to local signal generation
                    signals = self.generate_local_signals(symbol)
                    if not signals.empty:
                        all_signals.append(signals)
            else:
                # Local signal generation
                signals = self.generate_local_signals(symbol)
                if not signals.empty:
                    all_signals.append(signals)
        
        if all_signals:
            st.session_state.signals_data = pd.concat(all_signals, ignore_index=True)
            
            # Store signals in Snowflake if available
            if self.snowflake_available:
                try:
                    self.sf_manager.insert_trading_signals(st.session_state.signals_data)
                except Exception as e:
                    logger.warning(f"Failed to store signals: {str(e)}")
    
    def generate_local_signals(self, symbol: str) -> pd.DataFrame:
        """Generate signals locally as fallback"""
        if symbol not in st.session_state.stock_data:
            return pd.DataFrame()
        
        stock_data = st.session_state.stock_data[symbol]
        if stock_data.empty:
            return pd.DataFrame()
        
        try:
            # Simple signal generation logic
            latest_data = stock_data.iloc[0]
            
            # Technical analysis based signal
            rsi = latest_data.get('rsi', 50)
            macd = latest_data.get('macd', 0)
            macd_signal = latest_data.get('macd_signal', 0)
            
            # Determine signal
            if rsi < 30 and macd > macd_signal:
                signal_type = 'BUY'
                signal_strength = 0.8
            elif rsi > 70 and macd < macd_signal:
                signal_type = 'SELL'
                signal_strength = 0.8
            else:
                signal_type = 'HOLD'
                signal_strength = 0.3
            
            # News sentiment factor
            symbol_news = st.session_state.news_data[st.session_state.news_data['symbol'] == symbol]
            sentiment_score = 0.0
            if not symbol_news.empty:
                sentiment_score = symbol_news['impact_score'].mean()
            
            # Adjust signal based on sentiment
            confidence_score = (signal_strength + sentiment_score) / 2
            
            signal = {
                'id': f"{symbol}_{signal_type}_{int(datetime.now().timestamp())}",
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'confidence_score': confidence_score,
                'price_target': latest_data['close_price'] * (1.05 if signal_type == 'BUY' else 0.95),
                'stop_loss': latest_data['close_price'] * (0.95 if signal_type == 'BUY' else 1.05),
                'position_size': min(confidence_score, 0.1),
                'timeframe': '1D',
                'signal_source': 'LOCAL_TECHNICAL',
                'created_at': datetime.now().isoformat()
            }
            
            return pd.DataFrame([signal])
            
        except Exception as e:
            logger.error(f"Error generating local signals for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def render_main_dashboard(self):
        """Render main dashboard content"""
        
        # Market Overview
        st.header("🌍 Market Overview")
        self.render_market_overview()
        
        st.divider()
        
        # Trading Signals
        st.header("🎯 AI Trading Signals")
        self.render_trading_signals()
        
        st.divider()
        
        # Stock Analysis
        st.header("📊 Stock Analysis")
        self.render_stock_analysis()
        
        st.divider()
        
        # News Sentiment
        st.header("📰 News Sentiment Analysis")
        self.render_news_sentiment()
        
        st.divider()
        
        # Backtesting
        st.header("📈 Strategy Backtesting")
        self.render_backtesting()
    
    def render_market_overview(self):
        """Render market overview section"""
        try:
            market_data = self.data_ingester.get_market_overview()
            
            if market_data:
                cols = st.columns(4)
                
                for i, (symbol, data) in enumerate(market_data.items()):
                    with cols[i % 4]:
                        change_color = "success" if data['change'] >= 0 else "danger"
                        
                        st.markdown(f"""
                        <div class="metric-card {change_color}-metric">
                            <h4>{symbol}</h4>
                            <h3>${data['price']:.2f}</h3>
                            <p>Change: {data['change_percent']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("📊 Market overview data will be displayed here once available.")
                
        except Exception as e:
            st.error(f"Error loading market overview: {str(e)}")
    
    def render_trading_signals(self):
        """Render trading signals section"""
        if st.session_state.signals_data.empty:
            st.info("🎯 Generate signals by clicking 'Fetch Latest Data' in the sidebar.")
            return
        
        # Signal summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_signals = len(st.session_state.signals_data)
            st.metric("Total Signals", total_signals)
        
        with col2:
            buy_signals = len(st.session_state.signals_data[st.session_state.signals_data['signal_type'] == 'BUY'])
            st.metric("Buy Signals", buy_signals)
        
        with col3:
            sell_signals = len(st.session_state.signals_data[st.session_state.signals_data['signal_type'] == 'SELL'])
            st.metric("Sell Signals", sell_signals)
        
        with col4:
            avg_confidence = st.session_state.signals_data['confidence_score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Individual signals
        st.subheader("🔍 Signal Details")
        
        for _, signal in st.session_state.signals_data.iterrows():
            signal_class = f"{signal['signal_type'].lower()}-signal"
            
            st.markdown(f"""
            <div class="signal-card {signal_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4>{signal['symbol']} - {signal['signal_type']}</h4>
                        <p><strong>Confidence:</strong> {signal['confidence_score']:.2f} | 
                           <strong>Strength:</strong> {signal['signal_strength']:.2f}</p>
                        <p><strong>Target:</strong> ${signal['price_target']:.2f} | 
                           <strong>Stop Loss:</strong> ${signal['stop_loss']:.2f}</p>
                    </div>
                    <div style="text-align: right;">
                        <p><strong>Position Size:</strong> {signal['position_size']:.1%}</p>
                        <p><strong>Source:</strong> {signal.get('signal_source', 'AI')}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_stock_analysis(self):
        """Render stock analysis section"""
        if not st.session_state.stock_data:
            st.info("📊 Stock analysis will be displayed here once data is loaded.")
            return
        
        # Stock selector
        selected_stock = st.selectbox(
            "Select stock for detailed analysis:",
            list(st.session_state.stock_data.keys())
        )
        
        if selected_stock and selected_stock in st.session_state.stock_data:
            stock_data = st.session_state.stock_data[selected_stock]
            
            if not stock_data.empty:
                # Price chart with technical indicators
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=(f'{selected_stock} Price & Indicators', 'RSI', 'MACD'),
                    row_heights=[0.5, 0.25, 0.25]
                )
                
                # Price and Bollinger Bands
                fig.add_trace(
                    go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['open'],
                        high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        name='Price'
                    ),
                    row=1, col=1
                )
                
                if 'bb_upper' in stock_data.columns:
                    fig.add_trace(
                        go.Scatter(x=stock_data.index, y=stock_data['bb_upper'], 
                                 name='BB Upper', line=dict(color='gray', dash='dash')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=stock_data.index, y=stock_data['bb_lower'], 
                                 name='BB Lower', line=dict(color='gray', dash='dash')),
                        row=1, col=1
                    )
                
                # RSI
                if 'rsi' in stock_data.columns:
                    fig.add_trace(
                        go.Scatter(x=stock_data.index, y=stock_data['rsi'], 
                                 name='RSI', line=dict(color='orange')),
                        row=2, col=1
                    )
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                if 'macd' in stock_data.columns:
                    fig.add_trace(
                        go.Scatter(x=stock_data.index, y=stock_data['macd'], 
                                 name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    if 'macd_signal' in stock_data.columns:
                        fig.add_trace(
                            go.Scatter(x=stock_data.index, y=stock_data['macd_signal'], 
                                     name='MACD Signal', line=dict(color='red')),
                            row=3, col=1
                        )
                
                fig.update_layout(
                    height=800,
                    title=f"{selected_stock} - Technical Analysis",
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stock metrics
                col1, col2, col3, col4 = st.columns(4)
                
                latest_data = stock_data.iloc[0]
                
                with col1:
                    st.metric("Current Price", f"${latest_data['close']:.2f}")
                
                with col2:
                    if 'rsi' in latest_data:
                        st.metric("RSI", f"{latest_data['rsi']:.1f}")
                
                with col3:
                    if 'volatility' in latest_data:
                        st.metric("Volatility", f"{latest_data['volatility']:.1%}")
                
                with col4:
                    volume = latest_data.get('volume', 0)
                    st.metric("Volume", f"{volume:,.0f}")
    
    def render_news_sentiment(self):
        """Render news sentiment analysis section"""
        if st.session_state.news_data.empty:
            st.info("📰 News sentiment analysis will be displayed here once data is loaded.")
            return
        
        news_data = st.session_state.news_data
        
        # Sentiment overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_articles = len(news_data)
            st.metric("Total Articles", total_articles)
        
        with col2:
            avg_sentiment = news_data.get('sentiment_score', pd.Series([0])).mean()
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
        
        with col3:
            positive_news = len(news_data[news_data.get('sentiment_label', '') == 'POSITIVE'])
            st.metric("Positive News", positive_news)
        
        with col4:
            avg_impact = news_data['impact_score'].mean()
            st.metric("Avg Impact Score", f"{avg_impact:.2f}")
        
        # Sentiment distribution
        if 'sentiment_score' in news_data.columns:
            fig = px.histogram(
                news_data, x='sentiment_score', 
                title='News Sentiment Distribution',
                labels={'sentiment_score': 'Sentiment Score', 'count': 'Number of Articles'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent news
        st.subheader("📋 Recent Financial News")
        
        # Filter and sort news
        recent_news = news_data.sort_values('published_at', ascending=False).head(10)
        
        for _, article in recent_news.iterrows():
            sentiment_color = "🟢" if article.get('sentiment_label') == 'POSITIVE' else "🔴" if article.get('sentiment_label') == 'NEGATIVE' else "🟡"
            
            with st.expander(f"{sentiment_color} {article['title'][:100]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Source:** {article['source']}")
                    st.write(f"**Description:** {article.get('description', 'N/A')}")
                    if article.get('url'):
                        st.write(f"**Link:** {article['url']}")
                
                with col2:
                    st.write(f"**Symbol:** {article['symbol']}")
                    st.write(f"**Published:** {article['published_at']}")
                    st.write(f"**Sentiment:** {article.get('sentiment_score', 0):.2f}")
                    st.write(f"**Impact Score:** {article['impact_score']:.2f}")
    
    def render_backtesting(self):
        """Render backtesting section"""
        st.subheader("⚙️ Strategy Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_symbol = st.selectbox(
                "Select symbol for backtesting:",
                st.session_state.selected_symbols if st.session_state.selected_symbols else ['AAPL']
            )
            
            backtest_period = st.slider(
                "Backtesting period (days):",
                min_value=30,
                max_value=365,
                value=90
            )
        
        with col2:
            strategy_params = st.expander("📊 Strategy Parameters")
            with strategy_params:
                st.write("Current backtesting parameters:")
                st.write(f"• Initial Capital: ${self.backtester.initial_capital:,.0f}")
                st.write(f"• Commission: {self.backtester.commission:.2%}")
                st.write(f"• Max Position Size: {self.backtester.max_position_size:.1%}")
        
        if st.button("🚀 Run Backtest", type="primary"):
            self.run_backtest(backtest_symbol, backtest_period)
        
        # Display backtest results
        if st.session_state.backtest_results:
            self.display_backtest_results()
    
    def run_backtest(self, symbol: str, period_days: int):
        """Run backtesting for selected symbol and period"""
        try:
            with st.spinner("Running backtest..."):
                # Get historical data
                if symbol in st.session_state.stock_data:
                    stock_data = st.session_state.stock_data[symbol].copy()
                    
                    # Filter data for backtesting period
                    end_date = stock_data.index[0]
                    start_date = end_date - timedelta(days=period_days)
                    stock_data = stock_data[stock_data.index >= start_date]
                    
                    # Get signals for the period
                    signals_data = st.session_state.signals_data[
                        st.session_state.signals_data['symbol'] == symbol
                    ].copy()
                    
                    if not signals_data.empty and not stock_data.empty:
                        # Run backtest
                        results = self.backtester.run_backtest(
                            stock_data, signals_data, f"AI_Strategy_{symbol}"
                        )
                        
                        st.session_state.backtest_results = results
                        st.success("✅ Backtest completed successfully!")
                    else:
                        st.warning("⚠️ Insufficient data for backtesting.")
                else:
                    st.error("❌ No stock data available for selected symbol.")
                    
        except Exception as e:
            st.error(f"❌ Error running backtest: {str(e)}")
    
    def display_backtest_results(self):
        """Display comprehensive backtesting results"""
        results = st.session_state.backtest_results
        
        if not results:
            return
        
        st.subheader("📊 Backtest Results")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return", 
                f"{results.total_return_pct:.2f}%",
                delta=f"${results.total_return:,.0f}"
            )
        
        with col2:
            st.metric("Sharpe Ratio", f"{results.sharpe_ratio:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{results.max_drawdown:.2f}%")
        
        with col4:
            st.metric("Win Rate", f"{results.win_rate:.1f}%")
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Total Trades", results.total_trades)
        
        with col6:
            st.metric("Profit Factor", f"{results.profit_factor:.2f}")
        
        with col7:
            st.metric("Volatility", f"{results.volatility:.2f}%")
        
        with col8:
            st.metric("Calmar Ratio", f"{results.calmar_ratio:.2f}")
        
        # Equity curve
        if not results.equity_curve.empty:
            st.subheader("📈 Equity Curve")
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=results.equity_curve.index,
                    y=results.equity_curve['equity'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add benchmark (buy and hold)
            initial_price = results.equity_curve['equity'].iloc[0]
            benchmark = (results.equity_curve['equity'] / initial_price) * results.initial_capital
            
            fig.add_trace(
                go.Scatter(
                    x=results.equity_curve.index,
                    y=benchmark,
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', dash='dash')
                )
            )
            
            fig.update_layout(
                title='Portfolio Performance vs Buy & Hold',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        if not results.drawdown_curve.empty:
            st.subheader("📉 Drawdown Analysis")
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=results.drawdown_curve.index,
                    y=results.drawdown_curve['drawdown'] * 100,
                    mode='lines',
                    fill='tonexty',
                    name='Drawdown %',
                    line=dict(color='red'),
                    fillcolor='rgba(255, 0, 0, 0.3)'
                )
            )
            
            fig.update_layout(
                title='Drawdown Over Time',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade analysis
        if results.trades:
            st.subheader("📋 Trade Analysis")
            
            trades_df = pd.DataFrame([{
                'Symbol': trade.symbol,
                'Type': trade.signal_type.value,
                'Entry Price': f"${trade.entry_price:.2f}",
                'Exit Price': f"${trade.exit_price:.2f}" if trade.exit_price else "N/A",
                'P&L': f"${trade.pnl:.2f}",
                'Return %': f"{trade.return_pct:.2f}%",
                'Duration (h)': trade.holding_period,
                'Exit Reason': trade.exit_reason
            } for trade in results.trades])
            
            st.dataframe(trades_df, use_container_width=True)
        
        # Performance attribution
        st.subheader("🔍 Performance Attribution")
        
        attribution_col1, attribution_col2 = st.columns(2)
        
        with attribution_col1:
            st.write("**Risk Metrics:**")
            st.write(f"• Value at Risk (95%): {results.var_95:.2f}%")
            st.write(f"• Value at Risk (99%): {results.var_99:.2f}%")
            st.write(f"• Maximum Consecutive Losses: {results.consecutive_losses}")
            st.write(f"• Average Trade Return: ${results.avg_trade_return:.2f}")
        
        with attribution_col2:
            st.write("**Trade Statistics:**")
            st.write(f"• Winning Trades: {results.winning_trades}")
            st.write(f"• Losing Trades: {results.losing_trades}")
            st.write(f"• Average Winning Trade: ${results.avg_winning_trade:.2f}")
            st.write(f"• Average Losing Trade: ${results.avg_losing_trade:.2f}")
    
    def run(self):
        """Main application entry point"""
        self.render_header()
        self.render_sidebar()
        self.render_main_dashboard()

# Application entry point
def main():
    """Main function to run the Streamlit dashboard"""
    try:
        dashboard = FinancialDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"❌ Critical error: {str(e)}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()