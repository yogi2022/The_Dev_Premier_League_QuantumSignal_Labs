# 🚀 QuantumSignal Labs - AI Trading Platform
## Phase 2 Submission - The Dev Premier League

### 📋 Project Overview
An **AI-Powered Financial Signal Extraction & Trading Strategy Validation Platform** that combines real-time market data, advanced sentiment analysis, and sophisticated backtesting to generate actionable trading signals.

**Challenge:** Signal Extraction from Market & News Data  
**Team:** QuantumSignal Labs

---

## ✨ Key Features

### 🤖 AI Signal Generation
- **Multi-factor analysis** combining technical indicators and news sentiment
- **Real-time signal generation** with confidence scoring
- **Risk-reward optimization** with dynamic position sizing
- **Explainable AI** with full signal attribution

### 📊 Market Data Integration
- **Real-time stock data** from Alpha Vantage API
- **Comprehensive technical indicators**: RSI, MACD, Bollinger Bands, ATR, Moving Averages
- **Multi-timeframe analysis** with automated calculations
- **Rate-limiting optimization** for free API tier

### 📰 News Sentiment Analysis
- **Financial news ingestion** from News API
- **AI-powered sentiment scoring** using keyword analysis
- **Impact-weighted sentiment** calculation
- **Real-time news monitoring** with recency weighting

### 🎯 Advanced Backtesting
- **Risk-adjusted metrics**: Sharpe ratio, Max Drawdown, Win Rate
- **Commission & slippage modeling** for realistic results
- **Trade-by-trade analysis** with detailed logs
- **Equity curve visualization** with performance tracking

### 📈 Interactive Visualizations
- **Professional candlestick charts** with technical overlays
- **Real-time dashboards** with live updates
- **Sentiment gauges** and performance metrics
- **Responsive design** for all screen sizes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│           AI Trading Platform Architecture              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                 │
│  │ Alpha Vantage│    │   News API   │                 │
│  │  (Market Data│    │  (Sentiment) │                 │
│  └──────┬───────┘    └──────┬───────┘                 │
│         │                    │                          │
│         v                    v                          │
│  ┌──────────────────────────────────┐                 │
│  │   Data Ingestion Engine          │                 │
│  │  • Rate Limiting                 │                 │
│  │  • Caching (1hr)                 │                 │
│  │  • Technical Indicators          │                 │
│  └──────────┬───────────────────────┘                 │
│             │                                           │
│             v                                           │
│  ┌──────────────────────────────────┐                 │
│  │  Signal Generation Engine         │                 │
│  │  • Technical Analysis (65%)       │                 │
│  │  • Sentiment Analysis (35%)       │                 │
│  │  • Confidence Scoring             │                 │
│  │  • Risk/Reward Calculation        │                 │
│  └──────────┬───────────────────────┘                 │
│             │                                           │
│             v                                           │
│  ┌──────────────────────────────────┐                 │
│  │    Backtesting Engine             │                 │
│  │  • Portfolio Simulation           │                 │
│  │  • Performance Metrics            │                 │
│  │  • Trade Execution Logic          │                 │
│  └──────────┬───────────────────────┘                 │
│             │                                           │
│             v                                           │
│  ┌──────────────────────────────────┐                 │
│  │  Streamlit Dashboard (Frontend)   │                 │
│  │  • Interactive UI                 │                 │
│  │  • Real-time Charts               │                 │
│  │  • Signal Visualization           │                 │
│  └───────────────────────────────────┘                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| **Frontend** | Streamlit | Interactive web dashboard |
| **Data Processing** | Pandas, NumPy | Data manipulation & analysis |
| **Visualization** | Plotly | Interactive charts & graphs |
| **Market Data** | Alpha Vantage API | Real-time & historical stock data |
| **News Data** | News API | Financial news & articles |
| **Language** | Python 3.8+ | Core development |

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Internet connection (for API calls)
- API Keys (provided below)

### Step 1: Clone/Download Files
Download the following files:
- `ai_trading_dashboard.py` (main application)
- `requirements.txt` (dependencies)

### Step 2: Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
streamlit run ai_trading_dashboard.py
```

The application will automatically open in your browser at `http://localhost:8501`

---

## 🔑 API Configuration

The application uses the following API keys (already configured in code):

**Alpha Vantage API Key:** `2KNU40IVNB9EONNK`
- Free tier: 5 calls per minute, 500 calls per day
- Used for: Stock price data, technical indicators

**News API Key:** `fc9118ab4dab4d46817a011698909393`
- Free tier: 100 calls per day
- Used for: Financial news articles, sentiment analysis

**Note:** API keys are hardcoded in the application for easy demo purposes. The system automatically handles rate limiting.

---

## 🎮 How to Use

### 1. Select Stocks
- In the **sidebar**, select up to **3 stocks** from the dropdown
- Default selections: AAPL, MSFT, GOOGL
- Limited to 3 stocks to respect free API tier limits

### 2. Fetch Data
- Click the **"🚀 Fetch Latest Data"** button in the sidebar
- The system will:
  - Retrieve daily stock data with technical indicators
  - Fetch recent financial news
  - Analyze sentiment
  - Generate AI trading signals
- Progress bar shows fetching status

### 3. Explore Signals
Navigate through the tabs:

**📊 Market Overview**
- Real-time stock prices and metrics
- Quick performance indicators
- Volume and volatility stats

**🎯 Trading Signals**
- AI-generated BUY/SELL/HOLD signals
- Confidence scores and signal strength
- Price targets and stop-loss levels
- Risk/reward ratios
- Expandable detailed analysis

**📈 Technical Charts**
- Interactive candlestick charts
- Technical indicators overlay (RSI, MACD, Bollinger Bands)
- Multiple timeframe analysis
- Zoom and pan capabilities

**📰 News & Sentiment**
- Latest financial news articles
- Sentiment analysis (Positive/Negative/Neutral)
- Overall sentiment gauge
- Source credibility indicators

**🎯 Backtest Results**
- Historical strategy performance
- Key metrics: Returns, Sharpe Ratio, Max Drawdown
- Equity curve visualization
- Detailed trade log

### 4. Run Backtesting
- Configure parameters in sidebar:
  - **Initial Capital**: Starting portfolio value
  - **Lookback Period**: Historical period (30-365 days)
- Click **"▶️ Run Backtest"**
- View results in the Backtest Results tab

---

## 📊 Signal Generation Logic

### Technical Score (65% weight)
- **RSI Analysis**: Oversold (<30) = Bullish, Overbought (>70) = Bearish
- **MACD Crossover**: Signal line crossovers indicate trend changes
- **Bollinger Bands**: Price position relative to bands
- **Moving Averages**: 20/50/200 SMA crossovers and trends
- **Momentum**: Recent price change acceleration

### Sentiment Score (35% weight)
- **Keyword Analysis**: Positive/negative financial terms
- **Recency Weighting**: Recent news weighted higher
- **Source Quality**: Credible sources weighted higher

### Signal Types
- **BUY**: Combined score > 0.5
- **SELL**: Combined score < -0.5
- **HOLD**: Combined score between -0.5 and 0.5

### Position Sizing
- Based on signal confidence (0-100%)
- Maximum position size: 10% of capital
- Risk-adjusted based on volatility

---

## 📈 Performance Metrics

The backtesting engine calculates:

| Metric | Description |
|--------|-------------|
| **Total Return** | Absolute and percentage profit/loss |
| **Win Rate** | Percentage of profitable trades |
| **Sharpe Ratio** | Risk-adjusted return metric |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Profit Factor** | Ratio of gross profit to gross loss |
| **Average Win/Loss** | Mean profit and loss per trade |
| **Total Trades** | Number of completed trades |

---

## 🎨 Features Highlights

### ✅ Optimized for Free Tier
- **Intelligent caching** reduces API calls
- **Rate limiting** respects API constraints
- **Batch processing** maximizes efficiency
- **Compact data mode** for Alpha Vantage

### ✅ Production-Ready
- **Error handling** for API failures
- **Fallback mechanisms** for missing data
- **Logging system** for debugging
- **Session state management** for persistence

### ✅ User Experience
- **Professional UI** with gradient styling
- **Responsive design** for all devices
- **Interactive tooltips** and help text
- **Progress indicators** for operations
- **Real-time updates** without page refresh

### ✅ Scalability
- **Modular architecture** for easy extension
- **Class-based design** for maintainability
- **Type hints** for code clarity
- **Comprehensive documentation**

---

## 🏆 Competition Alignment

### Problem Understanding ✅
- Complete implementation of signal extraction from market & news data
- Real-time ingestion with sentiment integration
- Comprehensive backtesting framework

### Innovation & Relevance ✅
- Multi-modal AI signal generation (technical + sentiment)
- Explainable AI with full attribution
- Real-time risk monitoring
- Production-ready for actual trading

### Technical Excellence ✅
- Clean, modular code architecture
- Efficient API usage for free tiers
- Advanced analytics beyond basic metrics
- Professional error handling

### User Experience ✅
- Intuitive, professional dashboard
- Real-time updates and visualizations
- Clear workflow and navigation
- Comprehensive analytics

---

## 🚀 Future Enhancements

Potential additions for production deployment:

1. **Machine Learning Models**
   - LSTM for price prediction
   - Transformer models for news analysis
   - Reinforcement learning for strategy optimization

2. **Advanced Features**
   - Multi-asset portfolio optimization
   - Options and derivatives support
   - Social media sentiment integration
   - Real-time alert system

3. **Infrastructure**
   - Database integration (PostgreSQL/MongoDB)
   - Message queue for async processing
   - Containerization with Docker
   - Cloud deployment (AWS/Azure/GCP)

4. **Integration**
   - Broker API for live trading
   - Webhook notifications
   - Email/SMS alerts
   - Mobile app companion

---

## 📞 Support & Troubleshooting

### Common Issues

**"API Rate Limit Exceeded"**
- Wait 12 seconds between requests
- Reduce number of stocks analyzed
- Use cached data when available

**"No Data Available"**
- Check internet connection
- Verify API keys are correct
- Try different stock symbols
- Check API service status

**"Module Not Found"**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Activate virtual environment
- Check Python version (3.8+)

### Debug Mode
The application includes built-in logging. Check console output for detailed error messages.

---

## 📄 License & Attribution

**Built for:** The Dev Premier League - Phase 2  
**Powered by:** 
- Alpha Vantage (Market Data)
- News API (Financial News)
- Streamlit (Frontend Framework)
- Plotly (Visualizations)

**Note:** This is a demonstration project for the hackathon. For production use, consider premium API tiers and proper compliance with financial regulations.

---

## 🎯 Key Differentiators

What makes this solution stand out:

1. **Complete Integration**: Seamless combination of market data and news sentiment
2. **Free Tier Optimized**: Intelligent design for API limitations
3. **Production Quality**: Professional code architecture and error handling
4. **User-Centric**: Intuitive interface with comprehensive analytics
5. **Explainable AI**: Full transparency in signal generation
6. **Risk-First**: Comprehensive risk metrics and position sizing
7. **Single File Solution**: Easy deployment and demonstration

---

## 🏅 Demo Scenarios

### Scenario 1: Quick Analysis
1. Open application
2. Select 3 tech stocks (AAPL, MSFT, GOOGL)
3. Click "Fetch Latest Data"
4. Review AI-generated signals in seconds

### Scenario 2: Deep Dive
1. Fetch data for selected stocks
2. Explore technical charts tab
3. Analyze news sentiment
4. Review signal attribution details
5. Understand risk/reward ratios

### Scenario 3: Strategy Validation
1. Generate signals
2. Configure backtest parameters
3. Run historical simulation
4. Analyze performance metrics
5. Review trade-by-trade results

---

## 📧 Contact

**Team:** QuantumSignal Labs  
**Competition:** The Dev Premier League - Snowflake AI Challenge  
**Submission:** Phase 2 Prototype

---

## 🎊 Acknowledgments

Special thanks to:
- **Snowflake** for sponsoring The Dev Premier League
- **Alpha Vantage** for providing market data API
- **News API** for financial news access
- **Streamlit** for the amazing framework

---

**Built with ❤️ for The Dev Premier League**

*Empowering traders with AI-driven insights*
