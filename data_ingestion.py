"""
Financial Signal Extraction and Trading Strategy Validation System
Data ingestion module for Alpha Vantage and News API
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from newsapi import NewsApiClient
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionEngine:
    """
    Advanced data ingestion engine for real-time financial data and news
    """
    
    def __init__(self, alpha_vantage_key: str, news_api_key: str):
        self.av_key = alpha_vantage_key
        self.news_key = news_api_key
        self.av_ts = TimeSeries(key=self.av_key, output_format='pandas')
        self.av_ti = TechIndicators(key=self.av_key, output_format='pandas')
        self.news_client = NewsApiClient(api_key=self.news_key)
        self.rate_limit_delay = 12  # Alpha Vantage free tier: 5 calls per minute
        
    def get_intraday_data(self, symbol: str, interval: str = '5min', 
                         outputsize: str = 'full') -> pd.DataFrame:
        """
        Fetch intraday stock data with technical indicators
        """
        try:
            logger.info(f"Fetching intraday data for {symbol}")
            
            # Get price data
            data, meta_data = self.av_ts.get_intraday(
                symbol=symbol, 
                interval=interval, 
                outputsize=outputsize
            )
            
            # Clean column names
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data = data.sort_index()
            
            # Add technical indicators
            data = self._add_technical_indicators(data, symbol)
            
            # Add volatility metrics
            data = self._add_volatility_metrics(data)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            time.sleep(self.rate_limit_delay)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_daily_data(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        """
        Fetch daily stock data with comprehensive technical analysis
        """
        try:
            logger.info(f"Fetching daily data for {symbol}")
            
            # Get price data
            data, meta_data = self.av_ts.get_daily_adjusted(
                symbol=symbol, 
                outputsize=outputsize
            )
            
            # Clean column names
            data.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 
                           'volume', 'dividend_amount', 'split_coefficient']
            data = data.sort_index()
            
            # Add comprehensive technical indicators
            data = self._add_technical_indicators(data, symbol)
            data = self._add_volatility_metrics(data)
            data = self._add_momentum_indicators(data, symbol)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            time.sleep(self.rate_limit_delay)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add technical indicators using Alpha Vantage API
        """
        try:
            # RSI
            rsi_data, _ = self.av_ti.get_rsi(symbol=symbol, interval='daily')
            data = data.join(rsi_data.rename(columns={'RSI': 'rsi'}), how='left')
            time.sleep(self.rate_limit_delay)
            
            # MACD
            macd_data, _ = self.av_ti.get_macd(symbol=symbol, interval='daily')
            macd_data.columns = ['macd', 'macd_signal', 'macd_hist']
            data = data.join(macd_data, how='left')
            time.sleep(self.rate_limit_delay)
            
            # Bollinger Bands
            bb_data, _ = self.av_ti.get_bbands(symbol=symbol, interval='daily')
            bb_data.columns = ['bb_upper', 'bb_middle', 'bb_lower']
            data = data.join(bb_data, how='left')
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"Error adding technical indicators: {str(e)}")
        
        return data
    
    def _add_volatility_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based metrics
        """
        # True Range
        data['tr'] = pd.concat([
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        ], axis=1).max(axis=1)
        
        # Average True Range
        data['atr'] = data['tr'].rolling(window=14).mean()
        
        # Volatility (20-day rolling)
        data['volatility'] = data['close'].pct_change().rolling(window=20).std() * (252 ** 0.5)
        
        # Price channels
        data['price_channel_high'] = data['high'].rolling(window=20).max()
        data['price_channel_low'] = data['low'].rolling(window=20).min()
        
        return data
    
    def _add_momentum_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add momentum-based indicators
        """
        try:
            # Stochastic Oscillator
            stoch_data, _ = self.av_ti.get_stoch(symbol=symbol, interval='daily')
            stoch_data.columns = ['stoch_k', 'stoch_d']
            data = data.join(stoch_data, how='left')
            time.sleep(self.rate_limit_delay)
            
            # Williams %R
            willr_data, _ = self.av_ti.get_willr(symbol=symbol, interval='daily')
            data = data.join(willr_data.rename(columns={'WILLR': 'williams_r'}), how='left')
            time.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"Error adding momentum indicators: {str(e)}")
        
        return data
    
    def get_financial_news(self, symbols: List[str], hours_back: int = 24) -> pd.DataFrame:
        """
        Fetch financial news for given symbols with enhanced filtering
        """
        try:
            from_date = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            all_articles = []
            
            # Financial sources for better quality
            financial_sources = [
                'bloomberg', 'reuters', 'cnbc', 'marketwatch', 'financial-times',
                'the-wall-street-journal', 'fortune', 'business-insider'
            ]
            
            for symbol in symbols:
                try:
                    # Search for company-specific news
                    search_queries = [
                        f"{symbol} stock",
                        f"{symbol} earnings",
                        f"{symbol} financial results",
                        f"{symbol} market analysis"
                    ]
                    
                    for query in search_queries:
                        articles = self.news_client.get_everything(
                            q=query,
                            sources=','.join(financial_sources),
                            from_param=from_date,
                            language='en',
                            sort_by='publishedAt',
                            page_size=20
                        )
                        
                        for article in articles.get('articles', []):
                            all_articles.append({
                                'symbol': symbol,
                                'title': article['title'],
                                'description': article['description'],
                                'content': article['content'],
                                'source': article['source']['name'],
                                'url': article['url'],
                                'published_at': article['publishedAt'],
                                'query': query
                            })
                        
                        time.sleep(1)  # Rate limiting
                        
                except Exception as e:
                    logger.warning(f"Error fetching news for {symbol}: {str(e)}")
                    continue
            
            news_df = pd.DataFrame(all_articles)
            if not news_df.empty:
                # Remove duplicates
                news_df = news_df.drop_duplicates(subset=['title', 'source'])
                
                # Convert timestamp
                news_df['published_at'] = pd.to_datetime(news_df['published_at'])
                
                # Add text length and quality metrics
                news_df['title_length'] = news_df['title'].str.len()
                news_df['content_length'] = news_df['content'].fillna('').str.len()
                news_df['has_content'] = news_df['content'].notna()
                
                logger.info(f"Fetched {len(news_df)} news articles")
            
            return news_df
            
        except Exception as e:
            logger.error(f"Error fetching financial news: {str(e)}")
            return pd.DataFrame()
    
    def get_market_overview(self) -> Dict:
        """
        Get market overview data
        """
        try:
            # Market indices
            indices = ['SPY', 'QQQ', 'IWM', 'VIX']
            overview_data = {}
            
            for index in indices:
                try:
                    data, _ = self.av_ts.get_quote_endpoint(symbol=index)
                    overview_data[index] = {
                        'price': float(data['05. price']),
                        'change': float(data['09. change']),
                        'change_percent': data['10. change percent'].rstrip('%'),
                        'volume': int(data['06. volume']),
                        'timestamp': data['07. latest trading day']
                    }
                    time.sleep(self.rate_limit_delay)
                except Exception as e:
                    logger.warning(f"Error fetching data for {index}: {str(e)}")
                    continue
            
            return overview_data
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {str(e)}")
            return {}
    
    def get_sector_performance(self) -> pd.DataFrame:
        """
        Get sector performance data
        """
        try:
            sector_data, _ = self.av_ts.get_sector()
            
            # Transform the nested dict structure
            sectors_df = pd.DataFrame()
            for time_period, sectors in sector_data.items():
                temp_df = pd.DataFrame(list(sectors.items()), 
                                     columns=['sector', 'performance'])
                temp_df['time_period'] = time_period
                temp_df['performance'] = temp_df['performance'].str.rstrip('%').astype(float)
                sectors_df = pd.concat([sectors_df, temp_df], ignore_index=True)
            
            time.sleep(self.rate_limit_delay)
            return sectors_df
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {str(e)}")
            return pd.DataFrame()

class NewsEnhancer:
    """
    Enhanced news processing with advanced NLP features
    """
    
    def __init__(self):
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook',
            'acquisition', 'merger', 'ipo', 'dividend', 'buyback', 'split',
            'upgrade', 'downgrade', 'analyst', 'target', 'recommendation'
        ]
        
    def extract_financial_entities(self, text: str) -> Dict:
        """
        Extract financial entities and keywords from text
        """
        if not text:
            return {}
        
        text_lower = text.lower()
        entities = {
            'financial_keywords': [],
            'numbers': [],
            'percentages': [],
            'currencies': []
        }
        
        # Find financial keywords
        for keyword in self.financial_keywords:
            if keyword in text_lower:
                entities['financial_keywords'].append(keyword)
        
        # Extract numbers, percentages, currencies using regex
        import re
        
        # Numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', text)
        entities['numbers'] = [float(n) for n in numbers if float(n) > 0]
        
        # Percentages
        percentages = re.findall(r'\b\d+\.?\d*%', text)
        entities['percentages'] = percentages
        
        # Currency amounts
        currencies = re.findall(r'\$\d+\.?\d*[BMK]?', text)
        entities['currencies'] = currencies
        
        return entities
    
    def calculate_news_impact_score(self, article: Dict) -> float:
        """
        Calculate potential market impact score for news article
        """
        score = 0.0
        
        # Source credibility weight
        source_weights = {
            'Reuters': 1.0, 'Bloomberg': 1.0, 'Wall Street Journal': 0.9,
            'CNBC': 0.8, 'MarketWatch': 0.7, 'Financial Times': 0.9,
            'Fortune': 0.6, 'Business Insider': 0.5
        }
        
        source_weight = source_weights.get(article.get('source', ''), 0.3)
        score += source_weight * 0.3
        
        # Content quality
        if article.get('content') and len(article['content']) > 500:
            score += 0.2
        
        # Financial keyword density
        entities = self.extract_financial_entities(
            f"{article.get('title', '')} {article.get('description', '')}"
        )
        keyword_density = len(entities['financial_keywords']) / max(1, len(article.get('title', '').split()))
        score += min(keyword_density * 0.5, 0.3)
        
        # Recency boost
        if article.get('published_at'):
            hours_ago = (datetime.now() - pd.to_datetime(article['published_at'])).total_seconds() / 3600
            if hours_ago < 2:
                score += 0.2
            elif hours_ago < 6:
                score += 0.1
        
        return min(score, 1.0)

# Example usage and testing
if __name__ == "__main__":
    # Test data ingestion
    from dotenv import load_dotenv
    load_dotenv()
    
    av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    news_key = os.getenv('NEWS_API_KEY')
    
    if av_key and news_key:
        ingester = DataIngestionEngine(av_key, news_key)
        
        # Test data fetching
        print("Testing data ingestion...")
        
        # Fetch stock data
        stock_data = ingester.get_daily_data('AAPL')
        print(f"Stock data shape: {stock_data.shape}")
        if not stock_data.empty:
            print(f"Columns: {stock_data.columns.tolist()}")
        
        # Fetch news
        news_data = ingester.get_financial_news(['AAPL'], hours_back=24)
        print(f"News data shape: {news_data.shape}")
        
        # Market overview
        overview = ingester.get_market_overview()
        print(f"Market overview: {list(overview.keys())}")
    else:
        print("Please set API keys in .env file")