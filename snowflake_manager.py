"""
Snowflake Database Manager for Financial Signal Processing
Handles Snowflake connections, data storage, and Cortex AI integration
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import json

# Snowflake imports
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, when, lit, avg, stddev, max as sf_max, min as sf_min
from snowflake.snowpark.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from snowflake.connector import connect
import snowflake.connector as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeManager:
    """
    Advanced Snowflake manager with Cortex AI integration for financial signals
    """
    
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params
        self.session = None
        self.connection = None
        self._initialize_connection()
        self._setup_database_schema()
        
    def _initialize_connection(self):
        """Initialize Snowflake session and connection"""
        try:
            # Create Snowpark session
            self.session = Session.builder.configs(self.connection_params).create()
            
            # Create regular connection for some operations
            self.connection = connect(
                user=self.connection_params['user'],
                password=self.connection_params['password'],
                account=self.connection_params['account'],
                warehouse=self.connection_params['warehouse'],
                database=self.connection_params['database'],
                schema=self.connection_params['schema'],
                role=self.connection_params.get('role', 'ACCOUNTADMIN')
            )
            
            logger.info("Successfully connected to Snowflake")
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise
    
    def _setup_database_schema(self):
        """Setup database schema and tables for financial data"""
        try:
            # Create database and schema if they don't exist
            self.execute_query(f"CREATE DATABASE IF NOT EXISTS {self.connection_params['database']}")
            self.execute_query(f"CREATE SCHEMA IF NOT EXISTS {self.connection_params['database']}.{self.connection_params['schema']}")
            self.execute_query(f"USE DATABASE {self.connection_params['database']}")
            self.execute_query(f"USE SCHEMA {self.connection_params['schema']}")
            
            # Create tables
            self._create_financial_tables()
            
            logger.info("Database schema setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up database schema: {str(e)}")
            raise
    
    def _create_financial_tables(self):
        """Create tables for storing financial data"""
        
        # Stock price data table
        stock_data_ddl = """
        CREATE TABLE IF NOT EXISTS stock_data (
            symbol VARCHAR(10),
            timestamp TIMESTAMP_NTZ,
            open_price DOUBLE,
            high_price DOUBLE,
            low_price DOUBLE,
            close_price DOUBLE,
            adjusted_close DOUBLE,
            volume BIGINT,
            rsi DOUBLE,
            macd DOUBLE,
            macd_signal DOUBLE,
            macd_hist DOUBLE,
            bb_upper DOUBLE,
            bb_middle DOUBLE,
            bb_lower DOUBLE,
            atr DOUBLE,
            volatility DOUBLE,
            stoch_k DOUBLE,
            stoch_d DOUBLE,
            williams_r DOUBLE,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (symbol, timestamp)
        )
        """
        
        # News data table
        news_data_ddl = """
        CREATE TABLE IF NOT EXISTS news_data (
            id VARCHAR(255) PRIMARY KEY,
            symbol VARCHAR(10),
            title VARCHAR(1000),
            description VARCHAR(5000),
            content VARIANT,
            source VARCHAR(100),
            url VARCHAR(1000),
            published_at TIMESTAMP_NTZ,
            query_term VARCHAR(200),
            title_length INTEGER,
            content_length INTEGER,
            has_content BOOLEAN,
            impact_score DOUBLE,
            financial_keywords VARIANT,
            entities VARIANT,
            sentiment_score DOUBLE,
            sentiment_label VARCHAR(20),
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        
        # Trading signals table
        signals_ddl = """
        CREATE TABLE IF NOT EXISTS trading_signals (
            id VARCHAR(255) PRIMARY KEY,
            symbol VARCHAR(10),
            signal_type VARCHAR(50),
            signal_strength DOUBLE,
            confidence_score DOUBLE,
            price_target DOUBLE,
            stop_loss DOUBLE,
            position_size DOUBLE,
            timeframe VARCHAR(20),
            signal_source VARCHAR(100),
            technical_indicators VARIANT,
            news_sentiment VARIANT,
            risk_metrics VARIANT,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            expiry_at TIMESTAMP_NTZ,
            status VARCHAR(20) DEFAULT 'ACTIVE'
        )
        """
        
        # Backtesting results table
        backtest_results_ddl = """
        CREATE TABLE IF NOT EXISTS backtest_results (
            id VARCHAR(255) PRIMARY KEY,
            strategy_name VARCHAR(100),
            symbol VARCHAR(10),
            start_date DATE,
            end_date DATE,
            initial_capital DOUBLE,
            final_capital DOUBLE,
            total_return DOUBLE,
            sharpe_ratio DOUBLE,
            max_drawdown DOUBLE,
            win_rate DOUBLE,
            total_trades INTEGER,
            winning_trades INTEGER,
            losing_trades INTEGER,
            avg_trade_return DOUBLE,
            volatility DOUBLE,
            var_95 DOUBLE,
            calmar_ratio DOUBLE,
            parameters VARIANT,
            trade_log VARIANT,
            performance_metrics VARIANT,
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        
        # Market overview table
        market_overview_ddl = """
        CREATE TABLE IF NOT EXISTS market_overview (
            symbol VARCHAR(10),
            timestamp TIMESTAMP_NTZ,
            price DOUBLE,
            change_amount DOUBLE,
            change_percent DOUBLE,
            volume BIGINT,
            market_cap DOUBLE,
            sector VARCHAR(100),
            created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (symbol, timestamp)
        )
        """
        
        tables = [
            stock_data_ddl, news_data_ddl, signals_ddl, 
            backtest_results_ddl, market_overview_ddl
        ]
        
        for ddl in tables:
            self.execute_query(ddl)
        
        logger.info("All financial tables created successfully")
    
    def execute_query(self, query: str) -> Any:
        """Execute SQL query using Snowpark session"""
        try:
            result = self.session.sql(query).collect()
            return result
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def insert_stock_data(self, df: pd.DataFrame, symbol: str):
        """Insert stock data into Snowflake with enhanced processing"""
        if df.empty:
            logger.warning(f"Empty dataframe for symbol {symbol}")
            return
        
        try:
            # Prepare data for insertion
            df_clean = df.copy()
            df_clean = df_clean.reset_index()
            
            # Add symbol column
            df_clean['symbol'] = symbol
            
            # Rename columns to match table schema
            column_mapping = {
                'open': 'open_price',
                'high': 'high_price', 
                'low': 'low_price',
                'close': 'close_price',
                'volume': 'volume',
                'rsi': 'rsi',
                'macd': 'macd',
                'macd_signal': 'macd_signal',
                'macd_hist': 'macd_hist',
                'bb_upper': 'bb_upper',
                'bb_middle': 'bb_middle',
                'bb_lower': 'bb_lower',
                'atr': 'atr',
                'volatility': 'volatility',
                'stoch_k': 'stoch_k',
                'stoch_d': 'stoch_d',
                'williams_r': 'williams_r'
            }
            
            # Rename columns
            for old_col, new_col in column_mapping.items():
                if old_col in df_clean.columns:
                    df_clean = df_clean.rename(columns={old_col: new_col})
            
            # Ensure required columns exist
            required_columns = ['symbol', 'timestamp', 'open_price', 'high_price', 
                              'low_price', 'close_price', 'volume']
            
            missing_columns = [col for col in required_columns if col not in df_clean.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return
            
            # Convert timestamp column
            if 'timestamp' not in df_clean.columns and df_clean.index.name:
                df_clean = df_clean.reset_index()
                df_clean = df_clean.rename(columns={df_clean.columns[0]: 'timestamp'})
            
            # Fill NaN values
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)
            
            # Create Snowpark DataFrame
            snowpark_df = self.session.create_dataframe(df_clean)
            
            # Write to table (merge/upsert)
            snowpark_df.write.mode('append').save_as_table('stock_data')
            
            logger.info(f"Inserted {len(df_clean)} records for {symbol} into stock_data table")
            
        except Exception as e:
            logger.error(f"Error inserting stock data for {symbol}: {str(e)}")
            raise
    
    def insert_news_data(self, df: pd.DataFrame):
        """Insert news data with sentiment analysis using Cortex"""
        if df.empty:
            logger.warning("Empty news dataframe")
            return
        
        try:
            # Prepare data
            df_clean = df.copy()
            
            # Generate unique IDs
            df_clean['id'] = df_clean.apply(
                lambda row: f"{row['symbol']}_{hash(row['title'])}_{int(pd.to_datetime(row['published_at']).timestamp())}", 
                axis=1
            )
            
            # Process sentiment using Snowflake Cortex
            df_clean = self._process_news_sentiment(df_clean)
            
            # Convert complex columns to JSON strings for VARIANT fields
            variant_columns = ['financial_keywords', 'entities']
            for col in variant_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].apply(lambda x: json.dumps(x) if pd.notna(x) else None)
            
            # Create Snowpark DataFrame
            snowpark_df = self.session.create_dataframe(df_clean)
            
            # Write to table
            snowpark_df.write.mode('append').save_as_table('news_data')
            
            logger.info(f"Inserted {len(df_clean)} news records into news_data table")
            
        except Exception as e:
            logger.error(f"Error inserting news data: {str(e)}")
            raise
    
    def _process_news_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process news sentiment using Snowflake Cortex"""
        try:
            # Prepare text for sentiment analysis
            df['full_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
            
            # Create temporary table for sentiment processing
            temp_table_name = f"temp_news_sentiment_{int(datetime.now().timestamp())}"
            
            # Create Snowpark DataFrame and save as temp table
            temp_df = self.session.create_dataframe(df[['id', 'full_text']])
            temp_df.write.mode('overwrite').save_as_table(temp_table_name, table_type='temporary')
            
            # Apply Cortex sentiment analysis
            sentiment_query = f"""
            SELECT 
                id,
                full_text,
                SNOWFLAKE.CORTEX.SENTIMENT(full_text) as sentiment_score,
                CASE 
                    WHEN SNOWFLAKE.CORTEX.SENTIMENT(full_text) > 0.1 THEN 'POSITIVE'
                    WHEN SNOWFLAKE.CORTEX.SENTIMENT(full_text) < -0.1 THEN 'NEGATIVE'
                    ELSE 'NEUTRAL'
                END as sentiment_label
            FROM {temp_table_name}
            """
            
            # Execute sentiment analysis
            sentiment_results = self.session.sql(sentiment_query).to_pandas()
            
            # Merge sentiment results back to original dataframe
            df = df.merge(sentiment_results[['id', 'sentiment_score', 'sentiment_label']], 
                         on='id', how='left')
            
            # Clean up temporary table
            self.session.sql(f"DROP TABLE IF EXISTS {temp_table_name}").collect()
            
            logger.info("Successfully processed news sentiment using Snowflake Cortex")
            
        except Exception as e:
            logger.warning(f"Error processing sentiment with Cortex: {str(e)}")
            # Fallback to basic sentiment scoring
            df['sentiment_score'] = 0.0
            df['sentiment_label'] = 'NEUTRAL'
        
        return df
    
    def generate_trading_signals(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Generate trading signals using advanced analytics and Cortex AI"""
        try:
            # Fetch recent stock data with technical indicators
            stock_query = f"""
            SELECT *
            FROM stock_data
            WHERE symbol = '{symbol}'
            AND timestamp >= CURRENT_DATE() - {lookback_days}
            ORDER BY timestamp DESC
            LIMIT 1000
            """
            
            stock_data = self.session.sql(stock_query).to_pandas()
            
            if stock_data.empty:
                logger.warning(f"No stock data found for {symbol}")
                return pd.DataFrame()
            
            # Fetch recent news sentiment
            news_query = f"""
            SELECT symbol, sentiment_score, sentiment_label, impact_score, created_at
            FROM news_data
            WHERE symbol = '{symbol}'
            AND created_at >= CURRENT_DATE() - 7
            ORDER BY created_at DESC
            """
            
            news_data = self.session.sql(news_query).to_pandas()
            
            # Generate signals using combined analysis
            signals = self._generate_multi_factor_signals(stock_data, news_data, symbol)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _generate_multi_factor_signals(self, stock_data: pd.DataFrame, 
                                     news_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate multi-factor trading signals"""
        signals = []
        
        if stock_data.empty:
            return pd.DataFrame(signals)
        
        try:
            latest_data = stock_data.iloc[0]  # Most recent data
            
            # Technical signals
            technical_score = self._calculate_technical_score(stock_data)
            
            # Sentiment signals
            sentiment_score = self._calculate_sentiment_score(news_data)
            
            # Combined signal strength
            combined_score = (technical_score * 0.7) + (sentiment_score * 0.3)
            
            # Generate signal based on combined score
            if combined_score > 0.6:
                signal_type = 'BUY'
                position_size = min(combined_score, 1.0)
            elif combined_score < -0.6:
                signal_type = 'SELL'
                position_size = min(abs(combined_score), 1.0)
            else:
                signal_type = 'HOLD'
                position_size = 0.0
            
            # Calculate price targets and stop losses
            current_price = latest_data['close_price']
            atr = latest_data.get('atr', current_price * 0.02)  # Fallback to 2%
            
            if signal_type == 'BUY':
                price_target = current_price * (1 + (combined_score * 0.05))  # Up to 5% target
                stop_loss = current_price * (1 - (atr / current_price) * 2)
            elif signal_type == 'SELL':
                price_target = current_price * (1 - (abs(combined_score) * 0.05))  # Up to 5% target
                stop_loss = current_price * (1 + (atr / current_price) * 2)
            else:
                price_target = current_price
                stop_loss = current_price
            
            signal = {
                'id': f"{symbol}_{signal_type}_{int(datetime.now().timestamp())}",
                'symbol': symbol,
                'signal_type': signal_type,
                'signal_strength': abs(combined_score),
                'confidence_score': self._calculate_confidence_score(stock_data, news_data),
                'price_target': price_target,
                'stop_loss': stop_loss,
                'position_size': position_size,
                'timeframe': '1D',
                'signal_source': 'AI_MULTI_FACTOR',
                'technical_indicators': json.dumps({
                    'rsi': float(latest_data.get('rsi', 50)),
                    'macd': float(latest_data.get('macd', 0)),
                    'bb_position': self._calculate_bb_position(latest_data),
                    'technical_score': technical_score
                }),
                'news_sentiment': json.dumps({
                    'avg_sentiment': sentiment_score,
                    'news_count': len(news_data),
                    'impact_weighted_sentiment': self._calculate_impact_weighted_sentiment(news_data)
                }),
                'risk_metrics': json.dumps({
                    'volatility': float(latest_data.get('volatility', 0.2)),
                    'atr_percent': float(atr / current_price) if current_price > 0 else 0.02,
                    'risk_reward_ratio': self._calculate_risk_reward_ratio(current_price, price_target, stop_loss)
                }),
                'expiry_at': (datetime.now() + timedelta(days=1)).isoformat()
            }
            
            signals.append(signal)
            
        except Exception as e:
            logger.error(f"Error in multi-factor signal generation: {str(e)}")
        
        return pd.DataFrame(signals)
    
    def _calculate_technical_score(self, stock_data: pd.DataFrame) -> float:
        """Calculate technical analysis score"""
        if stock_data.empty:
            return 0.0
        
        latest = stock_data.iloc[0]
        score = 0.0
        
        # RSI score
        rsi = latest.get('rsi', 50)
        if rsi < 30:
            score += 0.3  # Oversold - bullish
        elif rsi > 70:
            score -= 0.3  # Overbought - bearish
        
        # MACD score
        macd = latest.get('macd', 0)
        macd_signal = latest.get('macd_signal', 0)
        if macd > macd_signal:
            score += 0.2
        else:
            score -= 0.2
        
        # Bollinger Bands score
        bb_position = self._calculate_bb_position(latest)
        if bb_position < 0.2:
            score += 0.2  # Near lower band - bullish
        elif bb_position > 0.8:
            score -= 0.2  # Near upper band - bearish
        
        # Price trend (simple moving average comparison)
        if len(stock_data) >= 20:
            recent_avg = stock_data.head(5)['close_price'].mean()
            older_avg = stock_data.iloc[15:20]['close_price'].mean()
            if recent_avg > older_avg:
                score += 0.3
            else:
                score -= 0.3
        
        return max(-1.0, min(1.0, score))
    
    def _calculate_bb_position(self, latest_data: pd.Series) -> float:
        """Calculate position within Bollinger Bands"""
        bb_upper = latest_data.get('bb_upper')
        bb_lower = latest_data.get('bb_lower')
        close_price = latest_data.get('close_price')
        
        if pd.isna(bb_upper) or pd.isna(bb_lower) or bb_upper <= bb_lower:
            return 0.5  # Neutral position
        
        return (close_price - bb_lower) / (bb_upper - bb_lower)
    
    def _calculate_sentiment_score(self, news_data: pd.DataFrame) -> float:
        """Calculate news sentiment score"""
        if news_data.empty:
            return 0.0
        
        # Weight recent news more heavily
        news_data['age_weight'] = 1.0 / (1 + (datetime.now() - pd.to_datetime(news_data['created_at'])).dt.total_seconds() / 86400)
        
        # Calculate weighted sentiment
        weighted_sentiment = (news_data['sentiment_score'] * news_data['age_weight'] * news_data['impact_score']).sum()
        total_weight = (news_data['age_weight'] * news_data['impact_score']).sum()
        
        if total_weight > 0:
            return weighted_sentiment / total_weight
        else:
            return 0.0
    
    def _calculate_impact_weighted_sentiment(self, news_data: pd.DataFrame) -> float:
        """Calculate impact-weighted sentiment score"""
        if news_data.empty:
            return 0.0
        
        total_impact = news_data['impact_score'].sum()
        if total_impact > 0:
            return (news_data['sentiment_score'] * news_data['impact_score']).sum() / total_impact
        else:
            return 0.0
    
    def _calculate_confidence_score(self, stock_data: pd.DataFrame, news_data: pd.DataFrame) -> float:
        """Calculate signal confidence score"""
        confidence = 0.5  # Base confidence
        
        # Data quality factors
        if len(stock_data) >= 20:
            confidence += 0.1
        
        if not news_data.empty:
            confidence += 0.1
        
        # Volatility factor (lower volatility = higher confidence)
        if not stock_data.empty:
            latest_vol = stock_data.iloc[0].get('volatility', 0.3)
            if latest_vol < 0.2:
                confidence += 0.1
            elif latest_vol > 0.4:
                confidence -= 0.1
        
        # News consistency factor
        if len(news_data) > 0:
            sentiment_std = news_data['sentiment_score'].std()
            if sentiment_std < 0.3:  # Consistent sentiment
                confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_risk_reward_ratio(self, current_price: float, target_price: float, stop_loss: float) -> float:
        """Calculate risk-reward ratio"""
        if current_price <= 0:
            return 0.0
        
        potential_gain = abs(target_price - current_price)
        potential_loss = abs(current_price - stop_loss)
        
        if potential_loss > 0:
            return potential_gain / potential_loss
        else:
            return 0.0
    
    def insert_trading_signals(self, signals_df: pd.DataFrame):
        """Insert trading signals into Snowflake"""
        if signals_df.empty:
            logger.warning("Empty signals dataframe")
            return
        
        try:
            # Create Snowpark DataFrame
            snowpark_df = self.session.create_dataframe(signals_df)
            
            # Write to table
            snowpark_df.write.mode('append').save_as_table('trading_signals')
            
            logger.info(f"Inserted {len(signals_df)} trading signals into trading_signals table")
            
        except Exception as e:
            logger.error(f"Error inserting trading signals: {str(e)}")
            raise
    
    def get_active_signals(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get active trading signals"""
        try:
            where_clause = "WHERE status = 'ACTIVE' AND expiry_at > CURRENT_TIMESTAMP()"
            if symbol:
                where_clause += f" AND symbol = '{symbol}'"
            
            query = f"""
            SELECT *
            FROM trading_signals
            {where_clause}
            ORDER BY created_at DESC
            """
            
            return self.session.sql(query).to_pandas()
            
        except Exception as e:
            logger.error(f"Error fetching active signals: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Close Snowflake connections"""
        try:
            if self.session:
                self.session.close()
            if self.connection:
                self.connection.close()
            logger.info("Snowflake connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")

# Configuration helper
def get_snowflake_config() -> Dict[str, str]:
    """Get Snowflake configuration from environment variables"""
    return {
        'user': os.getenv('SNOWFLAKE_USER'),
        'password': os.getenv('SNOWFLAKE_PASSWORD'),
        'account': os.getenv('SNOWFLAKE_ACCOUNT'),
        'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE'),
        'database': os.getenv('SNOWFLAKE_DATABASE'),
        'schema': os.getenv('SNOWFLAKE_SCHEMA'),
        'role': os.getenv('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
    }

# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test Snowflake connection
    config = get_snowflake_config()
    
    if all(config.values()):
        try:
            sf_manager = SnowflakeManager(config)
            print("Successfully connected to Snowflake")
            
            # Test query
            result = sf_manager.execute_query("SELECT CURRENT_TIMESTAMP() as current_time")
            print(f"Current Snowflake time: {result}")
            
            sf_manager.close()
            
        except Exception as e:
            print(f"Error testing Snowflake connection: {str(e)}")
    else:
        print("Please set all required Snowflake environment variables")