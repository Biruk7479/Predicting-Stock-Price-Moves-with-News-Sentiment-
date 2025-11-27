"""
Technical Indicators Module

This module calculates technical indicators for stock analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock data"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize Technical Indicators calculator
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        
    def calculate_sma(self, column: str = 'Close', window: int = 20) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            column: Column to calculate SMA on
            window: Window size
            
        Returns:
            Series with SMA values
        """
        return self.df[column].rolling(window=window).mean()
    
    def calculate_ema(self, column: str = 'Close', span: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            column: Column to calculate EMA on
            span: Span for EMA
            
        Returns:
            Series with EMA values
        """
        return self.df[column].ewm(span=span, adjust=False).mean()
    
    def calculate_rsi(self, column: str = 'Close', period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            column: Column to calculate RSI on
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        delta = self.df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, column: str = 'Close', 
                      fast_period: int = 12, 
                      slow_period: int = 26, 
                      signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            column: Column to calculate MACD on
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Dictionary with MACD, signal, and histogram
        """
        ema_fast = self.df[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = self.df[column].ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'MACD': macd_line,
            'MACD_Signal': signal_line,
            'MACD_Hist': histogram
        }
    
    def calculate_bollinger_bands(self, column: str = 'Close', 
                                  window: int = 20, 
                                  num_std: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            column: Column to calculate bands on
            window: Window size
            num_std: Number of standard deviations
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        sma = self.df[column].rolling(window=window).mean()
        std = self.df[column].rolling(window=window).std()
        
        return {
            'BB_Upper': sma + (std * num_std),
            'BB_Middle': sma,
            'BB_Lower': sma - (std * num_std)
        }
    
    def calculate_atr(self, high_col: str = 'High', 
                     low_col: str = 'Low', 
                     close_col: str = 'Close', 
                     period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            high_col: High price column
            low_col: Low price column
            close_col: Close price column
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high_low = self.df[high_col] - self.df[low_col]
        high_close = np.abs(self.df[high_col] - self.df[close_col].shift())
        low_close = np.abs(self.df[low_col] - self.df[close_col].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Returns:
            DataFrame with all indicators
        """
        logger.info("Calculating technical indicators...")
        
        result_df = self.df.copy()
        
        # Moving Averages
        result_df['SMA_20'] = self.calculate_sma(window=20)
        result_df['SMA_50'] = self.calculate_sma(window=50)
        result_df['EMA_12'] = self.calculate_ema(span=12)
        result_df['EMA_26'] = self.calculate_ema(span=26)
        
        # RSI
        result_df['RSI_14'] = self.calculate_rsi(period=14)
        
        # MACD
        macd_data = self.calculate_macd()
        result_df['MACD'] = macd_data['MACD']
        result_df['MACD_Signal'] = macd_data['MACD_Signal']
        result_df['MACD_Hist'] = macd_data['MACD_Hist']
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands()
        result_df['BB_Upper'] = bb_data['BB_Upper']
        result_df['BB_Middle'] = bb_data['BB_Middle']
        result_df['BB_Lower'] = bb_data['BB_Lower']
        
        # ATR
        if all(col in result_df.columns for col in ['High', 'Low', 'Close']):
            result_df['ATR_14'] = self.calculate_atr(period=14)
        
        # Daily Returns
        result_df['Daily_Return'] = result_df['Close'].pct_change() * 100
        
        # Volume Moving Average
        if 'Volume' in result_df.columns:
            result_df['Volume_SMA_20'] = result_df['Volume'].rolling(window=20).mean()
        
        logger.info("Technical indicators calculated successfully")
        
        return result_df
    
    def calculate_daily_returns(self, column: str = 'Close') -> pd.Series:
        """
        Calculate daily percentage returns
        
        Args:
            column: Column to calculate returns on
            
        Returns:
            Series with daily returns
        """
        return self.df[column].pct_change() * 100


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add technical indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with technical indicators
    """
    calculator = TechnicalIndicators(df)
    return calculator.calculate_all_indicators()
