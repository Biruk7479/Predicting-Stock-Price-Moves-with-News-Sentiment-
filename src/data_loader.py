"""
Data Loader Module

This module handles loading and preprocessing of financial news data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess financial news data"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            nrows: Number of rows to load (None for all)
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.df = pd.read_csv(
                self.data_path,
                nrows=nrows,
                parse_dates=['date']
            )
            
            # Drop the unnamed index column if it exists
            if 'Unnamed: 0' in self.df.columns:
                self.df = self.df.drop('Unnamed: 0', axis=1)
            
            logger.info(f"Loaded {len(self.df)} rows")
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the data
        
        Returns:
            Preprocessed DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        logger.info("Preprocessing data...")
        
        # Create a copy to avoid modifying original
        df = self.df.copy()
        
        # Convert date to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Extract date components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['date_only'] = df['date'].dt.date
        
        # Calculate headline length
        df['headline_length'] = df['headline'].str.len()
        df['word_count'] = df['headline'].str.split().str.len()
        
        # Clean publisher names
        df['publisher'] = df['publisher'].str.strip()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['headline', 'date', 'stock'])
        
        logger.info(f"Preprocessing complete. {len(df)} rows after cleaning.")
        
        self.df = df
        return df
    
    def get_stock_list(self) -> list:
        """Get list of unique stock tickers"""
        if self.df is None:
            raise ValueError("Data not loaded.")
        return sorted(self.df['stock'].unique().tolist())
    
    def filter_by_stock(self, stock_ticker: str) -> pd.DataFrame:
        """
        Filter data by stock ticker
        
        Args:
            stock_ticker: Stock ticker symbol
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
        return self.df[self.df['stock'] == stock_ticker].copy()
    
    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get date range of the dataset"""
        if self.df is None:
            raise ValueError("Data not loaded.")
        return self.df['date'].min(), self.df['date'].max()


def load_and_prepare_data(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function to load and prepare data
    
    Args:
        file_path: Path to CSV file
        nrows: Number of rows to load
        
    Returns:
        Preprocessed DataFrame
    """
    loader = DataLoader(file_path)
    loader.load_data(nrows=nrows)
    return loader.preprocess()
