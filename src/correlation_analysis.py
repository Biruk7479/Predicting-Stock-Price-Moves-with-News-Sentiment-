"""
Correlation Analysis Module

This module performs correlation analysis between sentiment and stock movements.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyze correlation between news sentiment and stock movements"""
    
    def __init__(self, sentiment_df: pd.DataFrame, stock_df: pd.DataFrame):
        """
        Initialize Correlation Analyzer
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            stock_df: DataFrame with stock price data
        """
        self.sentiment_df = sentiment_df
        self.stock_df = stock_df
        self.merged_df = None
        
    def align_data(self, stock_column: str = 'stock', 
                   date_column: str = 'Date') -> pd.DataFrame:
        """
        Align sentiment and stock data by date and ticker
        
        Args:
            stock_column: Column name for stock ticker
            date_column: Column name for date
            
        Returns:
            Merged DataFrame
        """
        logger.info("Aligning sentiment and stock data...")
        
        # Ensure date columns are in the same format
        sentiment_temp = self.sentiment_df.copy()
        stock_temp = self.stock_df.copy()
        
        # Convert dates to datetime
        if date_column in stock_temp.columns:
            stock_temp[date_column] = pd.to_datetime(stock_temp[date_column])
            stock_temp['date_only'] = stock_temp[date_column].dt.date
        
        if 'date_only' not in sentiment_temp.columns:
            sentiment_temp['date_only'] = pd.to_datetime(sentiment_temp['date']).dt.date
        
        # Merge on stock ticker and date
        self.merged_df = pd.merge(
            sentiment_temp,
            stock_temp,
            left_on=[stock_column, 'date_only'],
            right_on=[stock_column, 'date_only'],
            how='inner'
        )
        
        logger.info(f"Aligned {len(self.merged_df)} records")
        
        return self.merged_df
    
    def calculate_correlations(self, sentiment_cols: list, 
                              return_col: str = 'Daily_Return') -> Dict[str, Tuple[float, float]]:
        """
        Calculate correlations between sentiment and returns
        
        Args:
            sentiment_cols: List of sentiment column names
            return_col: Column name for returns
            
        Returns:
            Dictionary with correlation coefficients and p-values
        """
        if self.merged_df is None:
            raise ValueError("Data not aligned. Call align_data() first.")
        
        correlations = {}
        
        for col in sentiment_cols:
            if col in self.merged_df.columns and return_col in self.merged_df.columns:
                # Remove NaN values
                clean_data = self.merged_df[[col, return_col]].dropna()
                
                if len(clean_data) > 0:
                    # Pearson correlation
                    pearson_corr, pearson_pval = stats.pearsonr(
                        clean_data[col], 
                        clean_data[return_col]
                    )
                    
                    # Spearman correlation
                    spearman_corr, spearman_pval = stats.spearmanr(
                        clean_data[col], 
                        clean_data[return_col]
                    )
                    
                    correlations[col] = {
                        'pearson': (pearson_corr, pearson_pval),
                        'spearman': (spearman_corr, spearman_pval),
                        'n_samples': len(clean_data)
                    }
        
        return correlations
    
    def analyze_by_stock(self, sentiment_col: str = 'vader_compound',
                        return_col: str = 'Daily_Return') -> pd.DataFrame:
        """
        Analyze correlation for each stock separately
        
        Args:
            sentiment_col: Sentiment column to use
            return_col: Return column to use
            
        Returns:
            DataFrame with per-stock correlations
        """
        if self.merged_df is None:
            raise ValueError("Data not aligned. Call align_data() first.")
        
        results = []
        
        for stock in self.merged_df['stock'].unique():
            stock_data = self.merged_df[self.merged_df['stock'] == stock]
            clean_data = stock_data[[sentiment_col, return_col]].dropna()
            
            if len(clean_data) >= 10:  # Minimum sample size
                try:
                    corr, pval = stats.pearsonr(clean_data[sentiment_col], 
                                               clean_data[return_col])
                    
                    results.append({
                        'stock': stock,
                        'correlation': corr,
                        'p_value': pval,
                        'n_samples': len(clean_data),
                        'significant': pval < 0.05
                    })
                except Exception as e:
                    logger.warning(f"Error calculating correlation for {stock}: {e}")
        
        return pd.DataFrame(results).sort_values('correlation', ascending=False)
    
    def analyze_sentiment_impact(self, sentiment_col: str = 'vader_compound',
                                return_col: str = 'Daily_Return',
                                sentiment_threshold: float = 0.05) -> Dict:
        """
        Analyze impact of positive vs negative sentiment on returns
        
        Args:
            sentiment_col: Sentiment column to use
            return_col: Return column to use
            sentiment_threshold: Threshold for positive/negative classification
            
        Returns:
            Dictionary with impact analysis results
        """
        if self.merged_df is None:
            raise ValueError("Data not aligned. Call align_data() first.")
        
        clean_data = self.merged_df[[sentiment_col, return_col]].dropna()
        
        # Classify sentiment
        positive_mask = clean_data[sentiment_col] > sentiment_threshold
        negative_mask = clean_data[sentiment_col] < -sentiment_threshold
        neutral_mask = ~(positive_mask | negative_mask)
        
        # Calculate average returns
        positive_returns = clean_data.loc[positive_mask, return_col]
        negative_returns = clean_data.loc[negative_mask, return_col]
        neutral_returns = clean_data.loc[neutral_mask, return_col]
        
        # Statistical tests
        pos_vs_neg_ttest = stats.ttest_ind(positive_returns, negative_returns)
        pos_vs_neu_ttest = stats.ttest_ind(positive_returns, neutral_returns)
        
        results = {
            'positive_sentiment': {
                'count': len(positive_returns),
                'mean_return': positive_returns.mean(),
                'std_return': positive_returns.std(),
                'median_return': positive_returns.median()
            },
            'negative_sentiment': {
                'count': len(negative_returns),
                'mean_return': negative_returns.mean(),
                'std_return': negative_returns.std(),
                'median_return': negative_returns.median()
            },
            'neutral_sentiment': {
                'count': len(neutral_returns),
                'mean_return': neutral_returns.mean(),
                'std_return': neutral_returns.std(),
                'median_return': neutral_returns.median()
            },
            'statistical_tests': {
                'positive_vs_negative': {
                    't_statistic': pos_vs_neg_ttest.statistic,
                    'p_value': pos_vs_neg_ttest.pvalue
                },
                'positive_vs_neutral': {
                    't_statistic': pos_vs_neu_ttest.statistic,
                    'p_value': pos_vs_neu_ttest.pvalue
                }
            }
        }
        
        return results
    
    def calculate_lagged_correlation(self, sentiment_col: str = 'vader_compound',
                                    return_col: str = 'Daily_Return',
                                    max_lag: int = 5) -> pd.DataFrame:
        """
        Calculate correlation with different time lags
        
        Args:
            sentiment_col: Sentiment column to use
            return_col: Return column to use
            max_lag: Maximum lag to test (in days)
            
        Returns:
            DataFrame with lagged correlations
        """
        if self.merged_df is None:
            raise ValueError("Data not aligned. Call align_data() first.")
        
        results = []
        
        for lag in range(0, max_lag + 1):
            # Sort by date to ensure proper lagging
            temp_df = self.merged_df.sort_values('date_only').copy()
            
            # Create lagged return column
            temp_df[f'return_lag_{lag}'] = temp_df.groupby('stock')[return_col].shift(-lag)
            
            # Calculate correlation
            clean_data = temp_df[[sentiment_col, f'return_lag_{lag}']].dropna()
            
            if len(clean_data) > 0:
                corr, pval = stats.pearsonr(clean_data[sentiment_col], 
                                           clean_data[f'return_lag_{lag}'])
                
                results.append({
                    'lag_days': lag,
                    'correlation': corr,
                    'p_value': pval,
                    'n_samples': len(clean_data)
                })
        
        return pd.DataFrame(results)


def merge_sentiment_and_stock_data(sentiment_df: pd.DataFrame, 
                                  stock_df: pd.DataFrame,
                                  stock_col: str = 'stock',
                                  date_col: str = 'date') -> pd.DataFrame:
    """
    Convenience function to merge sentiment and stock data
    
    Args:
        sentiment_df: DataFrame with sentiment scores
        stock_df: DataFrame with stock prices
        stock_col: Column name for stock ticker
        date_col: Column name for date
        
    Returns:
        Merged DataFrame
    """
    analyzer = CorrelationAnalyzer(sentiment_df, stock_df)
    return analyzer.align_data(stock_column=stock_col, date_column=date_col)
