"""
Exploratory Data Analysis Module

This module contains functions for performing EDA on financial news data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Perform exploratory data analysis on financial news"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA Analyzer
        
        Args:
            df: DataFrame containing news data
        """
        self.df = df
        
    def descriptive_statistics(self) -> Dict:
        """
        Calculate descriptive statistics
        
        Returns:
            Dictionary containing statistics
        """
        stats = {
            'total_articles': len(self.df),
            'unique_stocks': self.df['stock'].nunique(),
            'unique_publishers': self.df['publisher'].nunique(),
            'date_range': (self.df['date'].min(), self.df['date'].max()),
            'headline_length_stats': self.df['headline_length'].describe(),
            'word_count_stats': self.df['word_count'].describe()
        }
        
        return stats
    
    def publisher_analysis(self) -> pd.DataFrame:
        """
        Analyze publisher activity
        
        Returns:
            DataFrame with publisher statistics
        """
        publisher_stats = self.df.groupby('publisher').agg({
            'headline': 'count',
            'stock': 'nunique'
        }).rename(columns={
            'headline': 'article_count',
            'stock': 'unique_stocks_covered'
        }).sort_values('article_count', ascending=False)
        
        return publisher_stats
    
    def stock_analysis(self) -> pd.DataFrame:
        """
        Analyze stock coverage
        
        Returns:
            DataFrame with stock statistics
        """
        stock_stats = self.df.groupby('stock').agg({
            'headline': 'count',
            'publisher': 'nunique'
        }).rename(columns={
            'headline': 'article_count',
            'publisher': 'unique_publishers'
        }).sort_values('article_count', ascending=False)
        
        return stock_stats
    
    def time_series_analysis(self) -> pd.DataFrame:
        """
        Analyze publication patterns over time
        
        Returns:
            DataFrame with time series statistics
        """
        # Daily article counts
        daily_counts = self.df.groupby('date_only').size()
        
        # Monthly article counts
        monthly_counts = self.df.groupby([self.df['date'].dt.year, 
                                          self.df['date'].dt.month]).size()
        
        # Hourly patterns
        hourly_counts = self.df.groupby('hour').size()
        
        # Day of week patterns
        dow_counts = self.df.groupby('dayofweek').size()
        
        return {
            'daily': daily_counts,
            'monthly': monthly_counts,
            'hourly': hourly_counts,
            'day_of_week': dow_counts
        }
    
    def extract_keywords(self, n_keywords: int = 50) -> List[Tuple[str, int]]:
        """
        Extract most common keywords from headlines
        
        Args:
            n_keywords: Number of top keywords to extract
            
        Returns:
            List of (keyword, count) tuples
        """
        # Common stop words to exclude
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                     'are', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
                     'did', 'will', 'would', 'could', 'should', 'may', 'might'}
        
        # Extract all words
        all_words = []
        for headline in self.df['headline'].dropna():
            words = headline.lower().split()
            words = [w.strip('.,!?;:()[]{}') for w in words]
            words = [w for w in words if w and w not in stop_words and len(w) > 2]
            all_words.extend(words)
        
        # Count frequencies
        word_counts = Counter(all_words)
        
        return word_counts.most_common(n_keywords)
    
    def plot_publisher_distribution(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot top publishers by article count
        
        Args:
            top_n: Number of top publishers to show
            figsize: Figure size
        """
        publisher_stats = self.publisher_analysis().head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(publisher_stats)), publisher_stats['article_count'])
        plt.yticks(range(len(publisher_stats)), publisher_stats.index)
        plt.xlabel('Number of Articles')
        plt.ylabel('Publisher')
        plt.title(f'Top {top_n} Publishers by Article Count')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_time_series(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot time series patterns
        
        Args:
            figsize: Figure size
        """
        time_stats = self.time_series_analysis()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Daily trends
        time_stats['daily'].plot(ax=axes[0, 0], title='Daily Article Count')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Articles')
        
        # Hourly patterns
        time_stats['hourly'].plot(kind='bar', ax=axes[0, 1], title='Articles by Hour of Day')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Number of Articles')
        
        # Day of week patterns
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        time_stats['day_of_week'].plot(kind='bar', ax=axes[1, 0], title='Articles by Day of Week')
        axes[1, 0].set_xticklabels(day_names, rotation=45)
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Number of Articles')
        
        # Headline length distribution
        axes[1, 1].hist(self.df['headline_length'], bins=50, edgecolor='black')
        axes[1, 1].set_title('Distribution of Headline Length')
        axes[1, 1].set_xlabel('Headline Length (characters)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_stock_distribution(self, top_n: int = 30, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot top stocks by article count
        
        Args:
            top_n: Number of top stocks to show
            figsize: Figure size
        """
        stock_stats = self.stock_analysis().head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(stock_stats)), stock_stats['article_count'])
        plt.yticks(range(len(stock_stats)), stock_stats.index)
        plt.xlabel('Number of Articles')
        plt.ylabel('Stock Ticker')
        plt.title(f'Top {top_n} Stocks by Article Coverage')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        return plt.gcf()
