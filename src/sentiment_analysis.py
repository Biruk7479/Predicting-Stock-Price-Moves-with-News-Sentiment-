"""
Sentiment Analysis Module

This module performs sentiment analysis on financial news headlines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Sentiment analysis libraries
from textblob import TextBlob
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
except ImportError:
    pass

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Perform sentiment analysis on news headlines"""
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.sia = None
        self._initialize_vader()
        
    def _initialize_vader(self):
        """Initialize VADER sentiment analyzer"""
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"VADER not available: {e}")
            
    def analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        blob = TextBlob(str(text))
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if self.sia is None:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
            
        scores = self.sia.polarity_scores(str(text))
        return scores
    
    def classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score
        
        Args:
            compound_score: VADER compound score
            
        Returns:
            Sentiment classification
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'headline') -> pd.DataFrame:
        """
        Perform sentiment analysis on a DataFrame
        
        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with sentiment scores
        """
        logger.info(f"Analyzing sentiment for {len(df)} rows...")
        
        # Create a copy
        result_df = df.copy()
        
        # TextBlob analysis
        textblob_results = df[text_column].apply(self.analyze_with_textblob)
        result_df['polarity'] = textblob_results.apply(lambda x: x['polarity'])
        result_df['subjectivity'] = textblob_results.apply(lambda x: x['subjectivity'])
        
        # VADER analysis
        if self.sia is not None:
            vader_results = df[text_column].apply(self.analyze_with_vader)
            result_df['vader_compound'] = vader_results.apply(lambda x: x['compound'])
            result_df['vader_pos'] = vader_results.apply(lambda x: x['pos'])
            result_df['vader_neu'] = vader_results.apply(lambda x: x['neu'])
            result_df['vader_neg'] = vader_results.apply(lambda x: x['neg'])
            result_df['sentiment_class'] = result_df['vader_compound'].apply(self.classify_sentiment)
        else:
            # Fallback to TextBlob-based classification
            result_df['sentiment_class'] = result_df['polarity'].apply(
                lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral')
            )
        
        logger.info("Sentiment analysis complete.")
        return result_df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of sentiment analysis
        
        Args:
            df: DataFrame with sentiment scores
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'sentiment_distribution': df['sentiment_class'].value_counts().to_dict(),
            'avg_polarity': df['polarity'].mean(),
            'avg_subjectivity': df['subjectivity'].mean(),
        }
        
        if 'vader_compound' in df.columns:
            summary['avg_vader_compound'] = df['vader_compound'].mean()
            summary['vader_positive_ratio'] = (df['vader_compound'] > 0.05).mean()
            summary['vader_negative_ratio'] = (df['vader_compound'] < -0.05).mean()
            summary['vader_neutral_ratio'] = ((df['vader_compound'] >= -0.05) & 
                                              (df['vader_compound'] <= 0.05)).mean()
        
        return summary
    
    def aggregate_daily_sentiment(self, df: pd.DataFrame, stock_column: str = 'stock',
                                  date_column: str = 'date_only') -> pd.DataFrame:
        """
        Aggregate sentiment scores by stock and date
        
        Args:
            df: DataFrame with sentiment scores
            stock_column: Column containing stock ticker
            date_column: Column containing date
            
        Returns:
            DataFrame with aggregated daily sentiment
        """
        agg_dict = {
            'polarity': 'mean',
            'subjectivity': 'mean',
            'headline': 'count'
        }
        
        if 'vader_compound' in df.columns:
            agg_dict.update({
                'vader_compound': 'mean',
                'vader_pos': 'mean',
                'vader_neu': 'mean',
                'vader_neg': 'mean'
            })
        
        daily_sentiment = df.groupby([stock_column, date_column]).agg(agg_dict)
        daily_sentiment.rename(columns={'headline': 'article_count'}, inplace=True)
        
        return daily_sentiment.reset_index()


def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
