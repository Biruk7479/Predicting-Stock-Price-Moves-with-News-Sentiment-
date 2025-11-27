
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_stock_data(ticker: str, 
                       start_date: str, 
                       end_date: str,
                       retry_count: int = 3) -> Optional[pd.DataFrame]:
    """
    Download stock data for a single ticker
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        retry_count: Number of retries on failure
        
    Returns:
        DataFrame with stock data or None if failed
    """
    for attempt in range(retry_count):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if not df.empty:
                df['stock'] = ticker
                df = df.reset_index()
                logger.info(f"Successfully downloaded data for {ticker}: {len(df)} rows")
                return df
            else:
                logger.warning(f"No data available for {ticker}")
                return None
                
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{retry_count} failed for {ticker}: {e}")
            if attempt < retry_count - 1:
                time.sleep(2)  # Wait before retry
            
    logger.error(f"Failed to download data for {ticker} after {retry_count} attempts")
    return None


def download_multiple_stocks(tickers: List[str], 
                            start_date: str, 
                            end_date: str,
                            output_file: Optional[str] = None,
                            delay: float = 0.5) -> pd.DataFrame:
    """
    Download stock data for multiple tickers
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_file: Optional path to save the combined data
        delay: Delay between downloads (seconds)
        
    Returns:
        Combined DataFrame with all stock data
    """
    all_data = []
    failed_tickers = []
    
    logger.info(f"Starting download for {len(tickers)} tickers")
    
    for i, ticker in enumerate(tickers):
        logger.info(f"Processing {ticker} ({i+1}/{len(tickers)})")
        
        df = download_stock_data(ticker, start_date, end_date)
        
        if df is not None:
            all_data.append(df)
        else:
            failed_tickers.append(ticker)
        
        # Add delay to avoid rate limiting
        if i < len(tickers) - 1:
            time.sleep(delay)
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Successfully downloaded data for {len(all_data)} tickers")
        
        if output_file:
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Saved combined data to {output_file}")
        
        if failed_tickers:
            logger.warning(f"Failed to download data for {len(failed_tickers)} tickers: {failed_tickers[:10]}...")
        
        return combined_df
    else:
        logger.error("No data downloaded successfully")
        return pd.DataFrame()


def get_tickers_from_news_data(news_data_path: str, 
                               limit: Optional[int] = None) -> List[str]:
    """
    Extract unique stock tickers from news dataset
    
    Args:
        news_data_path: Path to news CSV file
        limit: Optional limit on number of tickers
        
    Returns:
        List of unique stock tickers
    """
    logger.info(f"Reading news data from {news_data_path}")
    
    # Read only the stock column to save memory
    df = pd.read_csv(news_data_path, usecols=['stock'])
    
    tickers = df['stock'].unique().tolist()
    tickers = sorted([t for t in tickers if isinstance(t, str) and len(t) <= 5])
    
    logger.info(f"Found {len(tickers)} unique tickers")
    
    if limit:
        tickers = tickers[:limit]
        logger.info(f"Limited to {limit} tickers")
    
    return tickers


def main():
    """Main function to download stock data"""
    
    # Configuration
    NEWS_DATA_PATH = "../Data/newsData/raw_analyst_ratings.csv"
    OUTPUT_DIR = Path("../Data/stockData")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Date range (adjust based on your news data)
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    
    # Get tickers from news data
    # Start with top 50 most mentioned stocks for testing
    logger.info("Extracting tickers from news data...")
    
    try:
        # Read news data and get top stocks by article count
        news_df = pd.read_csv(NEWS_DATA_PATH, usecols=['stock'])
        stock_counts = news_df['stock'].value_counts()
        top_tickers = stock_counts.head(50).index.tolist()
        
        logger.info(f"Selected top 50 stocks: {top_tickers[:10]}...")
        
        # Download stock data
        output_file = OUTPUT_DIR / "stock_prices.csv"
        stock_data = download_multiple_stocks(
            tickers=top_tickers,
            start_date=START_DATE,
            end_date=END_DATE,
            output_file=str(output_file),
            delay=0.5
        )
        
        logger.info(f"Download complete! Total records: {len(stock_data)}")
        logger.info(f"Data saved to: {output_file}")
        
        # Print summary
        if not stock_data.empty:
            print("\n=== Download Summary ===")
            print(f"Total records: {len(stock_data)}")
            print(f"Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
            print(f"Stocks downloaded: {stock_data['stock'].nunique()}")
            print(f"\nFirst few rows:")
            print(stock_data.head())
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
