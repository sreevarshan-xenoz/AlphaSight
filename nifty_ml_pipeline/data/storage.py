# nifty_ml_pipeline/data/storage.py
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .models import PriceData, NewsData
from .validator import DataValidator


logger = logging.getLogger(__name__)


class DataStorage:
    """Local data storage system using Parquet format for efficient I/O.
    
    Implements data partitioning by symbol and date for optimal queries
    and automatic cleanup for data outside rolling window.
    """
    
    def __init__(self, base_path: str = "data", rolling_window_days: int = 365):
        """Initialize data storage system.
        
        Args:
            base_path: Base directory for data storage
            rolling_window_days: Number of days to retain data
        """
        self.base_path = Path(base_path)
        self.rolling_window_days = rolling_window_days
        self.validator = DataValidator()
        
        # Create directory structure
        self.price_data_path = self.base_path / "prices"
        self.news_data_path = self.base_path / "news"
        self.cache_path = self.base_path / "cache"
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for path in [self.price_data_path, self.news_data_path, self.cache_path]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
    
    def store_price_data(self, price_data: List[PriceData], symbol: str) -> bool:
        """Store price data with partitioning by symbol and date.
        
        Args:
            price_data: List of PriceData objects to store
            symbol: Symbol name for partitioning
            
        Returns:
            bool: True if storage successful
        """
        try:
            if not price_data:
                logger.warning("No price data provided for storage")
                return False
            
            # Validate data before storage
            self.validator.validate_price_data(price_data)
            
            # Convert to DataFrame
            df = self._price_data_to_dataframe(price_data)
            
            # Partition by date (year-month)
            for date_group, group_df in df.groupby(df['timestamp'].dt.to_period('M')):
                partition_path = self.price_data_path / symbol / f"year={date_group.year}" / f"month={date_group.month:02d}"
                partition_path.mkdir(parents=True, exist_ok=True)
                
                file_path = partition_path / f"{symbol}_{date_group}.parquet"
                
                # Write to Parquet with compression
                group_df.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )
                
                logger.info(f"Stored {len(group_df)} price records for {symbol} in {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store price data for {symbol}: {str(e)}")
            return False
    
    def store_news_data(self, news_data: List[NewsData]) -> bool:
        """Store news data with partitioning by date.
        
        Args:
            news_data: List of NewsData objects to store
            
        Returns:
            bool: True if storage successful
        """
        try:
            if not news_data:
                logger.warning("No news data provided for storage")
                return False
            
            # Validate data before storage
            self.validator.validate_news_data(news_data)
            
            # Convert to DataFrame
            df = self._news_data_to_dataframe(news_data)
            
            # Partition by date (year-month)
            for date_group, group_df in df.groupby(df['timestamp'].dt.to_period('M')):
                partition_path = self.news_data_path / f"year={date_group.year}" / f"month={date_group.month:02d}"
                partition_path.mkdir(parents=True, exist_ok=True)
                
                file_path = partition_path / f"news_{date_group}.parquet"
                
                # Write to Parquet with compression
                group_df.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )
                
                logger.info(f"Stored {len(group_df)} news records in {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store news data: {str(e)}")
            return False
    
    def retrieve_price_data(self, symbol: str, start_date: datetime, end_date: datetime) -> List[PriceData]:
        """Retrieve price data for specified symbol and date range.
        
        Args:
            symbol: Symbol to retrieve data for
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            List[PriceData]: Retrieved price data
        """
        try:
            symbol_path = self.price_data_path / symbol
            
            if not symbol_path.exists():
                logger.warning(f"No data found for symbol {symbol}")
                return []
            
            # Find relevant partition files
            parquet_files = []
            for year_dir in symbol_path.glob("year=*"):
                for month_dir in year_dir.glob("month=*"):
                    for parquet_file in month_dir.glob("*.parquet"):
                        parquet_files.append(parquet_file)
            
            if not parquet_files:
 