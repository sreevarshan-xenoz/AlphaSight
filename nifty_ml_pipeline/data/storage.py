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
                logger.warning(f"No parquet files found for symbol {symbol}")
                return []
            
            # Read and combine data
            dfs = []
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path, engine='pyarrow')
                    # Filter by date range
                    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {str(e)}")
                    continue
            
            if not dfs:
                logger.info(f"No data found for {symbol} in date range {start_date} to {end_date}")
                return []
            
            # Combine and sort
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').drop_duplicates()
            
            # Convert back to PriceData objects
            price_data_list = self._dataframe_to_price_data(combined_df, symbol)
            
            logger.info(f"Retrieved {len(price_data_list)} price records for {symbol}")
            return price_data_list
            
        except Exception as e:
            logger.error(f"Failed to retrieve price data for {symbol}: {str(e)}")
            return []
    
    def retrieve_news_data(self, start_date: datetime, end_date: datetime) -> List[NewsData]:
        """Retrieve news data for specified date range.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            List[NewsData]: Retrieved news data
        """
        try:
            # Find relevant partition files
            parquet_files = []
            for year_dir in self.news_data_path.glob("year=*"):
                for month_dir in year_dir.glob("month=*"):
                    for parquet_file in month_dir.glob("*.parquet"):
                        parquet_files.append(parquet_file)
            
            if not parquet_files:
                logger.warning("No news parquet files found")
                return []
            
            # Read and combine data
            dfs = []
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path, engine='pyarrow')
                    # Filter by date range
                    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                    if not df.empty:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {str(e)}")
                    continue
            
            if not dfs:
                logger.info(f"No news data found in date range {start_date} to {end_date}")
                return []
            
            # Combine and sort
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').drop_duplicates()
            
            # Convert back to NewsData objects
            news_data_list = self._dataframe_to_news_data(combined_df)
            
            logger.info(f"Retrieved {len(news_data_list)} news records")
            return news_data_list
            
        except Exception as e:
            logger.error(f"Failed to retrieve news data: {str(e)}")
            return []
    
    def cleanup_old_data(self) -> bool:
        """Remove data outside the rolling window.
        
        Returns:
            bool: True if cleanup successful
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.rolling_window_days)
            cutoff_period = cutoff_date.to_period('M')
            
            cleaned_files = 0
            
            # Clean price data
            for symbol_dir in self.price_data_path.glob("*"):
                if symbol_dir.is_dir():
                    cleaned_files += self._cleanup_directory(symbol_dir, cutoff_period)
            
            # Clean news data
            cleaned_files += self._cleanup_directory(self.news_data_path, cutoff_period)
            
            logger.info(f"Cleaned up {cleaned_files} old data files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")
            return False
    
    def _cleanup_directory(self, directory: Path, cutoff_period) -> int:
        """Clean up old files in a directory based on cutoff period.
        
        Args:
            directory: Directory to clean
            cutoff_period: Period before which files should be deleted
            
        Returns:
            int: Number of files cleaned
        """
        cleaned_files = 0
        
        for year_dir in directory.glob("year=*"):
            try:
                year = int(year_dir.name.split("=")[1])
                
                for month_dir in year_dir.glob("month=*"):
                    try:
                        month = int(month_dir.name.split("=")[1])
                        file_period = pd.Period(year=year, month=month, freq='M')
                        
                        if file_period < cutoff_period:
                            # Remove all files in this month directory
                            for file_path in month_dir.glob("*.parquet"):
                                file_path.unlink()
                                cleaned_files += 1
                                logger.debug(f"Removed old file: {file_path}")
                            
                            # Remove empty directory
                            if not any(month_dir.iterdir()):
                                month_dir.rmdir()
                                logger.debug(f"Removed empty directory: {month_dir}")
                    
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Invalid month directory name: {month_dir.name}")
                        continue
                
                # Remove empty year directory
                if not any(year_dir.iterdir()):
                    year_dir.rmdir()
                    logger.debug(f"Removed empty directory: {year_dir}")
            
            except (ValueError, IndexError) as e:
                logger.warning(f"Invalid year directory name: {year_dir.name}")
                continue
        
        return cleaned_files
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dict: Storage statistics
        """
        stats = {
            'price_data_files': 0,
            'news_data_files': 0,
            'total_size_mb': 0,
            'oldest_data': None,
            'newest_data': None
        }
        
        try:
            # Count price data files
            for parquet_file in self.price_data_path.rglob("*.parquet"):
                stats['price_data_files'] += 1
                stats['total_size_mb'] += parquet_file.stat().st_size / (1024 * 1024)
            
            # Count news data files
            for parquet_file in self.news_data_path.rglob("*.parquet"):
                stats['news_data_files'] += 1
                stats['total_size_mb'] += parquet_file.stat().st_size / (1024 * 1024)
            
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
        
        return stats
    
    def _price_data_to_dataframe(self, price_data: List[PriceData]) -> pd.DataFrame:
        """Convert PriceData objects to DataFrame."""
        data = []
        for item in price_data:
            data.append({
                'symbol': item.symbol,
                'timestamp': item.timestamp,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume
            })
        return pd.DataFrame(data)
    
    def _news_data_to_dataframe(self, news_data: List[NewsData]) -> pd.DataFrame:
        """Convert NewsData objects to DataFrame."""
        data = []
        for item in news_data:
            data.append({
                'source': item.source,
                'timestamp': item.timestamp,
                'headline': item.headline,
                'url': item.url,
                'sentiment_score': item.sentiment_score
            })
        return pd.DataFrame(data)
    
    def _dataframe_to_price_data(self, df: pd.DataFrame, symbol: str) -> List[PriceData]:
        """Convert DataFrame to PriceData objects."""
        price_data_list = []
        for _, row in df.iterrows():
            try:
                price_data = PriceData(
                    symbol=symbol,
                    timestamp=pd.to_datetime(row['timestamp']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )
                price_data_list.append(price_data)
            except Exception as e:
                logger.warning(f"Failed to convert row to PriceData: {str(e)}")
                continue
        return price_data_list
    
    def _dataframe_to_news_data(self, df: pd.DataFrame) -> List[NewsData]:
        """Convert DataFrame to NewsData objects."""
        news_data_list = []
        for _, row in df.iterrows():
            try:
                news_data = NewsData(
                    source=row['source'],
                    timestamp=pd.to_datetime(row['timestamp']),
                    headline=row['headline'],
                    url=row.get('url'),
                    sentiment_score=row.get('sentiment_score')
                )
                news_data_list.append(news_data)
            except Exception as e:
                logger.warning(f"Failed to convert row to NewsData: {str(e)}")
                continue
        return news_data_list


class DataCache:
    """In-memory cache for frequently accessed data."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize data cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None if not found
        """
        if key in self._cache:
            self._access_times[key] = datetime.now()
            return self._cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove oldest item if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[key] = value
        self._access_times[key] = datetime.now()
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._access_times.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)