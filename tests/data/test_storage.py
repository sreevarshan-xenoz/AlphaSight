# tests/data/test_storage.py
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from nifty_ml_pipeline.data.storage import DataStorage, DataCache
from nifty_ml_pipeline.data.models import PriceData, NewsData


class TestDataStorage:
    """Test suite for DataStorage class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = DataStorage(base_path=temp_dir, rolling_window_days=30)
        yield storage
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        return [
            PriceData(
                symbol='NIFTY 50',
                timestamp=datetime(2024, 1, 1),
                open=22000.0,
                high=22150.0,
                low=21950.0,
                close=22100.0,
                volume=1000000
            ),
            PriceData(
                symbol='NIFTY 50',
                timestamp=datetime(2024, 1, 2),
                open=22100.0,
                high=22200.0,
                low=22000.0,
                close=22050.0,
                volume=1100000
            ),
            PriceData(
                symbol='NIFTY 50',
                timestamp=datetime(2024, 2, 1),  # Different month for partitioning test
                open=22200.0,
                high=22300.0,
                low=22100.0,
                close=22250.0,
                volume=1200000
            )
        ]
    
    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data for testing."""
        return [
            NewsData(
                source='test_source',
                timestamp=datetime(2024, 1, 1),
                headline='NIFTY 50 shows strong performance',
                url='http://example.com/news/1'
            ),
            NewsData(
                source='test_source',
                timestamp=datetime(2024, 1, 2),
                headline='Market volatility affects indices',
                url='http://example.com/news/2'
            ),
            NewsData(
                source='test_source',
                timestamp=datetime(2024, 2, 1),  # Different month
                headline='Banking sector drives market gains',
                url='http://example.com/news/3'
            )
        ]
    
    def test_storage_initialization(self, temp_storage):
        """Test storage system initialization."""
        assert temp_storage.base_path.exists()
        assert temp_storage.price_data_path.exists()
        assert temp_storage.news_data_path.exists()
        assert temp_storage.cache_path.exists()
        assert temp_storage.rolling_window_days == 30
    
    def test_store_price_data(self, temp_storage, sample_price_data):
        """Test storing price data with partitioning."""
        result = temp_storage.store_price_data(sample_price_data, 'NIFTY 50')
        
        assert result is True
        
        # Check that partitioned files were created
        symbol_path = temp_storage.price_data_path / 'NIFTY 50'
        assert symbol_path.exists()
        
        # Check year/month partitions
        jan_partition = symbol_path / 'year=2024' / 'month=01'
        feb_partition = symbol_path / 'year=2024' / 'month=02'
        
        assert jan_partition.exists()
        assert feb_partition.exists()
        
        # Check parquet files exist
        jan_files = list(jan_partition.glob('*.parquet'))
        feb_files = list(feb_partition.glob('*.parquet'))
        
        assert len(jan_files) == 1
        assert len(feb_files) == 1
    
    def test_store_news_data(self, temp_storage, sample_news_data):
        """Test storing news data with partitioning."""
        result = temp_storage.store_news_data(sample_news_data)
        
        assert result is True
        
        # Check that partitioned files were created
        jan_partition = temp_storage.news_data_path / 'year=2024' / 'month=01'
        feb_partition = temp_storage.news_data_path / 'year=2024' / 'month=02'
        
        assert jan_partition.exists()
        assert feb_partition.exists()
        
        # Check parquet files exist
        jan_files = list(jan_partition.glob('*.parquet'))
        feb_files = list(feb_partition.glob('*.parquet'))
        
        assert len(jan_files) == 1
        assert len(feb_files) == 1
    
    def test_retrieve_price_data(self, temp_storage, sample_price_data):
        """Test retrieving price data."""
        # First store the data
        temp_storage.store_price_data(sample_price_data, 'NIFTY 50')
        
        # Retrieve data for January
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        retrieved_data = temp_storage.retrieve_price_data('NIFTY 50', start_date, end_date)
        
        assert len(retrieved_data) == 2  # Two January records
        assert all(isinstance(item, PriceData) for item in retrieved_data)
        assert all(item.symbol == 'NIFTY 50' for item in retrieved_data)
        
        # Check data is sorted by timestamp
        timestamps = [item.timestamp for item in retrieved_data]
        assert timestamps == sorted(timestamps)
    
    def test_retrieve_news_data(self, temp_storage, sample_news_data):
        """Test retrieving news data."""
        # First store the data
        temp_storage.store_news_data(sample_news_data)
        
        # Retrieve data for January
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        retrieved_data = temp_storage.retrieve_news_data(start_date, end_date)
        
        assert len(retrieved_data) == 2  # Two January records
        assert all(isinstance(item, NewsData) for item in retrieved_data)
        
        # Check data is sorted by timestamp
        timestamps = [item.timestamp for item in retrieved_data]
        assert timestamps == sorted(timestamps)
    
    def test_retrieve_nonexistent_data(self, temp_storage):
        """Test retrieving data that doesn't exist."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        # Try to retrieve price data for non-existent symbol
        price_data = temp_storage.retrieve_price_data('NONEXISTENT', start_date, end_date)
        assert len(price_data) == 0
        
        # Try to retrieve news data when no data exists
        news_data = temp_storage.retrieve_news_data(start_date, end_date)
        assert len(news_data) == 0
    
    def test_cleanup_old_data(self, temp_storage):
        """Test cleanup of old data outside rolling window."""
        # Create old data (outside rolling window)
        old_data = [
            PriceData(
                symbol='NIFTY 50',
                timestamp=datetime.now() - timedelta(days=40),  # Outside 30-day window
                open=20000.0,
                high=20100.0,
                low=19900.0,
                close=20050.0,
                volume=500000
            )
        ]
        
        # Create recent data (within rolling window)
        recent_data = [
            PriceData(
                symbol='NIFTY 50',
                timestamp=datetime.now() - timedelta(days=10),  # Within 30-day window
                open=22000.0,
                high=22100.0,
                low=21900.0,
                close=22050.0,
                volume=1000000
            )
        ]
        
        # Store both old and recent data
        temp_storage.store_price_data(old_data, 'NIFTY 50')
        temp_storage.store_price_data(recent_data, 'NIFTY 50')
        
        # Count files before cleanup
        files_before = len(list(temp_storage.price_data_path.rglob('*.parquet')))
        assert files_before > 0
        
        # Perform cleanup
        result = temp_storage.cleanup_old_data()
        assert result is True
        
        # Count files after cleanup (should be fewer)
        files_after = len(list(temp_storage.price_data_path.rglob('*.parquet')))
        assert files_after < files_before
    
    def test_get_storage_stats(self, temp_storage, sample_price_data, sample_news_data):
        """Test getting storage statistics."""
        # Store some data
        temp_storage.store_price_data(sample_price_data, 'NIFTY 50')
        temp_storage.store_news_data(sample_news_data)
        
        stats = temp_storage.get_storage_stats()
        
        assert isinstance(stats, dict)
        assert 'price_data_files' in stats
        assert 'news_data_files' in stats
        assert 'total_size_mb' in stats
        
        assert stats['price_data_files'] > 0
        assert stats['news_data_files'] > 0
        assert stats['total_size_mb'] > 0
    
    def test_store_empty_data(self, temp_storage):
        """Test storing empty data lists."""
        # Test empty price data
        result = temp_storage.store_price_data([], 'NIFTY 50')
        assert result is False
        
        # Test empty news data
        result = temp_storage.store_news_data([])
        assert result is False
    
    def test_data_validation_during_storage(self, temp_storage):
        """Test that data validation occurs during storage."""
        # Create invalid price data
        invalid_price_data = [
            PriceData(
                symbol='NIFTY 50',
                timestamp=datetime(2024, 1, 2),  # Later timestamp first
                open=22000.0,
                high=22100.0,
                low=21900.0,
                close=22050.0,
                volume=1000000
            ),
            PriceData(
                symbol='NIFTY 50',
                timestamp=datetime(2024, 1, 1),  # Earlier timestamp second (violates chronological order)
                open=21000.0,
                high=21100.0,
                low=20900.0,
                close=21050.0,
                volume=900000
            )
        ]
        
        # This should fail due to chronological ordering validation
        result = temp_storage.store_price_data(invalid_price_data, 'NIFTY 50')
        assert result is False


class TestDataCache:
    """Test suite for DataCache class."""
    
    @pytest.fixture
    def cache(self):
        """Create cache for testing."""
        return DataCache(max_size=3)
    
    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_size == 3
        assert cache.size() == 0
    
    def test_cache_put_and_get(self, cache):
        """Test putting and getting items from cache."""
        # Put item in cache
        cache.put('key1', 'value1')
        assert cache.size() == 1
        
        # Get item from cache
        value = cache.get('key1')
        assert value == 'value1'
        
        # Get non-existent item
        value = cache.get('nonexistent')
        assert value is None
    
    def test_cache_max_size_enforcement(self, cache):
        """Test that cache enforces maximum size."""
        # Fill cache to max size
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')
        assert cache.size() == 3
        
        # Add one more item (should evict oldest)
        cache.put('key4', 'value4')
        assert cache.size() == 3
        
        # Oldest item should be evicted
        assert cache.get('key1') is None
        assert cache.get('key4') == 'value4'
    
    def test_cache_access_time_update(self, cache):
        """Test that accessing items updates their access time."""
        # Add items
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        cache.put('key3', 'value3')
        
        # Access key1 to update its access time
        cache.get('key1')
        
        # Add new item (should evict key2, not key1)
        cache.put('key4', 'value4')
        
        assert cache.get('key1') == 'value1'  # Should still be there
        assert cache.get('key2') is None      # Should be evicted
        assert cache.get('key4') == 'value4'  # Should be there
    
    def test_cache_clear(self, cache):
        """Test clearing cache."""
        # Add items
        cache.put('key1', 'value1')
        cache.put('key2', 'value2')
        assert cache.size() == 2
        
        # Clear cache
        cache.clear()
        assert cache.size() == 0
        assert cache.get('key1') is None
        assert cache.get('key2') is None


if __name__ == '__main__':
    pytest.main([__file__])