# tests/data/test_collectors.py
import pytest
import pandas as pd
import requests
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from nifty_ml_pipeline.data.collectors import NSEDataCollector, NewsDataCollector
from nifty_ml_pipeline.data.models import PriceData, NewsData


class TestNSEDataCollector:
    """Test suite for NSEDataCollector class."""
    
    @pytest.fixture
    def collector(self):
        """Create NSEDataCollector instance for testing."""
        return NSEDataCollector(max_retries=2, base_delay=0.1)
    
    @pytest.fixture
    def mock_nse_data(self):
        """Create mock NSE data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        data = pd.DataFrame({
            'Open': [22000.0, 22100.0, 22050.0, 22200.0, 22150.0],
            'High': [22150.0, 22200.0, 22200.0, 22300.0, 22250.0],  # Fixed: High >= max(Open, Close)
            'Low': [21950.0, 22000.0, 22000.0, 22100.0, 22050.0],   # Fixed: Low <= min(Open, Close)
            'Close': [22100.0, 22050.0, 22200.0, 22150.0, 22200.0],
            'Volume': [1000000, 1100000, 950000, 1200000, 1050000]
        }, index=dates)
        return data
    
    @patch('nifty_ml_pipeline.data.collectors.get_history')
    def test_collect_data_success(self, mock_get_history, collector, mock_nse_data):
        """Test successful data collection from NSE."""
        mock_get_history.return_value = mock_nse_data
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        result = collector.collect_data('NIFTY 50', start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert list(result.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        mock_get_history.assert_called_once_with(
            symbol='NIFTY 50',
            start=start_date,
            end=end_date,
            index=True
        )
    
    @patch('nifty_ml_pipeline.data.collectors.get_history')
    def test_collect_data_empty_response(self, mock_get_history, collector):
        """Test handling of empty response from NSE API."""
        mock_get_history.return_value = pd.DataFrame()
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        with pytest.raises(ValueError, match="No data returned"):
            collector.collect_data('NIFTY 50', start_date, end_date)
    
    @patch('nifty_ml_pipeline.data.collectors.get_history')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_collect_data_retry_logic(self, mock_sleep, mock_get_history, collector, mock_nse_data):
        """Test retry logic with exponential backoff."""
        # First call fails, second succeeds
        mock_get_history.side_effect = [Exception("API Error"), mock_nse_data]
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        result = collector.collect_data('NIFTY 50', start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert mock_get_history.call_count == 2
        mock_sleep.assert_called_once_with(0.1)  # base_delay
    
    @patch('nifty_ml_pipeline.data.collectors.get_history')
    @patch('time.sleep')
    def test_collect_data_max_retries_exceeded(self, mock_sleep, mock_get_history, collector):
        """Test behavior when max retries are exceeded."""
        mock_get_history.side_effect = Exception("Persistent API Error")
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        with pytest.raises(Exception, match="Persistent API Error"):
            collector.collect_data('NIFTY 50', start_date, end_date)
        
        assert mock_get_history.call_count == 2  # max_retries
        assert mock_sleep.call_count == 1  # Only sleep between retries
    
    def test_validate_data_success(self, collector, mock_nse_data):
        """Test successful data validation."""
        result = collector.validate_data(mock_nse_data)
        assert result is True
    
    def test_validate_data_missing_columns(self, collector):
        """Test validation failure with missing columns."""
        invalid_data = pd.DataFrame({
            'Open': [22000.0, 22100.0],
            'High': [22150.0, 22200.0]
            # Missing Low, Close, Volume
        })
        
        result = collector.validate_data(invalid_data)
        assert result is False
    
    def test_validate_data_invalid_price_relationships(self, collector):
        """Test validation failure with invalid price relationships."""
        invalid_data = pd.DataFrame({
            'Open': [22000.0, 22100.0],
            'High': [21900.0, 22000.0],  # High < Open (invalid)
            'Low': [21950.0, 22000.0],
            'Close': [22100.0, 22050.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range(start='2024-01-01', periods=2))
        
        result = collector.validate_data(invalid_data)
        assert result is False
    
    def test_validate_data_negative_values(self, collector):
        """Test validation failure with negative values."""
        invalid_data = pd.DataFrame({
            'Open': [22000.0, 22100.0],
            'High': [22150.0, 22200.0],
            'Low': [21950.0, 22000.0],
            'Close': [22100.0, 22050.0],
            'Volume': [-1000000, 1100000]  # Negative volume (invalid)
        }, index=pd.date_range(start='2024-01-01', periods=2))
        
        result = collector.validate_data(invalid_data)
        assert result is False
    
    @patch('nifty_ml_pipeline.data.collectors.get_history')
    def test_collect_rolling_window_data(self, mock_get_history, collector, mock_nse_data):
        """Test collection of rolling window data."""
        mock_get_history.return_value = mock_nse_data
        
        result = collector.collect_rolling_window_data('NIFTY 50', window_days=30)
        
        assert isinstance(result, pd.DataFrame)
        # Verify the date range is approximately correct
        call_args = mock_get_history.call_args
        start_date = call_args[1]['start']
        end_date = call_args[1]['end']
        
        # Should be approximately 30 days difference
        date_diff = (end_date - start_date).days
        assert 29 <= date_diff <= 31  # Allow for some variation
    
    def test_convert_to_price_data_objects(self, collector, mock_nse_data):
        """Test conversion of DataFrame to PriceData objects."""
        price_data_list = collector.convert_to_price_data_objects(mock_nse_data, 'NIFTY 50')
        
        assert len(price_data_list) == 5
        assert all(isinstance(item, PriceData) for item in price_data_list)
        
        # Check first item
        first_item = price_data_list[0]
        assert first_item.symbol == 'NIFTY 50'
        assert first_item.open == 22000.0
        assert first_item.high == 22150.0
        assert first_item.low == 21950.0
        assert first_item.close == 22100.0
        assert first_item.volume == 1000000
    
    def test_convert_to_price_data_objects_with_invalid_data(self, collector):
        """Test conversion with some invalid rows."""
        # Create data with one invalid row (negative price)
        invalid_data = pd.DataFrame({
            'Open': [22000.0, -22100.0, 22050.0],  # Second row has negative open
            'High': [22150.0, 22200.0, 22200.0],   # Fixed: High >= max(Open, Close)
            'Low': [21950.0, 22000.0, 22000.0],    # Fixed: Low <= min(Open, Close)
            'Close': [22100.0, 22050.0, 22200.0],
            'Volume': [1000000, 1100000, 950000]
        }, index=pd.date_range(start='2024-01-01', periods=3))
        
        price_data_list = collector.convert_to_price_data_objects(invalid_data, 'NIFTY 50')
        
        # Should only have 2 valid items (invalid row skipped)
        assert len(price_data_list) == 2
        assert all(isinstance(item, PriceData) for item in price_data_list)


class TestNewsDataCollector:
    """Test suite for NewsDataCollector class."""
    
    @pytest.fixture
    def collector(self):
        """Create NewsDataCollector instance for testing."""
        return NewsDataCollector(max_retries=2, base_delay=0.1)
    
    @pytest.fixture
    def mock_news_response(self):
        """Create mock news response for testing."""
        return """
        <rss>
            <item>
                <title>NIFTY 50 shows strong performance</title>
                <pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate>
                <link>https://example.com/news/1</link>
            </item>
        </rss>
        """
    
    @patch('requests.get')
    def test_collect_data_success(self, mock_get, collector):
        """Test successful news data collection."""
        mock_response = Mock()
        mock_response.text = "mock response"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        result = collector.collect_data('NIFTY 50', start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert 'headline' in result.columns
        assert 'timestamp' in result.columns
        assert 'source' in result.columns
    
    @patch('requests.get')
    def test_collect_data_request_failure(self, mock_get, collector):
        """Test handling of request failures."""
        mock_get.side_effect = requests.RequestException("Network error")
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 5)
        
        # Should not raise exception, but return empty DataFrame
        result = collector.collect_data('NIFTY 50', start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_filter_relevant_news(self, collector):
        """Test filtering of news by relevance window."""
        # Create test data with mixed dates
        test_data = pd.DataFrame({
            'headline': ['Old news', 'Recent news', 'Very recent news'],
            'timestamp': [
                datetime.now() - timedelta(days=45),  # Too old
                datetime.now() - timedelta(days=15),  # Within window
                datetime.now() - timedelta(days=1)    # Very recent
            ],
            'source': ['test', 'test', 'test']
        })
        
        filtered = collector._filter_relevant_news(test_data, days=30)
        
        assert len(filtered) == 2  # Should exclude the 45-day old news
        assert 'Old news' not in filtered['headline'].values
        assert 'Recent news' in filtered['headline'].values
        assert 'Very recent news' in filtered['headline'].values
    
    def test_filter_relevant_news_empty_dataframe(self, collector):
        """Test filtering with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['headline', 'timestamp', 'source'])
        result = collector._filter_relevant_news(empty_df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_validate_data_success(self, collector):
        """Test successful news data validation."""
        valid_data = pd.DataFrame({
            'headline': ['News 1', 'News 2'],
            'timestamp': [datetime.now() - timedelta(days=1), datetime.now()],
            'source': ['test', 'test']
        })
        
        result = collector.validate_data(valid_data)
        assert result is True
    
    def test_validate_data_empty_headlines(self, collector):
        """Test validation failure with empty headlines."""
        invalid_data = pd.DataFrame({
            'headline': ['Valid headline', '   '],  # Second headline is empty
            'timestamp': [datetime.now() - timedelta(days=1), datetime.now()],
            'source': ['test', 'test']
        })
        
        result = collector.validate_data(invalid_data)
        assert result is False
    
    def test_validate_data_empty_dataframe(self, collector):
        """Test validation of empty DataFrame (should pass)."""
        empty_df = pd.DataFrame(columns=['headline', 'timestamp', 'source'])
        result = collector.validate_data(empty_df)
        assert result is True
    
    def test_convert_to_news_data_objects(self, collector):
        """Test conversion of DataFrame to NewsData objects."""
        test_data = pd.DataFrame({
            'headline': ['News 1', 'News 2'],
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'source': ['test1', 'test2'],
            'url': ['http://example.com/1', 'http://example.com/2']
        })
        
        news_data_list = collector.convert_to_news_data_objects(test_data)
        
        assert len(news_data_list) == 2
        assert all(isinstance(item, NewsData) for item in news_data_list)
        
        # Check first item
        first_item = news_data_list[0]
        assert first_item.headline == 'News 1'
        assert first_item.source == 'test1'
        assert first_item.url == 'http://example.com/1'
        assert first_item.sentiment_score is None  # Should be None initially
    
    def test_convert_to_news_data_objects_with_invalid_data(self, collector):
        """Test conversion with some invalid rows."""
        # Create data with one invalid row (empty headline)
        invalid_data = pd.DataFrame({
            'headline': ['Valid news', '', 'Another valid news'],
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            'source': ['test1', 'test2', 'test3']
        })
        
        news_data_list = collector.convert_to_news_data_objects(invalid_data)
        
        # Should only have 2 valid items (empty headline row skipped)
        assert len(news_data_list) == 2
        assert all(isinstance(item, NewsData) for item in news_data_list)
        assert all(item.headline.strip() != '' for item in news_data_list)
    
    def test_handle_missing_news_data(self, collector):
        """Test handling of missing news data."""
        symbol = 'NIFTY 50'
        date = datetime(2024, 1, 1)
        
        fallback_news = collector.handle_missing_news_data(symbol, date)
        
        assert len(fallback_news) == 1
        assert fallback_news[0]['headline'].startswith('Market update:')
        assert fallback_news[0]['timestamp'] == date
        assert fallback_news[0]['source'] == 'fallback'
        assert fallback_news[0]['is_fallback'] is True
    
    def test_check_news_freshness(self, collector):
        """Test news freshness checking."""
        current_time = datetime.now()
        news_items = [
            {
                'headline': 'Fresh news',
                'timestamp': current_time - timedelta(hours=1),  # Fresh
                'source': 'test'
            },
            {
                'headline': 'Stale news',
                'timestamp': current_time - timedelta(hours=25),  # Stale
                'source': 'test'
            },
            {
                'headline': 'Another fresh news',
                'timestamp': current_time - timedelta(hours=12),  # Fresh
                'source': 'test'
            }
        ]
        
        fresh_news = collector.check_news_freshness(news_items, max_age_hours=24)
        
        assert len(fresh_news) == 2
        assert 'Fresh news' in [item['headline'] for item in fresh_news]
        assert 'Another fresh news' in [item['headline'] for item in fresh_news]
        assert 'Stale news' not in [item['headline'] for item in fresh_news]


if __name__ == '__main__':
    pytest.main([__file__])