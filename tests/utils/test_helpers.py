"""
Tests for utility helper functions and common test utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Test utilities for generating consistent mock data
class TestDataGenerators:
    """Test utilities for generating mock data."""
    
    @staticmethod
    def generate_mock_price_data(symbol: str = "NIFTY50", days: int = 30, seed: int = 42) -> pd.DataFrame:
        """Generate realistic mock price data for testing."""
        np.random.seed(seed)
        dates = pd.date_range('2024-01-01', periods=days, freq='D')
        
        prices = []
        base_price = 100.0
        
        for date in dates:
            # Simulate realistic price movements
            change = np.random.normal(0, 0.015)  # 1.5% daily volatility
            base_price *= (1 + change)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.005)))
            low = base_price * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.normal(1000000, 150000))
            
            prices.append({
                'timestamp': date,
                'open': base_price,
                'high': max(base_price, high),
                'low': min(base_price, low),
                'close': base_price,
                'volume': max(volume, 50000)
            })
        
        return pd.DataFrame(prices)
    
    @staticmethod
    def generate_mock_news_data(symbol: str = "NIFTY50", days: int = 30, seed: int = 42) -> pd.DataFrame:
        """Generate realistic mock news data for testing."""
        np.random.seed(seed)
        dates = pd.date_range('2024-01-01', periods=days//2, freq='2D')
        
        headlines = [
            "Market shows strong bullish momentum amid positive sentiment",
            "Economic indicators point to sustained growth trajectory",
            "Concerns over inflation continue to impact market dynamics",
            "Technology sector leads market gains with robust performance",
            "Banking stocks show mixed performance in volatile session",
            "Regulatory changes create uncertainty in financial markets",
            "Global economic trends influence domestic market outlook",
            "Sector rotation continues as investors seek opportunities"
        ]
        
        news_data = []
        for i, date in enumerate(dates):
            headline = headlines[i % len(headlines)]
            news_data.append({
                'headline': headline,
                'timestamp': date,
                'source': 'test_source',
                'url': f'http://test.com/news/{i}'
            })
        
        return pd.DataFrame(news_data)
    
    @staticmethod
    def generate_mock_feature_data(samples: int = 30, seed: int = 42) -> pd.DataFrame:
        """Generate realistic mock feature data for testing."""
        np.random.seed(seed)
        dates = pd.date_range('2024-01-01', periods=samples, freq='D')
        
        return pd.DataFrame({
            'timestamp': dates,
            'lag1_return': np.random.normal(0, 0.02, samples),
            'lag2_return': np.random.normal(0, 0.02, samples),
            'sma_5_ratio': np.random.uniform(0.95, 1.05, samples),
            'rsi_14': np.random.uniform(20, 80, samples),
            'macd_hist': np.random.normal(0, 0.5, samples),
            'daily_sentiment': np.random.uniform(-0.5, 0.5, samples)
        })


class TestMockHelpers:
    """Test helper functions for creating mocks."""
    
    @staticmethod
    def create_mock_prediction(direction: str = "Buy", confidence: float = 0.8, 
                             timestamp: datetime = None) -> Mock:
        """Create a mock prediction object."""
        if timestamp is None:
            timestamp = datetime.now()
        
        mock_prediction = Mock()
        mock_prediction.predicted_direction = direction
        mock_prediction.confidence = confidence
        mock_prediction.timestamp = timestamp
        mock_prediction.symbol = "NIFTY50"
        
        return mock_prediction
    
    @staticmethod
    def create_mock_pipeline_result(success: bool = True, predictions: list = None,
                                  error_message: str = "") -> Mock:
        """Create a mock pipeline result object."""
        if predictions is None:
            predictions = [TestMockHelpers.create_mock_prediction()]
        
        mock_result = Mock()
        mock_result.was_successful.return_value = success
        mock_result.predictions = predictions
        mock_result.error_message = error_message
        mock_result.total_duration_ms = 1500.0
        mock_result.stage_results = [
            {
                'stage': 'data_collection',
                'status': 'completed',
                'duration_ms': 500.0,
                'data_count': 100,
                'metadata': {'price_records': 60, 'news_records': 40}
            },
            {
                'stage': 'feature_engineering',
                'status': 'completed',
                'duration_ms': 800.0,
                'data_count': 60,
                'metadata': {'features_created': 6}
            },
            {
                'stage': 'model_inference',
                'status': 'completed',
                'duration_ms': 200.0,
                'data_count': 1,
                'metadata': {'predictions_generated': 1, 'meets_latency_target': True}
            }
        ]
        
        return mock_result


class TestValidationHelpers:
    """Helper functions for test validation."""
    
    @staticmethod
    def validate_price_data_structure(df: pd.DataFrame) -> bool:
        """Validate that a DataFrame has the expected price data structure."""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check all required columns exist
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            return False
        
        # Check numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False
        
        # Check logical constraints
        if not (df['high'] >= df[['open', 'close']].max(axis=1)).all():
            return False
        
        if not (df['low'] <= df[['open', 'close']].min(axis=1)).all():
            return False
        
        if not (df['volume'] >= 0).all():
            return False
        
        return True
    
    @staticmethod
    def validate_news_data_structure(df: pd.DataFrame) -> bool:
        """Validate that a DataFrame has the expected news data structure."""
        required_columns = ['headline', 'timestamp', 'source', 'url']
        
        # Check all required columns exist
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            return False
        
        # Check string columns
        string_columns = ['headline', 'source', 'url']
        for col in string_columns:
            if not df[col].dtype == 'object':
                return False
        
        return True
    
    @staticmethod
    def validate_feature_data_structure(df: pd.DataFrame) -> bool:
        """Validate that a DataFrame has the expected feature data structure."""
        expected_features = ['lag1_return', 'lag2_return', 'sma_5_ratio', 
                           'rsi_14', 'macd_hist', 'daily_sentiment']
        
        # Check feature columns exist
        if not all(col in df.columns for col in expected_features):
            return False
        
        # Check all features are numeric
        for col in expected_features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False
        
        # Check value ranges for specific features
        if not df['rsi_14'].between(0, 100).all():
            return False
        
        if not df['daily_sentiment'].between(-1, 1).all():
            return False
        
        return True


class TestAssertionHelpers:
    """Custom assertion helpers for tests."""
    
    @staticmethod
    def assert_prediction_valid(prediction) -> None:
        """Assert that a prediction object is valid."""
        assert hasattr(prediction, 'predicted_direction')
        assert hasattr(prediction, 'confidence')
        assert hasattr(prediction, 'timestamp')
        
        assert prediction.predicted_direction in ['Buy', 'Sell', 'Hold']
        assert 0 <= prediction.confidence <= 1
        assert isinstance(prediction.timestamp, datetime)
    
    @staticmethod
    def assert_pipeline_result_valid(result) -> None:
        """Assert that a pipeline result object is valid."""
        assert hasattr(result, 'was_successful')
        assert hasattr(result, 'predictions')
        assert hasattr(result, 'stage_results')
        assert hasattr(result, 'total_duration_ms')
        
        assert isinstance(result.predictions, list)
        assert isinstance(result.stage_results, list)
        assert result.total_duration_ms >= 0
    
    @staticmethod
    def assert_stage_result_valid(stage_result: dict) -> None:
        """Assert that a stage result dictionary is valid."""
        required_keys = ['stage', 'status', 'duration_ms', 'data_count', 'metadata']
        
        assert all(key in stage_result for key in required_keys)
        assert stage_result['status'] in ['completed', 'failed', 'in_progress']
        assert stage_result['duration_ms'] >= 0
        assert stage_result['data_count'] >= 0
        assert isinstance(stage_result['metadata'], dict)


# Test the test utilities themselves
class TestTestUtilities:
    """Tests for the test utility functions."""
    
    def test_generate_mock_price_data(self):
        """Test mock price data generation."""
        data = TestDataGenerators.generate_mock_price_data(days=50)
        
        assert len(data) == 50
        assert TestValidationHelpers.validate_price_data_structure(data)
        
        # Test reproducibility
        data2 = TestDataGenerators.generate_mock_price_data(days=50, seed=42)
        pd.testing.assert_frame_equal(data, data2)
    
    def test_generate_mock_news_data(self):
        """Test mock news data generation."""
        data = TestDataGenerators.generate_mock_news_data(days=30)
        
        assert len(data) == 15  # Every 2 days
        assert TestValidationHelpers.validate_news_data_structure(data)
        
        # Test reproducibility
        data2 = TestDataGenerators.generate_mock_news_data(days=30, seed=42)
        pd.testing.assert_frame_equal(data, data2)
    
    def test_generate_mock_feature_data(self):
        """Test mock feature data generation."""
        data = TestDataGenerators.generate_mock_feature_data(samples=100)
        
        assert len(data) == 100
        assert TestValidationHelpers.validate_feature_data_structure(data)
        
        # Test reproducibility
        data2 = TestDataGenerators.generate_mock_feature_data(samples=100, seed=42)
        pd.testing.assert_frame_equal(data, data2)
    
    def test_create_mock_prediction(self):
        """Test mock prediction creation."""
        prediction = TestMockHelpers.create_mock_prediction()
        
        TestAssertionHelpers.assert_prediction_valid(prediction)
        assert prediction.predicted_direction == "Buy"
        assert prediction.confidence == 0.8
    
    def test_create_mock_pipeline_result(self):
        """Test mock pipeline result creation."""
        result = TestMockHelpers.create_mock_pipeline_result()
        
        TestAssertionHelpers.assert_pipeline_result_valid(result)
        assert result.was_successful() == True
        assert len(result.predictions) == 1
    
    def test_validation_helpers(self):
        """Test validation helper functions."""
        # Test price data validation
        valid_price_data = TestDataGenerators.generate_mock_price_data()
        assert TestValidationHelpers.validate_price_data_structure(valid_price_data)
        
        # Test invalid price data
        invalid_price_data = pd.DataFrame({'invalid': [1, 2, 3]})
        assert not TestValidationHelpers.validate_price_data_structure(invalid_price_data)
        
        # Test news data validation
        valid_news_data = TestDataGenerators.generate_mock_news_data()
        assert TestValidationHelpers.validate_news_data_structure(valid_news_data)
        
        # Test feature data validation
        valid_feature_data = TestDataGenerators.generate_mock_feature_data()
        assert TestValidationHelpers.validate_feature_data_structure(valid_feature_data)
    
    def test_assertion_helpers(self):
        """Test assertion helper functions."""
        # Test prediction assertion
        valid_prediction = TestMockHelpers.create_mock_prediction()
        TestAssertionHelpers.assert_prediction_valid(valid_prediction)  # Should not raise
        
        # Test pipeline result assertion
        valid_result = TestMockHelpers.create_mock_pipeline_result()
        TestAssertionHelpers.assert_pipeline_result_valid(valid_result)  # Should not raise
        
        # Test stage result assertion
        valid_stage = {
            'stage': 'test_stage',
            'status': 'completed',
            'duration_ms': 100.0,
            'data_count': 50,
            'metadata': {'test': 'value'}
        }
        TestAssertionHelpers.assert_stage_result_valid(valid_stage)  # Should not raise