# tests/features/test_feature_normalizer.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import os

from nifty_ml_pipeline.features.feature_normalizer import FeatureNormalizer, NormalizedFeatures
from nifty_ml_pipeline.data.models import FeatureVector


class TestNormalizedFeatures:
    """Test suite for NormalizedFeatures dataclass."""
    
    def test_normalized_features_creation(self):
        """Test creation of NormalizedFeatures object."""
        timestamp = datetime(2024, 1, 1)
        features = NormalizedFeatures(
            timestamp=timestamp,
            symbol='NIFTY',
            price_features={'lag1_return': 0.1, 'lag2_return': 0.05},
            technical_features={'rsi_14': 0.6, 'sma_5_ratio': 1.02},
            sentiment_features={'compound_score': 0.3},
            raw_features={'lag1_return': 0.02, 'rsi_14': 60.0}
        )
        
        assert features.timestamp == timestamp
        assert features.symbol == 'NIFTY'
        assert features.price_features['lag1_return'] == 0.1
        assert features.technical_features['rsi_14'] == 0.6
        assert features.sentiment_features['compound_score'] == 0.3
    
    def test_to_feature_vector(self):
        """Test conversion to FeatureVector."""
        timestamp = datetime(2024, 1, 1)
        features = NormalizedFeatures(
            timestamp=timestamp,
            symbol='NIFTY',
            price_features={'lag1_return': 0.1, 'lag2_return': 0.05},
            technical_features={'rsi_14': 0.6, 'sma_5_ratio': 1.02, 'macd_histogram': 0.1},
            sentiment_features={'compound_score': 0.3},
            raw_features={}
        )
        
        feature_vector = features.to_feature_vector()
        
        assert isinstance(feature_vector, FeatureVector)
        assert feature_vector.timestamp == timestamp
        assert feature_vector.symbol == 'NIFTY'
        assert feature_vector.lag1_return == 0.1
        assert feature_vector.lag2_return == 0.05
        assert feature_vector.rsi_14 == 0.6
        assert feature_vector.sma_5_ratio == 1.02
        assert feature_vector.macd_hist == 0.1
        assert feature_vector.daily_sentiment == 0.3
    
    def test_to_feature_vector_missing_features(self):
        """Test conversion to FeatureVector with missing features."""
        timestamp = datetime(2024, 1, 1)
        features = NormalizedFeatures(
            timestamp=timestamp,
            symbol='NIFTY',
            price_features={},  # Empty
            technical_features={'rsi_14': 0.6},
            sentiment_features={},  # Empty
            raw_features={}
        )
        
        feature_vector = features.to_feature_vector()
        
        # Should use default values
        assert feature_vector.lag1_return == 0.0
        assert feature_vector.lag2_return == 0.0
        assert feature_vector.sma_5_ratio == 1.0
        assert feature_vector.rsi_14 == 0.6
        assert feature_vector.macd_hist == 0.0
        assert feature_vector.daily_sentiment == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime(2024, 1, 1)
        features = NormalizedFeatures(
            timestamp=timestamp,
            symbol='NIFTY',
            price_features={'lag1_return': 0.1},
            technical_features={'rsi_14': 0.6},
            sentiment_features={'compound_score': 0.3},
            raw_features={'lag1_return': 0.02}
        )
        
        result_dict = features.to_dict()
        
        assert result_dict['timestamp'] == timestamp.isoformat()
        assert result_dict['symbol'] == 'NIFTY'
        assert result_dict['price_features']['lag1_return'] == 0.1
        assert result_dict['technical_features']['rsi_14'] == 0.6
        assert result_dict['sentiment_features']['compound_score'] == 0.3
        assert result_dict['raw_features']['lag1_return'] == 0.02


class TestFeatureNormalizer:
    """Test suite for FeatureNormalizer class."""
    
    @pytest.fixture
    def normalizer(self):
        """Create a FeatureNormalizer instance for testing."""
        return FeatureNormalizer(normalization_method='standard')
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        
        # Create realistic price data with trend
        base_price = 100.0
        data = []
        
        for i, date in enumerate(dates):
            close = base_price + i * 0.5 + np.random.normal(0, 1)
            open_price = close + np.random.normal(0, 0.5)
            high = max(open_price, close) + abs(np.random.normal(0, 0.5))
            low = min(open_price, close) - abs(np.random.normal(0, 0.5))
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def sample_news_data(self):
        """Create sample news data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        
        headlines = [
            "Market shows strong performance",
            "Stocks decline on economic concerns",
            "NIFTY reaches new highs",
            "Banking sector shows growth",
            "Tech stocks face pressure",
            "Market remains stable",
            "Positive earnings drive rally",
            "Inflation concerns weigh on market",
            "Strong GDP growth boosts sentiment",
            "Market consolidates gains"
        ]
        
        data = []
        for i, date in enumerate(dates):
            data.append({
                'timestamp': date,
                'headline': headlines[i],
                'source': 'ET'
            })
        
        return pd.DataFrame(data)
    
    def test_normalizer_initialization(self, normalizer):
        """Test FeatureNormalizer initialization."""
        assert normalizer.normalization_method == 'standard'
        assert not normalizer.is_fitted
        assert normalizer.scalers == {}
        assert normalizer.technical_calculator is not None
        assert normalizer.sentiment_analyzer is not None
        
        # Check default values
        assert 'lag1_return' in normalizer.default_values
        assert 'rsi_14' in normalizer.default_values
        assert 'compound_score' in normalizer.default_values
    
    def test_normalizer_initialization_different_methods(self):
        """Test initialization with different normalization methods."""
        methods = ['standard', 'minmax', 'robust']
        
        for method in methods:
            normalizer = FeatureNormalizer(normalization_method=method)
            assert normalizer.normalization_method == method
    
    def test_normalizer_initialization_invalid_method(self):
        """Test initialization with invalid normalization method."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalizer = FeatureNormalizer(normalization_method='invalid')
            normalizer._create_scaler()  # This will trigger the error
    
    def test_extract_price_features(self, normalizer, sample_price_data):
        """Test extraction of price features."""
        price_features = normalizer._extract_price_features(sample_price_data)
        
        assert not price_features.empty
        assert 'lag1_return' in price_features.columns
        assert 'lag2_return' in price_features.columns
        assert 'volatility' in price_features.columns
        
        # Check that returns are calculated correctly (first value should be filled with default)
        expected_returns = sample_price_data['close'].pct_change().fillna(0.0)
        pd.testing.assert_series_equal(
            price_features['lag1_return'], 
            expected_returns, 
            check_names=False
        )
    
    def test_extract_price_features_empty_data(self, normalizer):
        """Test price feature extraction with empty data."""
        empty_df = pd.DataFrame()
        
        price_features = normalizer._extract_price_features(empty_df)
        
        assert price_features.empty
    
    def test_extract_price_features_missing_close(self, normalizer):
        """Test price feature extraction with missing close column."""
        df_no_close = pd.DataFrame({'open': [100, 101], 'high': [102, 103]})
        
        price_features = normalizer._extract_price_features(df_no_close)
        
        assert price_features.empty
    
    def test_extract_technical_features(self, normalizer, sample_price_data):
        """Test extraction of technical features."""
        technical_features = normalizer._extract_technical_features(sample_price_data)
        
        assert not technical_features.empty
        assert 'rsi_14' in technical_features.columns
        assert 'sma_5_ratio' in technical_features.columns
        assert 'macd_histogram' in technical_features.columns
        
        # RSI should be normalized to 0-1 range
        rsi_values = technical_features['rsi_14'].dropna()
        if len(rsi_values) > 0:
            assert (rsi_values >= 0).all()
            assert (rsi_values <= 1).all()
    
    def test_extract_technical_features_empty_data(self, normalizer):
        """Test technical feature extraction with empty data."""
        empty_df = pd.DataFrame()
        
        technical_features = normalizer._extract_technical_features(empty_df)
        
        assert technical_features.empty
    
    def test_extract_sentiment_features(self, normalizer, sample_news_data):
        """Test extraction of sentiment features."""
        target_dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        
        sentiment_features = normalizer._extract_sentiment_features(
            sample_news_data, target_dates, symbol='NIFTY'
        )
        
        assert not sentiment_features.empty
        assert 'compound_score' in sentiment_features.columns
        assert len(sentiment_features) == len(target_dates)
        
        # Sentiment scores should be in valid range
        scores = sentiment_features['compound_score']
        assert (scores >= -1).all()
        assert (scores <= 1).all()
    
    def test_extract_sentiment_features_empty_news(self, normalizer):
        """Test sentiment feature extraction with empty news data."""
        empty_news = pd.DataFrame()
        target_dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
        
        sentiment_features = normalizer._extract_sentiment_features(
            empty_news, target_dates
        )
        
        assert not sentiment_features.empty
        assert len(sentiment_features) == len(target_dates)
        # Should use default neutral sentiment
        assert (sentiment_features['compound_score'] == 0.0).all()
    
    def test_fit_basic(self, normalizer, sample_price_data, sample_news_data):
        """Test basic fitting of the normalizer."""
        normalizer.fit(sample_price_data, sample_news_data, symbol='NIFTY')
        
        assert normalizer.is_fitted
        assert 'price' in normalizer.scalers
        assert 'technical' in normalizer.scalers
        assert 'sentiment' in normalizer.scalers
    
    def test_fit_price_only(self, normalizer, sample_price_data):
        """Test fitting with price data only."""
        normalizer.fit(sample_price_data)
        
        assert normalizer.is_fitted
        assert 'price' in normalizer.scalers
        assert 'technical' in normalizer.scalers
        # Sentiment scaler might not be created if no news data
    
    def test_fit_empty_data(self, normalizer):
        """Test fitting with empty data."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Price data cannot be empty"):
            normalizer.fit(empty_df)
    
    def test_transform_basic(self, normalizer, sample_price_data, sample_news_data):
        """Test basic transformation."""
        # First fit the normalizer
        normalizer.fit(sample_price_data, sample_news_data, symbol='NIFTY')
        
        # Then transform
        normalized_features = normalizer.transform(
            sample_price_data, sample_news_data, symbol='NIFTY'
        )
        
        assert len(normalized_features) > 0
        assert all(isinstance(f, NormalizedFeatures) for f in normalized_features)
        
        # Check first feature
        first_feature = normalized_features[0]
        assert first_feature.symbol == 'NIFTY'
        assert isinstance(first_feature.timestamp, datetime)
    
    def test_transform_not_fitted(self, normalizer, sample_price_data):
        """Test transformation without fitting first."""
        with pytest.raises(ValueError, match="Normalizer must be fitted"):
            normalizer.transform(sample_price_data)
    
    def test_transform_empty_data(self, normalizer, sample_price_data):
        """Test transformation with empty data."""
        normalizer.fit(sample_price_data)
        
        empty_df = pd.DataFrame()
        result = normalizer.transform(empty_df)
        
        assert result == []
    
    def test_fit_transform(self, normalizer, sample_price_data, sample_news_data):
        """Test fit_transform method."""
        normalized_features = normalizer.fit_transform(
            sample_price_data, sample_news_data, symbol='NIFTY'
        )
        
        assert normalizer.is_fitted
        assert len(normalized_features) > 0
        assert all(isinstance(f, NormalizedFeatures) for f in normalized_features)
    
    def test_save_and_load_scalers(self, normalizer, sample_price_data):
        """Test saving and loading scalers."""
        # Fit the normalizer
        normalizer.fit(sample_price_data)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save scalers
            normalizer.save_scalers(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Create new normalizer and load scalers
            new_normalizer = FeatureNormalizer()
            new_normalizer.load_scalers(tmp_path)
            
            assert new_normalizer.is_fitted
            assert len(new_normalizer.scalers) == len(normalizer.scalers)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_scalers_not_fitted(self, normalizer):
        """Test saving scalers when not fitted."""
        with tempfile.NamedTemporaryFile(suffix='.pkl') as tmp_file:
            with pytest.raises(ValueError, match="Cannot save unfitted normalizer"):
                normalizer.save_scalers(tmp_file.name)
    
    def test_get_feature_importance_weights(self, normalizer):
        """Test getting feature importance weights."""
        weights = normalizer.get_feature_importance_weights()
        
        assert isinstance(weights, dict)
        assert 'lag1_return' in weights
        assert 'rsi_14' in weights
        assert 'compound_score' in weights
        
        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # All weights should be positive
        assert all(w > 0 for w in weights.values())
    
    def test_validate_features_valid(self, normalizer):
        """Test validation of valid features."""
        features = NormalizedFeatures(
            timestamp=datetime(2024, 1, 1),
            symbol='NIFTY',
            price_features={'lag1_return': 0.1, 'lag2_return': 0.05},
            technical_features={'rsi_14': 0.6, 'sma_5_ratio': 1.02},
            sentiment_features={'compound_score': 0.3},
            raw_features={}
        )
        
        assert normalizer.validate_features(features) is True
    
    def test_validate_features_invalid_nan(self, normalizer):
        """Test validation with NaN values."""
        features = NormalizedFeatures(
            timestamp=datetime(2024, 1, 1),
            symbol='NIFTY',
            price_features={'lag1_return': np.nan},
            technical_features={'rsi_14': 0.6},
            sentiment_features={'compound_score': 0.3},
            raw_features={}
        )
        
        assert normalizer.validate_features(features) is False
    
    def test_validate_features_invalid_infinite(self, normalizer):
        """Test validation with infinite values."""
        features = NormalizedFeatures(
            timestamp=datetime(2024, 1, 1),
            symbol='NIFTY',
            price_features={'lag1_return': np.inf},
            technical_features={'rsi_14': 0.6},
            sentiment_features={'compound_score': 0.3},
            raw_features={}
        )
        
        assert normalizer.validate_features(features) is False
    
    def test_different_normalization_methods(self, sample_price_data):
        """Test different normalization methods."""
        methods = ['standard', 'minmax', 'robust']
        
        for method in methods:
            normalizer = FeatureNormalizer(normalization_method=method)
            normalized_features = normalizer.fit_transform(sample_price_data)
            
            assert len(normalized_features) > 0
            assert normalizer.is_fitted
    
    def test_feature_vector_conversion_integration(self, normalizer, sample_price_data):
        """Test integration with FeatureVector conversion."""
        normalized_features = normalizer.fit_transform(sample_price_data, symbol='NIFTY')
        
        # Convert to feature vectors
        feature_vectors = [f.to_feature_vector() for f in normalized_features]
        
        assert len(feature_vectors) == len(normalized_features)
        assert all(isinstance(fv, FeatureVector) for fv in feature_vectors)
        assert all(fv.symbol == 'NIFTY' for fv in feature_vectors)
    
    def test_missing_sentiment_data_handling(self, normalizer, sample_price_data):
        """Test handling of missing sentiment data."""
        # Fit and transform without news data
        normalized_features = normalizer.fit_transform(sample_price_data)
        
        assert len(normalized_features) > 0
        
        # All sentiment features should have default values
        for features in normalized_features:
            if 'compound_score' in features.sentiment_features:
                assert features.sentiment_features['compound_score'] == 0.0
    
    def test_robustness_with_extreme_values(self, normalizer):
        """Test robustness with extreme price values."""
        # Create data with extreme values
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        extreme_data = pd.DataFrame({
            'open': [100, 1000, 10, 100, 100, 100, 100, 100, 100, 100],
            'high': [105, 1100, 15, 105, 105, 105, 105, 105, 105, 105],
            'low': [95, 900, 5, 95, 95, 95, 95, 95, 95, 95],
            'close': [102, 1050, 12, 102, 102, 102, 102, 102, 102, 102],
            'volume': [1000000] * 10
        }, index=dates)
        
        # Should handle extreme values gracefully
        normalized_features = normalizer.fit_transform(extreme_data)
        
        assert len(normalized_features) > 0
        
        # Validate all features
        for features in normalized_features:
            assert normalizer.validate_features(features)