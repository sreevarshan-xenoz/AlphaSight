# tests/models/test_predictor.py
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nifty_ml_pipeline.data.models import FeatureVector
from nifty_ml_pipeline.models.predictor import XGBoostPredictor


class TestXGBoostPredictor:
    """Test suite for XGBoostPredictor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Create datetime index
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n_samples),
            periods=n_samples,
            freq='D'
        )
        
        # Generate synthetic feature data
        features = pd.DataFrame({
            'lag1_return': np.random.normal(0, 0.02, n_samples),
            'lag2_return': np.random.normal(0, 0.02, n_samples),
            'sma_5_ratio': np.random.normal(1.0, 0.1, n_samples),
            'rsi_14': np.random.uniform(20, 80, n_samples),
            'macd_hist': np.random.normal(0, 0.01, n_samples),
            'daily_sentiment': np.random.uniform(-0.5, 0.5, n_samples)
        }, index=dates)
        
        # Generate synthetic target (future returns)
        target = pd.Series(
            np.random.normal(0, 0.03, n_samples),
            index=dates,
            name='future_return'
        )
        
        return features, target
    
    @pytest.fixture
    def predictor(self):
        """Create XGBoostPredictor instance for testing."""
        return XGBoostPredictor()
    
    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.model is None
        assert not predictor.is_trained
        assert len(predictor.feature_names) == 6
        assert predictor.hyperparameters['n_jobs'] == 1
        assert predictor.hyperparameters['tree_method'] == 'exact'
        assert predictor.hyperparameters['max_depth'] == 6
    
    def test_train_basic(self, predictor, sample_data):
        """Test basic model training functionality."""
        X, y = sample_data
        
        metrics = predictor.train(X, y)
        
        assert predictor.is_trained
        assert predictor.model is not None
        assert 'train_rmse' in metrics
        assert 'cv_rmse_mean' in metrics
        assert 'training_time_seconds' in metrics
        assert metrics['n_features'] == 6
        assert metrics['n_samples'] == 100
    
    def test_train_with_unordered_data(self, predictor, sample_data):
        """Test training with chronologically unordered data."""
        X, y = sample_data
        
        # Shuffle the data to test reordering
        shuffle_idx = np.random.permutation(len(X))
        X_shuffled = X.iloc[shuffle_idx]
        y_shuffled = y.iloc[shuffle_idx]
        
        metrics = predictor.train(X_shuffled, y_shuffled)
        
        assert predictor.is_trained
        assert 'train_rmse' in metrics
    
    def test_train_validation_errors(self, predictor):
        """Test training validation and error handling."""
        # Test mismatched lengths
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        y = pd.Series([1, 2])  # Different length
        
        with pytest.raises(ValueError, match="same length"):
            predictor.train(X, y)
    
    def test_predict_dataframe(self, predictor, sample_data):
        """Test prediction with DataFrame input."""
        X, y = sample_data
        predictor.train(X, y)
        
        # Test prediction on training data
        predictions = predictor.predict(X[:10])
        
        assert len(predictions) == 10
        assert isinstance(predictions, np.ndarray)
        assert all(isinstance(p, (float, np.floating)) for p in predictions)
    
    def test_predict_feature_vector(self, predictor, sample_data):
        """Test prediction with single FeatureVector input."""
        X, y = sample_data
        predictor.train(X, y)
        
        # Create single FeatureVector
        feature_vector = FeatureVector(
            timestamp=datetime.now(),
            symbol="NIFTY",
            lag1_return=0.01,
            lag2_return=-0.005,
            sma_5_ratio=1.02,
            rsi_14=65.0,
            macd_hist=0.002,
            daily_sentiment=0.1
        )
        
        prediction = predictor.predict(feature_vector)
        
        assert isinstance(prediction, (float, np.floating))
    
    def test_predict_untrained_model(self, predictor):
        """Test prediction with untrained model raises error."""
        X = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="must be trained"):
            predictor.predict(X)
    
    def test_predict_invalid_input(self, predictor, sample_data):
        """Test prediction with invalid input types."""
        X, y = sample_data
        predictor.train(X, y)
        
        with pytest.raises(ValueError, match="Input must be"):
            predictor.predict("invalid_input")
    
    def test_predict_with_confidence(self, predictor, sample_data):
        """Test prediction with confidence scoring."""
        X, y = sample_data
        predictor.train(X, y)
        
        feature_vector = FeatureVector(
            timestamp=datetime.now(),
            symbol="NIFTY",
            lag1_return=0.01,
            lag2_return=-0.005,
            sma_5_ratio=1.02,
            rsi_14=65.0,
            macd_hist=0.002,
            daily_sentiment=0.1
        )
        
        prediction, confidence = predictor.predict_with_confidence(feature_vector)
        
        assert isinstance(prediction, (float, np.floating))
        assert 0.5 <= confidence <= 0.95
    
    def test_generate_signal(self, predictor):
        """Test trading signal generation."""
        current_price = 100.0
        
        # Test Buy signal
        predicted_price = 103.0  # 3% increase
        signal = predictor.generate_signal(current_price, predicted_price, 0.8)
        assert signal == "Buy"
        
        # Test Sell signal
        predicted_price = 97.0  # 3% decrease
        signal = predictor.generate_signal(current_price, predicted_price, 0.8)
        assert signal == "Sell"
        
        # Test Hold signal (small change)
        predicted_price = 101.0  # 1% increase (below threshold)
        signal = predictor.generate_signal(current_price, predicted_price, 0.8)
        assert signal == "Hold"
        
        # Test Hold signal (low confidence)
        predicted_price = 105.0  # 5% increase but low confidence
        signal = predictor.generate_signal(current_price, predicted_price, 0.5)
        assert signal == "Hold"
    
    def test_inference_latency(self, predictor, sample_data):
        """Test inference latency meets sub-10ms requirement."""
        X, y = sample_data
        predictor.train(X, y)
        
        feature_vector = FeatureVector(
            timestamp=datetime.now(),
            symbol="NIFTY",
            lag1_return=0.01,
            lag2_return=-0.005,
            sma_5_ratio=1.02,
            rsi_14=65.0,
            macd_hist=0.002,
            daily_sentiment=0.1
        )
        
        # Measure inference time
        start_time = time.perf_counter()
        prediction = predictor.predict(feature_vector)
        end_time = time.perf_counter()
        
        inference_time_ms = (end_time - start_time) * 1000
        
        # Should be well under 10ms for single prediction
        assert inference_time_ms < 10.0, f"Inference time {inference_time_ms:.2f}ms exceeds 10ms target"
        assert isinstance(prediction, (float, np.floating))
    
    def test_model_serialization(self, predictor, sample_data):
        """Test model saving and loading."""
        X, y = sample_data
        predictor.train(X, y)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            
            # Save model
            predictor.save_model(str(model_path))
            assert model_path.exists()
            
            # Create new predictor and load model
            new_predictor = XGBoostPredictor()
            new_predictor.load_model(str(model_path))
            
            assert new_predictor.is_trained
            assert new_predictor.model_version == predictor.model_version
            assert new_predictor.feature_names == predictor.feature_names
            
            # Test predictions are consistent
            test_features = X[:5]
            original_pred = predictor.predict(test_features)
            loaded_pred = new_predictor.predict(test_features)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_save_untrained_model(self, predictor):
        """Test saving untrained model raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            
            with pytest.raises(ValueError, match="Cannot save untrained model"):
                predictor.save_model(str(model_path))
    
    def test_load_nonexistent_model(self, predictor):
        """Test loading non-existent model raises error."""
        with pytest.raises(FileNotFoundError):
            predictor.load_model("nonexistent_model.pkl")
    
    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        predictor.train(X, y)
        
        importance = predictor.get_feature_importance()
        
        assert len(importance) == 6
        assert all(name in importance for name in predictor.feature_names)
        assert all(isinstance(score, (float, np.floating)) for score in importance.values())
        assert all(score >= 0 for score in importance.values())
    
    def test_feature_importance_untrained(self, predictor):
        """Test feature importance on untrained model raises error."""
        with pytest.raises(ValueError, match="must be trained"):
            predictor.get_feature_importance()
    
    def test_cpu_optimization_parameters(self, predictor):
        """Test that CPU optimization parameters are correctly set."""
        assert predictor.hyperparameters['n_jobs'] == 1
        assert predictor.hyperparameters['tree_method'] == 'exact'
        assert predictor.hyperparameters['max_depth'] == 6
        assert predictor.hyperparameters['objective'] == 'reg:squarederror'
    
    def test_time_series_split_validation(self, predictor, sample_data):
        """Test that TimeSeriesSplit is used for validation."""
        X, y = sample_data
        
        # Ensure data has proper time ordering
        assert X.index.is_monotonic_increasing
        
        metrics = predictor.train(X, y)
        
        # Should have cross-validation metrics
        assert 'cv_rmse_mean' in metrics
        assert 'cv_rmse_std' in metrics
        assert metrics['cv_rmse_mean'] > 0
        assert metrics['cv_rmse_std'] >= 0