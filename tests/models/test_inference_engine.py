# tests/models/test_inference_engine.py
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nifty_ml_pipeline.data.models import FeatureVector, PredictionResult
from nifty_ml_pipeline.models.inference_engine import InferenceEngine
from nifty_ml_pipeline.models.predictor import XGBoostPredictor


class TestInferenceEngine:
    """Test suite for InferenceEngine class."""
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained XGBoost predictor for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Create sample training data
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=n_samples),
            periods=n_samples,
            freq='D'
        )
        
        features = pd.DataFrame({
            'lag1_return': np.random.normal(0, 0.02, n_samples),
            'lag2_return': np.random.normal(0, 0.02, n_samples),
            'sma_5_ratio': np.random.normal(1.0, 0.1, n_samples),
            'rsi_14': np.random.uniform(20, 80, n_samples),
            'macd_hist': np.random.normal(0, 0.01, n_samples),
            'daily_sentiment': np.random.uniform(-0.5, 0.5, n_samples)
        }, index=dates)
        
        target = pd.Series(
            np.random.normal(0, 0.03, n_samples),
            index=dates,
            name='future_return'
        )
        
        # Train predictor
        predictor = XGBoostPredictor()
        predictor.train(features, target)
        
        return predictor
    
    @pytest.fixture
    def inference_engine(self, trained_predictor):
        """Create InferenceEngine with trained model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            trained_predictor.save_model(str(model_path))
            
            engine = InferenceEngine(str(model_path))
            yield engine
    
    @pytest.fixture
    def sample_feature_vector(self):
        """Create sample FeatureVector for testing."""
        return FeatureVector(
            timestamp=datetime.now(),
            symbol="NIFTY",
            lag1_return=0.01,
            lag2_return=-0.005,
            sma_5_ratio=1.02,
            rsi_14=65.0,
            macd_hist=0.002,
            daily_sentiment=0.1
        )
    
    def test_initialization_with_model(self, trained_predictor):
        """Test inference engine initialization with pre-trained model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            trained_predictor.save_model(str(model_path))
            
            engine = InferenceEngine(str(model_path))
            
            assert engine.predictor.is_trained
            assert engine.confidence_threshold == 0.6
            assert engine.performance_metrics['total_predictions'] == 0
    
    def test_initialization_without_model(self):
        """Test inference engine initialization without model."""
        engine = InferenceEngine()
        
        assert not engine.predictor.is_trained
        assert engine.confidence_threshold == 0.6
        assert len(engine.prediction_cache) == 0
    
    def test_predict_single_basic(self, inference_engine, sample_feature_vector):
        """Test basic single prediction functionality."""
        current_price = 100.0
        
        result = inference_engine.predict_single(sample_feature_vector, current_price)
        
        assert isinstance(result, PredictionResult)
        assert result.symbol == "NIFTY"
        assert result.signal in ["Buy", "Sell", "Hold"]
        assert 0.0 <= result.confidence <= 1.0
        assert result.predicted_close > 0
        assert len(result.features_used) == 6
    
    def test_predict_single_latency(self, inference_engine, sample_feature_vector):
        """Test single prediction meets latency requirements."""
        current_price = 100.0
        
        # Measure inference time
        start_time = time.perf_counter()
        result = inference_engine.predict_single(sample_feature_vector, current_price)
        end_time = time.perf_counter()
        
        inference_time_ms = (end_time - start_time) * 1000
        
        # Should be well under 10ms
        assert inference_time_ms < 10.0, f"Inference time {inference_time_ms:.2f}ms exceeds 10ms target"
        assert isinstance(result, PredictionResult)
    
    def test_predict_single_caching(self, inference_engine, sample_feature_vector):
        """Test prediction caching functionality."""
        current_price = 100.0
        
        # First prediction
        result1 = inference_engine.predict_single(sample_feature_vector, current_price)
        
        # Second prediction with same inputs should use cache
        result2 = inference_engine.predict_single(sample_feature_vector, current_price)
        
        assert result1.predicted_close == result2.predicted_close
        assert result1.confidence == result2.confidence
        assert result1.signal == result2.signal
        assert inference_engine.performance_metrics['cache_hits'] == 1
    
    def test_predict_single_untrained_model(self):
        """Test prediction with untrained model raises error."""
        engine = InferenceEngine()
        feature_vector = FeatureVector(
            timestamp=datetime.now(),
            symbol="TEST",
            lag1_return=0.0,
            lag2_return=0.0,
            sma_5_ratio=1.0,
            rsi_14=50.0,
            macd_hist=0.0,
            daily_sentiment=0.0
        )
        
        with pytest.raises(ValueError, match="must be trained"):
            engine.predict_single(feature_vector, 100.0)
    
    def test_predict_batch(self, inference_engine):
        """Test batch prediction functionality."""
        features_list = []
        current_prices = []
        
        for i in range(5):
            features = FeatureVector(
                timestamp=datetime.now() + timedelta(minutes=i),
                symbol=f"STOCK_{i}",
                lag1_return=np.random.normal(0, 0.02),
                lag2_return=np.random.normal(0, 0.02),
                sma_5_ratio=np.random.normal(1.0, 0.1),
                rsi_14=np.random.uniform(20, 80),
                macd_hist=np.random.normal(0, 0.01),
                daily_sentiment=np.random.uniform(-0.5, 0.5)
            )
            features_list.append(features)
            current_prices.append(100.0 + i)
        
        results = inference_engine.predict_batch(features_list, current_prices)
        
        assert len(results) == 5
        assert all(isinstance(r, PredictionResult) for r in results)
        assert all(r.symbol == f"STOCK_{i}" for i, r in enumerate(results))
    
    def test_predict_batch_mismatched_lengths(self, inference_engine):
        """Test batch prediction with mismatched input lengths."""
        features_list = [FeatureVector(
            timestamp=datetime.now(),
            symbol="TEST",
            lag1_return=0.0,
            lag2_return=0.0,
            sma_5_ratio=1.0,
            rsi_14=50.0,
            macd_hist=0.0,
            daily_sentiment=0.0
        )]
        current_prices = [100.0, 101.0]  # Different length
        
        with pytest.raises(ValueError, match="same length"):
            inference_engine.predict_batch(features_list, current_prices)
    
    def test_get_actionable_predictions(self, inference_engine):
        """Test filtering for actionable predictions."""
        features_list = []
        current_prices = []
        
        # Create features that should generate different confidence levels
        for i in range(10):
            features = FeatureVector(
                timestamp=datetime.now() + timedelta(minutes=i),
                symbol=f"STOCK_{i}",
                lag1_return=np.random.normal(0, 0.02),
                lag2_return=np.random.normal(0, 0.02),
                sma_5_ratio=np.random.normal(1.0, 0.1),
                rsi_14=np.random.uniform(20, 80),
                macd_hist=np.random.normal(0, 0.01),
                daily_sentiment=np.random.uniform(-0.5, 0.5)
            )
            features_list.append(features)
            current_prices.append(100.0 + i * 0.5)
        
        actionable = inference_engine.get_actionable_predictions(features_list, current_prices)
        
        # All actionable predictions should meet confidence threshold and not be Hold
        for pred in actionable:
            assert pred.confidence >= inference_engine.confidence_threshold
            assert pred.signal != "Hold"
    
    def test_performance_metrics_tracking(self, inference_engine, sample_feature_vector):
        """Test performance metrics are properly tracked."""
        current_price = 100.0
        
        # Make several predictions
        for i in range(5):
            features = FeatureVector(
                timestamp=datetime.now() + timedelta(minutes=i),
                symbol=f"TEST_{i}",
                lag1_return=0.01,
                lag2_return=-0.005,
                sma_5_ratio=1.02,
                rsi_14=65.0,
                macd_hist=0.002,
                daily_sentiment=0.1
            )
            inference_engine.predict_single(features, current_price)
        
        metrics = inference_engine.get_performance_summary()
        
        assert metrics['total_predictions'] == 5
        assert metrics['avg_latency_ms'] > 0
        assert 0 <= metrics['cache_hit_rate'] <= 1
        assert 0 <= metrics['high_confidence_rate'] <= 1
        assert isinstance(metrics['latency_target_met'], bool)
    
    def test_clear_cache(self, inference_engine, sample_feature_vector):
        """Test cache clearing functionality."""
        current_price = 100.0
        
        # Make prediction to populate cache
        inference_engine.predict_single(sample_feature_vector, current_price)
        assert len(inference_engine.prediction_cache) > 0
        
        # Clear cache
        inference_engine.clear_cache()
        assert len(inference_engine.prediction_cache) == 0
        assert inference_engine.performance_metrics['cache_hits'] == 0
    
    def test_reset_metrics(self, inference_engine, sample_feature_vector):
        """Test metrics reset functionality."""
        current_price = 100.0
        
        # Make predictions to populate metrics
        inference_engine.predict_single(sample_feature_vector, current_price)
        assert inference_engine.performance_metrics['total_predictions'] > 0
        
        # Reset metrics
        inference_engine.reset_metrics()
        assert inference_engine.performance_metrics['total_predictions'] == 0
        assert inference_engine.performance_metrics['avg_latency_ms'] == 0.0
        assert len(inference_engine.prediction_cache) == 0
    
    def test_validate_latency_requirement(self, inference_engine):
        """Test latency validation functionality."""
        validation_results = inference_engine.validate_latency_requirement(num_samples=50)
        
        assert 'mean_latency_ms' in validation_results
        assert 'median_latency_ms' in validation_results
        assert 'p95_latency_ms' in validation_results
        assert 'p99_latency_ms' in validation_results
        assert 'target_met_rate' in validation_results
        assert validation_results['num_samples'] == 50
        
        # Most predictions should meet the 10ms target
        assert validation_results['target_met_rate'] > 0.8
    
    def test_validate_latency_untrained_model(self):
        """Test latency validation with untrained model raises error."""
        engine = InferenceEngine()
        
        with pytest.raises(ValueError, match="must be trained"):
            engine.validate_latency_requirement()
    
    def test_format_prediction_for_output(self, inference_engine, sample_feature_vector):
        """Test prediction result formatting."""
        current_price = 100.0
        result = inference_engine.predict_single(sample_feature_vector, current_price)
        
        formatted = inference_engine.format_prediction_for_output(result)
        
        assert 'timestamp' in formatted
        assert 'symbol' in formatted
        assert 'prediction' in formatted
        assert 'metadata' in formatted
        
        assert 'price' in formatted['prediction']
        assert 'signal' in formatted['prediction']
        assert 'confidence' in formatted['prediction']
        
        assert 'model_version' in formatted['metadata']
        assert 'features_count' in formatted['metadata']
        assert 'is_actionable' in formatted['metadata']
    
    def test_load_model(self, inference_engine, trained_predictor):
        """Test loading a different model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_model_path = Path(temp_dir) / "new_model.pkl"
            trained_predictor.save_model(str(new_model_path))
            
            # Load new model
            inference_engine.load_model(str(new_model_path))
            
            assert inference_engine.predictor.is_trained
            assert inference_engine.performance_metrics['total_predictions'] == 0
    
    def test_set_confidence_threshold(self, inference_engine):
        """Test confidence threshold setting."""
        # Test valid threshold
        inference_engine.set_confidence_threshold(0.8)
        assert inference_engine.confidence_threshold == 0.8
        
        # Test invalid thresholds
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            inference_engine.set_confidence_threshold(-0.1)
        
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            inference_engine.set_confidence_threshold(1.1)
    
    def test_confidence_threshold_affects_actionable_predictions(self, inference_engine):
        """Test that confidence threshold affects actionable prediction filtering."""
        features_list = []
        current_prices = []
        
        for i in range(5):
            features = FeatureVector(
                timestamp=datetime.now() + timedelta(minutes=i),
                symbol=f"STOCK_{i}",
                lag1_return=np.random.normal(0, 0.02),
                lag2_return=np.random.normal(0, 0.02),
                sma_5_ratio=np.random.normal(1.0, 0.1),
                rsi_14=np.random.uniform(20, 80),
                macd_hist=np.random.normal(0, 0.01),
                daily_sentiment=np.random.uniform(-0.5, 0.5)
            )
            features_list.append(features)
            current_prices.append(100.0 + i * 2)  # Larger price differences
        
        # Test with lower threshold
        inference_engine.set_confidence_threshold(0.3)
        actionable_low = inference_engine.get_actionable_predictions(features_list, current_prices)
        
        # Test with higher threshold
        inference_engine.set_confidence_threshold(0.9)
        actionable_high = inference_engine.get_actionable_predictions(features_list, current_prices)
        
        # Higher threshold should result in fewer or equal actionable predictions
        assert len(actionable_high) <= len(actionable_low)