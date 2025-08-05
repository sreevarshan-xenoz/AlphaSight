# tests/models/test_validator.py
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nifty_ml_pipeline.data.models import FeatureVector, PredictionResult
from nifty_ml_pipeline.models.predictor import XGBoostPredictor
from nifty_ml_pipeline.models.validator import ModelValidator, PerformanceTracker


class TestModelValidator:
    """Test suite for ModelValidator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data for testing."""
        np.random.seed(42)
        n_samples = 200
        
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
        
        # Generate synthetic target with some correlation to features
        target = pd.Series(
            features['lag1_return'] * 0.3 + np.random.normal(0, 0.02, n_samples),
            index=dates,
            name='future_return'
        )
        
        return features, target
    
    @pytest.fixture
    def trained_predictor(self, sample_data):
        """Create a trained XGBoost predictor."""
        X, y = sample_data
        predictor = XGBoostPredictor()
        predictor.train(X, y)
        return predictor
    
    @pytest.fixture
    def validator(self):
        """Create ModelValidator instance."""
        return ModelValidator(n_splits=3)  # Use fewer splits for faster testing
    
    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator.n_splits == 3
        assert validator.validation_results == {}
    
    def test_validate_model_basic(self, validator, trained_predictor, sample_data):
        """Test basic model validation functionality."""
        X, y = sample_data
        
        metrics = validator.validate_model(trained_predictor, X, y)
        
        assert 'cv_rmse_mean' in metrics
        assert 'cv_rmse_std' in metrics
        assert 'cv_mae_mean' in metrics
        assert 'cv_mae_std' in metrics
        assert 'cv_directional_accuracy_mean' in metrics
        assert 'cv_directional_accuracy_std' in metrics
        assert 'n_folds' in metrics
        assert 'total_samples' in metrics
        
        assert metrics['n_folds'] == 3
        assert metrics['total_samples'] == len(X)
        assert 0 <= metrics['cv_directional_accuracy_mean'] <= 100
    
    def test_validate_model_untrained(self, validator, sample_data):
        """Test validation with untrained model raises error."""
        X, y = sample_data
        untrained_predictor = XGBoostPredictor()
        
        with pytest.raises(ValueError, match="must be trained"):
            validator.validate_model(untrained_predictor, X, y)
    
    def test_validate_model_unordered_data(self, validator, trained_predictor, sample_data):
        """Test validation with chronologically unordered data."""
        X, y = sample_data
        
        # Shuffle the data
        shuffle_idx = np.random.permutation(len(X))
        X_shuffled = X.iloc[shuffle_idx]
        y_shuffled = y.iloc[shuffle_idx]
        
        metrics = validator.validate_model(trained_predictor, X_shuffled, y_shuffled)
        
        assert 'cv_rmse_mean' in metrics
        assert metrics['total_samples'] == len(X)
    
    def test_calculate_directional_accuracy(self, validator):
        """Test directional accuracy calculation."""
        y_true = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        y_pred = np.array([0.005, -0.015, 0.025, 0.01, 0.015])  # 4/5 correct directions
        
        accuracy = validator._calculate_directional_accuracy(y_true, y_pred)
        
        assert accuracy == 80.0  # 4 out of 5 correct
    
    def test_validate_latency(self, validator, trained_predictor):
        """Test latency validation functionality."""
        latency_metrics = validator.validate_latency(trained_predictor, num_samples=50)
        
        assert 'mean_latency_ms' in latency_metrics
        assert 'median_latency_ms' in latency_metrics
        assert 'p95_latency_ms' in latency_metrics
        assert 'p99_latency_ms' in latency_metrics
        assert 'target_met_rate' in latency_metrics
        assert 'num_samples' in latency_metrics
        
        assert latency_metrics['num_samples'] == 50
        assert 0 <= latency_metrics['target_met_rate'] <= 1
        assert latency_metrics['mean_latency_ms'] > 0
    
    def test_validate_latency_untrained(self, validator):
        """Test latency validation with untrained model raises error."""
        untrained_predictor = XGBoostPredictor()
        
        with pytest.raises(ValueError, match="must be trained"):
            validator.validate_latency(untrained_predictor)
    
    def test_validate_accuracy_threshold(self, validator):
        """Test accuracy threshold validation."""
        # Test passing threshold
        result_pass = validator.validate_accuracy_threshold(85.0, 80.0)
        assert result_pass['meets_threshold'] is True
        assert result_pass['accuracy'] == 85.0
        assert result_pass['margin'] == 5.0
        
        # Test failing threshold
        result_fail = validator.validate_accuracy_threshold(75.0, 80.0)
        assert result_fail['meets_threshold'] is False
        assert result_fail['accuracy'] == 75.0
        assert result_fail['margin'] == -5.0
    
    def test_get_validation_summary_no_validation(self, validator):
        """Test validation summary when no validation performed."""
        summary = validator.get_validation_summary()
        
        assert summary['status'] == 'No validation performed'
    
    def test_get_validation_summary_with_validation(self, validator, trained_predictor, sample_data):
        """Test validation summary after validation."""
        X, y = sample_data
        validator.validate_model(trained_predictor, X, y)
        
        summary = validator.get_validation_summary()
        
        assert summary['status'] == 'Validation completed'
        assert 'metrics' in summary
        assert 'recommendations' in summary
        assert isinstance(summary['recommendations'], list)
    
    def test_generate_recommendations(self, validator, trained_predictor, sample_data):
        """Test recommendation generation."""
        X, y = sample_data
        validator.validate_model(trained_predictor, X, y)
        
        recommendations = validator._generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)


class TestPerformanceTracker:
    """Test suite for PerformanceTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create PerformanceTracker instance."""
        return PerformanceTracker(accuracy_threshold=75.0)
    
    @pytest.fixture
    def sample_prediction(self):
        """Create sample PredictionResult."""
        return PredictionResult(
            timestamp=datetime.now(),
            symbol="NIFTY",
            predicted_close=101.5,
            signal="Buy",
            confidence=0.8,
            model_version="test_v1",
            features_used=['lag1_return', 'lag2_return', 'sma_5_ratio', 'rsi_14', 'macd_hist', 'daily_sentiment']
        )
    
    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.accuracy_threshold == 75.0
        assert tracker.performance_history == []
        assert tracker.alerts == []
    
    def test_track_prediction_basic(self, tracker, sample_prediction):
        """Test basic prediction tracking."""
        tracker.track_prediction(sample_prediction, actual_return=0.02, inference_time_ms=5.0)
        
        assert len(tracker.performance_history) == 1
        
        record = tracker.performance_history[0]
        assert record['symbol'] == "NIFTY"
        assert record['actual_return'] == 0.02
        assert record['inference_time_ms'] == 5.0
        assert record['confidence'] == 0.8
    
    def test_track_prediction_without_actual(self, tracker, sample_prediction):
        """Test tracking prediction without actual return."""
        tracker.track_prediction(sample_prediction, inference_time_ms=3.0)
        
        assert len(tracker.performance_history) == 1
        
        record = tracker.performance_history[0]
        assert record['actual_return'] is None
        assert record['inference_time_ms'] == 3.0
    
    def test_calculate_directional_accuracy_no_data(self, tracker):
        """Test directional accuracy calculation with no data."""
        accuracy = tracker.calculate_directional_accuracy()
        assert accuracy == 0.0
    
    def test_calculate_directional_accuracy_with_data(self, tracker):
        """Test directional accuracy calculation with sample data."""
        # Add some test predictions
        predictions = [
            (0.01, 0.015, "Buy"),   # Correct direction
            (-0.02, -0.01, "Sell"), # Correct direction
            (0.03, -0.01, "Buy"),   # Wrong direction
            (-0.01, -0.02, "Sell"), # Correct direction
        ]
        
        for i, (pred_return, actual_return, signal) in enumerate(predictions):
            prediction = PredictionResult(
                timestamp=datetime.now() - timedelta(days=i),
                symbol="TEST",
                predicted_close=100 * (1 + pred_return),
                signal=signal,
                confidence=0.7,
                model_version="test_v1"
            )
            tracker.track_prediction(prediction, actual_return=actual_return)
        
        accuracy = tracker.calculate_directional_accuracy(lookback_days=30)
        assert accuracy == 75.0  # 3 out of 4 correct
    
    def test_calculate_average_latency_no_data(self, tracker):
        """Test average latency calculation with no data."""
        latency = tracker.calculate_average_latency()
        assert latency == 0.0
    
    def test_calculate_average_latency_with_data(self, tracker, sample_prediction):
        """Test average latency calculation with sample data."""
        latencies = [5.0, 7.0, 3.0, 8.0]
        
        for i, latency in enumerate(latencies):
            prediction = PredictionResult(
                timestamp=datetime.now() - timedelta(hours=i),
                symbol="TEST",
                predicted_close=100.0,
                signal="Hold",
                confidence=0.6,
                model_version="test_v1"
            )
            tracker.track_prediction(prediction, inference_time_ms=latency)
        
        avg_latency = tracker.calculate_average_latency(lookback_days=7)
        assert avg_latency == np.mean(latencies)
    
    def test_get_performance_metrics_no_data(self, tracker):
        """Test performance metrics with no data."""
        metrics = tracker.get_performance_metrics()
        
        assert metrics['total_predictions'] == 0
        assert metrics['directional_accuracy'] == 0.0
        assert metrics['avg_latency_ms'] == 0.0
        assert metrics['avg_confidence'] == 0.0
        assert metrics['signal_distribution'] == {}
    
    def test_get_performance_metrics_with_data(self, tracker):
        """Test performance metrics with sample data."""
        # Add test data
        signals = ["Buy", "Sell", "Hold", "Buy"]
        confidences = [0.8, 0.7, 0.6, 0.9]
        
        for i, (signal, confidence) in enumerate(zip(signals, confidences)):
            prediction = PredictionResult(
                timestamp=datetime.now() - timedelta(days=i),
                symbol="TEST",
                predicted_close=100.0,
                signal=signal,
                confidence=confidence,
                model_version="test_v1"
            )
            tracker.track_prediction(prediction, inference_time_ms=5.0)
        
        metrics = tracker.get_performance_metrics(lookback_days=30)
        
        assert metrics['total_predictions'] == 4
        assert metrics['avg_confidence'] == np.mean(confidences)
        assert metrics['avg_latency_ms'] == 5.0
        assert 'Buy' in metrics['signal_distribution']
        assert 'Sell' in metrics['signal_distribution']
        assert 'Hold' in metrics['signal_distribution']
    
    def test_performance_alerts_accuracy_degradation(self, tracker):
        """Test accuracy degradation alerts."""
        # Add more predictions with poor accuracy to trigger alert
        predictions = [
            (0.01, -0.01, "Buy"),   # Wrong direction
            (-0.02, 0.01, "Sell"),  # Wrong direction
            (0.03, -0.01, "Buy"),   # Wrong direction
            (-0.01, 0.02, "Sell"),  # Wrong direction
            (0.02, -0.015, "Buy"),  # Wrong direction
        ]
        
        for i, (pred_return, actual_return, signal) in enumerate(predictions):
            prediction = PredictionResult(
                timestamp=datetime.now() - timedelta(days=i),  # Use days instead of hours
                symbol="TEST",
                predicted_close=100 * (1 + pred_return),
                signal=signal,
                confidence=0.7,
                model_version="test_v1"
            )
            tracker.track_prediction(prediction, actual_return=actual_return)
        
        # Should generate accuracy alert (0% accuracy should be well below 75% threshold)
        alerts = tracker.get_active_alerts()
        accuracy_alerts = [a for a in alerts if a['type'] == 'accuracy_degradation']
        assert len(accuracy_alerts) > 0
    
    def test_performance_alerts_latency_degradation(self, tracker, sample_prediction):
        """Test latency degradation alerts."""
        # Add prediction with high latency
        tracker.track_prediction(sample_prediction, inference_time_ms=15.0)
        
        # Should generate latency alert
        alerts = tracker.get_active_alerts()
        latency_alerts = [a for a in alerts if a['type'] == 'latency_degradation']
        assert len(latency_alerts) > 0
    
    def test_get_active_alerts_time_filtering(self, tracker, sample_prediction):
        """Test active alerts time filtering."""
        # Add old alert
        tracker.alerts.append({
            'timestamp': datetime.now() - timedelta(days=2),
            'type': 'test_alert',
            'message': 'Old alert',
            'severity': 'low'
        })
        
        # Add recent alert
        tracker.track_prediction(sample_prediction, inference_time_ms=15.0)
        
        # Get alerts from last 24 hours
        recent_alerts = tracker.get_active_alerts(hours_back=24)
        old_alerts = tracker.get_active_alerts(hours_back=72)
        
        assert len(recent_alerts) >= 1
        assert len(old_alerts) >= len(recent_alerts)
    
    def test_clear_alerts(self, tracker, sample_prediction):
        """Test alert clearing functionality."""
        # Generate some alerts
        tracker.track_prediction(sample_prediction, inference_time_ms=15.0)
        assert len(tracker.alerts) > 0
        
        # Clear alerts
        tracker.clear_alerts()
        assert len(tracker.alerts) == 0
    
    def test_export_performance_data(self, tracker, sample_prediction):
        """Test performance data export."""
        # Add some data
        tracker.track_prediction(sample_prediction, actual_return=0.01, inference_time_ms=5.0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "performance_data.csv"
            
            tracker.export_performance_data(str(filepath))
            
            assert filepath.exists()
            
            # Read back and verify
            import pandas as pd
            df = pd.read_csv(filepath)
            assert len(df) == 1
            assert 'symbol' in df.columns
            assert 'confidence' in df.columns
    
    def test_export_performance_data_no_data(self, tracker):
        """Test export with no data raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / "performance_data.csv"
            
            with pytest.raises(ValueError, match="No performance data"):
                tracker.export_performance_data(str(filepath))
    
    def test_get_performance_summary_no_data(self, tracker):
        """Test performance summary with no data."""
        summary = tracker.get_performance_summary()
        
        assert summary['status'] == 'No performance data available'
    
    def test_get_performance_summary_with_data(self, tracker, sample_prediction):
        """Test performance summary with data."""
        tracker.track_prediction(sample_prediction, actual_return=0.01, inference_time_ms=5.0)
        
        summary = tracker.get_performance_summary()
        
        assert summary['status'] == 'Performance tracking active'
        assert summary['total_predictions'] == 1
        assert 'metrics_7d' in summary
        assert 'metrics_30d' in summary
        assert 'active_alerts' in summary
        assert 'accuracy_threshold' in summary
        assert summary['accuracy_threshold'] == 75.0