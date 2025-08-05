# tests/output/test_performance_reporter.py
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import numpy as np

from nifty_ml_pipeline.output.performance_reporter import PerformanceReporter
from nifty_ml_pipeline.output.prediction_storage import PredictionStorage
from nifty_ml_pipeline.data.models import PredictionResult


class TestPerformanceReporter:
    """Test suite for PerformanceReporter class."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage(self, temp_storage_dir):
        """Create a PredictionStorage instance for testing."""
        return PredictionStorage(base_path=temp_storage_dir)
    
    @pytest.fixture
    def reporter(self, storage):
        """Create a PerformanceReporter instance for testing."""
        return PerformanceReporter(storage, accuracy_threshold=0.75, drift_threshold=0.1)
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction results for testing."""
        base_time = datetime(2024, 1, 15, 15, 30)
        
        return [
            PredictionResult(
                timestamp=base_time,
                symbol='NIFTY50',
                predicted_close=21500.0,
                signal='Buy',
                confidence=0.85,
                model_version='v1.0.0',
                features_used=['rsi_14', 'sma_5_ratio']
            ),
            PredictionResult(
                timestamp=base_time + timedelta(minutes=1),
                symbol='BANKNIFTY',
                predicted_close=45000.0,
                signal='Sell',
                confidence=0.75,
                model_version='v1.0.0',
                features_used=['macd_hist', 'daily_sentiment']
            ),
            PredictionResult(
                timestamp=base_time + timedelta(minutes=2),
                symbol='RELIANCE',
                predicted_close=2500.0,
                signal='Hold',
                confidence=0.6,
                model_version='v1.0.0',
                features_used=['rsi_14']
            ),
            PredictionResult(
                timestamp=base_time + timedelta(minutes=3),
                symbol='NIFTY50',
                predicted_close=21600.0,
                signal='Buy',
                confidence=0.9,
                model_version='v1.1.0',
                features_used=['rsi_14', 'sma_5_ratio', 'macd_hist']
            )
        ]
    
    def test_initialization(self, storage):
        """Test PerformanceReporter initialization."""
        reporter = PerformanceReporter(storage, accuracy_threshold=0.8, drift_threshold=0.15)
        
        assert reporter.storage == storage
        assert reporter.accuracy_threshold == 0.8
        assert reporter.drift_threshold == 0.15
    
    def test_generate_performance_summary(self, reporter, storage, sample_predictions):
        """Test performance summary generation."""
        # Store sample predictions
        storage.store_predictions(sample_predictions, "test_exec_001")
        
        start_date = datetime(2024, 1, 15, 0, 0)
        end_date = datetime(2024, 1, 15, 23, 59)
        
        summary = reporter.generate_performance_summary(start_date, end_date)
        
        assert 'overview' in summary
        assert summary['overview']['total_predictions'] == 4
        assert summary['overview']['actionable_predictions'] == 3  # confidence >= 0.7
        assert summary['overview']['actionable_rate'] == 0.75
        
        assert 'signal_distribution' in summary
        assert summary['signal_distribution']['Buy'] == 2
        assert summary['signal_distribution']['Sell'] == 1
        assert summary['signal_distribution']['Hold'] == 1
        
        assert 'confidence_statistics' in summary
        assert 'temporal_analysis' in summary
        assert 'model_analysis' in summary
    
    def test_generate_performance_summary_with_symbol_filter(self, reporter, storage, sample_predictions):
        """Test performance summary with symbol filter."""
        storage.store_predictions(sample_predictions, "test_exec_002")
        
        start_date = datetime(2024, 1, 15, 0, 0)
        end_date = datetime(2024, 1, 15, 23, 59)
        
        summary = reporter.generate_performance_summary(start_date, end_date, symbol='NIFTY50')
        
        assert summary['overview']['total_predictions'] == 2
        assert summary['period']['symbol_filter'] == 'NIFTY50'
    
    def test_generate_performance_summary_no_data(self, reporter):
        """Test performance summary with no data."""
        start_date = datetime(2024, 1, 1, 0, 0)
        end_date = datetime(2024, 1, 1, 23, 59)
        
        summary = reporter.generate_performance_summary(start_date, end_date)
        
        assert summary['overview']['total_predictions'] == 0
        assert summary['overview']['actionable_predictions'] == 0
        assert summary['overview']['actionable_rate'] == 0.0
    
    def test_calculate_accuracy_metrics(self, reporter, sample_predictions):
        """Test accuracy metrics calculation."""
        # Mock actual prices
        actual_prices = {
            'NIFTY50_2024-01-15_15-30': 21450.0,  # Prediction was 21500, signal was Buy
            'BANKNIFTY_2024-01-15_15-31': 44800.0,  # Prediction was 45000, signal was Sell
            'RELIANCE_2024-01-15_15-32': 2520.0,   # Prediction was 2500, signal was Hold
            'NIFTY50_2024-01-15_15-33': 21650.0    # Prediction was 21600, signal was Buy
        }
        
        metrics = reporter.calculate_accuracy_metrics(sample_predictions, actual_prices)
        
        assert 'directional_accuracy' in metrics
        assert 'mean_absolute_error' in metrics
        assert 'root_mean_squared_error' in metrics
        assert 'mean_absolute_percentage_error' in metrics
        assert metrics['total_comparisons'] == 4
        assert metrics['directional_comparisons'] >= 0
    
    def test_calculate_accuracy_metrics_empty_data(self, reporter):
        """Test accuracy metrics with empty data."""
        metrics = reporter.calculate_accuracy_metrics([], {})
        
        assert metrics['directional_accuracy'] == 0.0
        assert metrics['mean_absolute_error'] == 0.0
        assert metrics['total_comparisons'] == 0
    
    def test_analyze_historical_performance(self, reporter, storage, sample_predictions):
        """Test historical performance analysis."""
        # Store predictions across multiple days
        for i, pred in enumerate(sample_predictions):
            pred.timestamp = datetime(2024, 1, 15 + i, 15, 30)
        
        storage.store_predictions(sample_predictions, "test_exec_003")
        
        with patch('nifty_ml_pipeline.output.performance_reporter.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 20, 12, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            analysis = reporter.analyze_historical_performance(days=10)
            
            assert 'daily_statistics' in analysis
            assert 'trends' in analysis
            assert 'patterns' in analysis
            assert analysis['period']['days'] == 10
    
    def test_detect_model_drift(self, reporter, storage):
        """Test model drift detection."""
        # Create baseline predictions (older)
        baseline_time = datetime(2024, 1, 1, 15, 30)
        baseline_predictions = [
            PredictionResult(
                timestamp=baseline_time + timedelta(days=i),
                symbol='NIFTY50',
                predicted_close=21000.0 + i * 10,
                signal='Buy',
                confidence=0.8,
                model_version='v1.0.0',
                features_used=['rsi_14']
            ) for i in range(5)
        ]
        
        # Create recent predictions with different characteristics (drift)
        recent_time = datetime(2024, 1, 25, 15, 30)
        recent_predictions = [
            PredictionResult(
                timestamp=recent_time + timedelta(days=i),
                symbol='NIFTY50',
                predicted_close=22000.0 + i * 10,
                signal='Hold',  # Different signal distribution
                confidence=0.6,  # Lower confidence
                model_version='v1.0.0',
                features_used=['rsi_14']
            ) for i in range(3)
        ]
        
        storage.store_predictions(baseline_predictions, "baseline_exec")
        storage.store_predictions(recent_predictions, "recent_exec")
        
        with patch('nifty_ml_pipeline.output.performance_reporter.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 30, 12, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            drift_analysis = reporter.detect_model_drift(baseline_days=30, comparison_days=7)
            
            assert 'baseline_metrics' in drift_analysis
            assert 'recent_metrics' in drift_analysis
            assert 'drift_indicators' in drift_analysis
            assert 'retraining_assessment' in drift_analysis
    
    def test_detect_model_drift_insufficient_data(self, reporter):
        """Test drift detection with insufficient data."""
        with patch('nifty_ml_pipeline.output.performance_reporter.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 30, 12, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            drift_analysis = reporter.detect_model_drift()
            
            assert 'error' in drift_analysis
            assert 'Insufficient data' in drift_analysis['error']
    
    def test_generate_dashboard_data(self, reporter, storage, sample_predictions):
        """Test dashboard data generation."""
        storage.store_predictions(sample_predictions, "dashboard_exec")
        
        with patch('nifty_ml_pipeline.output.performance_reporter.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 20, 12, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            dashboard = reporter.generate_dashboard_data()
            
            assert 'overview' in dashboard
            assert 'weekly_performance' in dashboard
            assert 'monthly_trends' in dashboard
            assert 'drift_detection' in dashboard
            assert 'charts' in dashboard
            assert 'alerts' in dashboard
    
    def test_calculate_signal_distribution(self, reporter, sample_predictions):
        """Test signal distribution calculation."""
        distribution = reporter._calculate_signal_distribution(sample_predictions)
        
        assert distribution['Buy'] == 2
        assert distribution['Sell'] == 1
        assert distribution['Hold'] == 1
    
    def test_calculate_confidence_stats(self, reporter, sample_predictions):
        """Test confidence statistics calculation."""
        stats = reporter._calculate_confidence_stats(sample_predictions)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        # Verify calculations
        confidences = [0.85, 0.75, 0.6, 0.9]
        assert stats['mean'] == round(np.mean(confidences), 3)
        assert stats['min'] == 0.6
        assert stats['max'] == 0.9
    
    def test_analyze_temporal_patterns(self, reporter, sample_predictions):
        """Test temporal pattern analysis."""
        patterns = reporter._analyze_temporal_patterns(sample_predictions)
        
        assert 'hourly_distribution' in patterns
        assert 'daily_distribution' in patterns
        assert 'peak_hour' in patterns
        assert 'peak_day' in patterns
        
        # All predictions are at 15:30, so peak hour should be 15
        assert patterns['peak_hour'] == 15
    
    def test_analyze_model_versions(self, reporter, sample_predictions):
        """Test model version analysis."""
        analysis = reporter._analyze_model_versions(sample_predictions)
        
        assert 'version_distribution' in analysis
        assert 'version_avg_confidence' in analysis
        assert 'most_used_version' in analysis
        
        # v1.0.0 appears 3 times, v1.1.0 appears 1 time
        assert analysis['version_distribution']['v1.0.0'] == 3
        assert analysis['version_distribution']['v1.1.0'] == 1
        assert analysis['most_used_version'] == 'v1.0.0'
    
    def test_calculate_daily_statistics(self, reporter, sample_predictions):
        """Test daily statistics calculation."""
        daily_stats = reporter._calculate_daily_statistics(sample_predictions)
        
        # All predictions are on the same day
        assert len(daily_stats) == 1
        
        date_key = '2024-01-15'
        assert date_key in daily_stats
        
        stats = daily_stats[date_key]
        assert stats['prediction_count'] == 4
        assert stats['actionable_count'] == 3  # confidence >= 0.7
        assert stats['actionable_rate'] == 0.75
        assert stats['signals']['Buy'] == 2
        assert stats['signals']['Sell'] == 1
        assert stats['signals']['Hold'] == 1
    
    def test_calculate_performance_trends(self, reporter):
        """Test performance trend calculation."""
        # Create mock daily stats with trend
        daily_stats = {
            '2024-01-15': {'prediction_count': 10, 'avg_confidence': 0.7, 'actionable_rate': 0.6},
            '2024-01-16': {'prediction_count': 12, 'avg_confidence': 0.75, 'actionable_rate': 0.7},
            '2024-01-17': {'prediction_count': 14, 'avg_confidence': 0.8, 'actionable_rate': 0.8}
        }
        
        trends = reporter._calculate_performance_trends(daily_stats)
        
        assert 'prediction_count_trend' in trends
        assert 'confidence_trend' in trends
        assert 'actionable_rate_trend' in trends
        assert trends['trend_period_days'] == 3
        
        # All trends should be positive (increasing)
        assert trends['prediction_count_trend'] > 0
        assert trends['confidence_trend'] > 0
        assert trends['actionable_rate_trend'] > 0
    
    def test_calculate_performance_trends_insufficient_data(self, reporter):
        """Test trend calculation with insufficient data."""
        daily_stats = {'2024-01-15': {'prediction_count': 10, 'avg_confidence': 0.7, 'actionable_rate': 0.6}}
        
        trends = reporter._calculate_performance_trends(daily_stats)
        
        assert 'error' in trends
    
    def test_identify_performance_patterns(self, reporter):
        """Test performance pattern identification."""
        daily_stats = {
            '2024-01-15': {'avg_confidence': 0.7},
            '2024-01-16': {'avg_confidence': 0.9},  # Best day
            '2024-01-17': {'avg_confidence': 0.5}   # Worst day
        }
        
        patterns = reporter._identify_performance_patterns(daily_stats)
        
        assert 'best_performance_day' in patterns
        assert 'worst_performance_day' in patterns
        assert 'consistency_score' in patterns
        assert 'performance_volatility' in patterns
        
        assert patterns['best_performance_day']['date'] == '2024-01-16'
        assert patterns['worst_performance_day']['date'] == '2024-01-17'
    
    def test_calculate_period_metrics(self, reporter, sample_predictions):
        """Test period metrics calculation."""
        metrics = reporter._calculate_period_metrics(sample_predictions)
        
        assert 'avg_confidence' in metrics
        assert 'confidence_std' in metrics
        assert 'prediction_count' in metrics
        assert 'actionable_rate' in metrics
        assert 'buy_rate' in metrics
        assert 'sell_rate' in metrics
        assert 'hold_rate' in metrics
        
        assert metrics['prediction_count'] == 4
        assert metrics['buy_rate'] == 0.5  # 2 out of 4
        assert metrics['sell_rate'] == 0.25  # 1 out of 4
        assert metrics['hold_rate'] == 0.25  # 1 out of 4
    
    def test_calculate_drift_indicators(self, reporter):
        """Test drift indicator calculation."""
        baseline = {'avg_confidence': 0.8, 'actionable_rate': 0.7, 'buy_rate': 0.5}
        recent = {'avg_confidence': 0.6, 'actionable_rate': 0.5, 'buy_rate': 0.3}
        
        indicators = reporter._calculate_drift_indicators(baseline, recent)
        
        assert 'avg_confidence_drift' in indicators
        assert 'actionable_rate_drift' in indicators
        assert 'buy_rate_drift' in indicators
        
        # Check confidence drift
        conf_drift = indicators['avg_confidence_drift']
        assert conf_drift['absolute'] == 0.2
        assert conf_drift['relative'] == 0.25  # 0.2 / 0.8
        assert conf_drift['direction'] == 'decrease'
    
    def test_assess_retraining_need(self, reporter):
        """Test retraining assessment."""
        # High drift scenario
        high_drift_indicators = {
            'avg_confidence_drift': {'relative': 0.15},  # Above 0.1 threshold
            'actionable_rate_drift': {'relative': 0.05}  # Below threshold
        }
        
        assessment = reporter._assess_retraining_need(high_drift_indicators)
        
        assert assessment['retraining_recommended'] is True
        assert 'avg_confidence_drift' in assessment['high_drift_metrics']
        assert 'actionable_rate_drift' not in assessment['high_drift_metrics']
        
        # Low drift scenario
        low_drift_indicators = {
            'avg_confidence_drift': {'relative': 0.05},  # Below threshold
            'actionable_rate_drift': {'relative': 0.03}  # Below threshold
        }
        
        assessment = reporter._assess_retraining_need(low_drift_indicators)
        
        assert assessment['retraining_recommended'] is False
        assert len(assessment['high_drift_metrics']) == 0
    
    def test_prepare_chart_data(self, reporter, sample_predictions):
        """Test chart data preparation."""
        chart_data = reporter._prepare_chart_data(sample_predictions)
        
        assert 'confidence_time_series' in chart_data
        assert 'signal_distribution' in chart_data
        assert 'data_points' in chart_data
        
        assert chart_data['data_points'] == 4
        assert len(chart_data['confidence_time_series']) == 4
        
        # Check signal distribution
        assert chart_data['signal_distribution']['Buy'] == 2
        assert chart_data['signal_distribution']['Sell'] == 1
        assert chart_data['signal_distribution']['Hold'] == 1
    
    def test_assess_data_freshness(self, reporter, sample_predictions):
        """Test data freshness assessment."""
        # Mock current time to be 30 minutes after last prediction
        with patch('nifty_ml_pipeline.output.performance_reporter.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 16, 0)  # 30 minutes later
            
            freshness = reporter._assess_data_freshness(sample_predictions)
            
            assert freshness['status'] == 'fresh'  # Less than 1 hour
            assert 'last_prediction' in freshness
            assert freshness['hours_since_last'] == 0.5
    
    def test_generate_alerts(self, reporter):
        """Test alert generation."""
        # Low accuracy scenario
        weekly_data = {
            'overview': {'actionable_rate': 0.6}  # Below 0.75 threshold
        }
        
        # High drift scenario
        drift_data = {
            'retraining_assessment': {
                'retraining_recommended': True,
                'high_drift_metrics': ['avg_confidence_drift']
            }
        }
        
        alerts = reporter._generate_alerts(weekly_data, drift_data)
        
        assert len(alerts) == 2
        
        # Check low accuracy alert
        accuracy_alert = next(a for a in alerts if a['type'] == 'low_accuracy')
        assert accuracy_alert['severity'] == 'warning'
        
        # Check drift alert
        drift_alert = next(a for a in alerts if a['type'] == 'model_drift')
        assert drift_alert['severity'] == 'critical'
    
    def test_empty_summary_structure(self, reporter):
        """Test empty summary structure."""
        empty = reporter._empty_summary()
        
        assert empty['overview']['total_predictions'] == 0
        assert empty['overview']['actionable_predictions'] == 0
        assert empty['overview']['actionable_rate'] == 0.0
        assert 'signal_distribution' in empty
        assert 'confidence_statistics' in empty
    
    def test_empty_accuracy_metrics_structure(self, reporter):
        """Test empty accuracy metrics structure."""
        empty = reporter._empty_accuracy_metrics()
        
        assert empty['directional_accuracy'] == 0.0
        assert empty['mean_absolute_error'] == 0.0
        assert empty['root_mean_squared_error'] == 0.0
        assert empty['total_comparisons'] == 0
    
    def test_error_handling_in_performance_summary(self, reporter):
        """Test error handling in performance summary generation."""
        # Mock storage to raise an exception
        with patch.object(reporter.storage, 'retrieve_predictions', side_effect=Exception("Storage error")):
            summary = reporter.generate_performance_summary(datetime.now(), datetime.now())
            
            assert 'error' in summary
            assert 'Storage error' in summary['error']
    
    def test_error_handling_in_accuracy_calculation(self, reporter):
        """Test error handling in accuracy calculation."""
        # Create valid predictions but with problematic actual prices
        valid_predictions = [
            PredictionResult(
                timestamp=datetime.now(),
                symbol='TEST',
                predicted_close=1000.0,
                signal='Buy',
                confidence=0.8,
                model_version='v1.0.0',
                features_used=[]
            )
        ]
        
        # Create actual prices that will cause division by zero
        problematic_actual_prices = {'invalid_key': 0}
        
        with patch('nifty_ml_pipeline.output.performance_reporter.logger') as mock_logger:
            # This should not cause an error but should handle gracefully
            metrics = reporter.calculate_accuracy_metrics(valid_predictions, problematic_actual_prices)
            
            # Should return empty metrics since no matching keys
            assert metrics['total_comparisons'] == 0