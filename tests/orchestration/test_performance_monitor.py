# tests/orchestration/test_performance_monitor.py
import pytest
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from nifty_ml_pipeline.orchestration.performance_monitor import (
    PerformanceMonitor, PerformanceMetric, Alert, AlertLevel, MetricType
)
from nifty_ml_pipeline.data.models import PipelineStage


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        'performance': {
            'MIN_ACCURACY': 0.75,
            'MAX_INFERENCE_LATENCY_MS': 10.0,
            'MAX_ERROR_RATE': 0.1
        }
    }


@pytest.fixture
def performance_monitor(mock_config):
    """Create performance monitor instance for testing."""
    return PerformanceMonitor(mock_config)


@pytest.fixture
def sample_stage_results():
    """Sample stage results for testing."""
    return [
        {
            'stage': 'data_collection',
            'status': 'completed',
            'duration_ms': 1000,
            'data_count': 100
        },
        {
            'stage': 'feature_engineering',
            'status': 'completed',
            'duration_ms': 500,
            'data_count': 100
        },
        {
            'stage': 'model_inference',
            'status': 'completed',
            'duration_ms': 5,
            'data_count': 1
        }
    ]


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing."""
    return [
        {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'NIFTY 50',
            'predicted_direction': 'Buy',
            'confidence': 0.85
        }
    ]


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def test_initialization(self, mock_config):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor(mock_config)
        
        assert monitor.config == mock_config
        assert monitor.min_accuracy_threshold == 0.75
        assert monitor.max_latency_threshold == 10.0
        assert monitor.max_error_rate_threshold == 0.1
        assert monitor.execution_count == 0
        assert monitor.successful_executions == 0
        assert monitor.failed_executions == 0
        assert len(monitor.metrics_history) == 0
        assert len(monitor.alerts_history) == 0
    
    def test_record_metric(self, performance_monitor):
        """Test recording individual metrics."""
        performance_monitor.record_metric(
            MetricType.LATENCY,
            15.5,
            "ms",
            stage=PipelineStage.MODEL_INFERENCE,
            metadata={'test': 'data'}
        )
        
        assert len(performance_monitor.metrics_history) == 1
        
        metric = performance_monitor.metrics_history[0]
        assert metric.metric_type == MetricType.LATENCY
        assert metric.value == 15.5
        assert metric.unit == "ms"
        assert metric.stage == PipelineStage.MODEL_INFERENCE
        assert metric.metadata == {'test': 'data'}
    
    def test_record_pipeline_execution_success(self, performance_monitor, sample_stage_results, sample_predictions):
        """Test recording successful pipeline execution."""
        performance_monitor.record_pipeline_execution(
            execution_id="test_123",
            success=True,
            duration_ms=1505.0,
            stage_results=sample_stage_results,
            predictions=sample_predictions
        )
        
        assert performance_monitor.execution_count == 1
        assert performance_monitor.successful_executions == 1
        assert performance_monitor.failed_executions == 0
        
        # Should have recorded multiple metrics
        assert len(performance_monitor.metrics_history) > 0
        
        # Check for overall latency metric
        latency_metrics = [m for m in performance_monitor.metrics_history if m.metric_type == MetricType.LATENCY]
        assert len(latency_metrics) >= 4  # Overall + 3 stages
        
        # Check for accuracy metric
        accuracy_metrics = [m for m in performance_monitor.metrics_history if m.metric_type == MetricType.ACCURACY]
        assert len(accuracy_metrics) == 1
        assert accuracy_metrics[0].value == 0.85  # From prediction confidence
        
        # Check for error rate metric
        error_rate_metrics = [m for m in performance_monitor.metrics_history if m.metric_type == MetricType.ERROR_RATE]
        assert len(error_rate_metrics) == 1
        assert error_rate_metrics[0].value == 0.0  # No failures yet
    
    def test_record_pipeline_execution_failure(self, performance_monitor, sample_stage_results):
        """Test recording failed pipeline execution."""
        performance_monitor.record_pipeline_execution(
            execution_id="test_failed",
            success=False,
            duration_ms=500.0,
            stage_results=sample_stage_results[:1],  # Only first stage completed
            predictions=[]
        )
        
        assert performance_monitor.execution_count == 1
        assert performance_monitor.successful_executions == 0
        assert performance_monitor.failed_executions == 1
        
        # Check error rate
        error_rate_metrics = [m for m in performance_monitor.metrics_history if m.metric_type == MetricType.ERROR_RATE]
        assert len(error_rate_metrics) == 1
        assert error_rate_metrics[0].value == 1.0  # 100% failure rate
    
    def test_accuracy_threshold_alert(self, performance_monitor):
        """Test alert generation for accuracy below threshold."""
        # Record low accuracy metric
        performance_monitor.record_metric(
            MetricType.ACCURACY,
            0.65,  # Below 0.75 threshold
            "ratio"
        )
        
        assert len(performance_monitor.alerts_history) == 1
        
        alert = performance_monitor.alerts_history[0]
        assert alert.level == AlertLevel.ERROR
        assert alert.metric_type == MetricType.ACCURACY
        assert alert.current_value == 0.65
        assert alert.threshold_value == 0.75
        assert "below threshold" in alert.message.lower()
    
    def test_latency_threshold_alert(self, performance_monitor):
        """Test alert generation for inference latency above threshold."""
        # Record high latency metric for inference stage
        performance_monitor.record_metric(
            MetricType.LATENCY,
            15.0,  # Above 10.0ms threshold
            "ms",
            stage=PipelineStage.MODEL_INFERENCE
        )
        
        assert len(performance_monitor.alerts_history) == 1
        
        alert = performance_monitor.alerts_history[0]
        assert alert.level == AlertLevel.WARNING
        assert alert.metric_type == MetricType.LATENCY
        assert alert.stage == PipelineStage.MODEL_INFERENCE
        assert alert.current_value == 15.0
        assert alert.threshold_value == 10.0
        assert "exceeds threshold" in alert.message.lower()
    
    def test_error_rate_threshold_alert(self, performance_monitor):
        """Test alert generation for error rate above threshold."""
        # Record high error rate metric
        performance_monitor.record_metric(
            MetricType.ERROR_RATE,
            0.15,  # Above 0.1 threshold
            "ratio"
        )
        
        assert len(performance_monitor.alerts_history) == 1
        
        alert = performance_monitor.alerts_history[0]
        assert alert.level == AlertLevel.CRITICAL
        assert alert.metric_type == MetricType.ERROR_RATE
        assert alert.current_value == 0.15
        assert alert.threshold_value == 0.1
    
    def test_cpu_usage_alert(self, performance_monitor):
        """Test alert generation for high CPU usage."""
        performance_monitor.record_metric(
            MetricType.CPU_USAGE,
            95.0,  # Above 90% threshold
            "percent"
        )
        
        assert len(performance_monitor.alerts_history) == 1
        
        alert = performance_monitor.alerts_history[0]
        assert alert.level == AlertLevel.WARNING
        assert alert.metric_type == MetricType.CPU_USAGE
        assert "High CPU usage" in alert.message
    
    def test_memory_usage_alert(self, performance_monitor):
        """Test alert generation for high memory usage."""
        performance_monitor.record_metric(
            MetricType.MEMORY_USAGE,
            90.0,  # Above 85% threshold
            "percent"
        )
        
        assert len(performance_monitor.alerts_history) == 1
        
        alert = performance_monitor.alerts_history[0]
        assert alert.level == AlertLevel.WARNING
        assert alert.metric_type == MetricType.MEMORY_USAGE
        assert "High memory usage" in alert.message
    
    def test_alert_callbacks(self, performance_monitor):
        """Test alert callback functionality."""
        callback_called = []
        
        def test_callback(alert):
            callback_called.append(alert)
        
        performance_monitor.add_alert_callback(test_callback)
        
        # Trigger an alert
        performance_monitor.record_metric(
            MetricType.ACCURACY,
            0.65,  # Below threshold
            "ratio"
        )
        
        assert len(callback_called) == 1
        assert callback_called[0].metric_type == MetricType.ACCURACY
        
        # Remove callback
        performance_monitor.remove_alert_callback(test_callback)
        
        # Trigger another alert
        performance_monitor.record_metric(
            MetricType.ACCURACY,
            0.60,  # Below threshold
            "ratio"
        )
        
        # Callback should not be called again
        assert len(callback_called) == 1
    
    def test_get_performance_summary(self, performance_monitor, sample_stage_results, sample_predictions):
        """Test performance summary generation."""
        # Record some executions
        performance_monitor.record_pipeline_execution(
            "exec_1", True, 1000.0, sample_stage_results, sample_predictions
        )
        performance_monitor.record_pipeline_execution(
            "exec_2", False, 500.0, sample_stage_results[:1], []
        )
        
        summary = performance_monitor.get_performance_summary(hours=24)
        
        assert summary.total_executions == 2
        assert summary.successful_executions == 1
        assert summary.failed_executions == 1
        assert summary.error_rate == 0.5
        assert summary.average_latency_ms > 0
        assert summary.average_accuracy is not None
        assert len(summary.stage_performance) > 0
    
    def test_get_recent_alerts(self, performance_monitor):
        """Test retrieving recent alerts."""
        # Generate some alerts
        performance_monitor.record_metric(MetricType.ACCURACY, 0.65, "ratio")  # ERROR
        performance_monitor.record_metric(MetricType.CPU_USAGE, 95.0, "percent")  # WARNING
        
        # Get all recent alerts
        all_alerts = performance_monitor.get_recent_alerts(hours=1)
        assert len(all_alerts) == 2
        
        # Get only error alerts
        error_alerts = performance_monitor.get_recent_alerts(hours=1, level=AlertLevel.ERROR)
        assert len(error_alerts) == 1
        assert error_alerts[0].metric_type == MetricType.ACCURACY
        
        # Get only warning alerts
        warning_alerts = performance_monitor.get_recent_alerts(hours=1, level=AlertLevel.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].metric_type == MetricType.CPU_USAGE
    
    def test_get_metrics_by_type(self, performance_monitor):
        """Test retrieving metrics by type."""
        # Record different types of metrics
        performance_monitor.record_metric(MetricType.LATENCY, 5.0, "ms")
        performance_monitor.record_metric(MetricType.LATENCY, 8.0, "ms")
        performance_monitor.record_metric(MetricType.ACCURACY, 0.85, "ratio")
        
        # Get latency metrics
        latency_metrics = performance_monitor.get_metrics_by_type(MetricType.LATENCY, hours=1)
        assert len(latency_metrics) == 2
        assert all(m.metric_type == MetricType.LATENCY for m in latency_metrics)
        
        # Get accuracy metrics
        accuracy_metrics = performance_monitor.get_metrics_by_type(MetricType.ACCURACY, hours=1)
        assert len(accuracy_metrics) == 1
        assert accuracy_metrics[0].metric_type == MetricType.ACCURACY
    
    def test_record_system_metrics(self, performance_monitor):
        """Test recording system metrics."""
        with patch('psutil.cpu_percent', return_value=45.5), \
             patch('psutil.cpu_count', return_value=4), \
             patch('psutil.virtual_memory') as mock_memory_func:
            
            mock_memory = Mock()
            mock_memory.percent = 60.0
            mock_memory.total = 8 * 1024**3  # 8GB
            mock_memory.available = 3 * 1024**3  # 3GB
            mock_memory_func.return_value = mock_memory
            
            performance_monitor.record_system_metrics()
        
        # Check that CPU and memory metrics were recorded
        cpu_metrics = [m for m in performance_monitor.metrics_history if m.metric_type == MetricType.CPU_USAGE]
        memory_metrics = [m for m in performance_monitor.metrics_history if m.metric_type == MetricType.MEMORY_USAGE]
        
        assert len(cpu_metrics) == 1
        assert len(memory_metrics) == 1
        assert cpu_metrics[0].value == 45.5
        assert memory_metrics[0].value == 60.0
    
    def test_record_system_metrics_no_psutil(self, performance_monitor):
        """Test system metrics recording when psutil is not available."""
        with patch('builtins.__import__', side_effect=lambda name, *args: ImportError() if name == 'psutil' else __import__(name, *args)):
            # Should not raise exception
            performance_monitor.record_system_metrics()
            
            # No system metrics should be recorded
            cpu_metrics = [m for m in performance_monitor.metrics_history if m.metric_type == MetricType.CPU_USAGE]
            memory_metrics = [m for m in performance_monitor.metrics_history if m.metric_type == MetricType.MEMORY_USAGE]
            
            assert len(cpu_metrics) == 0
            assert len(memory_metrics) == 0
    
    def test_clear_history(self, performance_monitor):
        """Test clearing monitoring history."""
        # Add some data
        performance_monitor.record_metric(MetricType.LATENCY, 5.0, "ms")
        performance_monitor.record_metric(MetricType.ACCURACY, 0.65, "ratio")  # Triggers alert
        performance_monitor.execution_count = 5
        performance_monitor.successful_executions = 3
        performance_monitor.failed_executions = 2
        
        assert len(performance_monitor.metrics_history) > 0
        assert len(performance_monitor.alerts_history) > 0
        assert performance_monitor.execution_count > 0
        
        performance_monitor.clear_history()
        
        assert len(performance_monitor.metrics_history) == 0
        assert len(performance_monitor.alerts_history) == 0
        assert performance_monitor.execution_count == 0
        assert performance_monitor.successful_executions == 0
        assert performance_monitor.failed_executions == 0
    
    def test_export_metrics(self, performance_monitor):
        """Test exporting metrics to JSON file."""
        # Add some test data
        performance_monitor.record_metric(MetricType.LATENCY, 5.0, "ms")
        performance_monitor.record_metric(MetricType.ACCURACY, 0.65, "ratio")  # Triggers alert
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            performance_monitor.export_metrics(temp_path, hours=1)
            
            # Read and verify exported data
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'export_timestamp' in exported_data
            assert 'time_range_hours' in exported_data
            assert 'summary' in exported_data
            assert 'metrics' in exported_data
            assert 'alerts' in exported_data
            
            assert exported_data['time_range_hours'] == 1
            assert len(exported_data['metrics']) > 0
            assert len(exported_data['alerts']) > 0
            
        finally:
            import os
            os.unlink(temp_path)
    
    @patch('nifty_ml_pipeline.orchestration.performance_monitor.logger')
    def test_structured_logging(self, mock_logger, performance_monitor):
        """Test structured logging of metrics and alerts."""
        # Record a metric that will trigger structured logging
        performance_monitor.record_metric(MetricType.LATENCY, 5.0, "ms")
        
        # Verify structured metric logging
        metric_log_calls = [call for call in mock_logger.info.call_args_list 
                           if 'performance_metric' in str(call)]
        assert len(metric_log_calls) > 0
        
        # Parse the logged JSON
        log_data = json.loads(metric_log_calls[0][0][0])
        assert log_data['event'] == 'performance_metric'
        assert log_data['metric_type'] == 'latency'
        assert log_data['value'] == 5.0
        assert log_data['unit'] == 'ms'
    
    def test_alert_callback_exception_handling(self, performance_monitor):
        """Test that alert callback exceptions don't break the system."""
        def failing_callback(alert):
            raise Exception("Callback failed")
        
        performance_monitor.add_alert_callback(failing_callback)
        
        # This should not raise an exception despite the failing callback
        performance_monitor.record_metric(MetricType.ACCURACY, 0.65, "ratio")
        
        # Alert should still be recorded
        assert len(performance_monitor.alerts_history) == 1