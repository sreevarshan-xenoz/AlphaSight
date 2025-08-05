# tests/orchestration/test_scheduler.py
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytz

from nifty_ml_pipeline.orchestration.scheduler import TaskScheduler
from nifty_ml_pipeline.orchestration.controller import PipelineController


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        'scheduling': {
            'execution_time': '17:30',
            'timezone': 'Asia/Kolkata'
        },
        'api': {
            'keys': {
                'ECONOMIC_TIMES_API_KEY': 'test_key'
            }
        },
        'data': {
            'retention_days': 365,
            'storage_format': 'parquet'
        },
        'performance': {
            'MAX_INFERENCE_LATENCY_MS': 10.0
        },
        'paths': {
            'data': '/tmp/test_data'
        }
    }


@pytest.fixture
def mock_pipeline_controller():
    """Mock pipeline controller for testing."""
    controller = Mock(spec=PipelineController)
    
    # Mock successful pipeline result
    mock_result = Mock()
    mock_result.was_successful.return_value = True
    mock_result.execution_id = "test_execution_123"
    mock_result.predictions = [{'signal': 'Buy', 'confidence': 0.85}]
    mock_result.stage_results = [
        {'stage': 'data_collection', 'status': 'completed', 'duration_ms': 1000},
        {'stage': 'feature_engineering', 'status': 'completed', 'duration_ms': 500},
        {'stage': 'model_inference', 'status': 'completed', 'duration_ms': 5}
    ]
    mock_result.to_dict.return_value = {
        'execution_id': 'test_execution_123',
        'status': 'completed',
        'predictions': [{'signal': 'Buy', 'confidence': 0.85}]
    }
    
    controller.execute_pipeline.return_value = mock_result
    return controller


class TestTaskScheduler:
    """Test cases for TaskScheduler."""
    
    def test_initialization(self, mock_config):
        """Test scheduler initialization."""
        scheduler = TaskScheduler(mock_config)
        
        assert scheduler.config == mock_config
        assert scheduler.execution_time == '17:30'
        assert scheduler.execution_hour == 17
        assert scheduler.execution_minute == 30
        assert scheduler.timezone.zone == 'Asia/Kolkata'
        assert not scheduler.is_running
        assert scheduler.scheduler_thread is None
    
    def test_get_next_execution_time_today(self, mock_config):
        """Test next execution time calculation when execution is later today."""
        scheduler = TaskScheduler(mock_config)
        
        # Mock current time to be before execution time
        with patch('nifty_ml_pipeline.orchestration.scheduler.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 10, 0, 0)  # 10:00 AM
            mock_datetime.now.return_value = mock_now.replace(tzinfo=scheduler.timezone)
            
            next_execution = scheduler._get_next_execution_time()
            
            expected = mock_now.replace(hour=17, minute=30, second=0, microsecond=0, tzinfo=scheduler.timezone)
            assert next_execution == expected
    
    def test_get_next_execution_time_tomorrow(self, mock_config):
        """Test next execution time calculation when execution is tomorrow."""
        scheduler = TaskScheduler(mock_config)
        
        # Mock current time to be after execution time
        with patch('nifty_ml_pipeline.orchestration.scheduler.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 18, 0, 0)  # 6:00 PM (after 5:30 PM)
            mock_datetime.now.return_value = mock_now.replace(tzinfo=scheduler.timezone)
            
            next_execution = scheduler._get_next_execution_time()
            
            expected = (mock_now + timedelta(days=1)).replace(
                hour=17, minute=30, second=0, microsecond=0, tzinfo=scheduler.timezone
            )
            assert next_execution == expected
    
    def test_wait_with_interrupt_normal(self, mock_config):
        """Test wait with interrupt under normal conditions."""
        scheduler = TaskScheduler(mock_config)
        
        start_time = time.time()
        interrupted = scheduler._wait_with_interrupt(0.1)  # Wait 100ms
        end_time = time.time()
        
        assert not interrupted  # Should not be interrupted
        assert 0.08 <= (end_time - start_time) <= 0.15  # Allow some timing variance
    
    def test_wait_with_interrupt_stop_signal(self, mock_config):
        """Test wait with interrupt when stop signal is set."""
        scheduler = TaskScheduler(mock_config)
        
        # Set stop event immediately
        scheduler.stop_event.set()
        
        start_time = time.time()
        interrupted = scheduler._wait_with_interrupt(1.0)  # Try to wait 1 second
        end_time = time.time()
        
        assert interrupted  # Should be interrupted
        assert (end_time - start_time) < 0.1  # Should return quickly
    
    @patch('nifty_ml_pipeline.orchestration.scheduler.PipelineController')
    def test_execute_now_success(self, mock_controller_class, mock_config, mock_pipeline_controller):
        """Test immediate pipeline execution."""
        mock_controller_class.return_value = mock_pipeline_controller
        
        scheduler = TaskScheduler(mock_config)
        result = scheduler.execute_now("NIFTY 50")
        
        assert result['execution_id'] == 'test_execution_123'
        assert result['status'] == 'completed'
        assert len(result['predictions']) == 1
        mock_pipeline_controller.execute_pipeline.assert_called_once_with("NIFTY 50")
    
    @patch('nifty_ml_pipeline.orchestration.scheduler.PipelineController')
    def test_execute_now_failure(self, mock_controller_class, mock_config):
        """Test immediate pipeline execution with failure."""
        mock_controller_class.side_effect = Exception("Pipeline initialization failed")
        
        scheduler = TaskScheduler(mock_config)
        result = scheduler.execute_now("NIFTY 50")
        
        assert result['status'] == 'failed'
        assert 'Pipeline initialization failed' in result['error']
        assert 'timestamp' in result
    
    def test_get_next_execution_info_not_running(self, mock_config):
        """Test getting next execution info when scheduler is not running."""
        scheduler = TaskScheduler(mock_config)
        
        info = scheduler.get_next_execution_info()
        
        assert not info['scheduler_running']
        assert 'Scheduler is not running' in info['message']
    
    def test_get_next_execution_info_running(self, mock_config):
        """Test getting next execution info when scheduler is running."""
        scheduler = TaskScheduler(mock_config)
        scheduler.is_running = True
        
        with patch('nifty_ml_pipeline.orchestration.scheduler.datetime') as mock_datetime:
            mock_now = datetime(2024, 1, 15, 10, 0, 0, tzinfo=scheduler.timezone)
            mock_datetime.now.return_value = mock_now
            
            info = scheduler.get_next_execution_info()
            
            assert info['scheduler_running']
            assert 'next_execution_time' in info
            assert 'current_time' in info
            assert 'time_until_execution_seconds' in info
            assert info['execution_time'] == '17:30'
            assert info['timezone'] == 'Asia/Kolkata'
    
    def test_get_scheduler_status(self, mock_config):
        """Test getting scheduler status."""
        scheduler = TaskScheduler(mock_config)
        
        status = scheduler.get_scheduler_status()
        
        assert 'is_running' in status
        assert 'execution_time' in status
        assert 'timezone' in status
        assert 'thread_alive' in status
        assert 'stop_event_set' in status
        assert 'next_execution' in status
        
        assert not status['is_running']
        assert status['execution_time'] == '17:30'
        assert status['timezone'] == 'Asia/Kolkata'
    
    def test_start_stop_scheduler(self, mock_config):
        """Test starting and stopping the scheduler."""
        scheduler = TaskScheduler(mock_config)
        
        # Test start
        with patch.object(scheduler, '_scheduler_loop') as mock_loop:
            # Make the mock loop block briefly to keep thread alive
            import threading
            mock_loop.side_effect = lambda: threading.Event().wait(0.1)
            
            scheduler.start_scheduler()
            
            assert scheduler.is_running
            assert scheduler.scheduler_thread is not None
            # Give thread a moment to start
            time.sleep(0.05)
            assert scheduler.scheduler_thread.is_alive()
        
        # Test stop
        scheduler.stop_scheduler()
        
        assert not scheduler.is_running
        assert scheduler.stop_event.is_set()
    
    def test_start_scheduler_already_running(self, mock_config):
        """Test starting scheduler when it's already running."""
        scheduler = TaskScheduler(mock_config)
        scheduler.is_running = True
        
        with patch('nifty_ml_pipeline.orchestration.scheduler.Thread') as mock_thread:
            scheduler.start_scheduler()
            
            # Should not create new thread
            mock_thread.assert_not_called()
    
    def test_stop_scheduler_not_running(self, mock_config):
        """Test stopping scheduler when it's not running."""
        scheduler = TaskScheduler(mock_config)
        
        # Should handle gracefully
        scheduler.stop_scheduler()
        
        assert not scheduler.is_running
    
    @patch('nifty_ml_pipeline.orchestration.scheduler.PipelineController')
    def test_execute_scheduled_pipeline_success(self, mock_controller_class, mock_config, mock_pipeline_controller):
        """Test scheduled pipeline execution success."""
        mock_controller_class.return_value = mock_pipeline_controller
        
        scheduler = TaskScheduler(mock_config)
        
        # Execute scheduled pipeline
        scheduler._execute_scheduled_pipeline()
        
        # Verify pipeline was executed
        mock_pipeline_controller.execute_pipeline.assert_called_once()
    
    @patch('nifty_ml_pipeline.orchestration.scheduler.PipelineController')
    def test_execute_scheduled_pipeline_failure(self, mock_controller_class, mock_config):
        """Test scheduled pipeline execution failure."""
        mock_controller_class.side_effect = Exception("Scheduled execution failed")
        
        scheduler = TaskScheduler(mock_config)
        
        # Should handle exception gracefully
        scheduler._execute_scheduled_pipeline()
        
        # No exception should be raised
        assert True  # Test passes if no exception is raised
    
    def test_scheduler_with_provided_controller(self, mock_config, mock_pipeline_controller):
        """Test scheduler with pre-initialized pipeline controller."""
        scheduler = TaskScheduler(mock_config, mock_pipeline_controller)
        
        assert scheduler.pipeline_controller == mock_pipeline_controller
        
        # Execute now should use provided controller
        result = scheduler.execute_now("NIFTY 50")
        
        assert result['execution_id'] == 'test_execution_123'
        mock_pipeline_controller.execute_pipeline.assert_called_once_with("NIFTY 50")