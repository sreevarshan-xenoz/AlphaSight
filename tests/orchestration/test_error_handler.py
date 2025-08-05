# tests/orchestration/test_error_handler.py
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from nifty_ml_pipeline.orchestration.error_handler import (
    ErrorHandler, ErrorContext, RecoveryResult, RecoveryAction, ErrorSeverity
)
from nifty_ml_pipeline.data.models import PipelineStage


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        'error_handling': {
            'max_retries': 3,
            'retry_delay_seconds': 1,
            'enable_fallback_data': True
        }
    }


@pytest.fixture
def error_handler(mock_config):
    """Create error handler instance for testing."""
    return ErrorHandler(mock_config)


@pytest.fixture
def sample_error_context():
    """Create sample error context for testing."""
    return ErrorContext(
        stage=PipelineStage.DATA_COLLECTION,
        error=ConnectionError("API connection failed"),
        timestamp=datetime.now(),
        execution_id="test_execution_123",
        attempt_number=1,
        metadata={'symbol': 'NIFTY 50'}
    )


class TestErrorHandler:
    """Test cases for ErrorHandler."""
    
    def test_initialization(self, mock_config):
        """Test error handler initialization."""
        handler = ErrorHandler(mock_config)
        
        assert handler.config == mock_config
        assert handler.max_retries == 3
        assert handler.retry_delay_seconds == 1
        assert handler.enable_fallback_data is True
        assert len(handler.error_history) == 0
        assert len(handler.recovery_strategies) == 3  # Three pipeline stages
    
    def test_handle_connection_error_retry(self, error_handler, sample_error_context):
        """Test handling connection error with retry strategy."""
        result = error_handler.handle_error(sample_error_context)
        
        assert result.action_taken == RecoveryAction.RETRY
        assert result.success is True
        assert result.should_continue is True
        assert "Retrying" in result.message
        assert len(error_handler.error_history) == 1
    
    def test_handle_max_retries_exceeded(self, error_handler):
        """Test handling when maximum retries are exceeded."""
        error_context = ErrorContext(
            stage=PipelineStage.DATA_COLLECTION,
            error=ConnectionError("API connection failed"),
            timestamp=datetime.now(),
            execution_id="test_execution_123",
            attempt_number=4,  # Exceeds max_retries (3)
            metadata={'symbol': 'NIFTY 50'}
        )
        
        result = error_handler.handle_error(error_context)
        
        assert result.action_taken == RecoveryAction.RETRY
        assert result.success is False
        assert result.should_continue is False
        assert "Maximum retries" in result.message
    
    def test_handle_value_error_fallback(self, error_handler):
        """Test handling ValueError with fallback strategy."""
        error_context = ErrorContext(
            stage=PipelineStage.DATA_COLLECTION,
            error=ValueError("Invalid data format"),
            timestamp=datetime.now(),
            execution_id="test_execution_123",
            attempt_number=1
        )
        
        result = error_handler.handle_error(error_context)
        
        assert result.action_taken == RecoveryAction.USE_FALLBACK
        assert result.success is True
        assert result.should_continue is True
        assert result.data is not None
        assert "fallback data" in result.message.lower()
    
    def test_handle_memory_error_abort(self, error_handler):
        """Test handling MemoryError with abort strategy."""
        error_context = ErrorContext(
            stage=PipelineStage.FEATURE_ENGINEERING,
            error=MemoryError("Out of memory"),
            timestamp=datetime.now(),
            execution_id="test_execution_123",
            attempt_number=1
        )
        
        result = error_handler.handle_error(error_context)
        
        assert result.action_taken == RecoveryAction.ABORT_PIPELINE
        assert result.success is False
        assert result.should_continue is False
        assert "Aborting pipeline" in result.message
    
    def test_handle_generic_error_continue_with_warning(self, error_handler):
        """Test handling generic error with continue warning strategy."""
        error_context = ErrorContext(
            stage=PipelineStage.FEATURE_ENGINEERING,
            error=RuntimeError("Generic runtime error"),
            timestamp=datetime.now(),
            execution_id="test_execution_123",
            attempt_number=1
        )
        
        result = error_handler.handle_error(error_context)
        
        assert result.action_taken == RecoveryAction.CONTINUE_WITH_WARNING
        assert result.success is True
        assert result.should_continue is True
        assert "warning" in result.message.lower()
    
    def test_fallback_data_disabled(self, mock_config):
        """Test fallback handling when fallback data is disabled."""
        mock_config['error_handling']['enable_fallback_data'] = False
        handler = ErrorHandler(mock_config)
        
        error_context = ErrorContext(
            stage=PipelineStage.DATA_COLLECTION,
            error=ValueError("Invalid data format"),
            timestamp=datetime.now(),
            execution_id="test_execution_123",
            attempt_number=1
        )
        
        result = handler.handle_error(error_context)
        
        assert result.action_taken == RecoveryAction.USE_FALLBACK
        assert result.success is False
        assert result.should_continue is False
        assert "disabled" in result.message.lower()
    
    def test_generate_fallback_data_collection(self, error_handler):
        """Test fallback data generation for data collection stage."""
        fallback_data = error_handler._generate_fallback_data(PipelineStage.DATA_COLLECTION)
        
        assert fallback_data is not None
        price_data, news_data = fallback_data
        
        # Check price data structure
        assert len(price_data) == 30
        assert all(col in price_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Check news data structure
        assert len(news_data) == 5
        assert all(col in news_data.columns for col in ['headline', 'timestamp', 'source', 'url'])
    
    def test_generate_fallback_feature_engineering(self, error_handler):
        """Test fallback data generation for feature engineering stage."""
        fallback_data = error_handler._generate_fallback_data(PipelineStage.FEATURE_ENGINEERING)
        
        assert fallback_data is not None
        assert len(fallback_data) == 30
        
        expected_columns = ['lag1_return', 'lag2_return', 'sma_5_ratio', 'rsi_14', 'macd_hist', 'daily_sentiment']
        assert all(col in fallback_data.columns for col in expected_columns)
    
    def test_generate_fallback_model_inference(self, error_handler):
        """Test fallback data generation for model inference stage."""
        fallback_data = error_handler._generate_fallback_data(PipelineStage.MODEL_INFERENCE)
        
        assert fallback_data is not None
        assert len(fallback_data) == 1
        
        prediction = fallback_data[0]
        assert 'timestamp' in prediction
        assert 'symbol' in prediction
        assert 'predicted_direction' in prediction
        assert 'confidence' in prediction
        assert prediction['predicted_direction'] == 'Hold'
        assert prediction['confidence'] == 0.5
    
    def test_error_summary_empty(self, error_handler):
        """Test error summary when no errors have occurred."""
        summary = error_handler.get_error_summary()
        
        assert summary['total_errors'] == 0
        assert summary['errors_by_stage'] == {}
        assert summary['errors_by_type'] == {}
    
    def test_error_summary_with_errors(self, error_handler):
        """Test error summary with multiple errors."""
        # Add some errors to history
        error1 = ErrorContext(
            stage=PipelineStage.DATA_COLLECTION,
            error=ConnectionError("Connection failed"),
            timestamp=datetime.now(),
            execution_id="test_1",
            attempt_number=1
        )
        
        error2 = ErrorContext(
            stage=PipelineStage.DATA_COLLECTION,
            error=ValueError("Invalid data"),
            timestamp=datetime.now(),
            execution_id="test_2",
            attempt_number=1
        )
        
        error3 = ErrorContext(
            stage=PipelineStage.FEATURE_ENGINEERING,
            error=ValueError("Feature error"),
            timestamp=datetime.now(),
            execution_id="test_3",
            attempt_number=1
        )
        
        error_handler.handle_error(error1)
        error_handler.handle_error(error2)
        error_handler.handle_error(error3)
        
        summary = error_handler.get_error_summary()
        
        assert summary['total_errors'] == 3
        assert summary['errors_by_stage']['data_collection'] == 2
        assert summary['errors_by_stage']['feature_engineering'] == 1
        assert summary['errors_by_type']['ConnectionError'] == 1
        assert summary['errors_by_type']['ValueError'] == 2
        assert 'first_error_time' in summary
        assert 'last_error_time' in summary
    
    def test_clear_error_history(self, error_handler, sample_error_context):
        """Test clearing error history."""
        error_handler.handle_error(sample_error_context)
        assert len(error_handler.error_history) == 1
        
        error_handler.clear_error_history()
        assert len(error_handler.error_history) == 0
    
    def test_should_abort_pipeline_no_errors(self, error_handler):
        """Test pipeline abort decision with no errors."""
        assert error_handler.should_abort_pipeline() is False
    
    def test_should_abort_pipeline_critical_errors(self, error_handler):
        """Test pipeline abort decision with critical errors."""
        # Add critical errors
        for i in range(3):
            error_context = ErrorContext(
                stage=PipelineStage.FEATURE_ENGINEERING,
                error=MemoryError("Out of memory"),
                timestamp=datetime.now(),
                execution_id=f"test_{i}",
                attempt_number=1
            )
            error_handler.handle_error(error_context)
        
        assert error_handler.should_abort_pipeline() is True
    
    def test_get_error_severity(self, error_handler):
        """Test error severity classification."""
        assert error_handler._get_error_severity(MemoryError()) == ErrorSeverity.CRITICAL
        assert error_handler._get_error_severity(ConnectionError()) == ErrorSeverity.HIGH
        assert error_handler._get_error_severity(ValueError()) == ErrorSeverity.MEDIUM
        assert error_handler._get_error_severity(RuntimeError()) == ErrorSeverity.LOW
    
    def test_determine_recovery_action_inheritance(self, error_handler):
        """Test recovery action determination with error inheritance."""
        # Test that subclasses of Exception are handled correctly
        class CustomConnectionError(ConnectionError):
            pass
        
        error_context = ErrorContext(
            stage=PipelineStage.DATA_COLLECTION,
            error=CustomConnectionError("Custom connection error"),
            timestamp=datetime.now(),
            execution_id="test_inheritance",
            attempt_number=1
        )
        
        action = error_handler._determine_recovery_action(error_context)
        assert action == RecoveryAction.RETRY  # Should inherit from ConnectionError strategy
    
    @patch('nifty_ml_pipeline.orchestration.error_handler.logger')
    def test_logging_error_details(self, mock_logger, error_handler, sample_error_context):
        """Test that error details are properly logged."""
        error_handler._log_error_details(sample_error_context)
        
        # Verify logging calls were made
        assert mock_logger.error.call_count >= 4  # Multiple error log calls
        
        # Check that key information was logged
        logged_messages = [call.args[0] for call in mock_logger.error.call_args_list]
        assert any("data_collection" in msg for msg in logged_messages)
        assert any("ConnectionError" in msg for msg in logged_messages)
        assert any("API connection failed" in msg for msg in logged_messages)
    
    @patch('nifty_ml_pipeline.orchestration.error_handler.logger')
    def test_logging_recovery_result(self, mock_logger, error_handler, sample_error_context):
        """Test that recovery results are properly logged."""
        recovery_result = RecoveryResult(
            action_taken=RecoveryAction.RETRY,
            success=True,
            message="Retrying operation",
            should_continue=True
        )
        
        error_handler._log_recovery_result(sample_error_context, recovery_result)
        
        # Verify logging calls were made
        assert mock_logger.log.call_count >= 3  # Multiple log calls
        
        # Check that key information was logged
        logged_messages = [call.args[1] for call in mock_logger.log.call_args_list]
        assert any("retry" in msg.lower() for msg in logged_messages)
        assert any("Retrying operation" in msg for msg in logged_messages)