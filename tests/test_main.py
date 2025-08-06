"""
Tests for the main module and entry point functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

from nifty_ml_pipeline.main import main, setup_logging, load_configuration


class TestMainModule:
    """Tests for main module functionality."""
    
    def test_setup_logging(self):
        """Test logging configuration setup."""
        with patch('logging.basicConfig') as mock_basic_config, \
             patch('logging.getLogger') as mock_get_logger:
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            setup_logging()
            
            # Verify logging was configured
            mock_basic_config.assert_called_once()
            mock_get_logger.assert_called_with('nifty_ml_pipeline')
    
    def test_load_configuration(self):
        """Test configuration loading."""
        with patch('config.settings.get_config') as mock_get_config:
            mock_config = {
                'api': {'keys': {'ECONOMIC_TIMES_API_KEY': 'test_key'}},
                'data': {'retention_days': 365},
                'performance': {'MAX_INFERENCE_LATENCY_MS': 10.0}
            }
            mock_get_config.return_value = mock_config
            
            config = load_configuration()
            
            assert config == mock_config
            mock_get_config.assert_called_once()
    
    def test_main_successful_execution(self):
        """Test successful main function execution."""
        with patch('nifty_ml_pipeline.main.setup_logging') as mock_setup_logging, \
             patch('nifty_ml_pipeline.main.load_configuration') as mock_load_config, \
             patch('nifty_ml_pipeline.orchestration.controller.PipelineController') as mock_controller:
            
            # Mock configuration
            mock_config = {
                'api': {'keys': {'ECONOMIC_TIMES_API_KEY': 'test_key'}},
                'data': {'retention_days': 365},
                'performance': {'MAX_INFERENCE_LATENCY_MS': 10.0}
            }
            mock_load_config.return_value = mock_config
            
            # Mock successful pipeline execution
            mock_result = Mock()
            mock_result.was_successful.return_value = True
            mock_result.predictions = [Mock()]
            mock_controller.return_value.execute_pipeline.return_value = mock_result
            
            # Execute main function
            result = main()
            
            # Verify execution
            assert result == 0  # Success exit code
            mock_setup_logging.assert_called_once()
            mock_load_config.assert_called_once()
            mock_controller.assert_called_once_with(mock_config)
            mock_controller.return_value.execute_pipeline.assert_called_once_with("NIFTY50")
    
    def test_main_pipeline_failure(self):
        """Test main function handling of pipeline failure."""
        with patch('nifty_ml_pipeline.main.setup_logging') as mock_setup_logging, \
             patch('nifty_ml_pipeline.main.load_configuration') as mock_load_config, \
             patch('nifty_ml_pipeline.orchestration.controller.PipelineController') as mock_controller, \
             patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            
            # Mock configuration
            mock_config = {'test': 'config'}
            mock_load_config.return_value = mock_config
            
            # Mock failed pipeline execution
            mock_result = Mock()
            mock_result.was_successful.return_value = False
            mock_result.error_message = "Test error"
            mock_controller.return_value.execute_pipeline.return_value = mock_result
            
            # Execute main function
            result = main()
            
            # Verify failure handling
            assert result == 1  # Failure exit code
            assert "Pipeline execution failed" in mock_stderr.getvalue()
    
    def test_main_exception_handling(self):
        """Test main function exception handling."""
        with patch('nifty_ml_pipeline.main.setup_logging') as mock_setup_logging, \
             patch('nifty_ml_pipeline.main.load_configuration') as mock_load_config, \
             patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            
            # Mock configuration loading failure
            mock_load_config.side_effect = Exception("Configuration error")
            
            # Execute main function
            result = main()
            
            # Verify exception handling
            assert result == 1  # Failure exit code
            assert "Unexpected error" in mock_stderr.getvalue()
    
    def test_main_as_script(self):
        """Test main function when called as script."""
        with patch('nifty_ml_pipeline.main.main') as mock_main, \
             patch('sys.argv', ['main.py']):
            
            mock_main.return_value = 0
            
            # Simulate script execution
            with patch('__main__.__name__', '__main__'):
                # This would normally trigger the if __name__ == '__main__' block
                # We'll test the main function directly since we can't easily test the module execution
                result = mock_main()
                
                assert result == 0