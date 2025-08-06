"""
Tests for the main module and entry point functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

from nifty_ml_pipeline.main import main, setup_logging


class TestMainModule:
    """Tests for main module functionality."""
    
    def test_setup_logging(self):
        """Test logging configuration setup."""
        with patch('logging.basicConfig') as mock_basic_config:
            setup_logging()
            
            # Verify logging was configured
            mock_basic_config.assert_called_once()
            call_args = mock_basic_config.call_args
            assert 'level' in call_args.kwargs
            assert 'format' in call_args.kwargs
            assert 'handlers' in call_args.kwargs
    
    def test_main_successful_execution(self):
        """Test successful main function execution."""
        with patch('nifty_ml_pipeline.main.setup_logging') as mock_setup_logging, \
             patch('config.settings.get_config') as mock_get_config, \
             patch('logging.getLogger') as mock_get_logger:
            
            # Mock configuration
            mock_config = {
                'scheduling': {'execution_time': '17:30 IST'},
                'api': {'keys': {'ECONOMIC_TIMES_API_KEY': 'test_key'}},
                'data': {'retention_days': 365}
            }
            mock_get_config.return_value = mock_config
            
            # Mock logger
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Execute main function (should not raise exception)
            main()
            
            # Verify execution
            mock_setup_logging.assert_called_once()
            mock_get_config.assert_called_once()
            mock_logger.info.assert_called()
    
    def test_main_configuration_failure(self):
        """Test main function handling of configuration failure."""
        with patch('nifty_ml_pipeline.main.setup_logging') as mock_setup_logging, \
             patch('config.settings.get_config') as mock_get_config, \
             patch('logging.getLogger') as mock_get_logger, \
             patch('sys.exit') as mock_exit:
            
            # Mock configuration loading failure
            mock_get_config.side_effect = Exception("Configuration error")
            
            # Mock logger
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Execute main function
            main()
            
            # Verify failure handling
            mock_setup_logging.assert_called_once()
            mock_logger.error.assert_called()
            mock_exit.assert_called_once_with(1)
    
    def test_main_logging_setup(self):
        """Test that main function sets up logging correctly."""
        with patch('nifty_ml_pipeline.main.setup_logging') as mock_setup_logging, \
             patch('config.settings.get_config') as mock_get_config, \
             patch('logging.getLogger') as mock_get_logger:
            
            # Mock successful configuration
            mock_config = {'scheduling': {'execution_time': '17:30 IST'}}
            mock_get_config.return_value = mock_config
            
            # Mock logger
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Execute main function
            main()
            
            # Verify logging setup was called first
            mock_setup_logging.assert_called_once()
            mock_get_logger.assert_called_with('nifty_ml_pipeline.main')
    
    def test_main_configuration_logging(self):
        """Test that main function logs configuration details."""
        with patch('nifty_ml_pipeline.main.setup_logging'), \
             patch('config.settings.get_config') as mock_get_config, \
             patch('logging.getLogger') as mock_get_logger:
            
            # Mock configuration with specific execution time
            execution_time = '17:30 IST'
            mock_config = {'scheduling': {'execution_time': execution_time}}
            mock_get_config.return_value = mock_config
            
            # Mock logger to capture calls
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            # Execute main function
            main()
            
            # Verify configuration-related logging
            info_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Configuration loaded successfully" in call for call in info_calls)
            assert any(execution_time in call for call in info_calls)
            assert any("Pipeline setup complete" in call for call in info_calls)