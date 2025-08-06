"""
Configuration validation module for NIFTY 50 ML Pipeline.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Validates configuration settings at startup and runtime."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate complete configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: True if validation passes, False otherwise
            
        Raises:
            ConfigValidationError: If critical validation errors are found
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Validate required settings
        self._validate_required_settings(config)
        
        # Validate API configuration
        self._validate_api_config(config.get('api', {}))
        
        # Validate data configuration
        self._validate_data_config(config.get('data', {}))
        
        # Validate model configuration
        self._validate_model_config(config.get('model', {}))
        
        # Validate performance settings
        self._validate_performance_config(config.get('performance', {}))
        
        # Validate paths
        self._validate_paths(config.get('paths', {}))
        
        # Validate logging configuration
        self._validate_logging_config(config.get('logging', {}))
        
        # Validate scheduling configuration
        self._validate_scheduling_config(config.get('scheduling', {}))
        
        # Log results
        if self.errors:
            for error in self.errors:
                logger.error(f"Configuration error: {error}")
            raise ConfigValidationError(f"Configuration validation failed with {len(self.errors)} errors")
        
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        logger.info("Configuration validation passed")
        return True
    
    def _validate_required_settings(self, config: Dict[str, Any]) -> None:
        """Validate required configuration settings."""
        required_paths = [
            'performance.MAX_INFERENCE_LATENCY_MS',
            'model.xgboost.n_jobs',
            'model.xgboost.tree_method',
            'data.retention_days',
            'logging.level'
        ]
        
        for path in required_paths:
            value = self._get_nested_value(config, path)
            if value is None:
                self.errors.append(f"Required setting missing: {path}")
    
    def _validate_api_config(self, api_config: Dict[str, Any]) -> None:
        """Validate API configuration."""
        # Check base URLs
        base_urls = {
            'nse_base_url': api_config.get('nse_base_url'),
            'et_base_url': api_config.get('et_base_url')
        }
        
        for url_name, url_value in base_urls.items():
            if url_value and not url_value.startswith(('http://', 'https://')):
                self.errors.append(f"Invalid URL format for {url_name}: {url_value}")
        
        # Check API keys (warnings only, as they might not be required)
        keys = api_config.get('keys', {})
        if not keys.get('ECONOMIC_TIMES_API_KEY'):
            self.warnings.append("Economic Times API key not configured - using fallback data")
        
        if not keys.get('NSE_API_KEY'):
            self.warnings.append("NSE API key not configured - using public endpoints")
    
    def _validate_data_config(self, data_config: Dict[str, Any]) -> None:
        """Validate data configuration."""
        # Check retention days
        retention_days = data_config.get('retention_days')
        if retention_days:
            if not isinstance(retention_days, int) or retention_days <= 0:
                self.errors.append(f"Invalid retention_days: {retention_days}")
            elif retention_days < 30:
                self.warnings.append(f"Low retention_days ({retention_days}) may affect model performance")
            elif retention_days > 1095:  # 3 years
                self.warnings.append(f"High retention_days ({retention_days}) may impact performance")
        
        # Check news relevance days
        news_days = data_config.get('news_relevance_days')
        if news_days:
            if not isinstance(news_days, int) or news_days <= 0:
                self.errors.append(f"Invalid news_relevance_days: {news_days}")
            elif news_days > 90:
                self.warnings.append(f"High news_relevance_days ({news_days}) may include stale news")
        
        # Check storage format
        storage_format = data_config.get('storage_format')
        if storage_format and storage_format not in ['parquet', 'csv', 'json']:
            self.errors.append(f"Unsupported storage_format: {storage_format}")
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        # Validate XGBoost configuration
        xgboost_config = model_config.get('xgboost', {})
        
        # Check n_jobs for CPU optimization
        n_jobs = xgboost_config.get('n_jobs')
        if n_jobs and n_jobs != 1:
            self.warnings.append(f"n_jobs={n_jobs} may not be optimal for CPU performance (recommended: 1)")
        
        # Check tree_method for CPU optimization
        tree_method = xgboost_config.get('tree_method')
        if tree_method and tree_method != 'exact':
            self.warnings.append(f"tree_method='{tree_method}' may not be optimal for CPU (recommended: 'exact')")
        
        # Check max_depth
        max_depth = xgboost_config.get('max_depth')
        if max_depth:
            if not isinstance(max_depth, int) or max_depth <= 0:
                self.errors.append(f"Invalid max_depth: {max_depth}")
            elif max_depth > 10:
                self.warnings.append(f"High max_depth ({max_depth}) may cause overfitting")
        
        # Check learning_rate
        learning_rate = xgboost_config.get('learning_rate')
        if learning_rate:
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
                self.errors.append(f"Invalid learning_rate: {learning_rate}")
        
        # Check n_estimators
        n_estimators = xgboost_config.get('n_estimators')
        if n_estimators:
            if not isinstance(n_estimators, int) or n_estimators <= 0:
                self.errors.append(f"Invalid n_estimators: {n_estimators}")
            elif n_estimators > 1000:
                self.warnings.append(f"High n_estimators ({n_estimators}) may impact training time")
        
        # Validate technical indicators
        tech_indicators = model_config.get('technical_indicators', {})
        
        # Check RSI period
        rsi_period = tech_indicators.get('RSI_PERIOD')
        if rsi_period and (not isinstance(rsi_period, int) or rsi_period <= 0):
            self.errors.append(f"Invalid RSI_PERIOD: {rsi_period}")
        
        # Check SMA period
        sma_period = tech_indicators.get('SMA_PERIOD')
        if sma_period and (not isinstance(sma_period, int) or sma_period <= 0):
            self.errors.append(f"Invalid SMA_PERIOD: {sma_period}")
        
        # Check MACD parameters
        macd_fast = tech_indicators.get('MACD_FAST')
        macd_slow = tech_indicators.get('MACD_SLOW')
        if macd_fast and macd_slow and macd_fast >= macd_slow:
            self.errors.append(f"MACD_FAST ({macd_fast}) must be less than MACD_SLOW ({macd_slow})")
    
    def _validate_performance_config(self, perf_config: Dict[str, Any]) -> None:
        """Validate performance configuration."""
        # Check inference latency target
        max_latency = perf_config.get('MAX_INFERENCE_LATENCY_MS')
        if max_latency:
            if not isinstance(max_latency, (int, float)) or max_latency <= 0:
                self.errors.append(f"Invalid MAX_INFERENCE_LATENCY_MS: {max_latency}")
            elif max_latency > 50:
                self.warnings.append(f"High inference latency target: {max_latency}ms")
        
        # Check accuracy thresholds
        min_accuracy = perf_config.get('MIN_ACCURACY')
        if min_accuracy:
            if not isinstance(min_accuracy, (int, float)) or min_accuracy < 0 or min_accuracy > 1:
                self.errors.append(f"Invalid MIN_ACCURACY: {min_accuracy}")
        
        target_accuracy = perf_config.get('TARGET_ACCURACY')
        if target_accuracy:
            if not isinstance(target_accuracy, (int, float)) or target_accuracy < 0 or target_accuracy > 1:
                self.errors.append(f"Invalid TARGET_ACCURACY: {target_accuracy}")
        
        # Check that target >= min
        if min_accuracy and target_accuracy and target_accuracy < min_accuracy:
            self.errors.append(f"TARGET_ACCURACY ({target_accuracy}) must be >= MIN_ACCURACY ({min_accuracy})")
        
        # Check sentiment processing time
        max_sentiment_time = perf_config.get('MAX_SENTIMENT_PROCESSING_TIME_S')
        if max_sentiment_time:
            if not isinstance(max_sentiment_time, (int, float)) or max_sentiment_time <= 0:
                self.errors.append(f"Invalid MAX_SENTIMENT_PROCESSING_TIME_S: {max_sentiment_time}")
    
    def _validate_paths(self, paths_config: Dict[str, Any]) -> None:
        """Validate file paths and create directories if needed."""
        path_names = ['data', 'models', 'logs']
        
        for path_name in path_names:
            path_value = paths_config.get(path_name)
            if path_value:
                path_obj = Path(path_value)
                
                # Create directory if it doesn't exist
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Ensured directory exists: {path_value}")
                except OSError as e:
                    self.errors.append(f"Cannot create directory {path_name} ({path_value}): {e}")
                
                # Check write permissions
                if path_obj.exists() and not os.access(path_value, os.W_OK):
                    self.errors.append(f"No write permission for {path_name}: {path_value}")
    
    def _validate_logging_config(self, logging_config: Dict[str, Any]) -> None:
        """Validate logging configuration."""
        # Check log level
        log_level = logging_config.get('level')
        if log_level:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if log_level.upper() not in valid_levels:
                self.errors.append(f"Invalid log level: {log_level}")
        
        # Check log format
        log_format = logging_config.get('format')
        if log_format:
            try:
                # Test format string
                logging.Formatter(log_format)
            except (ValueError, TypeError) as e:
                self.errors.append(f"Invalid log format: {e}")
    
    def _validate_scheduling_config(self, scheduling_config: Dict[str, Any]) -> None:
        """Validate scheduling configuration."""
        # Check execution time format
        execution_time = scheduling_config.get('execution_time')
        if execution_time:
            try:
                from datetime import datetime
                datetime.strptime(execution_time, '%H:%M')
            except ValueError:
                self.errors.append(f"Invalid execution_time format: {execution_time} (expected HH:MM)")
        
        # Check timezone
        timezone = scheduling_config.get('timezone')
        if timezone:
            try:
                import pytz
                pytz.timezone(timezone)
            except Exception:
                self.warnings.append(f"Timezone '{timezone}' may not be valid")
    
    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
        """Get nested configuration value using dot notation."""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'is_valid': len(self.errors) == 0
        }


def validate_config_at_startup(config: Dict[str, Any]) -> None:
    """
    Validate configuration at application startup.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ConfigValidationError: If validation fails
    """
    validator = ConfigValidator()
    validator.validate_config(config)
    
    summary = validator.get_validation_summary()
    logger.info(f"Configuration validation completed: "
                f"{summary['error_count']} errors, {summary['warning_count']} warnings")