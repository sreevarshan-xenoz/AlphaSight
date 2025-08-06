# Configuration Management Guide

## Overview

This guide covers comprehensive configuration management for the NIFTY 50 ML Pipeline, including environment variables, configuration files, and deployment-specific settings.

## Configuration Architecture

The pipeline uses a hierarchical configuration system:

1. **Default Configuration**: Built-in defaults in `config/settings.py`
2. **Environment Variables**: Override defaults via `.env` file or system environment
3. **Runtime Configuration**: Dynamic configuration during execution
4. **Deployment-Specific**: Environment-specific overrides

## Environment Variables

### Core Configuration

Create a `.env` file in the project root:

```bash
# =============================================================================
# NIFTY 50 ML Pipeline Configuration
# =============================================================================

# Environment Settings
ENVIRONMENT=production                    # development, staging, production
DEBUG=false                              # Enable debug mode
VERSION=1.0.0                           # Application version

# =============================================================================
# API Configuration
# =============================================================================

# Economic Times API
ECONOMIC_TIMES_API_KEY=your_api_key_here
ECONOMIC_TIMES_BASE_URL=https://economictimes.indiatimes.com
ECONOMIC_TIMES_TIMEOUT=30
ECONOMIC_TIMES_MAX_RETRIES=3

# NSE API Configuration
NSE_API_TIMEOUT=30
NSE_MAX_RETRIES=3
NSE_BASE_DELAY=1.0
NSE_RATE_LIMIT_DELAY=2.0

# =============================================================================
# Data Configuration
# =============================================================================

# Data Retention
DATA_RETENTION_DAYS=365                  # Rolling window size
NEWS_RETENTION_DAYS=30                   # News relevance window
CACHE_TTL_HOURS=24                       # Cache time-to-live

# Storage Settings
DATA_PATH=data                           # Base data directory
STORAGE_FORMAT=parquet                   # parquet, csv, json
COMPRESSION=snappy                       # snappy, gzip, lz4
ENABLE_CACHING=true                      # Enable data caching
CACHE_SIZE_MB=500                        # Maximum cache size

# Data Quality
ENABLE_DATA_VALIDATION=true              # Enable data validation
MIN_DATA_QUALITY_SCORE=0.8              # Minimum quality threshold
HANDLE_MISSING_DATA=interpolate          # interpolate, drop, fill

# =============================================================================
# Model Configuration
# =============================================================================

# Model Paths
MODEL_PATH=models/xgboost_model.pkl
MODEL_BACKUP_PATH=models/backup/
FEATURE_SCALER_PATH=models/feature_scaler.pkl

# XGBoost Parameters
XGBOOST_N_JOBS=1                        # CPU threads (1 for optimization)
XGBOOST_TREE_METHOD=exact               # exact, approx, hist
XGBOOST_MAX_DEPTH=6                     # Tree depth
XGBOOST_N_ESTIMATORS=100                # Number of trees
XGBOOST_LEARNING_RATE=0.1               # Learning rate
XGBOOST_SUBSAMPLE=0.8                   # Row sampling
XGBOOST_COLSAMPLE_BYTREE=0.8            # Column sampling

# Model Performance
MAX_INFERENCE_LATENCY_MS=10             # Maximum inference time
MIN_CONFIDENCE_THRESHOLD=0.7            # Minimum prediction confidence
TARGET_ACCURACY=0.8                     # Target directional accuracy
ENABLE_MODEL_MONITORING=true            # Enable performance monitoring

# =============================================================================
# Feature Engineering Configuration
# =============================================================================

# Technical Indicators
RSI_PERIOD=14                           # RSI calculation period
SMA_PERIOD=5                            # Simple moving average period
MACD_FAST=12                            # MACD fast EMA
MACD_SLOW=26                            # MACD slow EMA
MACD_SIGNAL=9                           # MACD signal line

# Sentiment Analysis
SENTIMENT_MODEL=vader                    # vader, textblob, custom
SENTIMENT_BATCH_SIZE=100                # Batch processing size
SENTIMENT_TIMEOUT_SECONDS=30            # Processing timeout
ENABLE_SENTIMENT_CACHING=true           # Cache sentiment scores

# Feature Normalization
NORMALIZATION_METHOD=standard           # standard, minmax, robust
HANDLE_OUTLIERS=clip                    # clip, remove, transform
OUTLIER_THRESHOLD=3.0                   # Standard deviations

# =============================================================================
# Performance Configuration
# =============================================================================

# CPU Settings
CPU_THREADS=1                           # Number of CPU threads
CPU_AFFINITY=0,1                        # CPU core affinity (comma-separated)
PROCESS_PRIORITY=0                      # Process priority (-20 to 19)

# Memory Settings
MEMORY_LIMIT_GB=4                       # Maximum memory usage
ENABLE_MEMORY_MONITORING=true           # Monitor memory usage
GC_THRESHOLD=700,10,10                  # Garbage collection thresholds
MEMORY_CLEANUP_INTERVAL=3600            # Cleanup interval (seconds)

# I/O Settings
IO_BUFFER_SIZE=8192                     # I/O buffer size
MAX_CONCURRENT_REQUESTS=5               # Maximum concurrent API requests
REQUEST_TIMEOUT=30                      # Request timeout (seconds)

# =============================================================================
# Logging Configuration
# =============================================================================

# Log Levels
LOG_LEVEL=INFO                          # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_DATE_FORMAT=%Y-%m-%d %H:%M:%S

# Log Files
LOG_FILE=pipeline.log                   # Main log file
ERROR_LOG_FILE=error.log               # Error-specific log
PERFORMANCE_LOG_FILE=performance.log    # Performance metrics log

# Log Rotation
LOG_MAX_SIZE_MB=100                     # Maximum log file size
LOG_BACKUP_COUNT=5                      # Number of backup files
LOG_ROTATION_INTERVAL=daily             # daily, weekly, monthly

# Structured Logging
ENABLE_JSON_LOGGING=false               # Enable JSON log format
LOG_CORRELATION_ID=true                 # Include correlation IDs
LOG_PERFORMANCE_METRICS=true           # Log performance data

# =============================================================================
# Scheduling Configuration
# =============================================================================

# Execution Schedule
EXECUTION_TIME=17:30                    # Daily execution time (HH:MM)
TIMEZONE=Asia/Kolkata                   # Timezone for scheduling
ENABLE_SCHEDULER=true                   # Enable automatic scheduling
SCHEDULE_RETRY_ATTEMPTS=3               # Retry attempts for failed runs

# GitHub Actions
GITHUB_ACTIONS_ENABLED=true             # Enable GitHub Actions scheduling
GITHUB_ACTIONS_CRON=30 12 * * *        # Cron expression (UTC)
GITHUB_ACTIONS_TIMEOUT=30               # Timeout in minutes

# Manual Execution
ALLOW_MANUAL_EXECUTION=true             # Allow manual pipeline runs
MANUAL_EXECUTION_COOLDOWN=300           # Cooldown between manual runs (seconds)

# =============================================================================
# Monitoring and Alerting
# =============================================================================

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING=true      # Enable performance tracking
PERFORMANCE_ALERT_THRESHOLD=15          # Alert if latency > threshold (ms)
ACCURACY_ALERT_THRESHOLD=0.75           # Alert if accuracy < threshold
MEMORY_ALERT_THRESHOLD=80               # Alert if memory usage > threshold (%)

# Health Checks
ENABLE_HEALTH_CHECKS=true               # Enable health monitoring
HEALTH_CHECK_INTERVAL=300               # Health check interval (seconds)
HEALTH_CHECK_TIMEOUT=10                 # Health check timeout (seconds)

# Alerting
ALERT_EMAIL=admin@example.com           # Email for alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
ENABLE_EMAIL_ALERTS=true                # Enable email notifications
ENABLE_SLACK_ALERTS=true                # Enable Slack notifications

# Metrics Collection
ENABLE_METRICS_COLLECTION=true          # Collect system metrics
METRICS_RETENTION_DAYS=30               # Metrics retention period
METRICS_EXPORT_FORMAT=json              # json, csv, prometheus

# =============================================================================
# Security Configuration
# =============================================================================

# API Security
API_RATE_LIMIT=100                      # Requests per minute
API_RATE_LIMIT_WINDOW=60                # Rate limit window (seconds)
ENABLE_API_KEY_ROTATION=false           # Enable automatic key rotation
API_KEY_ROTATION_DAYS=90                # Key rotation interval

# Data Security
ENCRYPT_SENSITIVE_DATA=false            # Encrypt sensitive data at rest
ENCRYPTION_KEY_PATH=keys/encryption.key # Path to encryption key
ENABLE_DATA_ANONYMIZATION=false         # Anonymize sensitive data

# Access Control
ENABLE_ACCESS_LOGGING=true              # Log access attempts
MAX_LOGIN_ATTEMPTS=5                    # Maximum failed login attempts
LOGIN_LOCKOUT_DURATION=300              # Lockout duration (seconds)

# =============================================================================
# Development Configuration
# =============================================================================

# Development Mode
ENABLE_DEBUG_MODE=false                 # Enable debug features
ENABLE_PROFILING=false                  # Enable performance profiling
ENABLE_MOCK_DATA=false                  # Use mock data for testing
MOCK_DATA_PATH=tests/fixtures/          # Path to mock data

# Testing
RUN_TESTS_ON_STARTUP=false              # Run tests before execution
TEST_DATA_PATH=tests/data/              # Path to test data
ENABLE_INTEGRATION_TESTS=true           # Enable integration tests
TEST_TIMEOUT=300                        # Test timeout (seconds)

# Development Tools
ENABLE_HOT_RELOAD=false                 # Enable hot reloading
ENABLE_DEBUG_TOOLBAR=false              # Enable debug toolbar
DEVELOPMENT_PORT=8000                   # Development server port
```

## Configuration Validation

### Startup Validation

The pipeline validates configuration at startup:

```python
# config/validator.py
from typing import Dict, Any, List
import os
import logging

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates configuration settings."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate complete configuration."""
        self.errors.clear()
        self.warnings.clear()
        
        # Validate required settings
        self._validate_required_settings(config)
        
        # Validate API configuration
        self._validate_api_config(config.get('api', {}))
        
        # Validate model configuration
        self._validate_model_config(config.get('model', {}))
        
        # Validate performance settings
        self._validate_performance_config(config.get('performance', {}))
        
        # Validate paths
        self._validate_paths(config.get('paths', {}))
        
        # Log results
        if self.errors:
            for error in self.errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        if self.warnings:
            for warning in self.warnings:
                logger.warning(f"Configuration warning: {warning}")
        
        logger.info("Configuration validation passed")
        return True
    
    def _validate_required_settings(self, config: Dict[str, Any]) -> None:
        """Validate required configuration settings."""
        required_keys = [
            'api.keys.ECONOMIC_TIMES_API_KEY',
            'model.path',
            'performance.MAX_INFERENCE_LATENCY_MS',
            'logging.level'
        ]
        
        for key_path in required_keys:
            if not self._get_nested_value(config, key_path):
                self.errors.append(f"Required setting missing: {key_path}")
    
    def _validate_api_config(self, api_config: Dict[str, Any]) -> None:
        """Validate API configuration."""
        # Check API timeouts
        timeouts = api_config.get('timeouts', {})
        for timeout_key, timeout_value in timeouts.items():
            if not isinstance(timeout_value, int) or timeout_value <= 0:
                self.errors.append(f"Invalid timeout value: {timeout_key}={timeout_value}")
        
        # Check API keys
        keys = api_config.get('keys', {})
        if not keys.get('ECONOMIC_TIMES_API_KEY'):
            self.warnings.append("Economic Times API key not configured")
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        # Check inference latency target
        max_latency = model_config.get('max_inference_latency_ms')
        if max_latency and max_latency > 50:
            self.warnings.append(f"High inference latency target: {max_latency}ms")
        
        # Check confidence threshold
        confidence = model_config.get('min_confidence_threshold')
        if confidence and (confidence < 0.0 or confidence > 1.0):
            self.errors.append(f"Invalid confidence threshold: {confidence}")
    
    def _validate_performance_config(self, perf_config: Dict[str, Any]) -> None:
        """Validate performance configuration."""
        # Check CPU threads
        cpu_threads = perf_config.get('cpu_threads')
        if cpu_threads and cpu_threads != 1:
            self.warnings.append("CPU threads != 1 may impact performance optimization")
        
        # Check memory limit
        memory_limit = perf_config.get('memory_limit_gb')
        if memory_limit and memory_limit < 2:
            self.warnings.append(f"Low memory limit: {memory_limit}GB")
    
    def _validate_paths(self, paths_config: Dict[str, Any]) -> None:
        """Validate file paths."""
        for path_name, path_value in paths_config.items():
            if path_value and not os.path.exists(os.path.dirname(path_value)):
                try:
                    os.makedirs(os.path.dirname(path_value), exist_ok=True)
                    logger.info(f"Created directory for {path_name}: {os.path.dirname(path_value)}")
                except OSError as e:
                    self.errors.append(f"Cannot create directory for {path_name}: {e}")
    
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

# Usage in main configuration loading
def load_and_validate_config() -> Dict[str, Any]:
    """Load and validate configuration."""
    config = get_config()
    validator = ConfigValidator()
    
    if not validator.validate_config(config):
        raise ValueError("Configuration validation failed")
    
    return config
```

## Environment-Specific Configurations

### Development Configuration

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_MOCK_DATA=true
ENABLE_PROFILING=true
ENABLE_HOT_RELOAD=true
MAX_INFERENCE_LATENCY_MS=50  # Relaxed for development
ENABLE_SCHEDULER=false       # Manual execution only
```

### Staging Configuration

```bash
# .env.staging
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
ENABLE_MOCK_DATA=false
ENABLE_PROFILING=false
MAX_INFERENCE_LATENCY_MS=15  # Slightly relaxed
ENABLE_SCHEDULER=true
ALERT_EMAIL=staging-alerts@example.com
```

### Production Configuration

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
ENABLE_MOCK_DATA=false
ENABLE_PROFILING=false
MAX_INFERENCE_LATENCY_MS=10  # Strict requirement
ENABLE_SCHEDULER=true
ENABLE_PERFORMANCE_MONITORING=true
ALERT_EMAIL=production-alerts@example.com
ENABLE_SLACK_ALERTS=true
```

## Configuration Management Best Practices

### Security Best Practices

1. **Never commit sensitive data** to version control
2. **Use environment variables** for secrets
3. **Rotate API keys** regularly
4. **Encrypt configuration files** in production
5. **Use secure key management** systems

### Configuration Organization

```python
# config/environments.py
import os
from typing import Dict, Any

class ConfigManager:
    """Manages environment-specific configurations."""
    
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.base_config = self._load_base_config()
        self.env_config = self._load_environment_config()
    
    def get_config(self) -> Dict[str, Any]:
        """Get merged configuration for current environment."""
        config = self.base_config.copy()
        config.update(self.env_config)
        return config
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration common to all environments."""
        return {
            'version': '1.0.0',
            'app_name': 'NIFTY 50 ML Pipeline',
            'default_timeout': 30,
            'default_retries': 3,
        }
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        env_configs = {
            'development': self._get_development_config(),
            'staging': self._get_staging_config(),
            'production': self._get_production_config(),
        }
        
        return env_configs.get(self.environment, {})
    
    def _get_development_config(self) -> Dict[str, Any]:
        """Development-specific configuration."""
        return {
            'debug': True,
            'log_level': 'DEBUG',
            'enable_profiling': True,
            'max_inference_latency_ms': 50,
            'enable_mock_data': True,
        }
    
    def _get_staging_config(self) -> Dict[str, Any]:
        """Staging-specific configuration."""
        return {
            'debug': False,
            'log_level': 'INFO',
            'enable_profiling': False,
            'max_inference_latency_ms': 15,
            'enable_mock_data': False,
        }
    
    def _get_production_config(self) -> Dict[str, Any]:
        """Production-specific configuration."""
        return {
            'debug': False,
            'log_level': 'INFO',
            'enable_profiling': False,
            'max_inference_latency_ms': 10,
            'enable_mock_data': False,
            'enable_monitoring': True,
        }
```

### Configuration Testing

```python
# tests/test_config.py
import pytest
from config.validator import ConfigValidator
from config.settings import get_config

class TestConfiguration:
    """Test configuration management."""
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = get_config()
        validator = ConfigValidator()
        
        assert validator.validate_config(config)
        assert len(validator.errors) == 0
    
    def test_required_settings(self):
        """Test that required settings are present."""
        config = get_config()
        
        # Check critical settings
        assert config['performance']['MAX_INFERENCE_LATENCY_MS'] <= 10
        assert config['model']['min_confidence_threshold'] >= 0.0
        assert config['model']['min_confidence_threshold'] <= 1.0
    
    def test_environment_specific_config(self):
        """Test environment-specific configurations."""
        import os
        
        # Test development config
        os.environ['ENVIRONMENT'] = 'development'
        dev_config = get_config()
        assert dev_config.get('debug') is True
        
        # Test production config
        os.environ['ENVIRONMENT'] = 'production'
        prod_config = get_config()
        assert prod_config.get('debug') is False
        assert prod_config['performance']['MAX_INFERENCE_LATENCY_MS'] == 10
```

## Configuration Monitoring

### Runtime Configuration Changes

```python
# config/monitor.py
import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class ConfigChangeHandler(FileSystemEventHandler):
    """Handle configuration file changes."""
    
    def __init__(self, callback):
        self.callback = callback
        self.last_modified = {}
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        if event.src_path.endswith('.env'):
            # Debounce rapid changes
            current_time = time.time()
            if (event.src_path not in self.last_modified or 
                current_time - self.last_modified[event.src_path] > 1.0):
                
                logger.info(f"Configuration file changed: {event.src_path}")
                self.last_modified[event.src_path] = current_time
                self.callback(event.src_path)

class ConfigMonitor:
    """Monitor configuration changes."""
    
    def __init__(self, config_path='.env'):
        self.config_path = config_path
        self.observer = Observer()
        self.handler = ConfigChangeHandler(self._on_config_change)
    
    def start_monitoring(self):
        """Start monitoring configuration changes."""
        self.observer.schedule(
            self.handler, 
            path=os.path.dirname(self.config_path) or '.', 
            recursive=False
        )
        self.observer.start()
        logger.info("Configuration monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring configuration changes."""
        self.observer.stop()
        self.observer.join()
        logger.info("Configuration monitoring stopped")
    
    def _on_config_change(self, file_path):
        """Handle configuration change."""
        try:
            # Reload configuration
            from config.settings import reload_config
            new_config = reload_config()
            
            # Validate new configuration
            validator = ConfigValidator()
            if validator.validate_config(new_config):
                logger.info("Configuration reloaded successfully")
                # Notify application components of config change
                self._notify_config_change(new_config)
            else:
                logger.error("Invalid configuration detected, keeping current config")
                
        except Exception as e:
            logger.error(f"Error reloading configuration: {e}")
    
    def _notify_config_change(self, new_config):
        """Notify application components of configuration change."""
        # Implementation depends on your application architecture
        # Could use events, signals, or direct method calls
        pass
```

This comprehensive configuration management system provides:

1. **Hierarchical Configuration**: Base defaults with environment-specific overrides
2. **Validation**: Startup and runtime configuration validation
3. **Security**: Best practices for handling sensitive data
4. **Monitoring**: Real-time configuration change detection
5. **Testing**: Comprehensive configuration testing
6. **Documentation**: Clear documentation of all configuration options