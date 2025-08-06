"""
Configuration settings for the NIFTY 50 ML Pipeline.

This module provides comprehensive configuration management with environment variable support,
validation, and deployment-specific settings.
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue with system environment variables
    pass

logger = logging.getLogger(__name__)

# Global configuration cache
_config_cache: Optional[Dict[str, Any]] = None


def _get_env_var(key: str, default: Any = None, var_type: type = str) -> Any:
    """
    Get environment variable with type conversion and default value.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        var_type: Type to convert to (str, int, float, bool)
        
    Returns:
        Converted environment variable value or default
    """
    value = os.getenv(key)
    
    if value is None:
        return default
    
    if var_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    elif var_type == int:
        try:
            return int(value)
        except ValueError:
            logger.warning(f"Invalid integer value for {key}: {value}, using default: {default}")
            return default
    elif var_type == float:
        try:
            return float(value)
        except ValueError:
            logger.warning(f"Invalid float value for {key}: {value}, using default: {default}")
            return default
    else:
        return value


def _get_base_paths() -> Dict[str, str]:
    """Get base directory paths."""
    base_dir = Path(__file__).parent.parent.absolute()
    
    return {
        "base": str(base_dir),
        "data": _get_env_var("DATA_PATH", str(base_dir / "data")),
        "models": _get_env_var("MODELS_PATH", str(base_dir / "models")),
        "logs": _get_env_var("LOGS_PATH", str(base_dir / "logs")),
        "cache": _get_env_var("CACHE_PATH", str(base_dir / "data" / "cache")),
        "config": str(base_dir / "config")
    }


def _get_api_config() -> Dict[str, Any]:
    """Get API configuration from environment variables."""
    return {
        "nse_base_url": _get_env_var("NSE_API_BASE_URL", "https://www.nseindia.com"),
        "et_base_url": _get_env_var("ECONOMIC_TIMES_API_BASE_URL", "https://economictimes.indiatimes.com"),
        "keys": {
            "NSE_API_KEY": _get_env_var("NSE_API_KEY"),
            "ECONOMIC_TIMES_API_KEY": _get_env_var("ECONOMIC_TIMES_API_KEY")
        },
        "timeouts": {
            "NSE_API_TIMEOUT": _get_env_var("NSE_API_TIMEOUT", 30, int),
            "ECONOMIC_TIMES_TIMEOUT": _get_env_var("ECONOMIC_TIMES_TIMEOUT", 30, int)
        },
        "retries": {
            "NSE_MAX_RETRIES": _get_env_var("NSE_MAX_RETRIES", 3, int),
            "ECONOMIC_TIMES_MAX_RETRIES": _get_env_var("ECONOMIC_TIMES_MAX_RETRIES", 3, int),
            "BASE_DELAY": _get_env_var("API_BASE_DELAY", 1.0, float)
        }
    }


def _get_data_config() -> Dict[str, Any]:
    """Get data configuration from environment variables."""
    return {
        "retention_days": _get_env_var("DATA_RETENTION_DAYS", 365, int),
        "news_relevance_days": _get_env_var("NEWS_RELEVANCE_DAYS", 30, int),
        "storage_format": _get_env_var("STORAGE_FORMAT", "parquet"),
        "compression": _get_env_var("COMPRESSION", "snappy"),
        "cache_enabled": _get_env_var("ENABLE_CACHING", True, bool),
        "cache_ttl_hours": _get_env_var("CACHE_TTL_HOURS", 24, int),
        "cache_size_mb": _get_env_var("CACHE_SIZE_MB", 500, int),
        "enable_validation": _get_env_var("ENABLE_DATA_VALIDATION", True, bool),
        "min_quality_score": _get_env_var("MIN_DATA_QUALITY_SCORE", 0.8, float),
        "missing_data_strategy": _get_env_var("HANDLE_MISSING_DATA", "interpolate")
    }


def _get_model_config() -> Dict[str, Any]:
    """Get model configuration from environment variables."""
    return {
        "path": _get_env_var("MODEL_PATH", "models/xgboost_model.pkl"),
        "backup_path": _get_env_var("MODEL_BACKUP_PATH", "models/backup/"),
        "feature_scaler_path": _get_env_var("FEATURE_SCALER_PATH", "models/feature_scaler.pkl"),
        "xgboost": {
            "n_jobs": _get_env_var("XGBOOST_N_JOBS", 1, int),
            "tree_method": _get_env_var("XGBOOST_TREE_METHOD", "exact"),
            "max_depth": _get_env_var("XGBOOST_MAX_DEPTH", 6, int),
            "learning_rate": _get_env_var("XGBOOST_LEARNING_RATE", 0.1, float),
            "n_estimators": _get_env_var("XGBOOST_N_ESTIMATORS", 100, int),
            "subsample": _get_env_var("XGBOOST_SUBSAMPLE", 0.8, float),
            "colsample_bytree": _get_env_var("XGBOOST_COLSAMPLE_BYTREE", 0.8, float),
            "random_state": _get_env_var("XGBOOST_RANDOM_STATE", 42, int),
            "objective": _get_env_var("XGBOOST_OBJECTIVE", "reg:squarederror"),
            "eval_metric": _get_env_var("XGBOOST_EVAL_METRIC", "rmse")
        },
        "technical_indicators": {
            "RSI_PERIOD": _get_env_var("RSI_PERIOD", 14, int),
            "SMA_PERIOD": _get_env_var("SMA_PERIOD", 5, int),
            "MACD_FAST": _get_env_var("MACD_FAST", 12, int),
            "MACD_SLOW": _get_env_var("MACD_SLOW", 26, int),
            "MACD_SIGNAL": _get_env_var("MACD_SIGNAL", 9, int)
        },
        "feature_engineering": {
            "normalization_method": _get_env_var("NORMALIZATION_METHOD", "standard"),
            "handle_outliers": _get_env_var("HANDLE_OUTLIERS", "clip"),
            "outlier_threshold": _get_env_var("OUTLIER_THRESHOLD", 3.0, float)
        },
        "sentiment": {
            "model": _get_env_var("SENTIMENT_MODEL", "vader"),
            "batch_size": _get_env_var("SENTIMENT_BATCH_SIZE", 100, int),
            "timeout_seconds": _get_env_var("SENTIMENT_TIMEOUT_SECONDS", 30, int),
            "enable_caching": _get_env_var("ENABLE_SENTIMENT_CACHING", True, bool)
        }
    }


def _get_performance_config() -> Dict[str, Any]:
    """Get performance configuration from environment variables."""
    return {
        "MIN_ACCURACY": _get_env_var("MIN_ACCURACY", 0.75, float),
        "TARGET_ACCURACY": _get_env_var("TARGET_ACCURACY", 0.80, float),
        "MAX_INFERENCE_LATENCY_MS": _get_env_var("MAX_INFERENCE_LATENCY_MS", 10.0, float),
        "MAX_SENTIMENT_PROCESSING_TIME_S": _get_env_var("MAX_SENTIMENT_PROCESSING_TIME_S", 0.01, float),
        "cpu_threads": _get_env_var("CPU_THREADS", 1, int),
        "memory_limit_gb": _get_env_var("MEMORY_LIMIT_GB", 4, int),
        "enable_monitoring": _get_env_var("ENABLE_PERFORMANCE_MONITORING", True, bool),
        "alert_threshold_ms": _get_env_var("PERFORMANCE_ALERT_THRESHOLD", 15, int),
        "accuracy_alert_threshold": _get_env_var("ACCURACY_ALERT_THRESHOLD", 0.75, float)
    }


def _get_logging_config() -> Dict[str, Any]:
    """Get logging configuration from environment variables."""
    return {
        "level": _get_env_var("LOG_LEVEL", "INFO"),
        "format": _get_env_var("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        "date_format": _get_env_var("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S"),
        "file": _get_env_var("LOG_FILE", "pipeline.log"),
        "error_file": _get_env_var("ERROR_LOG_FILE", "error.log"),
        "performance_file": _get_env_var("PERFORMANCE_LOG_FILE", "performance.log"),
        "max_size_mb": _get_env_var("LOG_MAX_SIZE_MB", 100, int),
        "backup_count": _get_env_var("LOG_BACKUP_COUNT", 5, int),
        "rotation_interval": _get_env_var("LOG_ROTATION_INTERVAL", "daily"),
        "enable_json": _get_env_var("ENABLE_JSON_LOGGING", False, bool),
        "correlation_id": _get_env_var("LOG_CORRELATION_ID", True, bool),
        "performance_metrics": _get_env_var("LOG_PERFORMANCE_METRICS", True, bool)
    }


def _get_scheduling_config() -> Dict[str, Any]:
    """Get scheduling configuration from environment variables."""
    return {
        "execution_time": _get_env_var("EXECUTION_TIME", "17:30"),
        "timezone": _get_env_var("TIMEZONE", "Asia/Kolkata"),
        "enabled": _get_env_var("ENABLE_SCHEDULER", True, bool),
        "retry_attempts": _get_env_var("SCHEDULE_RETRY_ATTEMPTS", 3, int),
        "github_actions": {
            "enabled": _get_env_var("GITHUB_ACTIONS_ENABLED", True, bool),
            "cron": _get_env_var("GITHUB_ACTIONS_CRON", "30 12 * * *"),
            "timeout": _get_env_var("GITHUB_ACTIONS_TIMEOUT", 30, int)
        },
        "manual_execution": {
            "allowed": _get_env_var("ALLOW_MANUAL_EXECUTION", True, bool),
            "cooldown_seconds": _get_env_var("MANUAL_EXECUTION_COOLDOWN", 300, int)
        }
    }


def _get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring and alerting configuration from environment variables."""
    return {
        "enabled": _get_env_var("ENABLE_PERFORMANCE_MONITORING", True, bool),
        "health_checks": {
            "enabled": _get_env_var("ENABLE_HEALTH_CHECKS", True, bool),
            "interval_seconds": _get_env_var("HEALTH_CHECK_INTERVAL", 300, int),
            "timeout_seconds": _get_env_var("HEALTH_CHECK_TIMEOUT", 10, int)
        },
        "alerts": {
            "email": _get_env_var("ALERT_EMAIL"),
            "slack_webhook": _get_env_var("SLACK_WEBHOOK_URL"),
            "enable_email": _get_env_var("ENABLE_EMAIL_ALERTS", True, bool),
            "enable_slack": _get_env_var("ENABLE_SLACK_ALERTS", True, bool)
        },
        "metrics": {
            "collection_enabled": _get_env_var("ENABLE_METRICS_COLLECTION", True, bool),
            "retention_days": _get_env_var("METRICS_RETENTION_DAYS", 30, int),
            "export_format": _get_env_var("METRICS_EXPORT_FORMAT", "json")
        }
    }


def _get_security_config() -> Dict[str, Any]:
    """Get security configuration from environment variables."""
    return {
        "api": {
            "rate_limit": _get_env_var("API_RATE_LIMIT", 100, int),
            "rate_limit_window": _get_env_var("API_RATE_LIMIT_WINDOW", 60, int),
            "enable_key_rotation": _get_env_var("ENABLE_API_KEY_ROTATION", False, bool),
            "key_rotation_days": _get_env_var("API_KEY_ROTATION_DAYS", 90, int)
        },
        "data": {
            "encrypt_sensitive": _get_env_var("ENCRYPT_SENSITIVE_DATA", False, bool),
            "encryption_key_path": _get_env_var("ENCRYPTION_KEY_PATH", "keys/encryption.key"),
            "enable_anonymization": _get_env_var("ENABLE_DATA_ANONYMIZATION", False, bool)
        },
        "access": {
            "enable_logging": _get_env_var("ENABLE_ACCESS_LOGGING", True, bool),
            "max_login_attempts": _get_env_var("MAX_LOGIN_ATTEMPTS", 5, int),
            "lockout_duration": _get_env_var("LOGIN_LOCKOUT_DURATION", 300, int)
        }
    }


def _get_development_config() -> Dict[str, Any]:
    """Get development-specific configuration from environment variables."""
    return {
        "debug_mode": _get_env_var("ENABLE_DEBUG_MODE", False, bool),
        "profiling": _get_env_var("ENABLE_PROFILING", False, bool),
        "mock_data": _get_env_var("ENABLE_MOCK_DATA", False, bool),
        "mock_data_path": _get_env_var("MOCK_DATA_PATH", "tests/fixtures/"),
        "testing": {
            "run_on_startup": _get_env_var("RUN_TESTS_ON_STARTUP", False, bool),
            "test_data_path": _get_env_var("TEST_DATA_PATH", "tests/data/"),
            "enable_integration": _get_env_var("ENABLE_INTEGRATION_TESTS", True, bool),
            "timeout": _get_env_var("TEST_TIMEOUT", 300, int)
        },
        "tools": {
            "hot_reload": _get_env_var("ENABLE_HOT_RELOAD", False, bool),
            "debug_toolbar": _get_env_var("ENABLE_DEBUG_TOOLBAR", False, bool),
            "port": _get_env_var("DEVELOPMENT_PORT", 8000, int)
        }
    }


def get_config() -> Dict[str, Any]:
    """
    Get complete configuration dictionary.
    
    Returns:
        Dict containing all configuration settings
    """
    global _config_cache
    
    # Return cached config if available and not in development mode
    if _config_cache and not _get_env_var("ENABLE_DEBUG_MODE", False, bool):
        return _config_cache
    
    # Build configuration
    config = {
        "version": _get_env_var("VERSION", "1.0.0"),
        "environment": _get_env_var("ENVIRONMENT", "development"),
        "debug": _get_env_var("DEBUG", False, bool),
        "api": _get_api_config(),
        "data": _get_data_config(),
        "model": _get_model_config(),
        "performance": _get_performance_config(),
        "logging": _get_logging_config(),
        "scheduling": _get_scheduling_config(),
        "monitoring": _get_monitoring_config(),
        "security": _get_security_config(),
        "development": _get_development_config(),
        "paths": _get_base_paths()
    }
    
    # Cache configuration
    _config_cache = config
    
    return config


def reload_config() -> Dict[str, Any]:
    """
    Reload configuration from environment variables.
    
    Returns:
        Dict containing reloaded configuration
    """
    global _config_cache
    _config_cache = None
    
    # Reload .env file if available
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    
    return get_config()


def validate_config_at_startup() -> None:
    """Validate configuration at application startup."""
    from .validator import validate_config_at_startup
    
    config = get_config()
    validate_config_at_startup(config)


# Backward compatibility
LOG_LEVEL = _get_env_var("LOG_LEVEL", "INFO")
LOG_FORMAT = _get_env_var("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")