"""
Production deployment configuration for NIFTY 50 ML Pipeline.

This module provides production-specific settings with security hardening,
performance optimization, and monitoring configuration.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Production environment settings
ENVIRONMENT = "production"
DEBUG = False
VERSION = "1.0.0"

# Security settings
SECURITY_CONFIG = {
    "api": {
        "rate_limit": int(os.getenv("API_RATE_LIMIT", "50")),  # Reduced for production
        "rate_limit_window": int(os.getenv("API_RATE_LIMIT_WINDOW", "60")),
        "enable_key_rotation": True,
        "key_rotation_days": int(os.getenv("API_KEY_ROTATION_DAYS", "30")),
        "timeout_seconds": int(os.getenv("API_TIMEOUT", "30")),
        "max_retries": int(os.getenv("API_MAX_RETRIES", "3")),
        "backoff_factor": float(os.getenv("API_BACKOFF_FACTOR", "2.0"))
    },
    "data": {
        "encrypt_sensitive": True,
        "encryption_key_path": os.getenv("ENCRYPTION_KEY_PATH", "/secure/keys/encryption.key"),
        "enable_anonymization": False,  # Not needed for financial data
        "data_retention_days": int(os.getenv("DATA_RETENTION_DAYS", "365")),
        "backup_retention_days": int(os.getenv("BACKUP_RETENTION_DAYS", "90"))
    },
    "access": {
        "enable_logging": True,
        "max_login_attempts": int(os.getenv("MAX_LOGIN_ATTEMPTS", "3")),
        "lockout_duration": int(os.getenv("LOGIN_LOCKOUT_DURATION", "900")),  # 15 minutes
        "session_timeout": int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour
    },
    "ssl": {
        "verify_certificates": True,
        "min_tls_version": "1.2",
        "cipher_suites": "HIGH:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA"
    }
}

# Performance optimization settings
PERFORMANCE_CONFIG = {
    "cpu": {
        "max_threads": int(os.getenv("MAX_CPU_THREADS", "2")),  # Conservative for production
        "thread_pool_size": int(os.getenv("THREAD_POOL_SIZE", "4")),
        "enable_cpu_affinity": bool(os.getenv("ENABLE_CPU_AFFINITY", "false").lower() == "true")
    },
    "memory": {
        "max_memory_gb": int(os.getenv("MAX_MEMORY_GB", "4")),
        "gc_threshold": int(os.getenv("GC_THRESHOLD", "1000")),
        "enable_memory_profiling": bool(os.getenv("ENABLE_MEMORY_PROFILING", "true").lower() == "true"),
        "memory_alert_threshold": float(os.getenv("MEMORY_ALERT_THRESHOLD", "0.85"))
    },
    "cache": {
        "enable_caching": True,
        "cache_size_mb": int(os.getenv("CACHE_SIZE_MB", "256")),
        "cache_ttl_hours": int(os.getenv("CACHE_TTL_HOURS", "6")),
        "cache_cleanup_interval": int(os.getenv("CACHE_CLEANUP_INTERVAL", "3600"))
    },
    "targets": {
        "max_inference_latency_ms": float(os.getenv("MAX_INFERENCE_LATENCY_MS", "10.0")),
        "target_accuracy": float(os.getenv("TARGET_ACCURACY", "0.80")),
        "min_accuracy_threshold": float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.75")),
        "max_error_rate": float(os.getenv("MAX_ERROR_RATE", "0.05"))
    }
}

# Monitoring and alerting configuration
MONITORING_CONFIG = {
    "enabled": True,
    "metrics": {
        "collection_interval": int(os.getenv("METRICS_COLLECTION_INTERVAL", "60")),  # seconds
        "retention_days": int(os.getenv("METRICS_RETENTION_DAYS", "30")),
        "export_format": os.getenv("METRICS_EXPORT_FORMAT", "json"),
        "export_path": os.getenv("METRICS_EXPORT_PATH", "/var/log/nifty-pipeline/metrics/")
    },
    "health_checks": {
        "enabled": True,
        "interval_seconds": int(os.getenv("HEALTH_CHECK_INTERVAL", "300")),  # 5 minutes
        "timeout_seconds": int(os.getenv("HEALTH_CHECK_TIMEOUT", "30")),
        "failure_threshold": int(os.getenv("HEALTH_CHECK_FAILURE_THRESHOLD", "3"))
    },
    "alerts": {
        "email": {
            "enabled": bool(os.getenv("ENABLE_EMAIL_ALERTS", "true").lower() == "true"),
            "smtp_server": os.getenv("SMTP_SERVER", "localhost"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "smtp_username": os.getenv("SMTP_USERNAME"),
            "smtp_password": os.getenv("SMTP_PASSWORD"),
            "from_email": os.getenv("ALERT_FROM_EMAIL", "nifty-pipeline@company.com"),
            "to_emails": os.getenv("ALERT_TO_EMAILS", "").split(",") if os.getenv("ALERT_TO_EMAILS") else [],
            "subject_prefix": "[NIFTY-PIPELINE-PROD]"
        },
        "slack": {
            "enabled": bool(os.getenv("ENABLE_SLACK_ALERTS", "false").lower() == "true"),
            "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
            "channel": os.getenv("SLACK_CHANNEL", "#alerts"),
            "username": os.getenv("SLACK_USERNAME", "NIFTY Pipeline")
        },
        "thresholds": {
            "latency_ms": float(os.getenv("ALERT_LATENCY_THRESHOLD", "15.0")),
            "accuracy": float(os.getenv("ALERT_ACCURACY_THRESHOLD", "0.75")),
            "error_rate": float(os.getenv("ALERT_ERROR_RATE_THRESHOLD", "0.10")),
            "memory_percent": float(os.getenv("ALERT_MEMORY_THRESHOLD", "85.0")),
            "cpu_percent": float(os.getenv("ALERT_CPU_THRESHOLD", "80.0"))
        }
    }
}

# Logging configuration for production
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": os.getenv("LOG_FILE", "/var/log/nifty-pipeline/pipeline.log"),
            "maxBytes": int(os.getenv("LOG_MAX_SIZE_BYTES", "104857600")),  # 100MB
            "backupCount": int(os.getenv("LOG_BACKUP_COUNT", "10"))
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "json",
            "filename": os.getenv("ERROR_LOG_FILE", "/var/log/nifty-pipeline/error.log"),
            "maxBytes": int(os.getenv("LOG_MAX_SIZE_BYTES", "104857600")),
            "backupCount": int(os.getenv("LOG_BACKUP_COUNT", "10"))
        },
        "performance_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": os.getenv("PERFORMANCE_LOG_FILE", "/var/log/nifty-pipeline/performance.log"),
            "maxBytes": int(os.getenv("LOG_MAX_SIZE_BYTES", "104857600")),
            "backupCount": int(os.getenv("LOG_BACKUP_COUNT", "10"))
        }
    },
    "loggers": {
        "nifty_ml_pipeline": {
            "level": "INFO",
            "handlers": ["console", "file", "error_file"],
            "propagate": False
        },
        "performance": {
            "level": "INFO",
            "handlers": ["performance_file"],
            "propagate": False
        },
        "security": {
            "level": "WARNING",
            "handlers": ["console", "file", "error_file"],
            "propagate": False
        }
    },
    "root": {
        "level": "WARNING",
        "handlers": ["console", "file"]
    }
}

# Data storage configuration
STORAGE_CONFIG = {
    "data_path": os.getenv("DATA_PATH", "/var/lib/nifty-pipeline/data/"),
    "models_path": os.getenv("MODELS_PATH", "/var/lib/nifty-pipeline/models/"),
    "cache_path": os.getenv("CACHE_PATH", "/var/cache/nifty-pipeline/"),
    "logs_path": os.getenv("LOGS_PATH", "/var/log/nifty-pipeline/"),
    "backup_path": os.getenv("BACKUP_PATH", "/var/backups/nifty-pipeline/"),
    "format": "parquet",
    "compression": "snappy",
    "partitioning": {
        "enabled": True,
        "partition_by": ["symbol", "date"],
        "partition_size_mb": int(os.getenv("PARTITION_SIZE_MB", "100"))
    },
    "backup": {
        "enabled": True,
        "schedule": os.getenv("BACKUP_SCHEDULE", "0 2 * * *"),  # Daily at 2 AM
        "retention_days": int(os.getenv("BACKUP_RETENTION_DAYS", "90")),
        "compression": True
    }
}

# Model configuration
MODEL_CONFIG = {
    "xgboost": {
        "n_jobs": 1,  # Single-threaded for consistent performance
        "tree_method": "exact",
        "max_depth": int(os.getenv("XGBOOST_MAX_DEPTH", "6")),
        "learning_rate": float(os.getenv("XGBOOST_LEARNING_RATE", "0.1")),
        "n_estimators": int(os.getenv("XGBOOST_N_ESTIMATORS", "100")),
        "subsample": float(os.getenv("XGBOOST_SUBSAMPLE", "0.8")),
        "colsample_bytree": float(os.getenv("XGBOOST_COLSAMPLE_BYTREE", "0.8")),
        "random_state": 42,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "early_stopping_rounds": int(os.getenv("XGBOOST_EARLY_STOPPING", "10")),
        "verbosity": 0  # Quiet for production
    },
    "validation": {
        "method": "time_series_split",
        "n_splits": int(os.getenv("VALIDATION_SPLITS", "5")),
        "test_size": float(os.getenv("VALIDATION_TEST_SIZE", "0.2")),
        "gap": int(os.getenv("VALIDATION_GAP", "1"))  # Days between train/test
    },
    "retraining": {
        "enabled": True,
        "schedule": os.getenv("RETRAIN_SCHEDULE", "0 1 * * 0"),  # Weekly on Sunday at 1 AM
        "trigger_accuracy_threshold": float(os.getenv("RETRAIN_ACCURACY_THRESHOLD", "0.70")),
        "min_data_points": int(os.getenv("MIN_RETRAIN_DATA_POINTS", "1000"))
    }
}

# Scheduling configuration
SCHEDULING_CONFIG = {
    "execution_time": os.getenv("EXECUTION_TIME", "17:30"),
    "timezone": os.getenv("TIMEZONE", "Asia/Kolkata"),
    "enabled": True,
    "retry_attempts": int(os.getenv("SCHEDULE_RETRY_ATTEMPTS", "3")),
    "retry_delay_minutes": int(os.getenv("SCHEDULE_RETRY_DELAY", "5")),
    "timeout_minutes": int(os.getenv("PIPELINE_TIMEOUT_MINUTES", "30")),
    "github_actions": {
        "enabled": bool(os.getenv("GITHUB_ACTIONS_ENABLED", "true").lower() == "true"),
        "cron": os.getenv("GITHUB_ACTIONS_CRON", "30 12 * * *"),  # 5:30 PM IST = 12:00 PM UTC
        "timeout": int(os.getenv("GITHUB_ACTIONS_TIMEOUT", "30"))
    }
}

# API configuration
API_CONFIG = {
    "nse": {
        "base_url": os.getenv("NSE_API_BASE_URL", "https://www.nseindia.com"),
        "timeout": int(os.getenv("NSE_API_TIMEOUT", "30")),
        "max_retries": int(os.getenv("NSE_MAX_RETRIES", "3")),
        "backoff_factor": float(os.getenv("NSE_BACKOFF_FACTOR", "2.0")),
        "rate_limit": int(os.getenv("NSE_RATE_LIMIT", "10")),  # requests per minute
        "user_agent": os.getenv("NSE_USER_AGENT", "NIFTY-ML-Pipeline/1.0")
    },
    "economic_times": {
        "base_url": os.getenv("ECONOMIC_TIMES_API_BASE_URL", "https://economictimes.indiatimes.com"),
        "api_key": os.getenv("ECONOMIC_TIMES_API_KEY"),
        "timeout": int(os.getenv("ECONOMIC_TIMES_TIMEOUT", "30")),
        "max_retries": int(os.getenv("ECONOMIC_TIMES_MAX_RETRIES", "3")),
        "rate_limit": int(os.getenv("ECONOMIC_TIMES_RATE_LIMIT", "20"))
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "technical_indicators": {
        "rsi_period": int(os.getenv("RSI_PERIOD", "14")),
        "sma_period": int(os.getenv("SMA_PERIOD", "5")),
        "macd_fast": int(os.getenv("MACD_FAST", "12")),
        "macd_slow": int(os.getenv("MACD_SLOW", "26")),
        "macd_signal": int(os.getenv("MACD_SIGNAL", "9"))
    },
    "sentiment": {
        "model": "vader",
        "batch_size": int(os.getenv("SENTIMENT_BATCH_SIZE", "50")),  # Smaller for production
        "timeout_seconds": int(os.getenv("SENTIMENT_TIMEOUT", "30")),
        "cache_enabled": True,
        "cache_ttl_hours": int(os.getenv("SENTIMENT_CACHE_TTL", "24"))
    },
    "normalization": {
        "method": os.getenv("NORMALIZATION_METHOD", "standard"),
        "handle_outliers": os.getenv("HANDLE_OUTLIERS", "clip"),
        "outlier_threshold": float(os.getenv("OUTLIER_THRESHOLD", "3.0"))
    }
}


def get_production_config() -> Dict[str, Any]:
    """Get complete production configuration.
    
    Returns:
        Dictionary containing all production settings
    """
    return {
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "version": VERSION,
        "security": SECURITY_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "logging": LOGGING_CONFIG,
        "storage": STORAGE_CONFIG,
        "model": MODEL_CONFIG,
        "scheduling": SCHEDULING_CONFIG,
        "api": API_CONFIG,
        "features": FEATURE_CONFIG
    }


def validate_production_config() -> None:
    """Validate production configuration for required settings."""
    required_env_vars = [
        "ECONOMIC_TIMES_API_KEY",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
        "ALERT_TO_EMAILS"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Validate paths exist or can be created
    paths_to_check = [
        STORAGE_CONFIG["data_path"],
        STORAGE_CONFIG["models_path"],
        STORAGE_CONFIG["cache_path"],
        STORAGE_CONFIG["logs_path"],
        STORAGE_CONFIG["backup_path"]
    ]
    
    for path_str in paths_to_check:
        path = Path(path_str)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ValueError(f"Cannot create directory: {path}")
    
    # Validate numeric ranges
    if not 0.5 <= PERFORMANCE_CONFIG["targets"]["min_accuracy_threshold"] <= 1.0:
        raise ValueError("min_accuracy_threshold must be between 0.5 and 1.0")
    
    if not 1.0 <= PERFORMANCE_CONFIG["targets"]["max_inference_latency_ms"] <= 100.0:
        raise ValueError("max_inference_latency_ms must be between 1.0 and 100.0")