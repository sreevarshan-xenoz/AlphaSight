"""
Secure production configuration for NIFTY 50 ML Pipeline.

This configuration prioritizes security and performance for production deployment.
All sensitive values must be provided via environment variables.
"""

import os
from pathlib import Path

# Validate required environment variables at startup
REQUIRED_ENV_VARS = [
    'ECONOMIC_TIMES_API_KEY',
    'SMTP_USERNAME', 
    'SMTP_PASSWORD',
    'ALERT_TO_EMAILS',
    'ENCRYPTION_KEY_PATH'
]

def validate_environment():
    """Validate all required environment variables are set."""
    missing = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Production settings with security hardening
PRODUCTION_CONFIG = {
    "environment": "production",
    "debug": False,
    "version": os.getenv("VERSION", "1.0.0"),
    
    # Security settings
    "security": {
        "api": {
            "rate_limit": int(os.getenv("API_RATE_LIMIT", "30")),
            "timeout_seconds": int(os.getenv("API_TIMEOUT", "30")),
            "max_retries": int(os.getenv("API_MAX_RETRIES", "3")),
            "enable_ssl_verification": True,
            "min_tls_version": "1.2"
        },
        "data": {
            "encrypt_sensitive": True,
            "encryption_key_path": os.getenv("ENCRYPTION_KEY_PATH"),
            "data_retention_days": int(os.getenv("DATA_RETENTION_DAYS", "365"))
        },
        "access": {
            "enable_audit_logging": True,
            "max_login_attempts": int(os.getenv("MAX_LOGIN_ATTEMPTS", "3")),
            "session_timeout": int(os.getenv("SESSION_TIMEOUT", "3600"))
        }
    },
    
    # Performance optimization
    "performance": {
        "max_inference_latency_ms": float(os.getenv("MAX_INFERENCE_LATENCY_MS", "10.0")),
        "target_accuracy": float(os.getenv("TARGET_ACCURACY", "0.80")),
        "min_accuracy_threshold": float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.75")),
        "cpu_threads": int(os.getenv("CPU_THREADS", "1")),
        "memory_limit_gb": int(os.getenv("MEMORY_LIMIT_GB", "4")),
        "enable_caching": True,
        "cache_size_mb": int(os.getenv("CACHE_SIZE_MB", "256"))
    },
    
    # Monitoring and alerting
    "monitoring": {
        "enabled": True,
        "metrics_retention_days": int(os.getenv("METRICS_RETENTION_DAYS", "30")),
        "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "300")),
        "alerts": {
            "email": {
                "enabled": True,
                "smtp_server": os.getenv("SMTP_SERVER", "localhost"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "smtp_username": os.getenv("SMTP_USERNAME"),
                "smtp_password": os.getenv("SMTP_PASSWORD"),
                "from_email": os.getenv("ALERT_FROM_EMAIL"),
                "to_emails": os.getenv("ALERT_TO_EMAILS", "").split(",")
            }
        }
    },
    
    # Logging configuration
    "logging": {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": os.getenv("LOG_FILE", "/var/log/nifty-pipeline/pipeline.log"),
        "max_size_mb": int(os.getenv("LOG_MAX_SIZE_MB", "100")),
        "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5")),
        "enable_json": True,
        "enable_audit": True
    },
    
    # Data storage
    "storage": {
        "data_path": os.getenv("DATA_PATH", "/var/lib/nifty-pipeline/data/"),
        "models_path": os.getenv("MODELS_PATH", "/var/lib/nifty-pipeline/models/"),
        "backup_path": os.getenv("BACKUP_PATH", "/var/backups/nifty-pipeline/"),
        "format": "parquet",
        "compression": "snappy",
        "enable_encryption": True
    },
    
    # API configuration
    "api": {
        "economic_times": {
            "api_key": os.getenv("ECONOMIC_TIMES_API_KEY"),
            "timeout": int(os.getenv("ECONOMIC_TIMES_TIMEOUT", "30")),
            "rate_limit": int(os.getenv("ECONOMIC_TIMES_RATE_LIMIT", "20"))
        }
    }
}

def get_production_config():
    """Get production configuration with validation."""
    validate_environment()
    return PRODUCTION_CONFIG
