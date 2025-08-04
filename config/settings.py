"""
Configuration settings for the NIFTY 50 ML Pipeline.
"""
import os
from typing import Dict, Any

# API Configuration
NSE_API_BASE_URL = "https://www.nseindia.com"
ECONOMIC_TIMES_API_BASE_URL = "https://economictimes.indiatimes.com"

# Data Configuration
DATA_RETENTION_DAYS = 365  # One year rolling window
NEWS_RELEVANCE_DAYS = 30   # 30-day news relevance window
DATA_STORAGE_FORMAT = "parquet"

# Model Configuration
XGBOOST_CONFIG = {
    "n_jobs": 1,
    "tree_method": "exact",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "random_state": 42
}

# Technical Indicators Configuration
TECHNICAL_INDICATORS = {
    "RSI_PERIOD": 14,
    "SMA_PERIOD": 5,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "MIN_ACCURACY": 0.75,
    "TARGET_ACCURACY": 0.80,
    "MAX_INFERENCE_LATENCY_MS": 10.0,
    "MAX_SENTIMENT_PROCESSING_TIME_S": 0.01
}

# Scheduling Configuration
EXECUTION_TIME = "17:30"  # 5:30 PM IST
TIMEZONE = "Asia/Kolkata"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Environment Variables
API_KEYS = {
    "NSE_API_KEY": os.getenv("NSE_API_KEY"),
    "ECONOMIC_TIMES_API_KEY": os.getenv("ECONOMIC_TIMES_API_KEY")
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "api": {
            "nse_base_url": NSE_API_BASE_URL,
            "et_base_url": ECONOMIC_TIMES_API_BASE_URL,
            "keys": API_KEYS
        },
        "data": {
            "retention_days": DATA_RETENTION_DAYS,
            "news_relevance_days": NEWS_RELEVANCE_DAYS,
            "storage_format": DATA_STORAGE_FORMAT
        },
        "model": {
            "xgboost": XGBOOST_CONFIG,
            "technical_indicators": TECHNICAL_INDICATORS
        },
        "performance": PERFORMANCE_THRESHOLDS,
        "scheduling": {
            "execution_time": EXECUTION_TIME,
            "timezone": TIMEZONE
        },
        "logging": {
            "level": LOG_LEVEL,
            "format": LOG_FORMAT
        },
        "paths": {
            "base": BASE_DIR,
            "data": DATA_DIR,
            "models": MODELS_DIR,
            "logs": LOGS_DIR
        }
    }