# Configuration Options Reference

## Overview

This document provides a comprehensive reference for all configuration options available in the NIFTY 50 ML Pipeline.

## Environment Variables

### Core Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENVIRONMENT` | string | `development` | Deployment environment (development, staging, production) |
| `DEBUG` | boolean | `false` | Enable debug mode |
| `VERSION` | string | `1.0.0` | Application version |

### API Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ECONOMIC_TIMES_API_KEY` | string | `None` | API key for Economic Times |
| `ECONOMIC_TIMES_API_BASE_URL` | string | `https://economictimes.indiatimes.com` | Base URL for Economic Times API |
| `ECONOMIC_TIMES_TIMEOUT` | integer | `30` | API timeout in seconds |
| `ECONOMIC_TIMES_MAX_RETRIES` | integer | `3` | Maximum retry attempts |
| `NSE_API_KEY` | string | `None` | API key for NSE (if required) |
| `NSE_API_BASE_URL` | string | `https://www.nseindia.com` | Base URL for NSE API |
| `NSE_API_TIMEOUT` | integer | `30` | API timeout in seconds |
| `NSE_MAX_RETRIES` | integer | `3` | Maximum retry attempts |
| `API_BASE_DELAY` | float | `1.0` | Base delay for exponential backoff |

### Data Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATA_RETENTION_DAYS` | integer | `365` | Rolling window size in days |
| `NEWS_RELEVANCE_DAYS` | integer | `30` | News relevance window in days |
| `DATA_PATH` | string | `data` | Base data directory |
| `STORAGE_FORMAT` | string | `parquet` | Data storage format (parquet, csv, json) |
| `COMPRESSION` | string | `snappy` | Compression algorithm (snappy, gzip, lz4) |
| `ENABLE_CACHING` | boolean | `true` | Enable data caching |
| `CACHE_SIZE_MB` | integer | `500` | Maximum cache size in MB |
| `CACHE_TTL_HOURS` | integer | `24` | Cache time-to-live in hours |
| `ENABLE_DATA_VALIDATION` | boolean | `true` | Enable data validation |
| `MIN_DATA_QUALITY_SCORE` | float | `0.8` | Minimum data quality threshold |
| `HANDLE_MISSING_DATA` | string | `interpolate` | Missing data strategy (interpolate, drop, fill) |

### Model Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_PATH` | string | `models/xgboost_model.pkl` | Path to trained model |
| `MODEL_BACKUP_PATH` | string | `models/backup/` | Model backup directory |
| `FEATURE_SCALER_PATH` | string | `models/feature_scaler.pkl` | Feature scaler path |

#### XGBoost Parameters

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `XGBOOST_N_JOBS` | integer | `1` | Number of CPU threads (1 for optimization) |
| `XGBOOST_TREE_METHOD` | string | `exact` | Tree construction method (exact, approx, hist) |
| `XGBOOST_MAX_DEPTH` | integer | `6` | Maximum tree depth |
| `XGBOOST_N_ESTIMATORS` | integer | `100` | Number of trees |
| `XGBOOST_LEARNING_RATE` | float | `0.1` | Learning rate |
| `XGBOOST_SUBSAMPLE` | float | `0.8` | Row sampling ratio |
| `XGBOOST_COLSAMPLE_BYTREE` | float | `0.8` | Column sampling ratio |
| `XGBOOST_RANDOM_STATE` | integer | `42` | Random seed |
| `XGBOOST_OBJECTIVE` | string | `reg:squarederror` | Objective function |
| `XGBOOST_EVAL_METRIC` | string | `rmse` | Evaluation metric |

#### Technical Indicators

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RSI_PERIOD` | integer | `14` | RSI calculation period |
| `SMA_PERIOD` | integer | `5` | Simple moving average period |
| `MACD_FAST` | integer | `12` | MACD fast EMA period |
| `MACD_SLOW` | integer | `26` | MACD slow EMA period |
| `MACD_SIGNAL` | integer | `9` | MACD signal line period |

#### Feature Engineering

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NORMALIZATION_METHOD` | string | `standard` | Normalization method (standard, minmax, robust) |
| `HANDLE_OUTLIERS` | string | `clip` | Outlier handling (clip, remove, transform) |
| `OUTLIER_THRESHOLD` | float | `3.0` | Outlier threshold in standard deviations |

#### Sentiment Analysis

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SENTIMENT_MODEL` | string | `vader` | Sentiment analysis model (vader, textblob, custom) |
| `SENTIMENT_BATCH_SIZE` | integer | `100` | Batch processing size |
| `SENTIMENT_TIMEOUT_SECONDS` | integer | `30` | Processing timeout |
| `ENABLE_SENTIMENT_CACHING` | boolean | `true` | Cache sentiment scores |

### Performance Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_INFERENCE_LATENCY_MS` | float | `10.0` | Maximum inference latency in milliseconds |
| `MIN_CONFIDENCE_THRESHOLD` | float | `0.7` | Minimum prediction confidence |
| `TARGET_ACCURACY` | float | `0.8` | Target directional accuracy |
| `MIN_ACCURACY` | float | `0.75` | Minimum acceptable accuracy |
| `MAX_SENTIMENT_PROCESSING_TIME_S` | float | `0.01` | Maximum sentiment processing time |
| `CPU_THREADS` | integer | `1` | Number of CPU threads |
| `MEMORY_LIMIT_GB` | integer | `4` | Maximum memory usage in GB |
| `ENABLE_PERFORMANCE_MONITORING` | boolean | `true` | Enable performance monitoring |
| `PERFORMANCE_ALERT_THRESHOLD` | integer | `15` | Performance alert threshold in ms |
| `ACCURACY_ALERT_THRESHOLD` | float | `0.75` | Accuracy alert threshold |

### Logging Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOG_LEVEL` | string | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FORMAT` | string | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Log format string |
| `LOG_DATE_FORMAT` | string | `%Y-%m-%d %H:%M:%S` | Date format for logs |
| `LOG_FILE` | string | `pipeline.log` | Main log file |
| `ERROR_LOG_FILE` | string | `error.log` | Error-specific log file |
| `PERFORMANCE_LOG_FILE` | string | `performance.log` | Performance metrics log |
| `LOG_MAX_SIZE_MB` | integer | `100` | Maximum log file size in MB |
| `LOG_BACKUP_COUNT` | integer | `5` | Number of backup log files |
| `LOG_ROTATION_INTERVAL` | string | `daily` | Log rotation interval (daily, weekly, monthly) |
| `ENABLE_JSON_LOGGING` | boolean | `false` | Enable JSON log format |
| `LOG_CORRELATION_ID` | boolean | `true` | Include correlation IDs in logs |
| `LOG_PERFORMANCE_METRICS` | boolean | `true` | Log performance metrics |

### Scheduling Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EXECUTION_TIME` | string | `17:30` | Daily execution time (HH:MM) |
| `TIMEZONE` | string | `Asia/Kolkata` | Timezone for scheduling |
| `ENABLE_SCHEDULER` | boolean | `true` | Enable automatic scheduling |
| `SCHEDULE_RETRY_ATTEMPTS` | integer | `3` | Retry attempts for failed runs |
| `GITHUB_ACTIONS_ENABLED` | boolean | `true` | Enable GitHub Actions scheduling |
| `GITHUB_ACTIONS_CRON` | string | `30 12 * * *` | Cron expression for GitHub Actions |
| `GITHUB_ACTIONS_TIMEOUT` | integer | `30` | GitHub Actions timeout in minutes |
| `ALLOW_MANUAL_EXECUTION` | boolean | `true` | Allow manual pipeline runs |
| `MANUAL_EXECUTION_COOLDOWN` | integer | `300` | Cooldown between manual runs in seconds |

### Monitoring and Alerting

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_HEALTH_CHECKS` | boolean | `true` | Enable health monitoring |
| `HEALTH_CHECK_INTERVAL` | integer | `300` | Health check interval in seconds |
| `HEALTH_CHECK_TIMEOUT` | integer | `10` | Health check timeout in seconds |
| `ALERT_EMAIL` | string | `None` | Email address for alerts |
| `SLACK_WEBHOOK_URL` | string | `None` | Slack webhook URL for alerts |
| `ENABLE_EMAIL_ALERTS` | boolean | `true` | Enable email notifications |
| `ENABLE_SLACK_ALERTS` | boolean | `true` | Enable Slack notifications |
| `ENABLE_METRICS_COLLECTION` | boolean | `true` | Collect system metrics |
| `METRICS_RETENTION_DAYS` | integer | `30` | Metrics retention period |
| `METRICS_EXPORT_FORMAT` | string | `json` | Metrics export format (json, csv, prometheus) |

### Security Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_RATE_LIMIT` | integer | `100` | API requests per minute |
| `API_RATE_LIMIT_WINDOW` | integer | `60` | Rate limit window in seconds |
| `ENABLE_API_KEY_ROTATION` | boolean | `false` | Enable automatic API key rotation |
| `API_KEY_ROTATION_DAYS` | integer | `90` | API key rotation interval |
| `ENCRYPT_SENSITIVE_DATA` | boolean | `false` | Encrypt sensitive data at rest |
| `ENCRYPTION_KEY_PATH` | string | `keys/encryption.key` | Path to encryption key |
| `ENABLE_DATA_ANONYMIZATION` | boolean | `false` | Anonymize sensitive data |
| `ENABLE_ACCESS_LOGGING` | boolean | `true` | Log access attempts |
| `MAX_LOGIN_ATTEMPTS` | integer | `5` | Maximum failed login attempts |
| `LOGIN_LOCKOUT_DURATION` | integer | `300` | Lockout duration in seconds |

### Development Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENABLE_DEBUG_MODE` | boolean | `false` | Enable debug features |
| `ENABLE_PROFILING` | boolean | `false` | Enable performance profiling |
| `ENABLE_MOCK_DATA` | boolean | `false` | Use mock data for testing |
| `MOCK_DATA_PATH` | string | `tests/fixtures/` | Path to mock data |
| `RUN_TESTS_ON_STARTUP` | boolean | `false` | Run tests before execution |
| `TEST_DATA_PATH` | string | `tests/data/` | Path to test data |
| `ENABLE_INTEGRATION_TESTS` | boolean | `true` | Enable integration tests |
| `TEST_TIMEOUT` | integer | `300` | Test timeout in seconds |
| `ENABLE_HOT_RELOAD` | boolean | `false` | Enable hot reloading |
| `ENABLE_DEBUG_TOOLBAR` | boolean | `false` | Enable debug toolbar |
| `DEVELOPMENT_PORT` | integer | `8000` | Development server port |

## Configuration Validation Rules

### Required Settings

The following settings are required and must be present:

- `performance.MAX_INFERENCE_LATENCY_MS`
- `model.xgboost.n_jobs`
- `model.xgboost.tree_method`
- `data.retention_days`
- `logging.level`

### Validation Rules

1. **Performance Settings**:
   - `MAX_INFERENCE_LATENCY_MS` must be > 0
   - `MIN_ACCURACY` and `TARGET_ACCURACY` must be between 0.0 and 1.0
   - `TARGET_ACCURACY` must be >= `MIN_ACCURACY`

2. **Model Settings**:
   - `n_jobs` should be 1 for CPU optimization
   - `tree_method` should be 'exact' for CPU optimization
   - `max_depth` must be > 0
   - `learning_rate` must be between 0.0 and 1.0
   - `n_estimators` must be > 0

3. **Data Settings**:
   - `retention_days` must be > 0
   - `news_relevance_days` must be > 0
   - `storage_format` must be one of: parquet, csv, json

4. **Technical Indicators**:
   - All periods must be > 0
   - `MACD_FAST` must be < `MACD_SLOW`

5. **Logging Settings**:
   - `level` must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
   - Log format must be a valid Python logging format string

## Environment-Specific Defaults

### Development Environment

- `DEBUG=true`
- `LOG_LEVEL=DEBUG`
- `MAX_INFERENCE_LATENCY_MS=50` (relaxed)
- `ENABLE_PROFILING=true`
- `ENABLE_MOCK_DATA=true`
- `ENABLE_SCHEDULER=false`

### Staging Environment

- `DEBUG=false`
- `LOG_LEVEL=INFO`
- `MAX_INFERENCE_LATENCY_MS=15` (slightly relaxed)
- `ENABLE_PROFILING=false`
- `ENABLE_MOCK_DATA=false`
- `ENABLE_SCHEDULER=true`

### Production Environment

- `DEBUG=false`
- `LOG_LEVEL=INFO`
- `MAX_INFERENCE_LATENCY_MS=10` (strict)
- `ENABLE_PROFILING=false`
- `ENABLE_MOCK_DATA=false`
- `ENABLE_SCHEDULER=true`
- `ENABLE_PERFORMANCE_MONITORING=true`

## Configuration Best Practices

1. **Use Environment Variables**: Store sensitive data in environment variables, not in code
2. **Validate at Startup**: Always validate configuration before starting the application
3. **Environment-Specific Files**: Use separate configuration files for different environments
4. **Document Changes**: Document any configuration changes and their impact
5. **Test Configuration**: Test configuration changes in staging before production
6. **Monitor Performance**: Monitor the impact of configuration changes on performance
7. **Backup Configuration**: Keep backups of working configurations
8. **Version Control**: Version control configuration files (except sensitive data)

## Troubleshooting Configuration Issues

### Common Issues

1. **Invalid Environment Variables**: Check data types and valid values
2. **Missing Required Settings**: Ensure all required settings are present
3. **File Permissions**: Check read/write permissions for data directories
4. **API Keys**: Verify API keys are valid and not expired
5. **Performance Issues**: Check CPU and memory settings
6. **Logging Issues**: Verify log file paths and permissions

### Debugging Configuration

Use the configuration test script to debug issues:

```bash
python scripts/test_config.py
```

This will run comprehensive tests on your configuration and report any issues.