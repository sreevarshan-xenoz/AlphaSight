# Troubleshooting Guide

## Common Issues and Solutions

### Data Collection Issues

#### Issue: NSE API Connection Failures
**Symptoms:**
- Connection timeout errors
- HTTP 429 (Too Many Requests) errors
- Empty data responses

**Solutions:**
```bash
# Check network connectivity
ping nseindia.com

# Verify API timeout settings
export NSE_API_TIMEOUT=60

# Check rate limiting
grep "rate limit" pipeline.log
```

**Configuration Fix:**
```python
# In .env file
NSE_API_TIMEOUT=60
NSE_MAX_RETRIES=5
NSE_BASE_DELAY=2.0
```

#### Issue: News Data Collection Failures
**Symptoms:**
- Empty news DataFrames
- RSS feed parsing errors
- Stale news data warnings

**Solutions:**
```bash
# Test RSS feed accessibility
curl -I https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms

# Check news data freshness
python -c "
from nifty_ml_pipeline.data.collectors import NewsDataCollector
collector = NewsDataCollector()
# Test collection logic
"
```

### Feature Engineering Issues

#### Issue: Technical Indicator Calculation Errors
**Symptoms:**
- NaN values in RSI/SMA/MACD columns
- "Insufficient data" warnings
- Calculation timeout errors

**Solutions:**
```python
# Check data sufficiency
import pandas as pd
from nifty_ml_pipeline.features.technical_indicators import TechnicalIndicatorCalculator

calculator = TechnicalIndicatorCalculator()
# Ensure minimum 26 days of data for MACD
if len(price_data) < 26:
    print("Insufficient data for MACD calculation")
```

#### Issue: Sentiment Analysis Failures
**Symptoms:**
- VADER import errors
- Sentiment scores all zero
- Processing timeout errors

**Solutions:**
```bash
# Reinstall NLTK data
python -c "
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
"

# Test sentiment analyzer
python -c "
from nifty_ml_pipeline.features.sentiment_analysis import SentimentAnalyzer
analyzer = SentimentAnalyzer()
score = analyzer.analyze_headline('NIFTY surges on positive sentiment')
print(f'Test sentiment: {score}')
"
```

### Model Training and Inference Issues

#### Issue: XGBoost Training Failures
**Symptoms:**
- Memory allocation errors
- Training timeout
- Model convergence issues

**Solutions:**
```python
# Reduce model complexity
hyperparameters = {
    'max_depth': 4,  # Reduce from 6
    'n_estimators': 50,  # Reduce from 100
    'learning_rate': 0.05  # Reduce learning rate
}

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

#### Issue: Inference Latency > 10ms
**Symptoms:**
- Latency warnings in logs
- Performance degradation alerts
- Timeout errors

**Solutions:**
```python
# Profile inference performance
import time
start = time.perf_counter()
prediction = model.predict(features)
latency = (time.perf_counter() - start) * 1000
print(f"Inference latency: {latency:.2f}ms")

# Optimize model parameters
model_params = {
    'n_jobs': 1,  # Single thread
    'tree_method': 'exact',  # CPU optimized
    'max_depth': 5  # Reduce complexity
}
```

### Pipeline Orchestration Issues

#### Issue: Pipeline Stage Failures
**Symptoms:**
- Stage timeout errors
- Partial pipeline completion
- Error handler activation

**Solutions:**
```bash
# Check stage execution logs
grep "stage.*failed" pipeline.log

# Test individual stages
python -c "
from nifty_ml_pipeline.orchestration.controller import PipelineController
from config.settings import get_config

controller = PipelineController(get_config())
# Test data collection stage only
price_data, news_data = controller._execute_data_collection_stage('NIFTY 50')
print(f'Data collection: {len(price_data)} price records, {len(news_data)} news records')
"
```

#### Issue: Scheduler Not Running
**Symptoms:**
- No scheduled executions
- GitHub Actions failures
- Cron job not triggering

**Solutions:**
```bash
# Check GitHub Actions status
# Go to repository -> Actions tab

# Test local scheduling
python -c "
from nifty_ml_pipeline.orchestration.scheduler import TaskScheduler
from config.settings import get_config

scheduler = TaskScheduler(get_config())
print(f'Next execution: {scheduler.get_next_execution_time()}')
"

# Verify cron syntax
# 30 12 * * * = 12:30 UTC daily (5:30 PM IST)
```

### Performance Issues

#### Issue: High Memory Usage
**Symptoms:**
- Out of memory errors
- System slowdown
- Process killed by OS

**Solutions:**
```bash
# Monitor memory usage
top -p $(pgrep -f "nifty_ml_pipeline")

# Check for memory leaks
python -c "
import gc
import psutil
import os

process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
gc.collect()
print(f'After GC: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

**Configuration Fix:**
```python
# In config/settings.py
'performance': {
    'memory_limit_gb': 2,  # Reduce from 4
    'batch_size': 100,     # Process in smaller batches
    'gc_frequency': 10     # More frequent garbage collection
}
```

#### Issue: CPU Utilization Too High
**Symptoms:**
- System unresponsive
- High CPU usage alerts
- Thermal throttling

**Solutions:**
```python
# Limit CPU usage
import os
os.nice(10)  # Lower process priority

# Configure XGBoost for single thread
xgb_params = {
    'n_jobs': 1,
    'nthread': 1
}
```

### Configuration Issues

#### Issue: Environment Variables Not Loading
**Symptoms:**
- Default values being used
- Configuration errors
- Missing API keys

**Solutions:**
```bash
# Check .env file exists and is readable
ls -la .env
cat .env | grep -v "^#" | grep -v "^$"

# Test environment loading
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('API Key loaded:', bool(os.getenv('ECONOMIC_TIMES_API_KEY')))
"
```

#### Issue: File Permission Errors
**Symptoms:**
- Cannot write to data directory
- Log file creation failures
- Model save/load errors

**Solutions:**
```bash
# Fix permissions
chmod 755 data/ logs/ models/
chmod 644 .env config/settings.py

# Check disk space
df -h

# Create missing directories
mkdir -p data/{cache,news,prices} logs models
```

### Debugging Tools

#### Enable Debug Logging
```python
# In .env file
LOG_LEVEL=DEBUG

# Or programmatically
import logging
logging.getLogger('nifty_ml_pipeline').setLevel(logging.DEBUG)
```

#### Performance Profiling
```python
import cProfile
import pstats

# Profile pipeline execution
profiler = cProfile.Profile()
profiler.enable()

# Run pipeline
result = controller.execute_pipeline("NIFTY 50")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

#### Memory Profiling
```python
from memory_profiler import profile

@profile
def run_pipeline():
    controller = PipelineController(get_config())
    return controller.execute_pipeline("NIFTY 50")

# Run with: python -m memory_profiler script.py
```

### Getting Help

#### Log Analysis
```bash
# Find recent errors
grep -i "error\|exception\|failed" pipeline.log | tail -20

# Check performance metrics
grep "duration_ms\|latency" pipeline.log | tail -10

# Monitor resource usage
grep "memory\|cpu" pipeline.log | tail -10
```

#### Health Check Script
```python
#!/usr/bin/env python3
"""Pipeline health check script."""

import sys
import traceback
from datetime import datetime
from nifty_ml_pipeline.orchestration.controller import PipelineController
from config.settings import get_config

def health_check():
    """Perform comprehensive health check."""
    checks = []
    
    try:
        # Test configuration loading
        config = get_config()
        checks.append(("Configuration", "OK"))
        
        # Test controller initialization
        controller = PipelineController(config)
        checks.append(("Controller Init", "OK"))
        
        # Test data collectors
        controller.nse_collector.validate_data(pd.DataFrame())
        checks.append(("NSE Collector", "OK"))
        
        # Test feature engineering
        calculator = controller.technical_calculator
        checks.append(("Technical Indicators", "OK"))
        
        # Test model loading
        if controller.predictor.is_trained:
            checks.append(("Model", "OK"))
        else:
            checks.append(("Model", "Not Trained"))
            
    except Exception as e:
        checks.append(("Error", str(e)))
        traceback.print_exc()
    
    # Print results
    print(f"Health Check - {datetime.now()}")
    print("-" * 40)
    for check, status in checks:
        print(f"{check:20}: {status}")
    
    return all(status == "OK" for _, status in checks)

if __name__ == "__main__":
    healthy = health_check()
    sys.exit(0 if healthy else 1)
```

#### Contact Information
- **GitHub Issues**: Create an issue with detailed error logs
- **Documentation**: Check API documentation for usage examples
- **Logs**: Always include relevant log excerpts when reporting issues