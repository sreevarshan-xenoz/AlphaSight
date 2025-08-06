# Performance Tuning Guide

## Overview

This guide provides recommendations for optimizing the NIFTY 50 ML Pipeline performance to achieve sub-10ms inference latency and efficient resource utilization.

## CPU Optimization

### XGBoost Configuration

**Optimal CPU Settings:**
```python
# In nifty_ml_pipeline/models/predictor.py
hyperparameters = {
    'n_jobs': 1,                    # Single-threaded for CPU optimization
    'tree_method': 'exact',         # CPU-optimized tree construction
    'max_depth': 6,                 # Balanced complexity
    'n_estimators': 100,            # Moderate ensemble size
    'learning_rate': 0.1,           # Standard learning rate
    'subsample': 0.8,               # Reduce overfitting
    'colsample_bytree': 0.8,        # Feature sampling
    'reg_alpha': 0.1,               # L1 regularization
    'reg_lambda': 1.0,              # L2 regularization
}
```

**CPU-Specific Optimizations:**
```python
# Set CPU affinity (Linux only)
import os
import psutil

def set_cpu_affinity():
    """Set CPU affinity for optimal performance."""
    process = psutil.Process()
    # Use only performance cores
    process.cpu_affinity([0, 1])  # Adjust based on your CPU

# Disable hyperthreading for consistent performance
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
```

### Memory Optimization

**Memory-Efficient Data Processing:**
```python
# Use memory-efficient data types
def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# Process data in chunks
def process_large_dataset(data, chunk_size=1000):
    """Process large datasets in chunks."""
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        yield process_chunk(chunk)
```

**Garbage Collection Optimization:**
```python
import gc

def optimize_memory():
    """Optimize memory usage."""
    # Force garbage collection
    gc.collect()
    
    # Adjust GC thresholds for better performance
    gc.set_threshold(700, 10, 10)
    
    # Disable GC during critical sections
    gc.disable()
    # ... critical code ...
    gc.enable()
```

## Inference Optimization

### Model Loading Optimization

**Pre-load Models:**
```python
class OptimizedInferenceEngine:
    """Optimized inference engine with pre-loaded models."""
    
    def __init__(self):
        # Pre-load model at initialization
        self.model = self._load_optimized_model()
        self.feature_scaler = self._load_feature_scaler()
        
        # Pre-allocate arrays for inference
        self.feature_array = np.zeros(6, dtype=np.float32)
        
    def _load_optimized_model(self):
        """Load model with optimized settings."""
        model = xgb.XGBRegressor()
        model.load_model('models/optimized_model.json')
        
        # Set inference-specific parameters
        model.set_params(n_jobs=1, nthread=1)
        return model
```

### Feature Engineering Optimization

**Vectorized Operations:**
```python
def calculate_indicators_vectorized(prices):
    """Calculate indicators using vectorized operations."""
    # Use pandas vectorized operations
    returns = prices.pct_change()
    
    # Vectorized RSI calculation
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Pre-compute commonly used values
class OptimizedTechnicalCalculator:
    def __init__(self):
        # Pre-compute constants
        self.rsi_alpha = 1.0 / 14
        self.sma_weights = np.ones(5) / 5
        
    def fast_rsi(self, prices):
        """Optimized RSI calculation."""
        # Use pre-computed alpha for exponential smoothing
        delta = np.diff(prices)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Exponential moving average
        avg_gain = self._ema(gains, self.rsi_alpha)
        avg_loss = self._ema(losses, self.rsi_alpha)
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
```

### Caching Strategies

**Feature Caching:**
```python
from functools import lru_cache
import hashlib

class CachedFeatureEngine:
    """Feature engine with intelligent caching."""
    
    def __init__(self, cache_size=1000):
        self.cache_size = cache_size
        self.feature_cache = {}
        
    @lru_cache(maxsize=1000)
    def cached_technical_indicators(self, price_hash):
        """Cache technical indicators by price data hash."""
        return self._compute_indicators(price_hash)
    
    def get_cache_key(self, data):
        """Generate cache key for data."""
        return hashlib.md5(str(data).encode()).hexdigest()
    
    def get_features_with_cache(self, price_data):
        """Get features with caching."""
        cache_key = self.get_cache_key(price_data.values.tobytes())
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        features = self._compute_features(price_data)
        self.feature_cache[cache_key] = features
        
        # Limit cache size
        if len(self.feature_cache) > self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
        
        return features
```

## Data Collection Optimization

### API Request Optimization

**Connection Pooling:**
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class OptimizedDataCollector:
    """Data collector with optimized HTTP requests."""
    
    def __init__(self):
        self.session = self._create_optimized_session()
    
    def _create_optimized_session(self):
        """Create optimized HTTP session."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set timeouts
        session.timeout = (5, 30)  # (connect, read)
        
        return session
```

### Data Storage Optimization

**Efficient File Formats:**
```python
# Use Parquet for efficient storage
def save_optimized_data(df, filepath):
    """Save data in optimized format."""
    df.to_parquet(
        filepath,
        compression='snappy',  # Fast compression
        index=False,
        engine='pyarrow'
    )

# Use memory mapping for large files
def load_large_dataset(filepath):
    """Load large dataset efficiently."""
    return pd.read_parquet(
        filepath,
        engine='pyarrow',
        use_pandas_metadata=True
    )
```

## System-Level Optimizations

### Operating System Tuning

**Linux Kernel Parameters:**
```bash
# Add to /etc/sysctl.conf
vm.swappiness=10                    # Reduce swapping
vm.dirty_ratio=15                   # Reduce dirty page ratio
vm.dirty_background_ratio=5         # Background writeback threshold
net.core.rmem_max=16777216         # Increase network buffer
net.core.wmem_max=16777216         # Increase network buffer
```

**CPU Governor Settings:**
```bash
# Set CPU governor to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

### Process Optimization

**Process Priority:**
```python
import os
import psutil

def optimize_process_priority():
    """Optimize process priority for better performance."""
    process = psutil.Process()
    
    # Set higher priority (lower nice value)
    process.nice(-5)  # Requires root privileges
    
    # Set CPU affinity to specific cores
    process.cpu_affinity([0, 1])  # Use first two cores
    
    # Set I/O priority
    process.ionice(psutil.IOPRIO_CLASS_RT, value=4)
```

## Monitoring and Profiling

### Performance Monitoring

**Real-time Performance Tracking:**
```python
import time
import psutil
from contextlib import contextmanager

@contextmanager
def performance_monitor(operation_name):
    """Monitor performance of operations."""
    process = psutil.Process()
    
    # Record initial state
    start_time = time.perf_counter()
    start_memory = process.memory_info().rss
    start_cpu = process.cpu_percent()
    
    try:
        yield
    finally:
        # Record final state
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss
        end_cpu = process.cpu_percent()
        
        # Log performance metrics
        duration = (end_time - start_time) * 1000
        memory_delta = (end_memory - start_memory) / 1024 / 1024
        
        print(f"{operation_name}:")
        print(f"  Duration: {duration:.2f}ms")
        print(f"  Memory Delta: {memory_delta:.2f}MB")
        print(f"  CPU Usage: {end_cpu:.1f}%")

# Usage
with performance_monitor("Model Inference"):
    prediction = model.predict(features)
```

### Profiling Tools

**CPU Profiling:**
```python
import cProfile
import pstats
from pstats import SortKey

def profile_inference():
    """Profile inference performance."""
    profiler = cProfile.Profile()
    
    # Profile the inference
    profiler.enable()
    for _ in range(1000):
        prediction = model.predict(sample_features)
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Top 20 functions
```

**Memory Profiling:**
```python
from memory_profiler import profile
import tracemalloc

@profile
def memory_intensive_function():
    """Function to profile memory usage."""
    # Your code here
    pass

# Alternative: tracemalloc
def trace_memory_usage():
    """Trace memory allocations."""
    tracemalloc.start()
    
    # Your code here
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f}MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f}MB")
    
    tracemalloc.stop()
```

## Configuration Recommendations

### Production Configuration

**Optimal Settings for Production:**
```python
# config/production_settings.py
PRODUCTION_CONFIG = {
    'performance': {
        'cpu_threads': 1,
        'memory_limit_gb': 4,
        'max_inference_latency_ms': 10,
        'batch_size': 1,  # Single predictions for lowest latency
        'cache_enabled': True,
        'cache_size': 1000,
    },
    'model': {
        'max_depth': 6,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'tree_method': 'exact',
        'objective': 'reg:squarederror',
    },
    'data': {
        'storage_format': 'parquet',
        'compression': 'snappy',
        'chunk_size': 10000,
        'memory_map': True,
    },
    'monitoring': {
        'enable_profiling': False,  # Disable in production
        'log_performance': True,
        'alert_threshold_ms': 15,
    }
}
```

### Benchmarking

**Performance Benchmarks:**
```python
def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    import numpy as np
    from datetime import datetime
    
    # Test data
    n_samples = 1000
    features = np.random.randn(n_samples, 6).astype(np.float32)
    
    # Benchmark inference
    times = []
    for i in range(n_samples):
        start = time.perf_counter()
        prediction = model.predict(features[i:i+1])
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    # Calculate statistics
    mean_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)
    
    print(f"Inference Performance Benchmark:")
    print(f"  Mean latency: {mean_time:.2f}ms")
    print(f"  95th percentile: {p95_time:.2f}ms")
    print(f"  99th percentile: {p99_time:.2f}ms")
    print(f"  Target met: {mean_time < 10.0}")
    
    return {
        'mean_latency_ms': mean_time,
        'p95_latency_ms': p95_time,
        'p99_latency_ms': p99_time,
        'meets_target': mean_time < 10.0
    }
```

## Hardware Recommendations

### CPU Requirements
- **Minimum**: 2 cores, 2.5GHz
- **Recommended**: 4+ cores, 3.0GHz+
- **Optimal**: Intel Core i7/i9 or AMD Ryzen 7/9

### Memory Requirements
- **Minimum**: 4GB RAM
- **Recommended**: 8GB RAM
- **Optimal**: 16GB+ RAM for large datasets

### Storage Requirements
- **Type**: SSD recommended for data I/O
- **Space**: 10GB+ for data and models
- **I/O**: High IOPS for real-time processing

### Network Requirements
- **Bandwidth**: Stable internet for data collection
- **Latency**: Low latency for API calls
- **Reliability**: Redundant connections for production