# NIFTY 50 ML Pipeline API Documentation

## Overview

The NIFTY 50 ML Pipeline provides a comprehensive API for collecting financial data, engineering features, and generating trading predictions using machine learning. The system is designed for CPU-optimized performance with sub-10ms inference latency.

## Quick Start

```python
from nifty_ml_pipeline.orchestration.controller import PipelineController
from config.settings import get_config

# Initialize pipeline
config = get_config()
controller = PipelineController(config)

# Execute complete pipeline
result = controller.execute_pipeline("NIFTY 50")

# Check results
if result.was_successful():
    print(f"Generated {len(result.predictions)} predictions")
    for prediction in result.predictions:
        print(f"Signal: {prediction['signal']}, Confidence: {prediction['confidence']:.3f}")
```

## Core Components

### Data Collection
- **NSEDataCollector**: Collects historical price data from NSE
- **NewsDataCollector**: Retrieves financial news for sentiment analysis

### Feature Engineering
- **TechnicalIndicatorCalculator**: Computes RSI, SMA, MACD indicators
- **SentimentAnalyzer**: Analyzes news sentiment using VADER
- **FeatureNormalizer**: Normalizes and combines features

### Machine Learning
- **XGBoostPredictor**: CPU-optimized XGBoost model for predictions
- **InferenceEngine**: High-performance inference with sub-10ms latency

### Orchestration
- **PipelineController**: Main orchestration for end-to-end execution
- **ErrorHandler**: Comprehensive error handling and recovery
- **PerformanceMonitor**: Real-time performance monitoring

## Data Models

### Core Data Types
- **PriceData**: OHLCV price data with validation
- **NewsData**: Financial news with sentiment scores
- **FeatureVector**: Normalized feature inputs for ML model
- **PredictionResult**: Structured prediction outputs

## API Reference

Detailed API documentation is organized by module:

- [Data Collection API](data/README.md)
- [Feature Engineering API](features/README.md)
- [Machine Learning API](models/README.md)
- [Orchestration API](orchestration/README.md)
- [Storage API](storage/README.md)

## Performance Specifications

- **Inference Latency**: < 10ms on standard CPU hardware
- **Data Collection**: Handles 1-year rolling window with retry logic
- **Feature Engineering**: O(n) algorithms for technical indicators
- **Model Training**: TimeSeriesSplit for chronological validation

## Error Handling

The pipeline includes comprehensive error handling:
- Automatic retry with exponential backoff
- Graceful degradation with fallback data
- Detailed error logging and recovery procedures

## Configuration

See [Configuration Guide](../deployment/configuration.md) for detailed setup instructions.

## Examples

See [Examples Directory](../examples/) for usage examples and tutorials.