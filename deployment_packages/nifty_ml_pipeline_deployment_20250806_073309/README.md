# NIFTY 50 ML Pipeline

A CPU-optimized machine learning pipeline for predicting NIFTY 50 index movements using technical indicators, sentiment analysis, and XGBoost modeling.

## Overview

This project implements a complete machine learning pipeline that:
- Collects NIFTY 50 historical price data from NSEpy API
- Retrieves financial news for sentiment analysis
- Engineers technical indicators (RSI, SMA, MACD) and sentiment features
- Trains CPU-optimized XGBoost models for directional predictions
- Provides automated daily predictions with confidence scoring
- Includes comprehensive monitoring, logging, and error handling

## Key Features

- **Multi-Source Data Integration**: NSE price data + Economic Times news sentiment
- **CPU-Optimized Performance**: Sub-10ms inference latency on standard hardware
- **Technical Analysis**: RSI(14), SMA(5), MACD(12,26) indicators with O(n) complexity
- **Sentiment Analysis**: VADER-based news sentiment scoring
- **Real-Time Predictions**: Daily automated execution at 5:30 PM IST
- **Performance Monitoring**: 80%+ directional accuracy target with comprehensive metrics
- **Robust Architecture**: Graceful error handling and recovery mechanisms

## Project Structure

```
nifty_ml_pipeline/
├── data/              # Data collection and validation modules
├── features/          # Feature engineering (technical indicators, sentiment)
├── models/            # XGBoost model training and inference
├── orchestration/     # Pipeline coordination and scheduling
├── utils/             # Utility functions and helpers
└── main.py           # Main entry point

config/
├── settings.py        # Configuration management
└── __init__.py

tests/
├── data/             # Data module tests
├── features/         # Feature engineering tests
├── models/           # Model tests
├── orchestration/    # Pipeline tests
└── __init__.py

docs/                 # Documentation (created as needed)
logs/                 # Application logs (created at runtime)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CPU-based execution (no GPU required)
- Internet connection for data fetching

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nifty-50-ml-pipeline
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Production dependencies
   pip install -r requirements.txt
   
   # Or install with development tools
   make install-dev
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys and preferences
   ```

## Usage

### Running the Pipeline

```bash
# Using Python module
python -m nifty_ml_pipeline.main

# Using make command
make run

# Using installed console script (after pip install -e .)
nifty-ml-pipeline
```

### Development Commands

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Lint code
make lint

# Clean build artifacts
make clean
```

## Configuration

The pipeline uses environment variables for configuration. Key settings include:

- **API Keys**: NSE and Economic Times API credentials (if required)
- **Performance Thresholds**: Accuracy targets and latency limits
- **Scheduling**: Execution time and timezone settings
- **Model Parameters**: XGBoost hyperparameters and technical indicator periods

See `.env.example` for all available configuration options.

## Architecture

The pipeline follows a modular architecture with clear separation of concerns:

1. **Data Layer**: Handles collection, validation, and storage of financial data
2. **Feature Engineering**: Computes technical indicators and sentiment scores
3. **Model Layer**: XGBoost training, inference, and performance tracking
4. **Orchestration**: Pipeline coordination, scheduling, and error handling

## Performance Targets

- **Inference Latency**: < 10ms per prediction
- **Directional Accuracy**: 80%+ on validation data
- **Data Processing**: O(n) complexity for technical indicators
- **Sentiment Analysis**: < 0.01 seconds per news headline

## Dependencies

### Core ML Stack
- **XGBoost 2.0.3**: CPU-optimized gradient boosting
- **pandas 2.1.4**: Data manipulation and analysis
- **numpy 1.24.4**: Numerical computing
- **scikit-learn 1.3.2**: ML utilities and validation

### Financial Data
- **nsepy 0.8**: NSE India stock data
- **yfinance 0.2.28**: Additional financial data source

### Sentiment Analysis
- **nltk 3.8.1**: Natural language processing
- **vaderSentiment 3.3.2**: Financial sentiment analysis

### Infrastructure
- **schedule 1.2.1**: Task scheduling
- **loguru 0.7.2**: Advanced logging
- **pyarrow 14.0.2**: Efficient data storage

## Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=nifty_ml_pipeline --cov-report=html

# Run specific test modules
pytest tests/data/
pytest tests/models/
```

## Contributing

1. Follow PEP 8 style guidelines
2. Maintain test coverage above 80%
3. Use type hints for all public functions
4. Update documentation for new features

## License

This project is for educational and research purposes. See LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It should not be used for actual trading without proper risk management and thorough backtesting. Past performance does not guarantee future results.