# Feature Engineering API

## Overview

The feature engineering module provides CPU-optimized algorithms for computing technical indicators, analyzing sentiment, and normalizing features for machine learning models.

## TechnicalIndicatorCalculator

### Class: `TechnicalIndicatorCalculator`

CPU-optimized technical indicator calculator using vectorized pandas operations.

#### Constructor

```python
TechnicalIndicatorCalculator()
```

Initializes with default parameters:
- RSI period: 14
- SMA period: 5
- MACD fast: 12, slow: 26, signal: 9

#### Methods

##### `calculate_rsi(prices, period)`

Calculate RSI (Relative Strength Index) using O(n) algorithm.

```python
def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series
```

**Parameters:**
- `prices`: Series of closing prices
- `period`: RSI period (default: 14)

**Returns:**
- `pd.Series`: RSI values (0-100)

**Example:**
```python
from nifty_ml_pipeline.features.technical_indicators import TechnicalIndicatorCalculator
import pandas as pd

calculator = TechnicalIndicatorCalculator()
prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
rsi = calculator.calculate_rsi(prices)
print(f"Latest RSI: {rsi.iloc[-1]:.2f}")
```

##### `calculate_sma(prices, period)`

Calculate Simple Moving Average using O(n) rolling window.

```python
def calculate_sma(self, prices: pd.Series, period: int = 5) -> pd.Series
```

**Parameters:**
- `prices`: Series of closing prices
- `period`: SMA period (default: 5)

**Returns:**
- `pd.Series`: SMA values

##### `calculate_macd(prices, fast, slow, signal)`

Calculate MACD (Moving Average Convergence Divergence) with O(n) complexity.

```python
def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]
```

**Parameters:**
- `prices`: Series of closing prices
- `fast`: Fast EMA period (default: 12)
- `slow`: Slow EMA period (default: 26)
- `signal`: Signal line EMA period (default: 9)

**Returns:**
- `Dict[str, pd.Series]`: Dictionary containing 'macd_line', 'signal_line', and 'histogram'

**Example:**
```python
macd_results = calculator.calculate_macd(prices)
print(f"MACD Line: {macd_results['macd_line'].iloc[-1]:.4f}")
print(f"Signal Line: {macd_results['signal_line'].iloc[-1]:.4f}")
print(f"Histogram: {macd_results['histogram'].iloc[-1]:.4f}")
```

##### `calculate_all_indicators(price_data)`

Calculate all technical indicators for a DataFrame of price data.

```python
def calculate_all_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `price_data`: DataFrame with columns ['open', 'high', 'low', 'close', 'volume'] and datetime index

**Returns:**
- `pd.DataFrame`: Original data plus technical indicators

**Example:**
```python
# Assuming price_data is a DataFrame with OHLCV data
indicators_df = calculator.calculate_all_indicators(price_data)
print(indicators_df[['close', 'rsi_14', 'sma_5', 'macd_histogram']].tail())
```

##### `get_latest_indicators(price_data)`

Get the most recent technical indicators from price data.

```python
def get_latest_indicators(self, price_data: pd.DataFrame) -> TechnicalIndicators
```

**Parameters:**
- `price_data`: DataFrame with price data and calculated indicators

**Returns:**
- `TechnicalIndicators`: Object with latest values

## SentimentAnalyzer

### Class: `SentimentAnalyzer`

Analyzes financial news sentiment using VADER sentiment analyzer.

#### Constructor

```python
SentimentAnalyzer()
```

Initializes VADER sentiment analyzer with financial lexicon enhancements.

#### Methods

##### `analyze_headline(headline)`

Analyze sentiment of a single headline.

```python
def analyze_headline(self, headline: str) -> float
```

**Parameters:**
- `headline`: News headline text

**Returns:**
- `float`: Compound sentiment score (-1 to +1)

**Example:**
```python
from nifty_ml_pipeline.features.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
headline = "NIFTY 50 surges to new highs on strong buying interest"
sentiment = analyzer.analyze_headline(headline)
print(f"Sentiment score: {sentiment:.3f}")
```

##### `analyze_dataframe(news_df)`

Analyze sentiment for a DataFrame of news articles.

```python
def analyze_dataframe(self, news_df: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `news_df`: DataFrame with 'headline' column

**Returns:**
- `pd.DataFrame`: Original data with added 'sentiment_score' column

##### `aggregate_daily_sentiment(news_df)`

Aggregate sentiment scores by date.

```python
def aggregate_daily_sentiment(self, news_df: pd.DataFrame) -> pd.Series
```

**Parameters:**
- `news_df`: DataFrame with 'timestamp' and 'sentiment_score' columns

**Returns:**
- `pd.Series`: Daily aggregated sentiment scores

## FeatureNormalizer

### Class: `FeatureNormalizer`

Normalizes and combines features for machine learning models.

#### Constructor

```python
FeatureNormalizer()
```

#### Methods

##### `normalize_features(features_df)`

Normalize features using StandardScaler.

```python
def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `features_df`: DataFrame with raw features

**Returns:**
- `pd.DataFrame`: Normalized features

##### `create_feature_vectors(price_data, news_data)`

Create unified feature vectors combining price, technical, and sentiment data.

```python
def create_feature_vectors(self, price_data: pd.DataFrame, news_data: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `price_data`: DataFrame with price data and technical indicators
- `news_data`: DataFrame with news data and sentiment scores

**Returns:**
- `pd.DataFrame`: Combined and normalized feature vectors

**Example:**
```python
from nifty_ml_pipeline.features.feature_normalizer import FeatureNormalizer

normalizer = FeatureNormalizer()
feature_vectors = normalizer.create_feature_vectors(price_data, news_data)
print(f"Created {len(feature_vectors)} feature vectors with {len(feature_vectors.columns)} features")
```

## Data Models

### TechnicalIndicators

Container for computed technical indicators.

```python
@dataclass
class TechnicalIndicators:
    rsi_14: Optional[float] = None
    sma_5: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
```

### FeatureVector

Unified input vector for the ML model.

```python
@dataclass
class FeatureVector:
    timestamp: datetime
    symbol: str
    lag1_return: float
    lag2_return: float
    sma_5_ratio: float
    rsi_14: float
    macd_hist: float
    daily_sentiment: float
```

**Methods:**
- `to_series() -> pd.Series`: Convert to pandas Series for model input
- `to_array() -> List[float]`: Convert to list of feature values

## Performance Optimizations

### Technical Indicators
- **O(n) Algorithms**: All indicators computed with linear time complexity
- **Vectorized Operations**: Uses pandas vectorized operations for speed
- **Memory Efficiency**: Minimal memory footprint with streaming calculations

### Sentiment Analysis
- **Target Performance**: 0.01 seconds per sentence processing
- **Batch Processing**: Efficient batch analysis for multiple headlines
- **Caching**: Sentiment scores cached to avoid recomputation

### Feature Normalization
- **StandardScaler**: Z-score normalization for consistent feature scales
- **Missing Data Handling**: Robust handling of missing values
- **Feature Integration**: Efficient combination of multi-source features

## Error Handling

- **Data Validation**: Comprehensive validation of input data
- **Graceful Degradation**: Continues processing with available features
- **Logging**: Detailed logging of processing steps and errors
- **Recovery**: Automatic recovery from transient failures