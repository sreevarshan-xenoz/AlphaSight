# Data Collection API

## Overview

The data collection module provides robust APIs for retrieving financial data from multiple sources with built-in error handling and validation.

## NSEDataCollector

### Class: `NSEDataCollector`

Collects NIFTY 50 historical price data from NSE using NSEpy API.

#### Constructor

```python
NSEDataCollector(max_retries: int = 3, base_delay: float = 1.0)
```

**Parameters:**
- `max_retries`: Maximum number of retry attempts (default: 3)
- `base_delay`: Base delay in seconds for exponential backoff (default: 1.0)

#### Methods

##### `collect_data(symbol, start_date, end_date)`

Collect OHLCV data from NSE with retry logic.

```python
def collect_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame
```

**Parameters:**
- `symbol`: Stock symbol (e.g., 'NIFTY 50')
- `start_date`: Start date for data collection
- `end_date`: End date for data collection

**Returns:**
- `pd.DataFrame`: OHLCV data with columns [Open, High, Low, Close, Volume]

**Raises:**
- `Exception`: If all retry attempts fail

**Example:**
```python
from datetime import datetime, timedelta
from nifty_ml_pipeline.data.collectors import NSEDataCollector

collector = NSEDataCollector(max_retries=3)
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

data = collector.collect_data("NIFTY 50", start_date, end_date)
print(f"Collected {len(data)} records")
```

##### `collect_rolling_window_data(symbol, window_days)`

Collect data for a rolling window from current date.

```python
def collect_rolling_window_data(self, symbol: str, window_days: int = 365) -> pd.DataFrame
```

**Parameters:**
- `symbol`: Stock symbol to collect data for
- `window_days`: Number of days to look back (default: 365)

**Returns:**
- `pd.DataFrame`: Historical data for the rolling window

##### `validate_data(data)`

Validate NSE data structure and content.

```python
def validate_data(self, data: pd.DataFrame) -> bool
```

**Parameters:**
- `data`: DataFrame to validate

**Returns:**
- `bool`: True if validation passes

##### `convert_to_price_data_objects(df, symbol)`

Convert DataFrame to list of PriceData objects.

```python
def convert_to_price_data_objects(self, df: pd.DataFrame, symbol: str) -> List[PriceData]
```

**Parameters:**
- `df`: DataFrame with OHLCV data
- `symbol`: Symbol name for the data

**Returns:**
- `List[PriceData]`: List of PriceData objects

## NewsDataCollector

### Class: `NewsDataCollector`

Collects financial news data from Economic Times API.

#### Constructor

```python
NewsDataCollector(api_key: Optional[str] = None, max_retries: int = 3, base_delay: float = 1.0)
```

**Parameters:**
- `api_key`: API key for Economic Times (if required)
- `max_retries`: Maximum number of retry attempts (default: 3)
- `base_delay`: Base delay in seconds for exponential backoff (default: 1.0)

#### Methods

##### `collect_data(symbol, start_date, end_date)`

Collect news data for the specified date range.

```python
def collect_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame
```

**Parameters:**
- `symbol`: Symbol to collect news for (used for filtering)
- `start_date`: Start date for news collection
- `end_date`: End date for news collection

**Returns:**
- `pd.DataFrame`: News data with columns [headline, timestamp, source, url]

**Example:**
```python
from nifty_ml_pipeline.data.collectors import NewsDataCollector

collector = NewsDataCollector()
news_data = collector.collect_data("NIFTY 50", start_date, end_date)
print(f"Collected {len(news_data)} news articles")
```

##### `check_news_freshness(news_items, max_age_hours)`

Check news freshness and filter out stale news.

```python
def check_news_freshness(self, news_items: List[Dict[str, Any]], max_age_hours: int = 24) -> List[Dict[str, Any]]
```

**Parameters:**
- `news_items`: List of news items to check
- `max_age_hours`: Maximum age in hours for news to be considered fresh (default: 24)

**Returns:**
- `List[Dict]`: Fresh news items only

##### `handle_missing_news_data(symbol, date)`

Handle missing or stale news data by providing fallback content.

```python
def handle_missing_news_data(self, symbol: str, date: datetime) -> List[Dict[str, Any]]
```

**Parameters:**
- `symbol`: Symbol for which news is missing
- `date`: Date for which news is needed

**Returns:**
- `List[Dict]`: Fallback news items

## Data Models

### PriceData

Represents OHLCV data for a single trading day.

```python
@dataclass
class PriceData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
```

**Methods:**
- `validate() -> bool`: Validate price data consistency

### NewsData

Represents a single financial news headline with metadata.

```python
@dataclass
class NewsData:
    source: str
    timestamp: datetime
    headline: str
    url: Optional[str] = None
    sentiment_score: Optional[float] = None
```

**Methods:**
- `is_recent(days: int = 30) -> bool`: Check if news is within specified recency window

## Error Handling

All data collectors implement comprehensive error handling:

- **Retry Logic**: Exponential backoff with configurable max retries
- **Data Validation**: Automatic validation of collected data
- **Fallback Mechanisms**: Graceful handling of missing or stale data
- **Logging**: Detailed logging of errors and recovery attempts

## Performance Considerations

- **API Rate Limits**: Built-in delays to respect API rate limits
- **Data Caching**: Local storage to reduce API calls
- **Efficient Validation**: O(n) validation algorithms
- **Memory Management**: Streaming data processing for large datasets