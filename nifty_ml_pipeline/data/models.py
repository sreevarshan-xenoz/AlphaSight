# nifty_ml_pipeline/data/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd


@dataclass
class PriceData:
    """Represents OHLCV data for a single trading day.
    
    Ensures numerical validation and timestamp integrity to prevent
    look-ahead bias and maintain data quality.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self):
        """Validate price data integrity after initialization."""
        if not (0 < self.open and 0 < self.high and 0 < self.low and 0 < self.close):
            raise ValueError("Price values must be positive.")
        
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("High must be >= open/close; Low must be <= open/close.")
        
        if self.volume < 0:
            raise ValueError("Volume cannot be negative.")

    def validate(self) -> bool:
        """Validate price data consistency."""
        return all([
            self.open > 0,
            self.high >= max(self.open, self.close),
            self.low <= min(self.open, self.close),
            self.volume >= 0
        ])


@dataclass
class NewsData:
    """Represents a single financial news headline with metadata.
    
    Sentiment score is added post-processing via VADER sentiment analyzer.
    """
    source: str
    timestamp: datetime
    headline: str
    url: Optional[str] = None
    sentiment_score: Optional[float] = None  # To be filled by VADER

    def __post_init__(self):
        """Validate news data after initialization."""
        if not self.headline.strip():
            raise ValueError("Headline cannot be empty.")
        
        if self.sentiment_score is not None and not (-1 <= self.sentiment_score <= 1):
            raise ValueError("Sentiment score must be in [-1, 1] if provided.")

    def is_recent(self, days: int = 30) -> bool:
        """Check if news is within specified recency window."""
        return (datetime.now() - self.timestamp).days <= days


@dataclass
class FeatureVector:
    """Unified input vector for the ML model.
    
    All features are numeric and normalized for optimal model performance.
    Represents the complete feature set at a specific timestamp.
    """
    timestamp: datetime
    symbol: str
    lag1_return: float
    lag2_return: float
    sma_5_ratio: float  # close / sma_5
    rsi_14: float
    macd_hist: float
    daily_sentiment: float  # mean compound score from news

    def to_series(self) -> pd.Series:
        """Convert to pandas Series for model input."""
        data = {k: v for k, v in self.__dict__.items() 
                if k not in ['timestamp', 'symbol']}
        return pd.Series(data, name=self.timestamp)

    def to_array(self) -> List[float]:
        """Convert to list of feature values for model input."""
        return [
            self.lag1_return,
            self.lag2_return,
            self.sma_5_ratio,
            self.rsi_14,
            self.macd_hist,
            self.daily_sentiment
        ]


@dataclass
class PredictionResult:
    """Structured output of the prediction pipeline.
    
    Contains prediction, confidence, and metadata for tracking
    and performance monitoring.
    """
    timestamp: datetime
    symbol: str
    predicted_close: float
    signal: str  # "Buy", "Hold", "Sell"
    confidence: float  # 0.0 to 1.0
    model_version: str
    features_used: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate prediction result after initialization."""
        if self.signal not in ["Buy", "Hold", "Sell"]:
            raise ValueError("Signal must be one of: Buy, Hold, Sell")
        
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.predicted_close <= 0:
            raise ValueError("Predicted close price must be positive")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "predicted_close": self.predicted_close,
            "signal": self.signal,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "features_used": self.features_used
        }

    def is_actionable(self, min_confidence: float = 0.7) -> bool:
        """Check if prediction confidence meets actionable threshold."""
        return self.confidence >= min_confidence