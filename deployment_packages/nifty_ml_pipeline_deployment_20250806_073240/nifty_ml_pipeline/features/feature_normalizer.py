# nifty_ml_pipeline/features/feature_normalizer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from nifty_ml_pipeline.data.models import FeatureVector, PriceData, NewsData
from nifty_ml_pipeline.features.technical_indicators import TechnicalIndicatorCalculator, TechnicalIndicators
from nifty_ml_pipeline.features.sentiment_analysis import SentimentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class NormalizedFeatures:
    """Container for normalized feature data."""
    timestamp: datetime
    symbol: str
    price_features: Dict[str, float]
    technical_features: Dict[str, float]
    sentiment_features: Dict[str, float]
    raw_features: Dict[str, float]  # Original unnormalized features
    
    def to_feature_vector(self) -> FeatureVector:
        """Convert to FeatureVector for model input."""
        # Extract specific features expected by the model
        return FeatureVector(
            timestamp=self.timestamp,
            symbol=self.symbol,
            lag1_return=self.price_features.get('lag1_return', 0.0),
            lag2_return=self.price_features.get('lag2_return', 0.0),
            sma_5_ratio=self.technical_features.get('sma_5_ratio', 1.0),
            rsi_14=self.technical_features.get('rsi_14', 50.0),
            macd_hist=self.technical_features.get('macd_histogram', 0.0),
            daily_sentiment=self.sentiment_features.get('compound_score', 0.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price_features': self.price_features,
            'technical_features': self.technical_features,
            'sentiment_features': self.sentiment_features,
            'raw_features': self.raw_features
        }


class FeatureNormalizer:
    """
    Feature normalization and integration system for ML pipeline.
    
    Combines price data, technical indicators, and sentiment analysis into
    normalized feature vectors suitable for model training and inference.
    """
    
    def __init__(self, normalization_method: str = 'standard'):
        """
        Initialize the feature normalizer.
        
        Args:
            normalization_method: Method for normalization ('standard', 'minmax', 'robust')
        """
        self.normalization_method = normalization_method
        self.scalers = {}
        self.is_fitted = False
        
        # Initialize components
        self.technical_calculator = TechnicalIndicatorCalculator()
        self.sentiment_analyzer = SentimentAnalyzer(download_nltk_data=False)
        
        # Feature configuration
        self.price_feature_names = ['lag1_return', 'lag2_return', 'volatility']
        self.technical_feature_names = ['rsi_14', 'sma_5_ratio', 'macd_histogram']
        self.sentiment_feature_names = ['compound_score']
        
        # Default values for missing data
        self.default_values = {
            'lag1_return': 0.0,
            'lag2_return': 0.0,
            'volatility': 0.0,
            'rsi_14': 50.0,  # Neutral RSI
            'sma_5_ratio': 1.0,  # Price equals SMA
            'macd_histogram': 0.0,
            'compound_score': 0.0  # Neutral sentiment
        }
    
    def _create_scaler(self) -> Any:
        """Create a scaler based on the normalization method."""
        if self.normalization_method == 'standard':
            return StandardScaler()
        elif self.normalization_method == 'minmax':
            return MinMaxScaler()
        elif self.normalization_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
    
    def _extract_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract price-based features from OHLCV data.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        if price_data.empty or 'close' not in price_data.columns:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=price_data.index)
        
        # Calculate returns
        returns = price_data['close'].pct_change()
        features['lag1_return'] = returns
        features['lag2_return'] = returns.shift(1)
        
        # Calculate volatility (rolling standard deviation of returns)
        features['volatility'] = returns.rolling(window=5, min_periods=1).std()
        
        # Fill NaN values with defaults
        for col in features.columns:
            features[col] = features[col].fillna(self.default_values.get(col, 0.0))
        
        return features
    
    def _extract_technical_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract technical indicator features.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        if price_data.empty:
            return pd.DataFrame()
        
        # Calculate all technical indicators
        data_with_indicators = self.technical_calculator.calculate_all_indicators(price_data)
        
        features = pd.DataFrame(index=price_data.index)
        
        # Extract RSI (already normalized to 0-100, convert to 0-1)
        if 'rsi_14' in data_with_indicators.columns:
            features['rsi_14'] = data_with_indicators['rsi_14'] / 100.0
        else:
            features['rsi_14'] = self.default_values['rsi_14'] / 100.0
        
        # Calculate SMA ratio (close price / SMA)
        if 'sma_5' in data_with_indicators.columns and 'close' in data_with_indicators.columns:
            features['sma_5_ratio'] = data_with_indicators['close'] / data_with_indicators['sma_5']
        else:
            features['sma_5_ratio'] = self.default_values['sma_5_ratio']
        
        # Extract MACD histogram
        if 'macd_histogram' in data_with_indicators.columns:
            features['macd_histogram'] = data_with_indicators['macd_histogram']
        else:
            features['macd_histogram'] = self.default_values['macd_histogram']
        
        # Fill NaN values with defaults
        for col in features.columns:
            default_val = self.default_values.get(col, 0.0)
            if col == 'rsi_14':
                default_val = default_val / 100.0  # Normalize RSI default
            features[col] = features[col].fillna(default_val)
        
        return features
    
    def _extract_sentiment_features(self, news_data: pd.DataFrame, 
                                  target_dates: pd.DatetimeIndex,
                                  symbol: str = None) -> pd.DataFrame:
        """
        Extract sentiment features for given dates.
        
        Args:
            news_data: DataFrame with news data
            target_dates: Dates to extract sentiment for
            symbol: Symbol to filter news for (optional)
            
        Returns:
            DataFrame with sentiment features
        """
        features = pd.DataFrame(index=target_dates)
        
        if news_data.empty:
            features['compound_score'] = self.default_values['compound_score']
            return features
        
        # Get sentiment for each date
        sentiment_scores = []
        for date in target_dates:
            try:
                sentiment = self.sentiment_analyzer.get_daily_sentiment(
                    news_data, date, ticker=symbol
                )
                sentiment_scores.append(sentiment)
            except Exception as e:
                logger.warning(f"Error getting sentiment for {date}: {e}")
                sentiment_scores.append(self.default_values['compound_score'])
        
        features['compound_score'] = sentiment_scores
        
        return features
    
    def fit(self, price_data: pd.DataFrame, news_data: pd.DataFrame = None, 
            symbol: str = None) -> 'FeatureNormalizer':
        """
        Fit the normalizer on training data.
        
        Args:
            price_data: DataFrame with OHLCV data
            news_data: DataFrame with news data (optional)
            symbol: Symbol identifier (optional)
            
        Returns:
            Self for method chaining
        """
        if price_data.empty:
            raise ValueError("Price data cannot be empty for fitting")
        
        logger.info(f"Fitting feature normalizer on {len(price_data)} price records")
        
        # Extract all features
        price_features = self._extract_price_features(price_data)
        technical_features = self._extract_technical_features(price_data)
        
        if news_data is not None and not news_data.empty:
            sentiment_features = self._extract_sentiment_features(
                news_data, price_data.index, symbol
            )
        else:
            sentiment_features = pd.DataFrame(index=price_data.index)
            sentiment_features['compound_score'] = self.default_values['compound_score']
        
        # Fit scalers for each feature group
        feature_groups = {
            'price': (price_features, self.price_feature_names),
            'technical': (technical_features, self.technical_feature_names),
            'sentiment': (sentiment_features, self.sentiment_feature_names)
        }
        
        for group_name, (features_df, feature_names) in feature_groups.items():
            if not features_df.empty:
                # Select only the features we want to normalize
                available_features = [f for f in feature_names if f in features_df.columns]
                
                if available_features:
                    scaler = self._create_scaler()
                    feature_data = features_df[available_features].values
                    
                    # Remove any infinite or extremely large values
                    feature_data = np.where(np.isfinite(feature_data), feature_data, 0.0)
                    
                    scaler.fit(feature_data)
                    self.scalers[group_name] = scaler
                    
                    logger.info(f"Fitted {group_name} scaler on {len(available_features)} features")
        
        self.is_fitted = True
        logger.info("Feature normalizer fitting completed")
        
        return self
    
    def transform(self, price_data: pd.DataFrame, news_data: pd.DataFrame = None,
                 symbol: str = None) -> List[NormalizedFeatures]:
        """
        Transform data into normalized features.
        
        Args:
            price_data: DataFrame with OHLCV data
            news_data: DataFrame with news data (optional)
            symbol: Symbol identifier (optional)
            
        Returns:
            List of NormalizedFeatures objects
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        if price_data.empty:
            return []
        
        # Extract raw features
        price_features = self._extract_price_features(price_data)
        technical_features = self._extract_technical_features(price_data)
        
        if news_data is not None and not news_data.empty:
            sentiment_features = self._extract_sentiment_features(
                news_data, price_data.index, symbol
            )
        else:
            sentiment_features = pd.DataFrame(index=price_data.index)
            sentiment_features['compound_score'] = self.default_values['compound_score']
        
        # Normalize features
        normalized_results = []
        
        for idx in price_data.index:
            try:
                # Get raw feature values for this timestamp
                raw_features = {}
                norm_price_features = {}
                norm_technical_features = {}
                norm_sentiment_features = {}
                
                # Process price features
                if not price_features.empty and idx in price_features.index:
                    price_row = price_features.loc[idx]
                    available_price_features = [f for f in self.price_feature_names 
                                              if f in price_features.columns]
                    
                    if available_price_features and 'price' in self.scalers:
                        price_values = price_row[available_price_features].values.reshape(1, -1)
                        price_values = np.where(np.isfinite(price_values), price_values, 0.0)
                        normalized_price = self.scalers['price'].transform(price_values)[0]
                        
                        for i, feature_name in enumerate(available_price_features):
                            raw_features[feature_name] = float(price_row[feature_name])
                            norm_price_features[feature_name] = float(normalized_price[i])
                
                # Process technical features
                if not technical_features.empty and idx in technical_features.index:
                    tech_row = technical_features.loc[idx]
                    available_tech_features = [f for f in self.technical_feature_names 
                                             if f in technical_features.columns]
                    
                    if available_tech_features and 'technical' in self.scalers:
                        tech_values = tech_row[available_tech_features].values.reshape(1, -1)
                        tech_values = np.where(np.isfinite(tech_values), tech_values, 0.0)
                        normalized_tech = self.scalers['technical'].transform(tech_values)[0]
                        
                        for i, feature_name in enumerate(available_tech_features):
                            raw_features[feature_name] = float(tech_row[feature_name])
                            norm_technical_features[feature_name] = float(normalized_tech[i])
                
                # Process sentiment features
                if not sentiment_features.empty and idx in sentiment_features.index:
                    sent_row = sentiment_features.loc[idx]
                    available_sent_features = [f for f in self.sentiment_feature_names 
                                             if f in sentiment_features.columns]
                    
                    if available_sent_features and 'sentiment' in self.scalers:
                        sent_values = sent_row[available_sent_features].values.reshape(1, -1)
                        sent_values = np.where(np.isfinite(sent_values), sent_values, 0.0)
                        normalized_sent = self.scalers['sentiment'].transform(sent_values)[0]
                        
                        for i, feature_name in enumerate(available_sent_features):
                            raw_features[feature_name] = float(sent_row[feature_name])
                            norm_sentiment_features[feature_name] = float(normalized_sent[i])
                
                # Create normalized features object
                normalized_features = NormalizedFeatures(
                    timestamp=idx,
                    symbol=symbol or 'UNKNOWN',
                    price_features=norm_price_features,
                    technical_features=norm_technical_features,
                    sentiment_features=norm_sentiment_features,
                    raw_features=raw_features
                )
                
                normalized_results.append(normalized_features)
                
            except Exception as e:
                logger.error(f"Error processing features for {idx}: {e}")
                continue
        
        logger.info(f"Transformed {len(normalized_results)} feature vectors")
        return normalized_results
    
    def fit_transform(self, price_data: pd.DataFrame, news_data: pd.DataFrame = None,
                     symbol: str = None) -> List[NormalizedFeatures]:
        """
        Fit the normalizer and transform data in one step.
        
        Args:
            price_data: DataFrame with OHLCV data
            news_data: DataFrame with news data (optional)
            symbol: Symbol identifier (optional)
            
        Returns:
            List of NormalizedFeatures objects
        """
        return self.fit(price_data, news_data, symbol).transform(price_data, news_data, symbol)
    
    def save_scalers(self, filepath: str) -> None:
        """
        Save fitted scalers to disk.
        
        Args:
            filepath: Path to save the scalers
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer")
        
        scaler_data = {
            'scalers': self.scalers,
            'normalization_method': self.normalization_method,
            'feature_names': {
                'price': self.price_feature_names,
                'technical': self.technical_feature_names,
                'sentiment': self.sentiment_feature_names
            },
            'default_values': self.default_values
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str) -> 'FeatureNormalizer':
        """
        Load fitted scalers from disk.
        
        Args:
            filepath: Path to load the scalers from
            
        Returns:
            Self for method chaining
        """
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scalers = scaler_data['scalers']
        self.normalization_method = scaler_data['normalization_method']
        
        if 'feature_names' in scaler_data:
            self.price_feature_names = scaler_data['feature_names']['price']
            self.technical_feature_names = scaler_data['feature_names']['technical']
            self.sentiment_feature_names = scaler_data['feature_names']['sentiment']
        
        if 'default_values' in scaler_data:
            self.default_values = scaler_data['default_values']
        
        self.is_fitted = True
        logger.info(f"Scalers loaded from {filepath}")
        
        return self
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """
        Get feature importance weights based on domain knowledge.
        
        Returns:
            Dictionary mapping feature names to importance weights
        """
        return {
            # Price features - high importance for trend following
            'lag1_return': 0.25,
            'lag2_return': 0.15,
            'volatility': 0.10,
            
            # Technical features - medium to high importance
            'rsi_14': 0.20,
            'sma_5_ratio': 0.15,
            'macd_histogram': 0.10,
            
            # Sentiment features - lower but significant importance
            'compound_score': 0.05
        }
    
    def validate_features(self, features: NormalizedFeatures) -> bool:
        """
        Validate that normalized features are within expected ranges.
        
        Args:
            features: NormalizedFeatures object to validate
            
        Returns:
            True if features are valid, False otherwise
        """
        try:
            # Check for NaN or infinite values
            all_features = {**features.price_features, **features.technical_features, 
                          **features.sentiment_features}
            
            for name, value in all_features.items():
                if not np.isfinite(value):
                    logger.warning(f"Invalid value for {name}: {value}")
                    return False
            
            # Check sentiment is in valid range
            if 'compound_score' in features.sentiment_features:
                sentiment = features.sentiment_features['compound_score']
                if not (-5 <= sentiment <= 5):  # Normalized sentiment might be outside [-1,1]
                    logger.warning(f"Sentiment score outside expected range: {sentiment}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating features: {e}")
            return False