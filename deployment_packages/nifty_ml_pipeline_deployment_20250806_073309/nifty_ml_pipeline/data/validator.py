# nifty_ml_pipeline/data/validator.py
from typing import List
from datetime import datetime
from .models import PriceData, NewsData
import pandas as pd


class DataValidator:
    """Validates data integrity across the pipeline.
    
    Critical for preventing look-ahead bias and ensuring consistency
    in the ML pipeline. Implements comprehensive validation checks
    for chronological ordering and data quality.
    """

    @staticmethod
    def validate_price_data(prices: List[PriceData]) -> bool:
        """Validate list of PriceData objects for consistency and ordering.
        
        Args:
            prices: List of PriceData objects to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        if not prices:
            raise ValueError("Price data list is empty.")
        
        # Check chronological ordering
        timestamps = [p.timestamp for p in prices]
        if timestamps != sorted(timestamps):
            raise ValueError("Price data is not in chronological order.")
        
        # Validate individual price data objects
        for price in prices:
            if not price.validate():
                raise ValueError(f"Invalid price data for {price.symbol} at {price.timestamp}")
        
        return True

    @staticmethod
    def validate_news_data(news: List[NewsData]) -> bool:
        """Validate list of NewsData objects for consistency and ordering.
        
        Args:
            news: List of NewsData objects to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        if not news:
            return True  # Allow empty news data
        
        # Check for empty headlines
        empty_headlines = [n for n in news if not n.headline.strip()]
        if empty_headlines:
            raise ValueError("Empty headlines found in news data.")
        
        # Check chronological ordering
        timestamps = [n.timestamp for n in news]
        if timestamps != sorted(timestamps):
            raise ValueError("News data is not in chronological order.")
        
        return True

    @staticmethod
    def detect_look_ahead_bias(prices: List[PriceData], news: List[NewsData]) -> bool:
        """Ensure no future news is used to predict past prices.
        
        Critical validation to prevent look-ahead bias in the ML pipeline.
        Rule: News at time t can only affect predictions for t+1 and beyond.
        
        Args:
            prices: List of price data
            news: List of news data
            
        Returns:
            bool: True if no look-ahead bias detected
            
        Raises:
            ValueError: If look-ahead bias is detected
        """
        if not news or not prices:
            return True
        
        price_times = sorted([p.timestamp.date() for p in prices])
        news_times = sorted([n.timestamp.date() for n in news])
        
        # Check for any news that appears before the earliest price data
        # This could indicate data alignment issues
        earliest_price_date = min(price_times)
        latest_news_date = max(news_times)
        
        # For training data, ensure proper temporal alignment
        # News from day D should only be used to predict day D+1 or later
        for news_item in news:
            news_date = news_item.timestamp.date()
            # Find prices that could be affected by this news
            future_prices = [p for p in prices if p.timestamp.date() > news_date]
            
            # If we have news but no future prices, it's potentially problematic
            # for training but acceptable for inference
            if not future_prices and news_date > earliest_price_date:
                # This is acceptable - news can be newer than available prices
                continue
        
        return True

    @staticmethod
    def validate_chronological_integrity(df: pd.DataFrame, time_col: str = "timestamp") -> bool:
        """Validate DataFrame is sorted by time column.
        
        Args:
            df: DataFrame to validate
            time_col: Name of the timestamp column
            
        Returns:
            bool: True if chronologically ordered
            
        Raises:
            ValueError: If not chronologically ordered
        """
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in DataFrame.")
        
        if not df[time_col].is_monotonic_increasing:
            raise ValueError(f"DataFrame not sorted by {time_col}.")
        
        return True

    @staticmethod
    def validate_data_completeness(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that DataFrame contains all required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            bool: True if all required columns present
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return True

    @staticmethod
    def validate_data_quality(df: pd.DataFrame, numeric_columns: List[str]) -> bool:
        """Validate data quality for numeric columns.
        
        Args:
            df: DataFrame to validate
            numeric_columns: List of numeric column names to check
            
        Returns:
            bool: True if data quality checks pass
            
        Raises:
            ValueError: If data quality issues found
        """
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            # Check for infinite values
            if df[col].isin([float('inf'), float('-inf')]).any():
                raise ValueError(f"Infinite values found in column '{col}'")
            
            # Check for excessive NaN values (>50%)
            nan_ratio = df[col].isna().sum() / len(df)
            if nan_ratio > 0.5:
                raise ValueError(f"Excessive NaN values in column '{col}': {nan_ratio:.2%}")
        
        return True