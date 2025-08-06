# nifty_ml_pipeline/features/technical_indicators.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """Container for computed technical indicators."""
    rsi_14: Optional[float] = None
    sma_5: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None


class TechnicalIndicatorCalculator:
    """
    CPU-optimized technical indicator calculator using vectorized pandas operations.
    
    Implements O(n) algorithms for RSI(14), SMA(5), and MACD(12,26) calculations
    with efficient memory usage and linear time complexity.
    """
    
    def __init__(self):
        """Initialize the calculator with default parameters."""
        self.rsi_period = 14
        self.sma_period = 5
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index) using O(n) algorithm.
        
        Args:
            prices: Series of closing prices
            period: RSI period (default 14)
            
        Returns:
            Series of RSI values (0-100)
        """
        if len(prices) < period + 1:
            logger.warning(f"Insufficient data for RSI calculation. Need {period + 1}, got {len(prices)}")
            return pd.Series(index=prices.index, dtype=float)
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Initialize result series with NaN
        rsi = pd.Series(index=prices.index, dtype=float)
        
        # Calculate initial averages using simple mean for first period
        # We need period+1 values to calculate the first RSI (period for average + 1 for the diff)
        if len(prices) >= period + 1:
            # Calculate first average gain and loss
            first_avg_gain = gains.iloc[1:period+1].mean()  # Skip first NaN from diff
            first_avg_loss = losses.iloc[1:period+1].mean()
            
            # Calculate first RSI
            if first_avg_loss != 0:
                rs = first_avg_gain / first_avg_loss
                rsi.iloc[period] = 100 - (100 / (1 + rs))
            else:
                rsi.iloc[period] = 100 if first_avg_gain > 0 else 50
            
            # Use Wilder's smoothing for subsequent values
            alpha = 1.0 / period
            avg_gain = first_avg_gain
            avg_loss = first_avg_loss
            
            for i in range(period + 1, len(prices)):
                avg_gain = alpha * gains.iloc[i] + (1 - alpha) * avg_gain
                avg_loss = alpha * losses.iloc[i] + (1 - alpha) * avg_loss
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi.iloc[i] = 100 - (100 / (1 + rs))
                else:
                    rsi.iloc[i] = 100 if avg_gain > 0 else 50
        
        return rsi
    
    def calculate_sma(self, prices: pd.Series, period: int = 5) -> pd.Series:
        """
        Calculate Simple Moving Average using O(n) rolling window.
        
        Args:
            prices: Series of closing prices
            period: SMA period (default 5)
            
        Returns:
            Series of SMA values
        """
        if len(prices) < period:
            logger.warning(f"Insufficient data for SMA calculation. Need {period}, got {len(prices)}")
            return pd.Series(index=prices.index, dtype=float)
        
        return prices.rolling(window=period, min_periods=period).mean()
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence) with O(n) complexity.
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            Dictionary containing 'macd_line', 'signal_line', and 'histogram'
        """
        if len(prices) < slow:
            logger.warning(f"Insufficient data for MACD calculation. Need {slow}, got {len(prices)}")
            empty_series = pd.Series(index=prices.index, dtype=float)
            return {
                'macd_line': empty_series,
                'signal_line': empty_series,
                'histogram': empty_series
            }
        
        # Calculate EMAs using pandas ewm for efficiency
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = ema_fast - ema_slow
        
        # Signal line = EMA of MACD line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # MACD histogram = MACD line - Signal line
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
    
    def calculate_all_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame of price data.
        
        Args:
            price_data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                       and datetime index
            
        Returns:
            DataFrame with original data plus technical indicators
        """
        if price_data.empty:
            logger.warning("Empty price data provided")
            return price_data
        
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")
        
        result = price_data.copy()
        close_prices = price_data['close']
        
        # Calculate RSI
        try:
            result['rsi_14'] = self.calculate_rsi(close_prices, self.rsi_period)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            result['rsi_14'] = np.nan
        
        # Calculate SMA
        try:
            result['sma_5'] = self.calculate_sma(close_prices, self.sma_period)
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            result['sma_5'] = np.nan
        
        # Calculate MACD
        try:
            macd_results = self.calculate_macd(close_prices, self.macd_fast, self.macd_slow, self.macd_signal)
            result['macd_line'] = macd_results['macd_line']
            result['macd_signal'] = macd_results['signal_line']
            result['macd_histogram'] = macd_results['histogram']
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            result['macd_line'] = np.nan
            result['macd_signal'] = np.nan
            result['macd_histogram'] = np.nan
        
        return result
    
    def get_latest_indicators(self, price_data: pd.DataFrame) -> TechnicalIndicators:
        """
        Get the most recent technical indicators from price data.
        
        Args:
            price_data: DataFrame with price data and calculated indicators
            
        Returns:
            TechnicalIndicators object with latest values
        """
        if price_data.empty:
            return TechnicalIndicators()
        
        # Calculate indicators if not present
        if 'rsi_14' not in price_data.columns:
            price_data = self.calculate_all_indicators(price_data)
        
        latest_row = price_data.iloc[-1]
        
        return TechnicalIndicators(
            rsi_14=latest_row.get('rsi_14'),
            sma_5=latest_row.get('sma_5'),
            macd_line=latest_row.get('macd_line'),
            macd_signal=latest_row.get('macd_signal'),
            macd_histogram=latest_row.get('macd_histogram')
        )
    
    def validate_indicators(self, indicators: TechnicalIndicators) -> bool:
        """
        Validate that calculated indicators are within expected ranges.
        
        Args:
            indicators: TechnicalIndicators object to validate
            
        Returns:
            True if all indicators are valid, False otherwise
        """
        # RSI should be between 0 and 100
        if indicators.rsi_14 is not None:
            if not (0 <= indicators.rsi_14 <= 100):
                logger.warning(f"RSI out of range: {indicators.rsi_14}")
                return False
        
        # SMA should be positive
        if indicators.sma_5 is not None:
            if indicators.sma_5 <= 0:
                logger.warning(f"SMA is not positive: {indicators.sma_5}")
                return False
        
        # Check for NaN values
        for field_name, value in indicators.__dict__.items():
            if value is not None and pd.isna(value):
                logger.warning(f"NaN value found in {field_name}")
                return False
        
        return True