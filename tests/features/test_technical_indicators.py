# tests/features/test_technical_indicators.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nifty_ml_pipeline.features.technical_indicators import TechnicalIndicatorCalculator, TechnicalIndicators


class TestTechnicalIndicatorCalculator:
    """Test suite for TechnicalIndicatorCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create a TechnicalIndicatorCalculator instance for testing."""
        return TechnicalIndicatorCalculator()
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data for testing."""
        # Create 30 days of sample price data with known patterns
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Create prices with a slight upward trend and some volatility
        base_price = 100.0
        prices = []
        for i in range(30):
            # Add trend and some randomness
            price = base_price + i * 0.5 + np.sin(i * 0.3) * 2
            prices.append(price)
        
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample OHLCV DataFrame for testing."""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        data = []
        base_price = 100.0
        
        for i in range(30):
            close = base_price + i * 0.5 + np.sin(i * 0.3) * 2
            open_price = close + np.random.uniform(-1, 1)
            high = max(open_price, close) + np.random.uniform(0, 2)
            low = min(open_price, close) - np.random.uniform(0, 2)
            volume = np.random.randint(1000000, 5000000)
            
            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    def test_rsi_calculation_basic(self, calculator, sample_prices):
        """Test basic RSI calculation."""
        rsi = calculator.calculate_rsi(sample_prices, period=14)
        
        # RSI should be calculated for periods after the initial 14
        assert len(rsi) == len(sample_prices)
        
        # First 14 values should be NaN
        assert pd.isna(rsi.iloc[:14]).all()
        
        # RSI values should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_rsi_known_values(self, calculator):
        """Test RSI calculation with known values."""
        # Create a simple test case with known RSI values
        # Prices that go up consistently should have high RSI
        upward_prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                                  110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
        
        rsi = calculator.calculate_rsi(upward_prices, period=14)
        
        # RSI should be high (>70) for consistently rising prices
        final_rsi = rsi.iloc[-1]
        assert final_rsi > 70, f"Expected RSI > 70 for rising prices, got {final_rsi}"
    
    def test_rsi_insufficient_data(self, calculator):
        """Test RSI calculation with insufficient data."""
        short_prices = pd.Series([100, 101, 102])  # Only 3 data points
        
        rsi = calculator.calculate_rsi(short_prices, period=14)
        
        # Should return empty series with warning
        assert len(rsi) == len(short_prices)
        assert pd.isna(rsi).all()
    
    def test_sma_calculation_basic(self, calculator, sample_prices):
        """Test basic SMA calculation."""
        sma = calculator.calculate_sma(sample_prices, period=5)
        
        # SMA should be calculated for all periods
        assert len(sma) == len(sample_prices)
        
        # First 4 values should be NaN
        assert pd.isna(sma.iloc[:4]).all()
        
        # SMA values should be positive
        valid_sma = sma.dropna()
        assert (valid_sma > 0).all()
    
    def test_sma_known_values(self, calculator):
        """Test SMA calculation with known values."""
        # Simple test case: [1, 2, 3, 4, 5] should have SMA(5) = 3
        test_prices = pd.Series([1, 2, 3, 4, 5])
        
        sma = calculator.calculate_sma(test_prices, period=5)
        
        # The 5th value should be the average of [1,2,3,4,5] = 3
        assert abs(sma.iloc[-1] - 3.0) < 0.001
    
    def test_sma_insufficient_data(self, calculator):
        """Test SMA calculation with insufficient data."""
        short_prices = pd.Series([100, 101])  # Only 2 data points
        
        sma = calculator.calculate_sma(short_prices, period=5)
        
        # Should return series with NaN values
        assert len(sma) == len(short_prices)
        assert pd.isna(sma).all()
    
    def test_macd_calculation_basic(self, calculator, sample_prices):
        """Test basic MACD calculation."""
        macd_results = calculator.calculate_macd(sample_prices, fast=12, slow=26, signal=9)
        
        # Should return dictionary with three components
        assert 'macd_line' in macd_results
        assert 'signal_line' in macd_results
        assert 'histogram' in macd_results
        
        # All components should have same length as input
        for component in macd_results.values():
            assert len(component) == len(sample_prices)
        
        # MACD histogram should equal MACD line - Signal line
        macd_line = macd_results['macd_line']
        signal_line = macd_results['signal_line']
        histogram = macd_results['histogram']
        
        # Check the relationship (allowing for floating point precision)
        diff = histogram - (macd_line - signal_line)
        assert (abs(diff) < 1e-10).all()
    
    def test_macd_insufficient_data(self, calculator):
        """Test MACD calculation with insufficient data."""
        short_prices = pd.Series([100, 101, 102])  # Only 3 data points
        
        macd_results = calculator.calculate_macd(short_prices, fast=12, slow=26, signal=9)
        
        # Should return empty series with warning
        for component in macd_results.values():
            assert len(component) == len(short_prices)
            assert pd.isna(component).all()
    
    def test_calculate_all_indicators(self, calculator, sample_dataframe):
        """Test calculation of all indicators together."""
        result = calculator.calculate_all_indicators(sample_dataframe)
        
        # Should contain all original columns plus indicators
        expected_columns = ['open', 'high', 'low', 'close', 'volume',
                           'rsi_14', 'sma_5', 'macd_line', 'macd_signal', 'macd_histogram']
        
        for col in expected_columns:
            assert col in result.columns
        
        # Should have same number of rows
        assert len(result) == len(sample_dataframe)
    
    def test_calculate_all_indicators_empty_data(self, calculator):
        """Test calculation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = calculator.calculate_all_indicators(empty_df)
        
        # Should return empty DataFrame
        assert result.empty
    
    def test_calculate_all_indicators_missing_close(self, calculator):
        """Test calculation with missing 'close' column."""
        df_no_close = pd.DataFrame({'open': [100, 101], 'high': [102, 103]})
        
        with pytest.raises(ValueError, match="Price data must contain 'close' column"):
            calculator.calculate_all_indicators(df_no_close)
    
    def test_get_latest_indicators(self, calculator, sample_dataframe):
        """Test getting latest indicator values."""
        # First calculate all indicators
        df_with_indicators = calculator.calculate_all_indicators(sample_dataframe)
        
        latest = calculator.get_latest_indicators(df_with_indicators)
        
        # Should return TechnicalIndicators object
        assert isinstance(latest, TechnicalIndicators)
        
        # Should have values for all indicators
        assert latest.rsi_14 is not None
        assert latest.sma_5 is not None
        assert latest.macd_line is not None
        assert latest.macd_signal is not None
        assert latest.macd_histogram is not None
    
    def test_get_latest_indicators_auto_calculate(self, calculator, sample_dataframe):
        """Test getting latest indicators with auto-calculation."""
        # Don't pre-calculate indicators
        latest = calculator.get_latest_indicators(sample_dataframe)
        
        # Should still return TechnicalIndicators object
        assert isinstance(latest, TechnicalIndicators)
    
    def test_get_latest_indicators_empty_data(self, calculator):
        """Test getting latest indicators with empty data."""
        empty_df = pd.DataFrame()
        
        latest = calculator.get_latest_indicators(empty_df)
        
        # Should return empty TechnicalIndicators
        assert isinstance(latest, TechnicalIndicators)
        assert latest.rsi_14 is None
        assert latest.sma_5 is None
    
    def test_validate_indicators_valid(self, calculator):
        """Test validation of valid indicators."""
        valid_indicators = TechnicalIndicators(
            rsi_14=50.0,
            sma_5=100.0,
            macd_line=1.5,
            macd_signal=1.2,
            macd_histogram=0.3
        )
        
        assert calculator.validate_indicators(valid_indicators) is True
    
    def test_validate_indicators_invalid_rsi(self, calculator):
        """Test validation with invalid RSI."""
        invalid_indicators = TechnicalIndicators(
            rsi_14=150.0,  # RSI should be <= 100
            sma_5=100.0
        )
        
        assert calculator.validate_indicators(invalid_indicators) is False
    
    def test_validate_indicators_negative_sma(self, calculator):
        """Test validation with negative SMA."""
        invalid_indicators = TechnicalIndicators(
            rsi_14=50.0,
            sma_5=-10.0  # SMA should be positive
        )
        
        assert calculator.validate_indicators(invalid_indicators) is False
    
    def test_validate_indicators_nan_values(self, calculator):
        """Test validation with NaN values."""
        invalid_indicators = TechnicalIndicators(
            rsi_14=np.nan,  # NaN values should be invalid
            sma_5=100.0
        )
        
        assert calculator.validate_indicators(invalid_indicators) is False
    
    def test_rsi_edge_cases(self, calculator):
        """Test RSI calculation edge cases."""
        # All prices the same (no change)
        flat_prices = pd.Series([100] * 20)
        rsi = calculator.calculate_rsi(flat_prices, period=14)
        
        # RSI should be around 50 for no change (or NaN due to division by zero)
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            # If RSI is calculated, it should be around 50
            assert abs(valid_rsi.iloc[-1] - 50) < 10 or pd.isna(valid_rsi.iloc[-1])
    
    def test_performance_large_dataset(self, calculator):
        """Test performance with larger dataset."""
        # Create 1000 data points
        large_prices = pd.Series(np.random.randn(1000).cumsum() + 100)
        
        # Should complete quickly (this is more of a smoke test)
        rsi = calculator.calculate_rsi(large_prices)
        sma = calculator.calculate_sma(large_prices)
        macd = calculator.calculate_macd(large_prices)
        
        # Basic validation
        assert len(rsi) == 1000
        assert len(sma) == 1000
        assert len(macd['macd_line']) == 1000


class TestTechnicalIndicators:
    """Test suite for TechnicalIndicators dataclass."""
    
    def test_technical_indicators_creation(self):
        """Test creation of TechnicalIndicators object."""
        indicators = TechnicalIndicators(
            rsi_14=50.0,
            sma_5=100.0,
            macd_line=1.5,
            macd_signal=1.2,
            macd_histogram=0.3
        )
        
        assert indicators.rsi_14 == 50.0
        assert indicators.sma_5 == 100.0
        assert indicators.macd_line == 1.5
        assert indicators.macd_signal == 1.2
        assert indicators.macd_histogram == 0.3
    
    def test_technical_indicators_defaults(self):
        """Test default values for TechnicalIndicators."""
        indicators = TechnicalIndicators()
        
        assert indicators.rsi_14 is None
        assert indicators.sma_5 is None
        assert indicators.macd_line is None
        assert indicators.macd_signal is None
        assert indicators.macd_histogram is None