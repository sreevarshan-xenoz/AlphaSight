# tests/test_data_models.py
import unittest
from datetime import datetime
import pandas as pd
from nifty_ml_pipeline.data.models import PriceData, NewsData, FeatureVector, PredictionResult
from nifty_ml_pipeline.data.validator import DataValidator


class TestPriceData(unittest.TestCase):
    """Test cases for PriceData model."""

    def test_price_data_valid(self):
        """Test valid price data creation."""
        price = PriceData(
            symbol="NIFTY 50",
            timestamp=datetime(2025, 8, 4),
            open=24500.0,
            high=24750.0,
            low=24450.0,
            close=24722.75,
            volume=1_000_000_000
        )
        self.assertEqual(price.symbol, "NIFTY 50")
        self.assertEqual(price.close, 24722.75)
        self.assertTrue(price.validate())

    def test_price_data_invalid_high_low(self):
        """Test price data with invalid high/low values."""
        with self.assertRaises(ValueError):
            PriceData(
                symbol="NIFTY 50",
                timestamp=datetime(2025, 8, 4),
                open=25000.0,
                high=24900.0,  # High < open
                low=24450.0,
                close=24722.75,
                volume=1_000_000_000
            )

    def test_price_data_negative_volume(self):
        """Test price data with negative volume."""
        with self.assertRaises(ValueError):
            PriceData(
                symbol="NIFTY 50",
                timestamp=datetime(2025, 8, 4),
                open=24500.0,
                high=24750.0,
                low=24450.0,
                close=24722.75,
                volume=-1000
            )

    def test_price_data_zero_prices(self):
        """Test price data with zero prices."""
        with self.assertRaises(ValueError):
            PriceData(
                symbol="NIFTY 50",
                timestamp=datetime(2025, 8, 4),
                open=0.0,  # Invalid zero price
                high=24750.0,
                low=24450.0,
                close=24722.75,
                volume=1_000_000_000
            )


class TestNewsData(unittest.TestCase):
    """Test cases for NewsData model."""

    def test_news_data_valid(self):
        """Test valid news data creation."""
        news = NewsData(
            source="Economic Times",
            timestamp=datetime(2025, 8, 4),
            headline="Markets surge on positive sentiment",
            url="https://example.com/news",
            sentiment_score=0.75
        )
        self.assertEqual(news.source, "Economic Times")
        self.assertEqual(news.sentiment_score, 0.75)

    def test_news_data_empty_headline(self):
        """Test news data with empty headline."""
        with self.assertRaises(ValueError):
            NewsData(
                source="Economic Times",
                timestamp=datetime(2025, 8, 4),
                headline="   ",  # Empty headline
                sentiment_score=0.5
            )

    def test_news_data_invalid_sentiment_range(self):
        """Test news data with sentiment score out of range."""
        with self.assertRaises(ValueError):
            NewsData(
                source="Economic Times",
                timestamp=datetime(2025, 8, 4),
                headline="Markets up",
                sentiment_score=1.5  # Out of [-1, 1] range
            )

    def test_news_data_is_recent(self):
        """Test news recency check."""
        recent_news = NewsData(
            source="Economic Times",
            timestamp=datetime.now(),
            headline="Recent news"
        )
        self.assertTrue(recent_news.is_recent(30))


class TestFeatureVector(unittest.TestCase):
    """Test cases for FeatureVector model."""

    def test_feature_vector_creation(self):
        """Test valid feature vector creation."""
        fv = FeatureVector(
            timestamp=datetime(2025, 8, 4),
            symbol="NIFTY 50",
            lag1_return=0.01,
            lag2_return=-0.005,
            sma_5_ratio=1.002,
            rsi_14=62.0,
            macd_hist=5.3,
            daily_sentiment=0.34
        )
        self.assertEqual(fv.symbol, "NIFTY 50")
        self.assertEqual(fv.lag1_return, 0.01)

    def test_feature_vector_to_series(self):
        """Test conversion to pandas Series."""
        fv = FeatureVector(
            timestamp=datetime(2025, 8, 4),
            symbol="NIFTY 50",
            lag1_return=0.01,
            lag2_return=-0.005,
            sma_5_ratio=1.002,
            rsi_14=62.0,
            macd_hist=5.3,
            daily_sentiment=0.34
        )
        series = fv.to_series()
        self.assertEqual(len(series), 6)  # Excludes timestamp and symbol
        self.assertEqual(series["lag1_return"], 0.01)
        self.assertEqual(series["rsi_14"], 62.0)

    def test_feature_vector_to_array(self):
        """Test conversion to array."""
        fv = FeatureVector(
            timestamp=datetime(2025, 8, 4),
            symbol="NIFTY 50",
            lag1_return=0.01,
            lag2_return=-0.005,
            sma_5_ratio=1.002,
            rsi_14=62.0,
            macd_hist=5.3,
            daily_sentiment=0.34
        )
        array = fv.to_array()
        self.assertEqual(len(array), 6)
        self.assertEqual(array[0], 0.01)  # lag1_return
        self.assertEqual(array[3], 62.0)  # rsi_14


class TestPredictionResult(unittest.TestCase):
    """Test cases for PredictionResult model."""

    def test_prediction_result_valid(self):
        """Test valid prediction result creation."""
        pr = PredictionResult(
            timestamp=datetime(2025, 8, 4),
            symbol="NIFTY 50",
            predicted_close=24900.0,
            signal="Buy",
            confidence=0.85,
            model_version="xgb-v1",
            features_used=["lag1_return", "rsi_14"]
        )
        self.assertEqual(pr.signal, "Buy")
        self.assertEqual(pr.confidence, 0.85)
        self.assertTrue(pr.is_actionable(0.7))

    def test_prediction_result_invalid_signal(self):
        """Test prediction result with invalid signal."""
        with self.assertRaises(ValueError):
            PredictionResult(
                timestamp=datetime(2025, 8, 4),
                symbol="NIFTY 50",
                predicted_close=24900.0,
                signal="Invalid",  # Invalid signal
                confidence=0.85,
                model_version="xgb-v1"
            )

    def test_prediction_result_invalid_confidence(self):
        """Test prediction result with invalid confidence."""
        with self.assertRaises(ValueError):
            PredictionResult(
                timestamp=datetime(2025, 8, 4),
                symbol="NIFTY 50",
                predicted_close=24900.0,
                signal="Buy",
                confidence=1.5,  # Invalid confidence > 1.0
                model_version="xgb-v1"
            )

    def test_prediction_result_to_dict(self):
        """Test conversion to dictionary."""
        pr = PredictionResult(
            timestamp=datetime(2025, 8, 4),
            symbol="NIFTY 50",
            predicted_close=24900.0,
            signal="Buy",
            confidence=0.85,
            model_version="xgb-v1"
        )
        result_dict = pr.to_dict()
        self.assertEqual(result_dict["signal"], "Buy")
        self.assertEqual(result_dict["confidence"], 0.85)
        self.assertIn("timestamp", result_dict)


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""

    def test_validate_price_data_chronological(self):
        """Test chronological validation of price data."""
        prices = [
            PriceData("NIFTY", datetime(2025, 8, 3), 24500, 24600, 24400, 24550, 900_000_000),
            PriceData("NIFTY", datetime(2025, 8, 4), 24550, 24750, 24500, 24722, 1_000_000_000),
        ]
        self.assertTrue(DataValidator.validate_price_data(prices))

    def test_validate_price_data_chronological_failure(self):
        """Test chronological validation failure."""
        prices = [
            PriceData("NIFTY", datetime(2025, 8, 4), 24722, 24750, 24500, 24722, 1_000_000_000),
            PriceData("NIFTY", datetime(2025, 8, 3), 24550, 24600, 24400, 24550, 900_000_000),
        ]
        with self.assertRaises(ValueError):
            DataValidator.validate_price_data(prices)

    def test_validate_empty_price_data(self):
        """Test validation of empty price data."""
        with self.assertRaises(ValueError):
            DataValidator.validate_price_data([])

    def test_validate_news_data_valid(self):
        """Test validation of valid news data."""
        news = [
            NewsData("ET", datetime(2025, 8, 3), "Markets rise"),
            NewsData("ET", datetime(2025, 8, 4), "Positive sentiment"),
        ]
        self.assertTrue(DataValidator.validate_news_data(news))

    def test_validate_news_data_empty_headlines(self):
        """Test validation failure with empty headlines."""
        # Empty headline should be caught at NewsData creation
        with self.assertRaises(ValueError):
            NewsData("ET", datetime(2025, 8, 4), "   ")  # Empty headline

    def test_validate_chronological_integrity_dataframe(self):
        """Test DataFrame chronological integrity validation."""
        df = pd.DataFrame({
            'timestamp': [datetime(2025, 8, 3), datetime(2025, 8, 4)],
            'value': [100, 200]
        })
        self.assertTrue(DataValidator.validate_chronological_integrity(df))

    def test_validate_data_completeness(self):
        """Test data completeness validation."""
        df = pd.DataFrame({
            'timestamp': [datetime(2025, 8, 3)],
            'open': [24500],
            'close': [24600]
        })
        required_cols = ['timestamp', 'open', 'close']
        self.assertTrue(DataValidator.validate_data_completeness(df, required_cols))

    def test_validate_data_completeness_failure(self):
        """Test data completeness validation failure."""
        df = pd.DataFrame({
            'timestamp': [datetime(2025, 8, 3)],
            'open': [24500]
        })
        required_cols = ['timestamp', 'open', 'close']
        with self.assertRaises(ValueError):
            DataValidator.validate_data_completeness(df, required_cols)

    def test_detect_look_ahead_bias(self):
        """Test look-ahead bias detection."""
        prices = [
            PriceData("NIFTY", datetime(2025, 8, 3), 24500, 24600, 24400, 24550, 900_000_000),
            PriceData("NIFTY", datetime(2025, 8, 4), 24550, 24750, 24500, 24722, 1_000_000_000),
        ]
        news = [
            NewsData("ET", datetime(2025, 8, 3), "Markets expected to rise"),
        ]
        self.assertTrue(DataValidator.detect_look_ahead_bias(prices, news))


if __name__ == '__main__':
    unittest.main()