# tests/features/test_sentiment_analysis.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from nifty_ml_pipeline.features.sentiment_analysis import SentimentAnalyzer, SentimentScore


class TestSentimentScore:
    """Test suite for SentimentScore dataclass."""
    
    def test_sentiment_score_creation(self):
        """Test creation of SentimentScore object."""
        score = SentimentScore(
            compound=0.5,
            positive=0.7,
            negative=0.1,
            neutral=0.2,
            processing_time_ms=5.0
        )
        
        assert score.compound == 0.5
        assert score.positive == 0.7
        assert score.negative == 0.1
        assert score.neutral == 0.2
        assert score.processing_time_ms == 5.0
    
    def test_sentiment_score_defaults(self):
        """Test default values for SentimentScore."""
        score = SentimentScore(
            compound=0.0,
            positive=0.0,
            negative=0.0,
            neutral=1.0
        )
        
        assert score.processing_time_ms == 0.0


class TestSentimentAnalyzer:
    """Test suite for SentimentAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a SentimentAnalyzer instance for testing."""
        # Don't download NLTK data during testing
        return SentimentAnalyzer(download_nltk_data=False)
    
    @pytest.fixture
    def sample_headlines(self):
        """Create sample financial headlines for testing."""
        return [
            "NIFTY 50 surges to new all-time high on strong earnings",
            "Market crashes as inflation fears grip investors",
            "Reliance Industries reports steady quarterly results",
            "Banking stocks rally on positive RBI policy",
            "Tech stocks plummet amid global sell-off"
        ]
    
    @pytest.fixture
    def sample_news_dataframe(self):
        """Create sample news DataFrame for testing."""
        data = {
            'timestamp': [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 1, 1, 14, 0),
                datetime(2024, 1, 2, 9, 0),
                datetime(2024, 1, 2, 15, 0),
                datetime(2024, 1, 3, 11, 0)
            ],
            'headline': [
                "NIFTY 50 surges to new all-time high on strong earnings",
                "Market crashes as inflation fears grip investors",
                "Reliance Industries reports steady quarterly results",
                "Banking stocks rally on positive RBI policy",
                "Tech stocks plummet amid global sell-off"
            ],
            'source': ['ET', 'ET', 'ET', 'ET', 'ET']
        }
        return pd.DataFrame(data)
    
    def test_analyzer_initialization(self, analyzer):
        """Test SentimentAnalyzer initialization."""
        assert analyzer.analyzer is not None
        assert analyzer.total_sentences_processed == 0
        assert analyzer.total_processing_time_ms == 0.0
        assert 'positive' in analyzer.financial_keywords
        assert 'negative' in analyzer.financial_keywords
    
    def test_preprocess_headline_basic(self, analyzer):
        """Test basic headline preprocessing."""
        headline = "  NIFTY 50   surges   to new high  "
        processed = analyzer._preprocess_headline(headline)
        
        assert processed == "NIFTY 50 surges to new high"
    
    def test_preprocess_headline_special_chars(self, analyzer):
        """Test preprocessing with special characters."""
        headline = "Market @#$% crashes!!! What's next???"
        processed = analyzer._preprocess_headline(headline)
        
        # Should remove special chars but keep basic punctuation
        assert "@#$%" not in processed
        assert "crashes" in processed
    
    def test_preprocess_headline_empty(self, analyzer):
        """Test preprocessing with empty or invalid input."""
        assert analyzer._preprocess_headline("") == ""
        assert analyzer._preprocess_headline(None) == ""
        assert analyzer._preprocess_headline(123) == ""
    
    def test_analyze_headline_positive(self, analyzer):
        """Test sentiment analysis of positive headline."""
        headline = "NIFTY 50 surges to new all-time high on strong earnings"
        
        score = analyzer.analyze_headline(headline)
        
        assert isinstance(score, SentimentScore)
        assert score.compound > 0  # Should be positive
        assert score.positive > 0
        assert -1 <= score.compound <= 1
        assert 0 <= score.positive <= 1
        assert 0 <= score.negative <= 1
        assert 0 <= score.neutral <= 1
        assert score.processing_time_ms >= 0
    
    def test_analyze_headline_negative(self, analyzer):
        """Test sentiment analysis of negative headline."""
        headline = "Market crashes as inflation fears grip investors"
        
        score = analyzer.analyze_headline(headline)
        
        assert isinstance(score, SentimentScore)
        assert score.compound < 0  # Should be negative
        assert score.negative > 0
        assert -1 <= score.compound <= 1
    
    def test_analyze_headline_neutral(self, analyzer):
        """Test sentiment analysis of neutral headline."""
        headline = "Company reports quarterly results"
        
        score = analyzer.analyze_headline(headline)
        
        assert isinstance(score, SentimentScore)
        assert -0.5 <= score.compound <= 0.5  # Should be relatively neutral
        assert -1 <= score.compound <= 1
    
    def test_analyze_headline_empty(self, analyzer):
        """Test sentiment analysis of empty headline."""
        score = analyzer.analyze_headline("")
        
        assert score.compound == 0.0
        assert score.positive == 0.0
        assert score.negative == 0.0
        assert score.neutral == 1.0
        assert score.processing_time_ms == 0.0
    
    def test_analyze_headlines_batch(self, analyzer, sample_headlines):
        """Test batch processing of headlines."""
        scores = analyzer.analyze_headlines_batch(sample_headlines)
        
        assert len(scores) == len(sample_headlines)
        assert all(isinstance(score, SentimentScore) for score in scores)
        
        # Check that different headlines have different sentiments
        compound_scores = [score.compound for score in scores]
        assert len(set(compound_scores)) > 1  # Should have variety
    
    def test_analyze_headlines_batch_empty(self, analyzer):
        """Test batch processing with empty list."""
        scores = analyzer.analyze_headlines_batch([])
        
        assert scores == []
    
    def test_analyze_headlines_batch_performance_tracking(self, analyzer, sample_headlines):
        """Test that batch processing updates performance stats."""
        initial_count = analyzer.total_sentences_processed
        
        analyzer.analyze_headlines_batch(sample_headlines)
        
        assert analyzer.total_sentences_processed == initial_count + len(sample_headlines)
        assert analyzer.total_processing_time_ms > 0
    
    def test_aggregate_ticker_sentiment_basic(self, analyzer, sample_news_dataframe):
        """Test basic ticker sentiment aggregation."""
        metrics = analyzer.aggregate_ticker_sentiment(sample_news_dataframe)
        
        expected_keys = ['mean_compound', 'mean_positive', 'mean_negative', 
                        'mean_neutral', 'headline_count', 'sentiment_volatility']
        
        for key in expected_keys:
            assert key in metrics
        
        assert metrics['headline_count'] == len(sample_news_dataframe)
        assert -1 <= metrics['mean_compound'] <= 1
        assert 0 <= metrics['mean_positive'] <= 1
        assert 0 <= metrics['mean_negative'] <= 1
        assert 0 <= metrics['mean_neutral'] <= 1
        assert metrics['sentiment_volatility'] >= 0
    
    def test_aggregate_ticker_sentiment_empty_data(self, analyzer):
        """Test aggregation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        metrics = analyzer.aggregate_ticker_sentiment(empty_df)
        
        assert metrics['mean_compound'] == 0.0
        assert metrics['headline_count'] == 0
        assert metrics['sentiment_volatility'] == 0.0
    
    def test_aggregate_ticker_sentiment_with_ticker_filter(self, analyzer):
        """Test aggregation with ticker filtering."""
        # Create DataFrame with ticker column
        data = {
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            'headline': ['RELIANCE stock surges', 'TCS reports growth'],
            'ticker': ['RELIANCE', 'TCS']
        }
        df = pd.DataFrame(data)
        
        metrics = analyzer.aggregate_ticker_sentiment(df, ticker='RELIANCE')
        
        assert metrics['headline_count'] == 1
    
    def test_aggregate_ticker_sentiment_headline_filter(self, analyzer, sample_news_dataframe):
        """Test aggregation with ticker mentioned in headlines."""
        # Filter for headlines mentioning "NIFTY"
        metrics = analyzer.aggregate_ticker_sentiment(sample_news_dataframe, ticker='NIFTY')
        
        # Should find at least one headline mentioning NIFTY
        assert metrics['headline_count'] >= 1
    
    def test_get_daily_sentiment_basic(self, analyzer, sample_news_dataframe):
        """Test getting daily sentiment."""
        target_date = datetime(2024, 1, 1)
        
        sentiment = analyzer.get_daily_sentiment(sample_news_dataframe, target_date)
        
        assert isinstance(sentiment, float)
        assert -1 <= sentiment <= 1
    
    def test_get_daily_sentiment_no_data(self, analyzer, sample_news_dataframe):
        """Test getting daily sentiment for date with no news."""
        target_date = datetime(2024, 1, 10)  # Date not in sample data
        
        sentiment = analyzer.get_daily_sentiment(sample_news_dataframe, target_date)
        
        assert sentiment == 0.0  # Should return neutral
    
    def test_get_daily_sentiment_empty_dataframe(self, analyzer):
        """Test getting daily sentiment with empty DataFrame."""
        empty_df = pd.DataFrame()
        target_date = datetime(2024, 1, 1)
        
        sentiment = analyzer.get_daily_sentiment(empty_df, target_date)
        
        assert sentiment == 0.0
    
    def test_get_daily_sentiment_string_dates(self, analyzer):
        """Test getting daily sentiment with string date column."""
        data = {
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 14:00:00'],
            'headline': ['Positive news', 'Negative news']
        }
        df = pd.DataFrame(data)
        target_date = datetime(2024, 1, 1)
        
        sentiment = analyzer.get_daily_sentiment(df, target_date)
        
        assert isinstance(sentiment, float)
        assert -1 <= sentiment <= 1
    
    def test_get_performance_stats_initial(self, analyzer):
        """Test performance stats before processing any data."""
        stats = analyzer.get_performance_stats()
        
        expected_keys = ['total_sentences', 'total_time_ms', 
                        'avg_time_per_sentence_ms', 'sentences_per_second']
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['total_sentences'] == 0
        assert stats['total_time_ms'] == 0.0
        assert stats['avg_time_per_sentence_ms'] == 0.0
        assert stats['sentences_per_second'] == 0.0
    
    def test_get_performance_stats_after_processing(self, analyzer, sample_headlines):
        """Test performance stats after processing data."""
        analyzer.analyze_headlines_batch(sample_headlines)
        
        stats = analyzer.get_performance_stats()
        
        assert stats['total_sentences'] == len(sample_headlines)
        assert stats['total_time_ms'] > 0
        assert stats['avg_time_per_sentence_ms'] > 0
        assert stats['sentences_per_second'] > 0
    
    def test_validate_performance_target_no_data(self, analyzer):
        """Test performance validation with no processed data."""
        result = analyzer.validate_performance_target(10.0)
        
        assert result is True  # Should pass with no data
    
    def test_validate_performance_target_with_data(self, analyzer, sample_headlines):
        """Test performance validation after processing data."""
        analyzer.analyze_headlines_batch(sample_headlines)
        
        # Test with generous target (should pass)
        result = analyzer.validate_performance_target(100.0)
        assert result is True
        
        # Test with very strict target (might fail)
        result = analyzer.validate_performance_target(0.001)
        # Don't assert the result as it depends on system performance
        assert isinstance(result, bool)
    
    def test_reset_performance_stats(self, analyzer, sample_headlines):
        """Test resetting performance statistics."""
        # Process some data first
        analyzer.analyze_headlines_batch(sample_headlines)
        
        assert analyzer.total_sentences_processed > 0
        assert analyzer.total_processing_time_ms > 0
        
        # Reset stats
        analyzer.reset_performance_stats()
        
        assert analyzer.total_sentences_processed == 0
        assert analyzer.total_processing_time_ms == 0.0
    
    def test_performance_target_requirement(self, analyzer):
        """Test that analyzer meets the 0.01 seconds (10ms) per sentence target."""
        # Test with a single sentence
        headline = "Market shows steady performance today"
        
        score = analyzer.analyze_headline(headline)
        
        # Should process within 10ms target
        assert score.processing_time_ms <= 10.0, \
            f"Processing time {score.processing_time_ms}ms exceeds 10ms target"
    
    def test_compound_score_range(self, analyzer, sample_headlines):
        """Test that all compound scores are within valid range."""
        scores = analyzer.analyze_headlines_batch(sample_headlines)
        
        for score in scores:
            assert -1 <= score.compound <= 1, \
                f"Compound score {score.compound} outside valid range [-1, 1]"
            assert 0 <= score.positive <= 1
            assert 0 <= score.negative <= 1
            assert 0 <= score.neutral <= 1
    
    def test_financial_keywords_impact(self, analyzer):
        """Test that financial keywords appropriately impact sentiment."""
        positive_headline = "Stock surges with strong profit growth"
        negative_headline = "Market crashes with heavy losses"
        
        pos_score = analyzer.analyze_headline(positive_headline)
        neg_score = analyzer.analyze_headline(negative_headline)
        
        # Positive headline should have higher compound score
        assert pos_score.compound > neg_score.compound
        assert pos_score.positive > neg_score.positive
        assert pos_score.negative < neg_score.negative
    
    @patch('nifty_ml_pipeline.features.sentiment_analysis.SentimentIntensityAnalyzer')
    def test_error_handling_in_analysis(self, mock_analyzer_class, analyzer):
        """Test error handling when VADER analysis fails."""
        # Mock the analyzer to raise an exception
        mock_analyzer = MagicMock()
        mock_analyzer.polarity_scores.side_effect = Exception("VADER error")
        analyzer.analyzer = mock_analyzer
        
        score = analyzer.analyze_headline("Test headline")
        
        # Should return neutral score on error
        assert score.compound == 0.0
        assert score.positive == 0.0
        assert score.negative == 0.0
        assert score.neutral == 1.0