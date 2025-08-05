# nifty_ml_pipeline/features/sentiment_analysis.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import time
import re
from datetime import datetime, timedelta

# NLTK and VADER imports
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Container for sentiment analysis results."""
    compound: float  # Overall sentiment score (-1 to +1)
    positive: float  # Positive sentiment component (0 to 1)
    negative: float  # Negative sentiment component (0 to 1)
    neutral: float   # Neutral sentiment component (0 to 1)
    processing_time_ms: float = 0.0


class SentimentAnalyzer:
    """
    High-performance sentiment analyzer using VADER for financial news headlines.
    
    Optimized for 0.01 seconds per sentence processing target with efficient
    text preprocessing and batch processing capabilities.
    """
    
    def __init__(self, download_nltk_data: bool = True):
        """
        Initialize the sentiment analyzer with VADER.
        
        Args:
            download_nltk_data: Whether to download required NLTK data
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self._setup_nltk(download_nltk_data)
        
        # Performance tracking
        self.total_sentences_processed = 0
        self.total_processing_time_ms = 0.0
        
        # Financial keywords that might affect sentiment
        self.financial_keywords = {
            'positive': ['profit', 'gain', 'growth', 'rise', 'surge', 'bull', 'rally', 
                        'increase', 'up', 'high', 'strong', 'beat', 'exceed', 'outperform'],
            'negative': ['loss', 'fall', 'drop', 'decline', 'bear', 'crash', 'down', 
                        'low', 'weak', 'miss', 'underperform', 'cut', 'reduce']
        }
    
    def _setup_nltk(self, download_data: bool) -> None:
        """Setup NLTK data if needed."""
        if download_data:
            try:
                # Download required NLTK data silently
                nltk.download('punkt', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK data: {e}")
    
    def _preprocess_headline(self, headline: str) -> str:
        """
        Preprocess headline for optimal sentiment analysis.
        
        Args:
            headline: Raw headline text
            
        Returns:
            Cleaned headline text
        """
        if not headline or not isinstance(headline, str):
            return ""
        
        # Remove extra whitespace and normalize
        headline = re.sub(r'\s+', ' ', headline.strip())
        
        # Remove special characters that might interfere with sentiment
        headline = re.sub(r'[^\w\s\-\.\,\!\?\:\;]', '', headline)
        
        return headline
    
    def analyze_headline(self, headline: str) -> SentimentScore:
        """
        Analyze sentiment of a single headline with performance tracking.
        
        Args:
            headline: News headline text
            
        Returns:
            SentimentScore object with compound score and components
        """
        start_time = time.perf_counter()
        
        try:
            # Preprocess the headline
            clean_headline = self._preprocess_headline(headline)
            
            if not clean_headline:
                return SentimentScore(
                    compound=0.0,
                    positive=0.0,
                    negative=0.0,
                    neutral=1.0,
                    processing_time_ms=0.0
                )
            
            # Get VADER sentiment scores
            scores = self.analyzer.polarity_scores(clean_headline)
            
            # Calculate processing time
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000
            
            # Update performance tracking
            self.total_sentences_processed += 1
            self.total_processing_time_ms += processing_time_ms
            
            return SentimentScore(
                compound=scores['compound'],
                positive=scores['pos'],
                negative=scores['neg'],
                neutral=scores['neu'],
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Error analyzing headline '{headline}': {e}")
            return SentimentScore(
                compound=0.0,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                processing_time_ms=0.0
            )
    
    def analyze_headlines_batch(self, headlines: List[str]) -> List[SentimentScore]:
        """
        Analyze multiple headlines efficiently in batch.
        
        Args:
            headlines: List of headline strings
            
        Returns:
            List of SentimentScore objects
        """
        if not headlines:
            return []
        
        results = []
        batch_start_time = time.perf_counter()
        
        for headline in headlines:
            score = self.analyze_headline(headline)
            results.append(score)
        
        batch_end_time = time.perf_counter()
        batch_time_ms = (batch_end_time - batch_start_time) * 1000
        
        logger.info(f"Processed {len(headlines)} headlines in {batch_time_ms:.2f}ms "
                   f"({batch_time_ms/len(headlines):.2f}ms per headline)")
        
        return results
    
    def aggregate_ticker_sentiment(self, news_data: pd.DataFrame, 
                                 ticker: str = None,
                                 date_column: str = 'timestamp',
                                 headline_column: str = 'headline') -> Dict[str, float]:
        """
        Aggregate sentiment scores for a specific ticker or all news.
        
        Args:
            news_data: DataFrame with news data
            ticker: Specific ticker to filter for (optional)
            date_column: Name of the date column
            headline_column: Name of the headline column
            
        Returns:
            Dictionary with aggregated sentiment metrics
        """
        if news_data.empty:
            return {
                'mean_compound': 0.0,
                'mean_positive': 0.0,
                'mean_negative': 0.0,
                'mean_neutral': 1.0,
                'headline_count': 0,
                'sentiment_volatility': 0.0
            }
        
        # Filter by ticker if specified
        if ticker:
            # Assuming ticker is mentioned in headlines or there's a ticker column
            if 'ticker' in news_data.columns:
                filtered_data = news_data[news_data['ticker'] == ticker]
            else:
                # Filter by ticker mention in headline
                filtered_data = news_data[
                    news_data[headline_column].str.contains(ticker, case=False, na=False)
                ]
        else:
            filtered_data = news_data
        
        if filtered_data.empty:
            return {
                'mean_compound': 0.0,
                'mean_positive': 0.0,
                'mean_negative': 0.0,
                'mean_neutral': 1.0,
                'headline_count': 0,
                'sentiment_volatility': 0.0
            }
        
        # Analyze all headlines
        headlines = filtered_data[headline_column].tolist()
        sentiment_scores = self.analyze_headlines_batch(headlines)
        
        # Extract compound scores for aggregation
        compound_scores = [score.compound for score in sentiment_scores]
        positive_scores = [score.positive for score in sentiment_scores]
        negative_scores = [score.negative for score in sentiment_scores]
        neutral_scores = [score.neutral for score in sentiment_scores]
        
        # Calculate aggregated metrics
        return {
            'mean_compound': np.mean(compound_scores),
            'mean_positive': np.mean(positive_scores),
            'mean_negative': np.mean(negative_scores),
            'mean_neutral': np.mean(neutral_scores),
            'headline_count': len(compound_scores),
            'sentiment_volatility': np.std(compound_scores) if len(compound_scores) > 1 else 0.0
        }
    
    def get_daily_sentiment(self, news_data: pd.DataFrame,
                           target_date: datetime,
                           ticker: str = None,
                           date_column: str = 'timestamp',
                           headline_column: str = 'headline') -> float:
        """
        Get aggregated sentiment score for a specific date.
        
        Args:
            news_data: DataFrame with news data
            target_date: Date to get sentiment for
            ticker: Specific ticker to filter for (optional)
            date_column: Name of the date column
            headline_column: Name of the headline column
            
        Returns:
            Mean compound sentiment score for the date (-1 to +1)
        """
        if news_data.empty:
            return 0.0
        
        # Filter by date
        if isinstance(news_data[date_column].iloc[0], str):
            # Convert string dates to datetime
            news_data[date_column] = pd.to_datetime(news_data[date_column])
        
        # Filter for the target date
        target_date_str = target_date.strftime('%Y-%m-%d')
        daily_news = news_data[
            news_data[date_column].dt.strftime('%Y-%m-%d') == target_date_str
        ]
        
        if daily_news.empty:
            logger.info(f"No news found for {target_date_str}, returning neutral sentiment")
            return 0.0
        
        # Get aggregated sentiment for the day
        sentiment_metrics = self.aggregate_ticker_sentiment(
            daily_news, ticker, date_column, headline_column
        )
        
        return sentiment_metrics['mean_compound']
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics for the analyzer.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.total_sentences_processed == 0:
            return {
                'total_sentences': 0,
                'total_time_ms': 0.0,
                'avg_time_per_sentence_ms': 0.0,
                'sentences_per_second': 0.0
            }
        
        avg_time_ms = self.total_processing_time_ms / self.total_sentences_processed
        sentences_per_second = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0
        
        return {
            'total_sentences': self.total_sentences_processed,
            'total_time_ms': self.total_processing_time_ms,
            'avg_time_per_sentence_ms': avg_time_ms,
            'sentences_per_second': sentences_per_second
        }
    
    def validate_performance_target(self, target_ms: float = 10.0) -> bool:
        """
        Validate that processing meets performance target.
        
        Args:
            target_ms: Target processing time per sentence in milliseconds
            
        Returns:
            True if performance target is met, False otherwise
        """
        stats = self.get_performance_stats()
        avg_time = stats['avg_time_per_sentence_ms']
        
        if avg_time == 0.0:
            return True  # No data processed yet
        
        meets_target = avg_time <= target_ms
        
        if not meets_target:
            logger.warning(f"Performance target not met: {avg_time:.2f}ms > {target_ms}ms")
        else:
            logger.info(f"Performance target met: {avg_time:.2f}ms <= {target_ms}ms")
        
        return meets_target
    
    def reset_performance_stats(self) -> None:
        """Reset performance tracking statistics."""
        self.total_sentences_processed = 0
        self.total_processing_time_ms = 0.0