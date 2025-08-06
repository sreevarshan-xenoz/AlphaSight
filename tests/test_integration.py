"""
Integration tests for end-to-end pipeline validation.

This module contains integration tests that validate the complete pipeline
functionality with realistic data flows and component interactions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from nifty_ml_pipeline.orchestration.controller import PipelineController
from nifty_ml_pipeline.data.models import PriceData, NewsData, FeatureVector, PredictionResult
from nifty_ml_pipeline.data.collectors import NSEDataCollector, NewsDataCollector
from nifty_ml_pipeline.features.technical_indicators import TechnicalIndicatorCalculator
from nifty_ml_pipeline.features.sentiment_analysis import SentimentAnalyzer
from nifty_ml_pipeline.features.feature_normalizer import FeatureNormalizer
from nifty_ml_pipeline.models.predictor import XGBoostPredictor
from nifty_ml_pipeline.models.inference_engine import InferenceEngine


class TestEndToEndIntegration:
    """Integration tests for complete pipeline execution."""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration for integration tests."""
        return {
            'api': {
                'keys': {
                    'ECONOMIC_TIMES_API_KEY': 'test_key'
                }
            },
            'data': {
                'retention_days': 365,
                'storage_format': 'parquet'
            },
            'paths': {
                'data': tempfile.mkdtemp()
            },
            'performance': {
                'MAX_INFERENCE_LATENCY_MS': 10.0
            }
        }
    
    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for integration tests."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        prices = []
        base_price = 100.0
        
        for i, date in enumerate(dates):
            # Simulate realistic price movements
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            base_price *= (1 + change)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.normal(1000000, 200000))
            
            prices.append({
                'timestamp': date,
                'open': base_price,
                'high': max(base_price, high),
                'low': min(base_price, low),
                'close': base_price,
                'volume': max(volume, 100000)
            })
        
        return pd.DataFrame(prices)
    
    @pytest.fixture
    def sample_news_data(self):
        """Sample news data for integration tests."""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        headlines = [
            "Market shows strong bullish momentum",
            "Economic indicators point to growth",
            "Concerns over inflation impact markets",
            "Technology sector leads market gains",
            "Banking stocks show mixed performance"
        ]
        
        news_data = []
        for i, date in enumerate(dates):
            headline = headlines[i % len(headlines)]
            news_data.append({
                'headline': f"{headline} - Day {i+1}",
                'timestamp': date,
                'source': 'test_source',
                'url': f'http://test.com/{i}'
            })
        
        return pd.DataFrame(news_data)
    
    def test_complete_pipeline_with_real_components(self, integration_config, 
                                                  sample_price_data, sample_news_data):
        """Test complete pipeline execution with real component instances."""
        # Use real components instead of mocks for integration testing
        with patch('nifty_ml_pipeline.data.collectors.NSEDataCollector') as mock_nse, \
             patch('nifty_ml_pipeline.data.collectors.NewsDataCollector') as mock_news, \
             patch('nifty_ml_pipeline.data.storage.DataStorage') as mock_storage:
            
            # Configure mocks to return sample data
            mock_nse.return_value.collect_data.return_value = sample_price_data
            mock_news.return_value.collect_data.return_value = sample_news_data
            mock_storage.return_value.store_price_data.return_value = None
            mock_storage.return_value.store_news_data.return_value = None
            mock_storage.return_value.store_predictions.return_value = None
            
            # Execute pipeline
            controller = PipelineController(integration_config)
            result = controller.execute_pipeline("NIFTY50")
            
            # Verify successful execution
            assert result.was_successful()
            assert len(result.stage_results) >= 3  # At least data, features, inference
            assert len(result.predictions) > 0
            
            # Verify stage completion
            stage_names = [stage['stage'] for stage in result.stage_results]
            assert 'data_collection' in stage_names
            assert 'feature_engineering' in stage_names
            assert 'model_inference' in stage_names
    
    def test_pipeline_data_flow_integrity(self, integration_config,
                                        sample_price_data, sample_news_data):
        """Test that data flows correctly through pipeline stages."""
        with patch('nifty_ml_pipeline.data.collectors.NSEDataCollector') as mock_nse, \
             patch('nifty_ml_pipeline.data.collectors.NewsDataCollector') as mock_news, \
             patch('nifty_ml_pipeline.data.storage.DataStorage') as mock_storage:
            
            mock_nse.return_value.collect_data.return_value = sample_price_data
            mock_news.return_value.collect_data.return_value = sample_news_data
            mock_storage.return_value.store_price_data.return_value = None
            mock_storage.return_value.store_news_data.return_value = None
            mock_storage.return_value.store_predictions.return_value = None
            
            controller = PipelineController(integration_config)
            result = controller.execute_pipeline("NIFTY50")
            
            # Verify data integrity through stages
            assert result.was_successful()
            
            # Check that each stage processed reasonable amounts of data
            for stage in result.stage_results:
                if stage['stage'] == 'data_collection':
                    assert stage['data_count'] > 0
                    assert 'price_records' in stage['metadata']
                    assert 'news_records' in stage['metadata']
                elif stage['stage'] == 'feature_engineering':
                    assert stage['data_count'] > 0
                    assert 'features_created' in stage['metadata']
                elif stage['stage'] == 'model_inference':
                    assert stage['data_count'] > 0
                    assert 'predictions_generated' in stage['metadata']
    
    def test_pipeline_error_recovery(self, integration_config):
        """Test pipeline error handling and recovery mechanisms."""
        with patch('nifty_ml_pipeline.data.collectors.NSEDataCollector') as mock_nse, \
             patch('nifty_ml_pipeline.data.collectors.NewsDataCollector') as mock_news, \
             patch('nifty_ml_pipeline.data.storage.DataStorage') as mock_storage:
            
            # Simulate data collection failure
            mock_nse.return_value.collect_data.side_effect = Exception("API Error")
            mock_news.return_value.collect_data.return_value = pd.DataFrame()
            mock_storage.return_value.store_price_data.return_value = None
            mock_storage.return_value.store_news_data.return_value = None
            
            controller = PipelineController(integration_config)
            result = controller.execute_pipeline("NIFTY50")
            
            # Verify graceful failure handling
            assert not result.was_successful()
            assert result.error_message != ""
            assert len(result.stage_results) > 0  # Should have attempted at least one stage
    
    def test_pipeline_performance_tracking(self, integration_config,
                                         sample_price_data, sample_news_data):
        """Test that pipeline tracks performance metrics correctly."""
        with patch('nifty_ml_pipeline.data.collectors.NSEDataCollector') as mock_nse, \
             patch('nifty_ml_pipeline.data.collectors.NewsDataCollector') as mock_news, \
             patch('nifty_ml_pipeline.data.storage.DataStorage') as mock_storage:
            
            mock_nse.return_value.collect_data.return_value = sample_price_data
            mock_news.return_value.collect_data.return_value = sample_news_data
            mock_storage.return_value.store_price_data.return_value = None
            mock_storage.return_value.store_news_data.return_value = None
            mock_storage.return_value.store_predictions.return_value = None
            
            controller = PipelineController(integration_config)
            result = controller.execute_pipeline("NIFTY50")
            
            # Verify performance metrics are tracked
            assert result.was_successful()
            assert result.total_duration_ms > 0
            
            for stage in result.stage_results:
                assert stage['duration_ms'] >= 0
                assert 'metadata' in stage
                
                # Check stage-specific performance metrics
                if stage['stage'] == 'model_inference':
                    assert 'meets_latency_target' in stage['metadata']


class TestComponentIntegration:
    """Integration tests for component interactions."""
    
    def test_data_collector_to_feature_engineering_flow(self):
        """Test data flow from collectors to feature engineering."""
        # Create sample data
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
            'open': np.random.uniform(95, 105, 30),
            'high': np.random.uniform(100, 110, 30),
            'low': np.random.uniform(90, 100, 30),
            'close': np.random.uniform(95, 105, 30),
            'volume': np.random.randint(500000, 2000000, 30)
        })
        
        news_data = pd.DataFrame({
            'headline': [f"Market news {i}" for i in range(10)],
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='3D'),
            'source': ['test_source'] * 10,
            'url': [f'http://test.com/{i}' for i in range(10)]
        })
        
        # Test technical indicators calculation
        tech_calc = TechnicalIndicatorCalculator()
        price_with_indicators = tech_calc.calculate_all_indicators(price_data)
        
        # Verify indicators were added
        assert 'rsi_14' in price_with_indicators.columns
        assert 'sma_5' in price_with_indicators.columns
        assert 'macd_hist' in price_with_indicators.columns
        
        # Test sentiment analysis
        sentiment_analyzer = SentimentAnalyzer()
        news_with_sentiment = sentiment_analyzer.analyze_dataframe(news_data)
        
        # Verify sentiment scores were added
        assert 'sentiment_score' in news_with_sentiment.columns
        assert not news_with_sentiment['sentiment_score'].isna().all()
        
        # Test feature normalization
        normalizer = FeatureNormalizer()
        feature_vectors = normalizer.create_feature_vectors(
            price_with_indicators, news_with_sentiment
        )
        
        # Verify feature vectors were created
        assert len(feature_vectors) > 0
        assert 'lag1_return' in feature_vectors.columns
        assert 'daily_sentiment' in feature_vectors.columns
    
    def test_feature_engineering_to_model_flow(self):
        """Test data flow from feature engineering to model inference."""
        # Create sample feature data
        feature_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=20, freq='D'),
            'lag1_return': np.random.normal(0, 0.02, 20),
            'lag2_return': np.random.normal(0, 0.02, 20),
            'sma_5_ratio': np.random.uniform(0.95, 1.05, 20),
            'rsi_14': np.random.uniform(30, 70, 20),
            'macd_hist': np.random.normal(0, 0.5, 20),
            'daily_sentiment': np.random.uniform(-0.5, 0.5, 20)
        })
        
        # Test model prediction flow
        with patch('xgboost.XGBRegressor') as mock_xgb:
            # Mock XGBoost model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.02])  # 2% predicted return
            mock_xgb.return_value = mock_model
            
            # Create predictor and inference engine
            predictor = XGBoostPredictor()
            inference_engine = InferenceEngine(predictor)
            
            # Test single prediction
            latest_features = feature_data.iloc[-1:].drop('timestamp', axis=1)
            prediction = inference_engine.predict_single(latest_features)
            
            # Verify prediction structure
            assert hasattr(prediction, 'predicted_direction')
            assert hasattr(prediction, 'confidence')
            assert hasattr(prediction, 'timestamp')
            assert prediction.predicted_direction in ['Buy', 'Sell', 'Hold']
            assert 0 <= prediction.confidence <= 1


class TestMockDataGenerators:
    """Test utilities for generating consistent mock data."""
    
    def test_price_data_generator(self):
        """Test price data generation for consistent testing."""
        def generate_price_data(symbol: str, days: int = 30, seed: int = 42) -> pd.DataFrame:
            """Generate realistic price data for testing."""
            np.random.seed(seed)
            dates = pd.date_range('2024-01-01', periods=days, freq='D')
            
            prices = []
            base_price = 100.0
            
            for date in dates:
                # Simulate price movements with some volatility
                change = np.random.normal(0, 0.015)  # 1.5% daily volatility
                base_price *= (1 + change)
                
                high = base_price * (1 + abs(np.random.normal(0, 0.005)))
                low = base_price * (1 - abs(np.random.normal(0, 0.005)))
                volume = int(np.random.normal(1000000, 150000))
                
                prices.append({
                    'timestamp': date,
                    'open': base_price,
                    'high': max(base_price, high),
                    'low': min(base_price, low),
                    'close': base_price,
                    'volume': max(volume, 50000)
                })
            
            return pd.DataFrame(prices)
        
        # Test data generation
        data = generate_price_data("NIFTY50", days=50)
        
        # Verify data structure
        assert len(data) == 50
        assert all(col in data.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        assert data['high'].min() >= data[['open', 'close']].min().min()
        assert data['low'].max() <= data[['open', 'close']].max().max()
        assert data['volume'].min() > 0
    
    def test_news_data_generator(self):
        """Test news data generation for consistent testing."""
        def generate_news_data(symbol: str, days: int = 30, seed: int = 42) -> pd.DataFrame:
            """Generate realistic news data for testing."""
            np.random.seed(seed)
            dates = pd.date_range('2024-01-01', periods=days//2, freq='2D')  # News every 2 days
            
            positive_headlines = [
                "Strong earnings boost market confidence",
                "Economic growth exceeds expectations",
                "Technology sector shows robust performance",
                "Banking stocks rally on positive outlook"
            ]
            
            negative_headlines = [
                "Market concerns over inflation persist",
                "Geopolitical tensions impact trading",
                "Regulatory changes create uncertainty",
                "Supply chain disruptions affect sectors"
            ]
            
            neutral_headlines = [
                "Market closes mixed in volatile session",
                "Analysts maintain cautious outlook",
                "Trading volumes remain steady",
                "Sector rotation continues in markets"
            ]
            
            all_headlines = positive_headlines + negative_headlines + neutral_headlines
            
            news_data = []
            for i, date in enumerate(dates):
                headline = all_headlines[i % len(all_headlines)]
                news_data.append({
                    'headline': headline,
                    'timestamp': date,
                    'source': 'test_source',
                    'url': f'http://test.com/news/{i}'
                })
            
            return pd.DataFrame(news_data)
        
        # Test data generation
        data = generate_news_data("NIFTY50", days=30)
        
        # Verify data structure
        assert len(data) == 15  # Every 2 days for 30 days
        assert all(col in data.columns for col in ['headline', 'timestamp', 'source', 'url'])
        assert all(isinstance(headline, str) for headline in data['headline'])
        assert data['source'].nunique() == 1  # All from test_source