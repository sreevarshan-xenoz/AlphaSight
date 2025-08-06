# tests/orchestration/test_controller.py
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from nifty_ml_pipeline.orchestration.controller import PipelineController, PipelineStatus
from nifty_ml_pipeline.data.models import PipelineStage


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
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
        'performance': {
            'MAX_INFERENCE_LATENCY_MS': 10.0
        },
        'paths': {
            'data': '/tmp/test_data'
        }
    }


@pytest.fixture
def mock_price_data():
    """Mock price data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'open': [100.0 + i for i in range(10)],
        'high': [105.0 + i for i in range(10)],
        'low': [95.0 + i for i in range(10)],
        'close': [102.0 + i for i in range(10)],
        'volume': [1000000 + i*10000 for i in range(10)]
    }, index=dates)


@pytest.fixture
def mock_news_data():
    """Mock news data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    return pd.DataFrame({
        'headline': [f'Market news {i}' for i in range(5)],
        'timestamp': dates,
        'source': ['test_source'] * 5,
        'url': [f'http://test.com/{i}' for i in range(5)]
    })


@pytest.fixture
def mock_feature_data():
    """Mock feature data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'lag1_return': [0.01 * i for i in range(10)],
        'lag2_return': [0.005 * i for i in range(10)],
        'sma_5_ratio': [1.0 + 0.01 * i for i in range(10)],
        'rsi_14': [50.0 + i for i in range(10)],
        'macd_hist': [0.1 * i for i in range(10)],
        'daily_sentiment': [0.1 * (i % 3 - 1) for i in range(10)]
    }, index=dates)


class TestPipelineController:
    """Test cases for PipelineController."""
    
    @patch('nifty_ml_pipeline.orchestration.controller.NSEDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.NewsDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.TechnicalIndicatorCalculator')
    @patch('nifty_ml_pipeline.orchestration.controller.SentimentAnalyzer')
    @patch('nifty_ml_pipeline.orchestration.controller.FeatureNormalizer')
    @patch('nifty_ml_pipeline.orchestration.controller.XGBoostPredictor')
    @patch('nifty_ml_pipeline.orchestration.controller.InferenceEngine')
    @patch('nifty_ml_pipeline.orchestration.controller.DataStorage')
    def test_initialization(self, mock_storage, mock_inference, mock_predictor,
                           mock_normalizer, mock_sentiment, mock_technical,
                           mock_news_collector, mock_nse_collector, mock_config):
        """Test pipeline controller initialization."""
        controller = PipelineController(mock_config)
        
        assert controller.config == mock_config
        assert controller.status == PipelineStatus.NOT_STARTED
        assert controller.start_time is None
        assert controller.end_time is None
        assert len(controller.stage_results) == 0
        assert controller.execution_id.startswith('pipeline_')
        
        # Verify components were initialized
        mock_nse_collector.assert_called_once()
        mock_news_collector.assert_called_once()
        mock_technical.assert_called_once()
        mock_sentiment.assert_called_once()
        mock_normalizer.assert_called_once()
        mock_predictor.assert_called_once()
        mock_inference.assert_called_once()
        mock_storage.assert_called_once()
    
    @patch('nifty_ml_pipeline.orchestration.controller.NSEDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.NewsDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.TechnicalIndicatorCalculator')
    @patch('nifty_ml_pipeline.orchestration.controller.SentimentAnalyzer')
    @patch('nifty_ml_pipeline.orchestration.controller.FeatureNormalizer')
    @patch('nifty_ml_pipeline.orchestration.controller.XGBoostPredictor')
    @patch('nifty_ml_pipeline.orchestration.controller.InferenceEngine')
    @patch('nifty_ml_pipeline.orchestration.controller.DataStorage')
    def test_successful_pipeline_execution(self, mock_storage, mock_inference, mock_predictor,
                                         mock_normalizer, mock_sentiment, mock_technical,
                                         mock_news_collector, mock_nse_collector, mock_config,
                                         mock_price_data, mock_news_data, mock_feature_data):
        """Test successful end-to-end pipeline execution."""
        # Setup mocks
        mock_nse_collector.return_value.collect_data.return_value = mock_price_data
        mock_news_collector.return_value.collect_data.return_value = mock_news_data
        mock_technical.return_value.calculate_all_indicators.return_value = mock_price_data
        mock_sentiment.return_value.analyze_dataframe.return_value = mock_news_data
        mock_normalizer.return_value.create_feature_vectors.return_value = mock_feature_data
        
        # Mock storage operations
        mock_storage.return_value.store_price_data.return_value = None
        mock_storage.return_value.store_news_data.return_value = None
        mock_storage.return_value.store_predictions.return_value = None
        
        # Mock prediction result
        mock_prediction = MagicMock()
        mock_prediction.timestamp = datetime.now()
        mock_prediction.symbol = 'NIFTY 50'
        mock_prediction.predicted_direction = 'Buy'
        mock_prediction.confidence = 0.85
        mock_inference.return_value.predict_single.return_value = mock_prediction
        
        # Execute pipeline
        controller = PipelineController(mock_config)
        result = controller.execute_pipeline("NIFTY 50")
        
        # Verify execution
        assert result.symbol == "NIFTY 50"
        assert result.status == "completed"
        assert result.was_successful()
        assert len(result.stage_results) == 3  # Three stages
        assert len(result.predictions) == 1
        assert result.error_message is None
        
        # Verify stage execution order
        stages = [stage['stage'] for stage in result.stage_results]
        expected_stages = [
            PipelineStage.DATA_COLLECTION.value,
            PipelineStage.FEATURE_ENGINEERING.value,
            PipelineStage.MODEL_INFERENCE.value
        ]
        assert stages == expected_stages
        
        # Verify all stages completed successfully
        for stage_result in result.stage_results:
            assert stage_result['status'] == 'completed'
            assert stage_result['duration_ms'] > 0
    
    @patch('nifty_ml_pipeline.orchestration.controller.NSEDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.NewsDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.TechnicalIndicatorCalculator')
    @patch('nifty_ml_pipeline.orchestration.controller.SentimentAnalyzer')
    @patch('nifty_ml_pipeline.orchestration.controller.FeatureNormalizer')
    @patch('nifty_ml_pipeline.orchestration.controller.XGBoostPredictor')
    @patch('nifty_ml_pipeline.orchestration.controller.InferenceEngine')
    @patch('nifty_ml_pipeline.orchestration.controller.DataStorage')
    def test_data_collection_failure(self, mock_storage, mock_inference, mock_predictor,
                                   mock_normalizer, mock_sentiment, mock_technical,
                                   mock_news_collector, mock_nse_collector, mock_config):
        """Test pipeline behavior when data collection fails."""
        # Setup mock to raise exception
        mock_nse_collector.return_value.collect_data.side_effect = Exception("API Error")
        
        # Execute pipeline
        controller = PipelineController(mock_config)
        result = controller.execute_pipeline("NIFTY 50")
        
        # Verify failure handling
        assert result.status == "failed"
        assert not result.was_successful()
        assert result.error_message == "API Error"
        assert len(result.stage_results) == 1  # Only data collection stage
        assert result.stage_results[0]['status'] == 'failed'
        assert result.stage_results[0]['stage'] == PipelineStage.DATA_COLLECTION.value
    
    @patch('nifty_ml_pipeline.orchestration.controller.NSEDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.NewsDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.TechnicalIndicatorCalculator')
    @patch('nifty_ml_pipeline.orchestration.controller.SentimentAnalyzer')
    @patch('nifty_ml_pipeline.orchestration.controller.FeatureNormalizer')
    @patch('nifty_ml_pipeline.orchestration.controller.XGBoostPredictor')
    @patch('nifty_ml_pipeline.orchestration.controller.InferenceEngine')
    @patch('nifty_ml_pipeline.orchestration.controller.DataStorage')
    def test_feature_engineering_failure(self, mock_storage, mock_inference, mock_predictor,
                                       mock_normalizer, mock_sentiment, mock_technical,
                                       mock_news_collector, mock_nse_collector, mock_config,
                                       mock_price_data, mock_news_data):
        """Test pipeline behavior when feature engineering fails."""
        # Setup successful data collection
        mock_nse_collector.return_value.collect_data.return_value = mock_price_data
        mock_news_collector.return_value.collect_data.return_value = mock_news_data
        
        # Setup feature engineering failure
        mock_technical.return_value.calculate_all_indicators.side_effect = Exception("Feature Error")
        
        # Execute pipeline
        controller = PipelineController(mock_config)
        result = controller.execute_pipeline("NIFTY 50")
        
        # Verify failure handling
        assert result.status == "failed"
        assert not result.was_successful()
        assert result.error_message == "Feature Error"
        assert len(result.stage_results) == 2  # Data collection + feature engineering
        assert result.stage_results[0]['status'] == 'completed'  # Data collection succeeded
        assert result.stage_results[1]['status'] == 'failed'     # Feature engineering failed
    
    @patch('nifty_ml_pipeline.orchestration.controller.NSEDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.NewsDataCollector')
    @patch('nifty_ml_pipeline.orchestration.controller.TechnicalIndicatorCalculator')
    @patch('nifty_ml_pipeline.orchestration.controller.SentimentAnalyzer')
    @patch('nifty_ml_pipeline.orchestration.controller.FeatureNormalizer')
    @patch('nifty_ml_pipeline.orchestration.controller.XGBoostPredictor')
    @patch('nifty_ml_pipeline.orchestration.controller.InferenceEngine')
    @patch('nifty_ml_pipeline.orchestration.controller.DataStorage')
    def test_inference_latency_warning(self, mock_storage, mock_inference, mock_predictor,
                                     mock_normalizer, mock_sentiment, mock_technical,
                                     mock_news_collector, mock_nse_collector, mock_config,
                                     mock_price_data, mock_news_data, mock_feature_data):
        """Test successful pipeline execution with inference latency tracking."""
        # Setup successful pipeline
        mock_nse_collector.return_value.collect_data.return_value = mock_price_data
        mock_news_collector.return_value.collect_data.return_value = mock_news_data
        mock_technical.return_value.calculate_all_indicators.return_value = mock_price_data
        mock_sentiment.return_value.analyze_dataframe.return_value = mock_news_data
        mock_normalizer.return_value.create_feature_vectors.return_value = mock_feature_data
        
        # Mock storage operations
        mock_storage.return_value.store_price_data.return_value = None
        mock_storage.return_value.store_news_data.return_value = None
        mock_storage.return_value.store_predictions.return_value = None
        
        # Mock prediction
        mock_prediction = MagicMock()
        mock_prediction.timestamp = datetime.now()
        mock_prediction.symbol = 'NIFTY 50'
        mock_prediction.predicted_direction = 'Buy'
        mock_prediction.confidence = 0.85
        
        # Mock inference to return prediction
        mock_inference.return_value.predict_single.return_value = mock_prediction
        
        controller = PipelineController(mock_config)
        result = controller.execute_pipeline("NIFTY 50")
        
        # Verify execution completed successfully
        assert result.was_successful()
        inference_stage = next(s for s in result.stage_results if s['stage'] == 'model_inference')
        assert 'meets_latency_target' in inference_stage['metadata']
    
    def test_get_execution_summary(self, mock_config):
        """Test execution summary generation."""
        with patch.multiple(
            'nifty_ml_pipeline.orchestration.controller',
            NSEDataCollector=Mock(),
            NewsDataCollector=Mock(),
            TechnicalIndicatorCalculator=Mock(),
            SentimentAnalyzer=Mock(),
            FeatureNormalizer=Mock(),
            XGBoostPredictor=Mock(),
            InferenceEngine=Mock(),
            DataStorage=Mock()
        ):
            controller = PipelineController(mock_config)
            summary = controller.get_execution_summary()
            
            assert 'execution_id' in summary
            assert 'status' in summary
            assert 'start_time' in summary
            assert 'end_time' in summary
            assert 'total_duration_ms' in summary
            assert 'stages' in summary
            assert summary['status'] == 'not_started'
            assert len(summary['stages']) == 0