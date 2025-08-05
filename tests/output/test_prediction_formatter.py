# tests/output/test_prediction_formatter.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from nifty_ml_pipeline.output.prediction_formatter import PredictionFormatter
from nifty_ml_pipeline.data.models import PredictionResult


class TestPredictionFormatter:
    """Test suite for PredictionFormatter class."""
    
    @pytest.fixture
    def formatter(self):
        """Create a PredictionFormatter instance for testing."""
        return PredictionFormatter(min_confidence=0.7)
    
    @pytest.fixture
    def sample_prediction_data(self):
        """Sample prediction data for testing."""
        return {
            'symbol': 'NIFTY50',
            'timestamp': datetime(2024, 1, 15, 15, 30),
            'predicted_price': 21500.0,
            'current_price': 21000.0,
            'confidence': 0.85,
            'model_version': 'v1.0.0',
            'features_used': ['rsi_14', 'sma_5_ratio', 'macd_hist']
        }
    
    def test_initialization(self):
        """Test PredictionFormatter initialization."""
        formatter = PredictionFormatter(min_confidence=0.8)
        assert formatter.min_confidence == 0.8
        assert 'buy_threshold' in formatter.signal_thresholds
        assert 'sell_threshold' in formatter.signal_thresholds
    
    def test_format_prediction_buy_signal(self, formatter, sample_prediction_data):
        """Test formatting prediction with buy signal."""
        result = formatter.format_prediction(**sample_prediction_data)
        
        assert isinstance(result, PredictionResult)
        assert result.symbol == 'NIFTY50'
        assert result.signal == 'Buy'  # 2.38% return > 2% threshold
        assert result.confidence == 0.85
        assert result.predicted_close == 21500.0
        assert result.model_version == 'v1.0.0'
        assert result.features_used == ['rsi_14', 'sma_5_ratio', 'macd_hist']
    
    def test_format_prediction_sell_signal(self, formatter, sample_prediction_data):
        """Test formatting prediction with sell signal."""
        sample_prediction_data['predicted_price'] = 20500.0  # -2.38% return
        
        result = formatter.format_prediction(**sample_prediction_data)
        
        assert result.signal == 'Sell'
        assert result.predicted_close == 20500.0
    
    def test_format_prediction_hold_signal(self, formatter, sample_prediction_data):
        """Test formatting prediction with hold signal."""
        sample_prediction_data['predicted_price'] = 21100.0  # 0.48% return
        
        result = formatter.format_prediction(**sample_prediction_data)
        
        assert result.signal == 'Hold'
        assert result.predicted_close == 21100.0
    
    def test_format_prediction_low_confidence(self, formatter, sample_prediction_data):
        """Test formatting prediction with low confidence."""
        sample_prediction_data['confidence'] = 0.5  # Below 0.7 threshold
        
        result = formatter.format_prediction(**sample_prediction_data)
        
        assert result.signal == 'Hold'  # Should be Hold regardless of return
        assert result.confidence == 0.5
    
    def test_format_prediction_error_handling(self, formatter):
        """Test error handling in format_prediction."""
        with patch('nifty_ml_pipeline.output.prediction_formatter.logger') as mock_logger:
            # Missing required field
            result = formatter.format_prediction(
                symbol='NIFTY50',
                timestamp=datetime.now(),
                predicted_price=21000.0,
                current_price=None,  # This will cause an error
                confidence=0.8,
                model_version='v1.0.0',
                features_used=[]
            )
            
            # Should return neutral prediction on error
            assert result.signal == 'Hold'
            assert result.confidence == 0.0
            mock_logger.error.assert_called_once()
    
    def test_format_batch_predictions(self, formatter):
        """Test batch prediction formatting."""
        predictions_data = [
            {
                'symbol': 'NIFTY50',
                'timestamp': datetime(2024, 1, 15, 15, 30),
                'predicted_price': 21500.0,
                'current_price': 21000.0,
                'confidence': 0.85,
                'model_version': 'v1.0.0',
                'features_used': ['rsi_14']
            },
            {
                'symbol': 'BANKNIFTY',
                'timestamp': datetime(2024, 1, 15, 15, 30),
                'predicted_price': 45000.0,
                'current_price': 46000.0,
                'confidence': 0.75,
                'model_version': 'v1.0.0',
                'features_used': ['sma_5_ratio']
            }
        ]
        
        results = formatter.format_batch_predictions(predictions_data)
        
        assert len(results) == 2
        assert results[0].symbol == 'NIFTY50'
        assert results[0].signal == 'Buy'
        assert results[1].symbol == 'BANKNIFTY'
        assert results[1].signal == 'Sell'
    
    def test_format_batch_predictions_with_errors(self, formatter):
        """Test batch formatting with some invalid data."""
        predictions_data = [
            {
                'symbol': 'NIFTY50',
                'timestamp': datetime(2024, 1, 15, 15, 30),
                'predicted_price': 21500.0,
                'current_price': 21000.0,
                'confidence': 0.85,
                'model_version': 'v1.0.0',
                'features_used': ['rsi_14']
            },
            {
                # Missing required fields
                'symbol': 'BANKNIFTY',
                'timestamp': datetime(2024, 1, 15, 15, 30)
            }
        ]
        
        with patch('nifty_ml_pipeline.output.prediction_formatter.logger') as mock_logger:
            results = formatter.format_batch_predictions(predictions_data)
            
            assert len(results) == 1  # Only valid prediction processed
            assert results[0].symbol == 'NIFTY50'
            mock_logger.error.assert_called()
    
    def test_filter_actionable_signals(self, formatter):
        """Test filtering actionable signals."""
        predictions = [
            PredictionResult(
                timestamp=datetime.now(),
                symbol='NIFTY50',
                predicted_close=21500.0,
                signal='Buy',
                confidence=0.85,  # Above threshold
                model_version='v1.0.0',
                features_used=[]
            ),
            PredictionResult(
                timestamp=datetime.now(),
                symbol='BANKNIFTY',
                predicted_close=45000.0,
                signal='Sell',
                confidence=0.6,  # Below threshold
                model_version='v1.0.0',
                features_used=[]
            )
        ]
        
        actionable = formatter.filter_actionable_signals(predictions)
        
        assert len(actionable) == 1
        assert actionable[0].symbol == 'NIFTY50'
        assert actionable[0].confidence == 0.85
    
    def test_to_dataframe(self, formatter):
        """Test conversion to DataFrame."""
        predictions = [
            PredictionResult(
                timestamp=datetime(2024, 1, 15, 15, 30),
                symbol='NIFTY50',
                predicted_close=21500.0,
                signal='Buy',
                confidence=0.85,
                model_version='v1.0.0',
                features_used=['rsi_14']
            ),
            PredictionResult(
                timestamp=datetime(2024, 1, 15, 15, 31),
                symbol='BANKNIFTY',
                predicted_close=45000.0,
                signal='Sell',
                confidence=0.75,
                model_version='v1.0.0',
                features_used=['sma_5_ratio']
            )
        ]
        
        df = formatter.to_dataframe(predictions)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'symbol' in df.columns
        assert 'signal' in df.columns
        assert 'confidence' in df.columns
        assert df.iloc[0]['symbol'] == 'NIFTY50'  # Should be sorted by timestamp
    
    def test_to_dataframe_empty(self, formatter):
        """Test DataFrame conversion with empty list."""
        df = formatter.to_dataframe([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    def test_to_json_serializable(self, formatter):
        """Test conversion to JSON-serializable format."""
        predictions = [
            PredictionResult(
                timestamp=datetime(2024, 1, 15, 15, 30),
                symbol='NIFTY50',
                predicted_close=21500.0,
                signal='Buy',
                confidence=0.85,
                model_version='v1.0.0',
                features_used=['rsi_14']
            )
        ]
        
        json_data = formatter.to_json_serializable(predictions)
        
        assert isinstance(json_data, list)
        assert len(json_data) == 1
        assert json_data[0]['symbol'] == 'NIFTY50'
        assert json_data[0]['signal'] == 'Buy'
        assert isinstance(json_data[0]['timestamp'], str)  # Should be ISO format
    
    def test_generate_summary_stats(self, formatter):
        """Test summary statistics generation."""
        predictions = [
            PredictionResult(
                timestamp=datetime(2024, 1, 15, 15, 30),
                symbol='NIFTY50',
                predicted_close=21500.0,
                signal='Buy',
                confidence=0.85,
                model_version='v1.0.0',
                features_used=[]
            ),
            PredictionResult(
                timestamp=datetime(2024, 1, 15, 15, 31),
                symbol='BANKNIFTY',
                predicted_close=45000.0,
                signal='Hold',
                confidence=0.6,
                model_version='v1.0.0',
                features_used=[]
            ),
            PredictionResult(
                timestamp=datetime(2024, 1, 15, 15, 32),
                symbol='RELIANCE',
                predicted_close=2500.0,
                signal='Sell',
                confidence=0.9,
                model_version='v1.0.0',
                features_used=[]
            )
        ]
        
        stats = formatter.generate_summary_stats(predictions)
        
        assert stats['total_predictions'] == 3
        assert stats['actionable_predictions'] == 2  # confidence >= 0.7
        assert stats['signal_distribution']['Buy'] == 1
        assert stats['signal_distribution']['Hold'] == 1
        assert stats['signal_distribution']['Sell'] == 1
        assert 0.7 < stats['avg_confidence'] < 0.8
        assert 'confidence_distribution' in stats
        assert 'timestamp_range' in stats
    
    def test_generate_summary_stats_empty(self, formatter):
        """Test summary statistics with empty predictions."""
        stats = formatter.generate_summary_stats([])
        
        assert stats['total_predictions'] == 0
        assert stats['actionable_predictions'] == 0
        assert stats['signal_distribution'] == {}
        assert stats['avg_confidence'] == 0.0
    
    def test_update_thresholds(self, formatter):
        """Test updating signal thresholds."""
        new_thresholds = {
            'buy_threshold': 0.03,
            'sell_threshold': -0.03
        }
        
        formatter.update_thresholds(new_thresholds)
        
        assert formatter.signal_thresholds['buy_threshold'] == 0.03
        assert formatter.signal_thresholds['sell_threshold'] == -0.03
    
    def test_update_min_confidence(self, formatter):
        """Test updating minimum confidence threshold."""
        formatter.update_min_confidence(0.8)
        assert formatter.min_confidence == 0.8
        
        # Test invalid confidence
        with pytest.raises(ValueError):
            formatter.update_min_confidence(1.5)
        
        with pytest.raises(ValueError):
            formatter.update_min_confidence(-0.1)
    
    def test_generate_signal_logic(self, formatter):
        """Test internal signal generation logic."""
        # Test buy signal
        signal = formatter._generate_signal(0.025, 0.8)  # 2.5% return, high confidence
        assert signal == 'Buy'
        
        # Test sell signal
        signal = formatter._generate_signal(-0.025, 0.8)  # -2.5% return, high confidence
        assert signal == 'Sell'
        
        # Test hold signal (low return)
        signal = formatter._generate_signal(0.01, 0.8)  # 1% return, high confidence
        assert signal == 'Hold'
        
        # Test hold signal (low confidence)
        signal = formatter._generate_signal(0.05, 0.5)  # 5% return, low confidence
        assert signal == 'Hold'
    
    def test_custom_signal_thresholds(self):
        """Test formatter with custom signal thresholds."""
        custom_thresholds = {
            'buy_threshold': 0.01,  # 1% threshold
            'sell_threshold': -0.01
        }
        
        formatter = PredictionFormatter(
            min_confidence=0.6,
            signal_thresholds=custom_thresholds
        )
        
        assert formatter.signal_thresholds['buy_threshold'] == 0.01
        assert formatter.signal_thresholds['sell_threshold'] == -0.01
        
        # Test signal generation with custom thresholds
        signal = formatter._generate_signal(0.015, 0.7)  # 1.5% return
        assert signal == 'Buy'