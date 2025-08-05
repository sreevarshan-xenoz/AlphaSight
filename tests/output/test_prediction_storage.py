# tests/output/test_prediction_storage.py
import pytest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from nifty_ml_pipeline.output.prediction_storage import PredictionStorage
from nifty_ml_pipeline.data.models import PredictionResult


class TestPredictionStorage:
    """Test suite for PredictionStorage class."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def storage(self, temp_storage_dir):
        """Create a PredictionStorage instance for testing."""
        return PredictionStorage(base_path=temp_storage_dir, retention_days=30)
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample prediction results for testing."""
        return [
            PredictionResult(
                timestamp=datetime(2024, 1, 15, 15, 30),
                symbol='NIFTY50',
                predicted_close=21500.0,
                signal='Buy',
                confidence=0.85,
                model_version='v1.0.0',
                features_used=['rsi_14', 'sma_5_ratio']
            ),
            PredictionResult(
                timestamp=datetime(2024, 1, 15, 15, 31),
                symbol='BANKNIFTY',
                predicted_close=45000.0,
                signal='Sell',
                confidence=0.75,
                model_version='v1.0.0',
                features_used=['macd_hist', 'daily_sentiment']
            )
        ]
    
    def test_initialization(self, temp_storage_dir):
        """Test PredictionStorage initialization."""
        storage = PredictionStorage(base_path=temp_storage_dir, retention_days=60)
        
        assert storage.base_path == Path(temp_storage_dir)
        assert storage.retention_days == 60
        
        # Check that directories are created
        assert storage.predictions_path.exists()
        assert storage.summaries_path.exists()
        assert storage.metadata_path.exists()
    
    def test_store_predictions(self, storage, sample_predictions):
        """Test storing prediction results."""
        execution_id = "test_exec_001"
        metadata = {"model_version": "v1.0.0", "features_count": 6}
        
        result = storage.store_predictions(sample_predictions, execution_id, metadata)
        
        assert result is True
        
        # Check that files are created
        date_partition = storage.predictions_path / "date=2024-01-15"
        assert date_partition.exists()
        
        parquet_files = list(date_partition.glob("*.parquet"))
        assert len(parquet_files) == 1
        assert execution_id in parquet_files[0].name
        
        # Check metadata file
        metadata_file = storage.metadata_path / f"metadata_{execution_id}.json"
        assert metadata_file.exists()
    
    def test_store_predictions_empty_list(self, storage):
        """Test storing empty predictions list."""
        result = storage.store_predictions([], "test_exec_002")
        
        assert result is False
    
    def test_store_predictions_error_handling(self, storage, sample_predictions):
        """Test error handling in store_predictions."""
        # Create invalid predictions that will cause errors
        invalid_predictions = [
            PredictionResult(
                timestamp=None,  # This will cause an error
                symbol='NIFTY50',
                predicted_close=21500.0,
                signal='Buy',
                confidence=0.85,
                model_version='v1.0.0',
                features_used=[]
            )
        ]
        
        with patch('nifty_ml_pipeline.output.prediction_storage.logger') as mock_logger:
            result = storage.store_predictions(invalid_predictions, "test_exec_error")
            
            assert result is False
            mock_logger.error.assert_called()
    
    def test_retrieve_predictions(self, storage, sample_predictions):
        """Test retrieving prediction results."""
        execution_id = "test_exec_003"
        
        # Store predictions first
        storage.store_predictions(sample_predictions, execution_id)
        
        # Retrieve predictions
        start_date = datetime(2024, 1, 15, 0, 0)
        end_date = datetime(2024, 1, 15, 23, 59)
        
        retrieved = storage.retrieve_predictions(start_date, end_date)
        
        assert len(retrieved) == 2
        assert retrieved[0].symbol in ['NIFTY50', 'BANKNIFTY']
        assert retrieved[1].symbol in ['NIFTY50', 'BANKNIFTY']
    
    def test_retrieve_predictions_with_symbol_filter(self, storage, sample_predictions):
        """Test retrieving predictions with symbol filter."""
        execution_id = "test_exec_004"
        storage.store_predictions(sample_predictions, execution_id)
        
        start_date = datetime(2024, 1, 15, 0, 0)
        end_date = datetime(2024, 1, 15, 23, 59)
        
        retrieved = storage.retrieve_predictions(start_date, end_date, symbol='NIFTY50')
        
        assert len(retrieved) == 1
        assert retrieved[0].symbol == 'NIFTY50'
    
    def test_retrieve_predictions_with_execution_id_filter(self, storage, sample_predictions):
        """Test retrieving predictions with execution ID filter."""
        execution_id = "test_exec_005"
        storage.store_predictions(sample_predictions, execution_id)
        
        start_date = datetime(2024, 1, 15, 0, 0)
        end_date = datetime(2024, 1, 15, 23, 59)
        
        retrieved = storage.retrieve_predictions(start_date, end_date, execution_id=execution_id)
        
        assert len(retrieved) == 2
    
    def test_retrieve_predictions_no_data(self, storage):
        """Test retrieving predictions when no data exists."""
        start_date = datetime(2024, 1, 1, 0, 0)
        end_date = datetime(2024, 1, 1, 23, 59)
        
        retrieved = storage.retrieve_predictions(start_date, end_date)
        
        assert len(retrieved) == 0
    
    def test_get_prediction_history(self, storage, sample_predictions):
        """Test getting prediction history for a symbol."""
        execution_id = "test_exec_006"
        storage.store_predictions(sample_predictions, execution_id)
        
        # Mock datetime.now() to control the date range
        with patch('nifty_ml_pipeline.output.prediction_storage.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 20, 12, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            history = storage.get_prediction_history('NIFTY50', days=30)
            
            assert len(history) == 1
            assert history[0].symbol == 'NIFTY50'
    
    def test_get_latest_predictions(self, storage, sample_predictions):
        """Test getting latest predictions."""
        execution_id = "test_exec_007"
        storage.store_predictions(sample_predictions, execution_id)
        
        # Mock datetime.now() to be within 7 days of sample data
        with patch('nifty_ml_pipeline.output.prediction_storage.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 20, 12, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            latest = storage.get_latest_predictions(limit=10)
            
            assert len(latest) == 2
            # Should be sorted by timestamp descending
            assert latest[0].timestamp >= latest[1].timestamp
    
    def test_store_prediction_summary(self, storage):
        """Test storing prediction summary."""
        execution_id = "test_exec_008"
        summary_data = {
            'total_predictions': 5,
            'actionable_predictions': 3,
            'avg_confidence': 0.75
        }
        
        result = storage.store_prediction_summary(execution_id, summary_data)
        
        assert result is True
        
        # Check that summary file is created
        summary_file = storage.summaries_path / f"summary_{execution_id}.json"
        assert summary_file.exists()
        
        # Verify content
        with open(summary_file, 'r') as f:
            stored_data = json.load(f)
        
        assert stored_data['total_predictions'] == 5
        assert stored_data['execution_id'] == execution_id
        assert 'generated_at' in stored_data
    
    def test_get_prediction_summary(self, storage):
        """Test retrieving prediction summary."""
        execution_id = "test_exec_009"
        summary_data = {
            'total_predictions': 3,
            'actionable_predictions': 2
        }
        
        # Store summary first
        storage.store_prediction_summary(execution_id, summary_data)
        
        # Retrieve summary
        retrieved_summary = storage.get_prediction_summary(execution_id)
        
        assert retrieved_summary is not None
        assert retrieved_summary['total_predictions'] == 3
        assert retrieved_summary['execution_id'] == execution_id
    
    def test_get_prediction_summary_not_found(self, storage):
        """Test retrieving non-existent summary."""
        result = storage.get_prediction_summary("non_existent_id")
        
        assert result is None
    
    def test_cleanup_old_predictions(self, storage, sample_predictions):
        """Test cleanup of old predictions."""
        # Create old predictions (beyond retention period)
        old_predictions = [
            PredictionResult(
                timestamp=datetime(2023, 1, 15, 15, 30),  # Old date
                symbol='NIFTY50',
                predicted_close=20000.0,
                signal='Buy',
                confidence=0.8,
                model_version='v0.9.0',
                features_used=[]
            )
        ]
        
        # Store old and new predictions
        storage.store_predictions(old_predictions, "old_exec")
        storage.store_predictions(sample_predictions, "new_exec")
        
        # Mock datetime.now() to simulate current time
        with patch('nifty_ml_pipeline.output.prediction_storage.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 2, 15, 12, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            mock_datetime.strptime = datetime.strptime
            mock_datetime.fromtimestamp = datetime.fromtimestamp
            
            result = storage.cleanup_old_predictions()
            
            assert result is True
    
    def test_get_storage_stats(self, storage, sample_predictions):
        """Test getting storage statistics."""
        execution_id = "test_exec_010"
        storage.store_predictions(sample_predictions, execution_id)
        storage.store_prediction_summary(execution_id, {'test': 'data'})
        
        stats = storage.get_storage_stats()
        
        assert 'prediction_files' in stats
        assert 'summary_files' in stats
        assert 'total_size_mb' in stats
        assert 'date_range' in stats
        assert stats['prediction_files'] >= 1
        assert stats['summary_files'] >= 1
    
    def test_predictions_to_dataframe_conversion(self, storage, sample_predictions):
        """Test internal conversion methods."""
        df = storage._predictions_to_dataframe(sample_predictions)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'symbol' in df.columns
        assert 'signal' in df.columns
        assert 'confidence' in df.columns
        assert 'features_used' in df.columns
        
        # Test conversion back
        converted_predictions = storage._dataframe_to_predictions(df)
        
        assert len(converted_predictions) == 2
        assert isinstance(converted_predictions[0], PredictionResult)
        assert converted_predictions[0].symbol == sample_predictions[0].symbol
    
    def test_dataframe_to_predictions_error_handling(self, storage):
        """Test error handling in DataFrame to predictions conversion."""
        # Create DataFrame with invalid data
        invalid_df = pd.DataFrame([
            {
                'timestamp': 'invalid_timestamp',
                'symbol': 'NIFTY50',
                'predicted_close': 'invalid_price',
                'signal': 'Buy',
                'confidence': 0.8,
                'model_version': 'v1.0.0',
                'features_used': '[]'
            }
        ])
        
        with patch('nifty_ml_pipeline.output.prediction_storage.logger') as mock_logger:
            predictions = storage._dataframe_to_predictions(invalid_df)
            
            assert len(predictions) == 0  # Invalid row should be skipped
            mock_logger.warning.assert_called()
    
    def test_store_execution_metadata(self, storage):
        """Test storing execution metadata."""
        execution_id = "test_exec_011"
        metadata = {
            'model_version': 'v1.0.0',
            'execution_time': 1500,
            'features_count': 6
        }
        
        storage._store_execution_metadata(execution_id, metadata, 5)
        
        # Check that metadata file is created
        metadata_file = storage.metadata_path / f"metadata_{execution_id}.json"
        assert metadata_file.exists()
        
        # Verify content
        with open(metadata_file, 'r') as f:
            stored_metadata = json.load(f)
        
        assert stored_metadata['model_version'] == 'v1.0.0'
        assert stored_metadata['prediction_count'] == 5
        assert stored_metadata['execution_id'] == execution_id
        assert 'stored_at' in stored_metadata
    
    def test_multiple_date_partitions(self, storage):
        """Test storing predictions across multiple dates."""
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
                timestamp=datetime(2024, 1, 16, 15, 30),  # Different date
                symbol='BANKNIFTY',
                predicted_close=45000.0,
                signal='Sell',
                confidence=0.75,
                model_version='v1.0.0',
                features_used=[]
            )
        ]
        
        execution_id = "test_exec_012"
        result = storage.store_predictions(predictions, execution_id)
        
        assert result is True
        
        # Check that both date partitions are created
        date1_partition = storage.predictions_path / "date=2024-01-15"
        date2_partition = storage.predictions_path / "date=2024-01-16"
        
        assert date1_partition.exists()
        assert date2_partition.exists()
        
        # Verify files in each partition
        assert len(list(date1_partition.glob("*.parquet"))) == 1
        assert len(list(date2_partition.glob("*.parquet"))) == 1