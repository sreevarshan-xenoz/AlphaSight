# nifty_ml_pipeline/output/prediction_storage.py
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..data.models import PredictionResult


logger = logging.getLogger(__name__)


class PredictionStorage:
    """Handles storage and retrieval of prediction results with history tracking.
    
    Implements efficient storage using Parquet format with partitioning
    for optimal query performance and automatic cleanup.
    """
    
    def __init__(self, base_path: str = "data/predictions", retention_days: int = 365):
        """Initialize prediction storage system.
        
        Args:
            base_path: Base directory for prediction storage
            retention_days: Number of days to retain prediction history
        """
        self.base_path = Path(base_path)
        self.retention_days = retention_days
        
        # Create directory structure
        self.predictions_path = self.base_path / "predictions"
        self.summaries_path = self.base_path / "summaries"
        self.metadata_path = self.base_path / "metadata"
        
        self._ensure_directories()
        logger.info(f"Initialized PredictionStorage at {self.base_path}")
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for path in [self.predictions_path, self.summaries_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {path}")
    
    def store_predictions(self, 
                         predictions: List[PredictionResult], 
                         execution_id: str,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store prediction results with execution metadata.
        
        Args:
            predictions: List of prediction results to store
            execution_id: Unique identifier for this execution
            metadata: Optional metadata about the execution
            
        Returns:
            bool: True if storage successful
        """
        try:
            if not predictions:
                logger.warning("No predictions provided for storage")
                return False
            
            # Convert predictions to DataFrame
            df = self._predictions_to_dataframe(predictions)
            
            # Add execution metadata
            df['execution_id'] = execution_id
            df['stored_at'] = datetime.now()
            
            # Partition by date for efficient queries
            for date_group, group_df in df.groupby(df['timestamp'].dt.date):
                partition_path = self.predictions_path / f"date={date_group}"
                partition_path.mkdir(parents=True, exist_ok=True)
                
                file_path = partition_path / f"predictions_{execution_id}_{date_group}.parquet"
                
                # Write to Parquet with compression
                group_df.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )
                
                logger.debug(f"Stored {len(group_df)} predictions for {date_group} in {file_path}")
            
            # Store execution metadata
            if metadata:
                self._store_execution_metadata(execution_id, metadata, len(predictions))
            
            logger.info(f"Successfully stored {len(predictions)} predictions for execution {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store predictions for execution {execution_id}: {str(e)}")
            return False
    
    def retrieve_predictions(self, 
                           start_date: datetime, 
                           end_date: datetime,
                           symbol: Optional[str] = None,
                           execution_id: Optional[str] = None) -> List[PredictionResult]:
        """Retrieve prediction results for specified criteria.
        
        Args:
            start_date: Start date for retrieval
            end_date: End date for retrieval
            symbol: Optional symbol filter
            execution_id: Optional execution ID filter
            
        Returns:
            List[PredictionResult]: Retrieved predictions
        """
        try:
            # Find relevant partition files
            parquet_files = []
            current_date = start_date.date()
            end_date_only = end_date.date()
            
            while current_date <= end_date_only:
                partition_path = self.predictions_path / f"date={current_date}"
                if partition_path.exists():
                    for parquet_file in partition_path.glob("*.parquet"):
                        # Filter by execution_id if specified
                        if execution_id and execution_id not in parquet_file.name:
                            continue
                        parquet_files.append(parquet_file)
                
                current_date += timedelta(days=1)
            
            if not parquet_files:
                logger.info(f"No prediction files found for date range {start_date} to {end_date}")
                return []
            
            # Read and combine data
            dfs = []
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path, engine='pyarrow')
                    
                    # Apply filters
                    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                    
                    if symbol:
                        df = df[df['symbol'] == symbol]
                    
                    if not df.empty:
                        dfs.append(df)
                        
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {str(e)}")
                    continue
            
            if not dfs:
                logger.info("No matching predictions found after filtering")
                return []
            
            # Combine and sort
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values(['timestamp', 'symbol']).drop_duplicates()
            
            # Convert back to PredictionResult objects
            predictions = self._dataframe_to_predictions(combined_df)
            
            logger.info(f"Retrieved {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to retrieve predictions: {str(e)}")
            return []
    
    def get_prediction_history(self, 
                              symbol: str, 
                              days: int = 30) -> List[PredictionResult]:
        """Get prediction history for a specific symbol.
        
        Args:
            symbol: Symbol to get history for
            days: Number of days of history to retrieve
            
        Returns:
            List[PredictionResult]: Historical predictions
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.retrieve_predictions(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol
        )
    
    def get_latest_predictions(self, limit: int = 100) -> List[PredictionResult]:
        """Get the most recent predictions across all symbols.
        
        Args:
            limit: Maximum number of predictions to return
            
        Returns:
            List[PredictionResult]: Latest predictions
        """
        try:
            # Look for recent prediction files (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            predictions = self.retrieve_predictions(start_date, end_date)
            
            # Sort by timestamp descending and limit
            predictions.sort(key=lambda p: p.timestamp, reverse=True)
            
            return predictions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get latest predictions: {str(e)}")
            return []
    
    def store_prediction_summary(self, 
                                execution_id: str, 
                                summary_data: Dict[str, Any]) -> bool:
        """Store prediction summary for an execution.
        
        Args:
            execution_id: Execution identifier
            summary_data: Summary statistics and metadata
            
        Returns:
            bool: True if storage successful
        """
        try:
            summary_file = self.summaries_path / f"summary_{execution_id}.json"
            
            # Add timestamp to summary
            summary_data['generated_at'] = datetime.now().isoformat()
            summary_data['execution_id'] = execution_id
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            
            logger.info(f"Stored prediction summary for execution {execution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store prediction summary: {str(e)}")
            return False
    
    def get_prediction_summary(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve prediction summary for an execution.
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            Dict: Summary data or None if not found
        """
        try:
            summary_file = self.summaries_path / f"summary_{execution_id}.json"
            
            if not summary_file.exists():
                logger.warning(f"Summary file not found for execution {execution_id}")
                return None
            
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            
            logger.debug(f"Retrieved summary for execution {execution_id}")
            return summary_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve prediction summary: {str(e)}")
            return None
    
    def cleanup_old_predictions(self) -> bool:
        """Remove predictions older than retention period.
        
        Returns:
            bool: True if cleanup successful
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cleaned_files = 0
            
            # Clean prediction files
            for date_dir in self.predictions_path.glob("date=*"):
                try:
                    date_str = date_dir.name.split("=")[1]
                    dir_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if dir_date < cutoff_date:
                        # Remove all files in this date directory
                        for file_path in date_dir.glob("*.parquet"):
                            file_path.unlink()
                            cleaned_files += 1
                            logger.debug(f"Removed old prediction file: {file_path}")
                        
                        # Remove empty directory
                        if not any(date_dir.iterdir()):
                            date_dir.rmdir()
                            logger.debug(f"Removed empty directory: {date_dir}")
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid date directory name: {date_dir.name}")
                    continue
            
            # Clean old summaries
            for summary_file in self.summaries_path.glob("summary_*.json"):
                try:
                    file_stat = summary_file.stat()
                    file_date = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    if file_date < cutoff_date:
                        summary_file.unlink()
                        cleaned_files += 1
                        logger.debug(f"Removed old summary file: {summary_file}")
                
                except Exception as e:
                    logger.warning(f"Failed to check summary file {summary_file}: {str(e)}")
                    continue
            
            logger.info(f"Cleaned up {cleaned_files} old prediction files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old predictions: {str(e)}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dict: Storage statistics
        """
        stats = {
            'prediction_files': 0,
            'summary_files': 0,
            'total_size_mb': 0,
            'date_range': {'earliest': None, 'latest': None},
            'symbols_tracked': set()
        }
        
        try:
            # Count prediction files and gather stats
            for parquet_file in self.predictions_path.rglob("*.parquet"):
                stats['prediction_files'] += 1
                stats['total_size_mb'] += parquet_file.stat().st_size / (1024 * 1024)
                
                # Extract date from path for range calculation
                try:
                    date_str = parquet_file.parent.name.split("=")[1]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if stats['date_range']['earliest'] is None or file_date < stats['date_range']['earliest']:
                        stats['date_range']['earliest'] = file_date
                    if stats['date_range']['latest'] is None or file_date > stats['date_range']['latest']:
                        stats['date_range']['latest'] = file_date
                
                except (ValueError, IndexError):
                    continue
            
            # Count summary files
            stats['summary_files'] = len(list(self.summaries_path.glob("summary_*.json")))
            
            # Convert set to list for JSON serialization
            stats['symbols_tracked'] = list(stats['symbols_tracked'])
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            
            # Convert dates to ISO format
            if stats['date_range']['earliest']:
                stats['date_range']['earliest'] = stats['date_range']['earliest'].isoformat()
            if stats['date_range']['latest']:
                stats['date_range']['latest'] = stats['date_range']['latest'].isoformat()
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            stats['error'] = str(e)
        
        return stats
    
    def _predictions_to_dataframe(self, predictions: List[PredictionResult]) -> pd.DataFrame:
        """Convert PredictionResult objects to DataFrame."""
        data = []
        for pred in predictions:
            pred_dict = {
                'timestamp': pred.timestamp,
                'symbol': pred.symbol,
                'predicted_close': pred.predicted_close,
                'signal': pred.signal,
                'confidence': pred.confidence,
                'model_version': pred.model_version,
                'features_used': json.dumps(pred.features_used)  # Store as JSON string
            }
            data.append(pred_dict)
        
        return pd.DataFrame(data)
    
    def _dataframe_to_predictions(self, df: pd.DataFrame) -> List[PredictionResult]:
        """Convert DataFrame to PredictionResult objects."""
        predictions = []
        
        for _, row in df.iterrows():
            try:
                # Parse features_used from JSON string
                features_used = json.loads(row['features_used']) if 'features_used' in row else []
                
                prediction = PredictionResult(
                    timestamp=pd.to_datetime(row['timestamp']),
                    symbol=row['symbol'],
                    predicted_close=float(row['predicted_close']),
                    signal=row['signal'],
                    confidence=float(row['confidence']),
                    model_version=row['model_version'],
                    features_used=features_used
                )
                predictions.append(prediction)
                
            except Exception as e:
                logger.warning(f"Failed to convert row to PredictionResult: {str(e)}")
                continue
        
        return predictions
    
    def _store_execution_metadata(self, 
                                 execution_id: str, 
                                 metadata: Dict[str, Any], 
                                 prediction_count: int) -> None:
        """Store metadata for an execution."""
        try:
            metadata_file = self.metadata_path / f"metadata_{execution_id}.json"
            
            metadata_with_stats = {
                **metadata,
                'execution_id': execution_id,
                'prediction_count': prediction_count,
                'stored_at': datetime.now().isoformat()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_with_stats, f, indent=2, default=str)
            
            logger.debug(f"Stored execution metadata for {execution_id}")
            
        except Exception as e:
            logger.error(f"Failed to store execution metadata: {str(e)}")