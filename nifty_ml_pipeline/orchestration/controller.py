# nifty_ml_pipeline/orchestration/controller.py
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ..data.collectors import NSEDataCollector, NewsDataCollector
from ..features.technical_indicators import TechnicalIndicatorCalculator
from ..features.sentiment_analysis import SentimentAnalyzer
from ..features.feature_normalizer import FeatureNormalizer
from ..models.predictor import XGBoostPredictor
from ..models.inference_engine import InferenceEngine
from ..data.storage import DataStorage
from ..data.models import PipelineResult, PipelineStage
from .error_handler import ErrorHandler, ErrorContext, RecoveryAction


logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL_SUCCESS = "partial_success"


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""
    stage: PipelineStage
    status: PipelineStatus
    duration_ms: float
    data_count: int
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PipelineController:
    """
    Main orchestration class for end-to-end pipeline execution.
    
    Coordinates sequential execution of data collection, feature engineering,
    and model inference stages with comprehensive error handling and monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline controller with configuration.
        
        Args:
            config: Configuration dictionary containing all pipeline settings
        """
        self.config = config
        self.execution_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = PipelineStatus.NOT_STARTED
        self.stage_results: List[StageResult] = []
        
        # Initialize components
        self._initialize_components()
        
        # Initialize error handler
        self.error_handler = ErrorHandler(config)
        
        logger.info(f"Pipeline controller initialized with execution ID: {self.execution_id}")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Data collectors
            self.nse_collector = NSEDataCollector(
                max_retries=3,
                base_delay=1.0
            )
            self.news_collector = NewsDataCollector(
                api_key=self.config['api']['keys'].get('ECONOMIC_TIMES_API_KEY'),
                max_retries=3,
                base_delay=1.0
            )
            
            # Feature engineering components
            self.technical_calculator = TechnicalIndicatorCalculator()
            self.sentiment_analyzer = SentimentAnalyzer()
            self.feature_normalizer = FeatureNormalizer()
            
            # Model components
            self.predictor = XGBoostPredictor()
            self.inference_engine = InferenceEngine(self.predictor)
            
            # Storage
            self.storage = DataStorage(
                base_path=self.config['paths']['data'],
                format=self.config['data']['storage_format']
            )
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def execute_pipeline(self, symbol: str = "NIFTY 50") -> PipelineResult:
        """
        Execute the complete pipeline for the specified symbol.
        
        Args:
            symbol: Stock symbol to process (default: "NIFTY 50")
            
        Returns:
            PipelineResult containing execution summary and results
        """
        self.start_time = datetime.now()
        self.status = PipelineStatus.RUNNING
        
        logger.info(f"Starting pipeline execution for {symbol} (ID: {self.execution_id})")
        
        try:
            # Stage 1: Data Collection
            price_data, news_data = self._execute_data_collection_stage(symbol)
            
            # Stage 2: Feature Engineering
            feature_data = self._execute_feature_engineering_stage(price_data, news_data)
            
            # Stage 3: Model Inference
            predictions = self._execute_inference_stage(feature_data, symbol)
            
            # Pipeline completed successfully
            self.status = PipelineStatus.COMPLETED
            self.end_time = datetime.now()
            
            result = self._create_pipeline_result(symbol, predictions)
            logger.info(f"Pipeline execution completed successfully in {self._get_total_duration():.2f}ms")
            
            return result
            
        except Exception as e:
            self.status = PipelineStatus.FAILED
            self.end_time = datetime.now()
            
            logger.error(f"Pipeline execution failed: {e}")
            
            # Return partial result with error information
            return self._create_error_result(symbol, str(e))
    
    def _execute_data_collection_stage(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Execute data collection stage with error handling and recovery.
        
        Args:
            symbol: Symbol to collect data for
            
        Returns:
            Tuple of (price_data, news_data) DataFrames
        """
        stage_start = time.perf_counter()
        attempt = 1
        max_attempts = 3
        
        while attempt <= max_attempts:
            try:
                logger.info(f"Starting data collection stage (attempt {attempt}/{max_attempts})")
                
                # Calculate date range for rolling window
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.config['data']['retention_days'])
                
                # Collect price data
                logger.info(f"Collecting price data for {symbol}")
                price_data = self.nse_collector.collect_data(symbol, start_date, end_date)
                
                # Collect news data
                logger.info(f"Collecting news data for {symbol}")
                news_data = self.news_collector.collect_data(symbol, start_date, end_date)
                
                # Store collected data
                self.storage.store_price_data(price_data, symbol)
                self.storage.store_news_data(news_data, symbol)
                
                stage_end = time.perf_counter()
                duration_ms = (stage_end - stage_start) * 1000
                
                # Record successful stage result
                stage_result = StageResult(
                    stage=PipelineStage.DATA_COLLECTION,
                    status=PipelineStatus.COMPLETED,
                    duration_ms=duration_ms,
                    data_count=len(price_data) + len(news_data),
                    metadata={
                        'price_records': len(price_data),
                        'news_records': len(news_data),
                        'date_range': f"{start_date.date()} to {end_date.date()}",
                        'attempts_made': attempt
                    }
                )
                self.stage_results.append(stage_result)
                
                logger.info(f"Data collection completed in {duration_ms:.2f}ms")
                logger.info(f"Collected {len(price_data)} price records and {len(news_data)} news records")
                
                return price_data, news_data
                
            except Exception as e:
                # Create error context
                error_context = ErrorContext(
                    stage=PipelineStage.DATA_COLLECTION,
                    error=e,
                    timestamp=datetime.now(),
                    execution_id=self.execution_id,
                    attempt_number=attempt,
                    metadata={'symbol': symbol}
                )
                
                # Handle error with recovery strategy
                recovery_result = self.error_handler.handle_error(error_context)
                
                if recovery_result.action_taken == RecoveryAction.RETRY and attempt < max_attempts:
                    attempt += 1
                    time.sleep(2)  # Brief delay before retry
                    continue
                elif recovery_result.action_taken == RecoveryAction.USE_FALLBACK and recovery_result.data:
                    # Use fallback data
                    price_data, news_data = recovery_result.data
                    
                    stage_end = time.perf_counter()
                    duration_ms = (stage_end - stage_start) * 1000
                    
                    stage_result = StageResult(
                        stage=PipelineStage.DATA_COLLECTION,
                        status=PipelineStatus.PARTIAL_SUCCESS,
                        duration_ms=duration_ms,
                        data_count=len(price_data) + len(news_data),
                        metadata={
                            'used_fallback_data': True,
                            'original_error': str(e),
                            'attempts_made': attempt
                        }
                    )
                    self.stage_results.append(stage_result)
                    
                    logger.warning(f"Using fallback data for data collection stage")
                    return price_data, news_data
                else:
                    # Record failed stage result
                    stage_end = time.perf_counter()
                    duration_ms = (stage_end - stage_start) * 1000
                    
                    stage_result = StageResult(
                        stage=PipelineStage.DATA_COLLECTION,
                        status=PipelineStatus.FAILED,
                        duration_ms=duration_ms,
                        data_count=0,
                        error_message=str(e),
                        metadata={'attempts_made': attempt}
                    )
                    self.stage_results.append(stage_result)
                    
                    logger.error(f"Data collection stage failed after {attempt} attempts: {e}")
                    raise
    
    def _execute_feature_engineering_stage(self, price_data: pd.DataFrame, 
                                         news_data: pd.DataFrame) -> pd.DataFrame:
        """Execute feature engineering stage.
        
        Args:
            price_data: Historical price data
            news_data: News data for sentiment analysis
            
        Returns:
            DataFrame with engineered features
        """
        stage_start = time.perf_counter()
        
        try:
            logger.info("Starting feature engineering stage")
            
            # Calculate technical indicators
            logger.info("Computing technical indicators")
            price_with_indicators = self.technical_calculator.calculate_all_indicators(price_data)
            
            # Analyze sentiment from news data
            logger.info("Analyzing news sentiment")
            news_with_sentiment = self.sentiment_analyzer.analyze_dataframe(news_data)
            
            # Combine and normalize features
            logger.info("Combining and normalizing features")
            feature_data = self.feature_normalizer.create_feature_vectors(
                price_with_indicators, 
                news_with_sentiment
            )
            
            # Store feature data
            self.storage.store_feature_data(feature_data, "NIFTY50")
            
            stage_end = time.perf_counter()
            duration_ms = (stage_end - stage_start) * 1000
            
            # Record stage result
            stage_result = StageResult(
                stage=PipelineStage.FEATURE_ENGINEERING,
                status=PipelineStatus.COMPLETED,
                duration_ms=duration_ms,
                data_count=len(feature_data),
                metadata={
                    'feature_count': len(feature_data.columns),
                    'technical_indicators': ['rsi_14', 'sma_5', 'macd_line', 'macd_signal', 'macd_histogram'],
                    'sentiment_features': ['daily_sentiment']
                }
            )
            self.stage_results.append(stage_result)
            
            logger.info(f"Feature engineering completed in {duration_ms:.2f}ms")
            logger.info(f"Generated {len(feature_data)} feature vectors with {len(feature_data.columns)} features")
            
            return feature_data
            
        except Exception as e:
            stage_end = time.perf_counter()
            duration_ms = (stage_end - stage_start) * 1000
            
            stage_result = StageResult(
                stage=PipelineStage.FEATURE_ENGINEERING,
                status=PipelineStatus.FAILED,
                duration_ms=duration_ms,
                data_count=0,
                error_message=str(e)
            )
            self.stage_results.append(stage_result)
            
            logger.error(f"Feature engineering stage failed: {e}")
            raise
    
    def _execute_inference_stage(self, feature_data: pd.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Execute model inference stage.
        
        Args:
            feature_data: Engineered features for prediction
            symbol: Symbol being processed
            
        Returns:
            List of prediction results
        """
        stage_start = time.perf_counter()
        
        try:
            logger.info("Starting model inference stage")
            
            # Generate predictions using inference engine
            predictions = []
            
            # Use the latest feature vector for prediction
            if not feature_data.empty:
                latest_features = feature_data.iloc[-1]
                
                # Generate prediction with confidence
                prediction_result = self.inference_engine.predict_single(
                    features=latest_features.to_dict(),
                    symbol=symbol
                )
                
                predictions.append(prediction_result.__dict__)
                
                logger.info(f"Generated prediction: {prediction_result.predicted_direction} "
                           f"(confidence: {prediction_result.confidence:.3f})")
            
            stage_end = time.perf_counter()
            duration_ms = (stage_end - stage_start) * 1000
            
            # Validate inference latency requirement
            if duration_ms > self.config['performance']['MAX_INFERENCE_LATENCY_MS']:
                logger.warning(f"Inference latency {duration_ms:.2f}ms exceeds target "
                             f"{self.config['performance']['MAX_INFERENCE_LATENCY_MS']}ms")
            
            # Record stage result
            stage_result = StageResult(
                stage=PipelineStage.MODEL_INFERENCE,
                status=PipelineStatus.COMPLETED,
                duration_ms=duration_ms,
                data_count=len(predictions),
                metadata={
                    'predictions_generated': len(predictions),
                    'inference_latency_ms': duration_ms,
                    'meets_latency_target': duration_ms <= self.config['performance']['MAX_INFERENCE_LATENCY_MS']
                }
            )
            self.stage_results.append(stage_result)
            
            logger.info(f"Model inference completed in {duration_ms:.2f}ms")
            
            return predictions
            
        except Exception as e:
            stage_end = time.perf_counter()
            duration_ms = (stage_end - stage_start) * 1000
            
            stage_result = StageResult(
                stage=PipelineStage.MODEL_INFERENCE,
                status=PipelineStatus.FAILED,
                duration_ms=duration_ms,
                data_count=0,
                error_message=str(e)
            )
            self.stage_results.append(stage_result)
            
            logger.error(f"Model inference stage failed: {e}")
            raise
    
    def _create_pipeline_result(self, symbol: str, predictions: List[Dict[str, Any]]) -> PipelineResult:
        """Create pipeline result object.
        
        Args:
            symbol: Symbol that was processed
            predictions: Generated predictions
            
        Returns:
            PipelineResult object
        """
        # Convert stage results to dictionaries with string values
        stage_results_dict = []
        for result in self.stage_results:
            result_dict = result.__dict__.copy()
            result_dict['stage'] = result.stage.value
            result_dict['status'] = result.status.value
            stage_results_dict.append(result_dict)
        
        return PipelineResult(
            execution_id=self.execution_id,
            symbol=symbol,
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status.value,
            total_duration_ms=self._get_total_duration(),
            stage_results=stage_results_dict,
            predictions=predictions,
            metadata={
                'config_version': self.config.get('version', 'unknown'),
                'stages_completed': len([r for r in self.stage_results if r.status == PipelineStatus.COMPLETED]),
                'total_stages': len(self.stage_results)
            }
        )
    
    def _create_error_result(self, symbol: str, error_message: str) -> PipelineResult:
        """Create error result object.
        
        Args:
            symbol: Symbol that was being processed
            error_message: Error description
            
        Returns:
            PipelineResult object with error information
        """
        # Convert stage results to dictionaries with string values
        stage_results_dict = []
        for result in self.stage_results:
            result_dict = result.__dict__.copy()
            result_dict['stage'] = result.stage.value
            result_dict['status'] = result.status.value
            stage_results_dict.append(result_dict)
        
        return PipelineResult(
            execution_id=self.execution_id,
            symbol=symbol,
            start_time=self.start_time,
            end_time=self.end_time,
            status=self.status.value,
            total_duration_ms=self._get_total_duration(),
            stage_results=stage_results_dict,
            predictions=[],
            error_message=error_message,
            metadata={
                'failed_at_stage': self.stage_results[-1].stage.value if self.stage_results else 'initialization',
                'stages_completed': len([r for r in self.stage_results if r.status == PipelineStatus.COMPLETED]),
                'total_stages': len(self.stage_results)
            }
        )
    
    def _get_total_duration(self) -> float:
        """Get total pipeline execution duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution.
        
        Returns:
            Dictionary containing execution summary
        """
        return {
            'execution_id': self.execution_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_duration_ms': self._get_total_duration(),
            'stages': [
                {
                    'stage': result.stage.value,
                    'status': result.status.value,
                    'duration_ms': result.duration_ms,
                    'data_count': result.data_count,
                    'error': result.error_message
                }
                for result in self.stage_results
            ]
        }