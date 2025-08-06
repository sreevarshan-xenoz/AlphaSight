# nifty_ml_pipeline/output/prediction_formatter.py
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict
import pandas as pd
import numpy as np

from ..data.models import PredictionResult


logger = logging.getLogger(__name__)


class PredictionFormatter:
    """Formats prediction results with buy/sell/hold signals and confidence-based filtering.
    
    Handles conversion of raw model outputs to actionable trading signals
    with proper formatting for storage and reporting.
    """
    
    def __init__(self, min_confidence: float = 0.7, signal_thresholds: Optional[Dict[str, float]] = None):
        """Initialize prediction formatter.
        
        Args:
            min_confidence: Minimum confidence threshold for actionable signals
            signal_thresholds: Custom thresholds for buy/sell signals
        """
        self.min_confidence = min_confidence
        self.signal_thresholds = signal_thresholds or {
            'buy_threshold': 0.02,   # 2% predicted upward movement
            'sell_threshold': -0.02  # 2% predicted downward movement
        }
        
        logger.info(f"Initialized PredictionFormatter with min_confidence={min_confidence}")
    
    def format_prediction(self, 
                         symbol: str,
                         timestamp: datetime,
                         predicted_price: float,
                         current_price: float,
                         confidence: float,
                         model_version: str,
                         features_used: List[str]) -> PredictionResult:
        """Format a single prediction result with trading signal.
        
        Args:
            symbol: Stock symbol
            timestamp: Prediction timestamp
            predicted_price: Model predicted price
            current_price: Current market price
            confidence: Model confidence score (0.0 to 1.0)
            model_version: Version of the model used
            features_used: List of features used in prediction
            
        Returns:
            PredictionResult: Formatted prediction with signal
        """
        try:
            # Calculate predicted return
            predicted_return = (predicted_price - current_price) / current_price
            
            # Generate trading signal based on predicted return
            signal = self._generate_signal(predicted_return, confidence)
            
            # Create prediction result
            prediction = PredictionResult(
                timestamp=timestamp,
                symbol=symbol,
                predicted_close=predicted_price,
                signal=signal,
                confidence=confidence,
                model_version=model_version,
                features_used=features_used.copy()
            )
            
            logger.debug(f"Formatted prediction for {symbol}: {signal} (confidence: {confidence:.3f})")
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to format prediction for {symbol}: {str(e)}")
            # Return neutral prediction on error
            safe_price = current_price if current_price and current_price > 0 else 1000.0
            return PredictionResult(
                timestamp=timestamp,
                symbol=symbol,
                predicted_close=safe_price,
                signal="Hold",
                confidence=0.0,
                model_version=model_version,
                features_used=features_used.copy()
            )
    
    def format_batch_predictions(self, 
                                predictions_data: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Format multiple predictions in batch.
        
        Args:
            predictions_data: List of prediction data dictionaries
            
        Returns:
            List[PredictionResult]: Formatted predictions
        """
        formatted_predictions = []
        
        for pred_data in predictions_data:
            try:
                prediction = self.format_prediction(
                    symbol=pred_data['symbol'],
                    timestamp=pred_data['timestamp'],
                    predicted_price=pred_data['predicted_price'],
                    current_price=pred_data['current_price'],
                    confidence=pred_data['confidence'],
                    model_version=pred_data['model_version'],
                    features_used=pred_data['features_used']
                )
                formatted_predictions.append(prediction)
                
            except KeyError as e:
                logger.error(f"Missing required field in prediction data: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"Failed to format prediction: {str(e)}")
                continue
        
        logger.info(f"Formatted {len(formatted_predictions)} predictions from {len(predictions_data)} inputs")
        return formatted_predictions
    
    def filter_actionable_signals(self, predictions: List[PredictionResult]) -> List[PredictionResult]:
        """Filter predictions to only include actionable signals.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            List[PredictionResult]: Filtered actionable predictions
        """
        actionable_predictions = []
        
        for prediction in predictions:
            if prediction.is_actionable(self.min_confidence):
                actionable_predictions.append(prediction)
                logger.debug(f"Actionable signal: {prediction.symbol} - {prediction.signal}")
            else:
                logger.debug(f"Low confidence signal filtered: {prediction.symbol} - {prediction.confidence:.3f}")
        
        logger.info(f"Filtered to {len(actionable_predictions)} actionable signals from {len(predictions)} total")
        return actionable_predictions
    
    def to_dataframe(self, predictions: List[PredictionResult]) -> pd.DataFrame:
        """Convert predictions to pandas DataFrame for analysis.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            pd.DataFrame: Predictions as DataFrame
        """
        if not predictions:
            return pd.DataFrame()
        
        try:
            # Convert to list of dictionaries
            data = []
            for pred in predictions:
                pred_dict = asdict(pred)
                # Convert timestamp to string for JSON serialization
                pred_dict['timestamp'] = pred.timestamp.isoformat()
                data.append(pred_dict)
            
            df = pd.DataFrame(data)
            
            # Convert timestamp back to datetime for proper sorting
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            logger.info(f"Converted {len(predictions)} predictions to DataFrame")
            return df
            
        except Exception as e:
            logger.error(f"Failed to convert predictions to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def to_json_serializable(self, predictions: List[PredictionResult]) -> List[Dict[str, Any]]:
        """Convert predictions to JSON-serializable format.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            List[Dict]: JSON-serializable prediction data
        """
        serializable_data = []
        
        for prediction in predictions:
            try:
                pred_dict = prediction.to_dict()
                serializable_data.append(pred_dict)
            except Exception as e:
                logger.error(f"Failed to serialize prediction: {str(e)}")
                continue
        
        logger.info(f"Converted {len(serializable_data)} predictions to JSON format")
        return serializable_data
    
    def generate_summary_stats(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Generate summary statistics for predictions.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dict: Summary statistics
        """
        if not predictions:
            return {
                'total_predictions': 0,
                'actionable_predictions': 0,
                'signal_distribution': {},
                'avg_confidence': 0.0,
                'confidence_distribution': {}
            }
        
        try:
            actionable_preds = self.filter_actionable_signals(predictions)
            
            # Signal distribution
            signal_counts = {}
            for pred in predictions:
                signal_counts[pred.signal] = signal_counts.get(pred.signal, 0) + 1
            
            # Confidence statistics
            confidences = [pred.confidence for pred in predictions]
            avg_confidence = np.mean(confidences)
            
            # Confidence distribution (binned)
            confidence_bins = {
                'low (0.0-0.5)': sum(1 for c in confidences if 0.0 <= c < 0.5),
                'medium (0.5-0.7)': sum(1 for c in confidences if 0.5 <= c < 0.7),
                'high (0.7-1.0)': sum(1 for c in confidences if 0.7 <= c <= 1.0)
            }
            
            summary = {
                'total_predictions': len(predictions),
                'actionable_predictions': len(actionable_preds),
                'signal_distribution': signal_counts,
                'avg_confidence': round(avg_confidence, 3),
                'confidence_distribution': confidence_bins,
                'timestamp_range': {
                    'earliest': min(pred.timestamp for pred in predictions).isoformat(),
                    'latest': max(pred.timestamp for pred in predictions).isoformat()
                }
            }
            
            logger.info(f"Generated summary stats for {len(predictions)} predictions")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary stats: {str(e)}")
            return {'error': str(e)}
    
    def _generate_signal(self, predicted_return: float, confidence: float) -> str:
        """Generate trading signal based on predicted return and confidence.
        
        Args:
            predicted_return: Predicted price return (percentage)
            confidence: Model confidence score
            
        Returns:
            str: Trading signal ('Buy', 'Sell', 'Hold')
        """
        # Only generate buy/sell signals if confidence is above threshold
        if confidence < self.min_confidence:
            return "Hold"
        
        # Generate signal based on predicted return thresholds
        if predicted_return >= self.signal_thresholds['buy_threshold']:
            return "Buy"
        elif predicted_return <= self.signal_thresholds['sell_threshold']:
            return "Sell"
        else:
            return "Hold"
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update signal generation thresholds.
        
        Args:
            new_thresholds: New threshold values
        """
        self.signal_thresholds.update(new_thresholds)
        logger.info(f"Updated signal thresholds: {self.signal_thresholds}")
    
    def update_min_confidence(self, new_min_confidence: float) -> None:
        """Update minimum confidence threshold.
        
        Args:
            new_min_confidence: New minimum confidence threshold
        """
        if not (0.0 <= new_min_confidence <= 1.0):
            raise ValueError("Minimum confidence must be between 0.0 and 1.0")
        
        self.min_confidence = new_min_confidence
        logger.info(f"Updated minimum confidence threshold to {new_min_confidence}")