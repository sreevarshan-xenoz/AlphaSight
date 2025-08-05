# nifty_ml_pipeline/models/inference_engine.py
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.models import FeatureVector, PredictionResult
from .predictor import XGBoostPredictor


class InferenceEngine:
    """Real-time inference engine optimized for sub-10ms latency predictions.
    
    Provides optimized single-sample prediction capabilities with confidence
    scoring and result formatting for trading applications.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 confidence_threshold: float = 0.6):
        """Initialize inference engine with pre-trained model.
        
        Args:
            model_path: Path to pre-trained XGBoost model
            confidence_threshold: Minimum confidence for actionable predictions
        """
        self.predictor = XGBoostPredictor(model_path)
        self.confidence_threshold = confidence_threshold
        self.prediction_cache = {}  # Simple cache for repeated predictions
        self.performance_metrics = {
            'total_predictions': 0,
            'avg_latency_ms': 0.0,
            'cache_hits': 0,
            'high_confidence_predictions': 0
        }
        
        # Pre-warm the model for consistent latency
        if self.predictor.is_trained:
            self._warm_up_model()
    
    def _warm_up_model(self) -> None:
        """Pre-warm the model to ensure consistent inference latency."""
        dummy_features = FeatureVector(
            timestamp=datetime.now(),
            symbol="WARMUP",
            lag1_return=0.0,
            lag2_return=0.0,
            sma_5_ratio=1.0,
            rsi_14=50.0,
            macd_hist=0.0,
            daily_sentiment=0.0
        )
        
        # Run a few dummy predictions to warm up
        for _ in range(3):
            self.predictor.predict(dummy_features)
    
    def predict_single(self, features: FeatureVector, 
                      current_price: float) -> PredictionResult:
        """Generate single real-time prediction with sub-10ms target latency.
        
        Args:
            features: Input feature vector
            current_price: Current market price for signal generation
            
        Returns:
            PredictionResult with prediction, signal, and confidence
        """
        if not self.predictor.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        start_time = time.perf_counter()
        
        # Check cache for repeated predictions (same timestamp + symbol)
        cache_key = f"{features.symbol}_{features.timestamp.isoformat()}"
        if cache_key in self.prediction_cache:
            self.performance_metrics['cache_hits'] += 1
            return self.prediction_cache[cache_key]
        
        # Generate prediction with confidence (returns future return, not price)
        predicted_return, confidence = self.predictor.predict_with_confidence(features)
        
        # Convert return to price
        predicted_price = current_price * (1 + predicted_return)
        
        # Generate trading signal
        signal = self.predictor.generate_signal(
            current_price, predicted_price, confidence
        )
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000
        
        # Create prediction result
        result = PredictionResult(
            timestamp=features.timestamp,
            symbol=features.symbol,
            predicted_close=predicted_price,
            signal=signal,
            confidence=confidence,
            model_version=self.predictor.model_version,
            features_used=self.predictor.feature_names
        )
        
        # Update performance metrics
        self._update_metrics(inference_time_ms, confidence)
        
        # Cache result for potential reuse
        self.prediction_cache[cache_key] = result
        
        # Log warning if latency exceeds target
        if inference_time_ms > 10.0:
            print(f"Warning: Inference latency {inference_time_ms:.2f}ms exceeds 10ms target")
        
        return result
    
    def predict_batch(self, features_list: List[FeatureVector], 
                     current_prices: List[float]) -> List[PredictionResult]:
        """Generate batch predictions with optimized processing.
        
        Args:
            features_list: List of feature vectors
            current_prices: List of current prices corresponding to features
            
        Returns:
            List of PredictionResult objects
        """
        if len(features_list) != len(current_prices):
            raise ValueError("Features and prices lists must have same length")
        
        results = []
        
        # Process each prediction individually for now
        # Could be optimized for true batch processing if needed
        for features, current_price in zip(features_list, current_prices):
            result = self.predict_single(features, current_price)
            results.append(result)
        
        return results
    
    def get_actionable_predictions(self, features_list: List[FeatureVector],
                                 current_prices: List[float]) -> List[PredictionResult]:
        """Get only high-confidence, actionable predictions.
        
        Args:
            features_list: List of feature vectors
            current_prices: List of current prices
            
        Returns:
            List of high-confidence PredictionResult objects
        """
        all_predictions = self.predict_batch(features_list, current_prices)
        
        actionable = [
            pred for pred in all_predictions 
            if pred.confidence >= self.confidence_threshold and pred.signal != "Hold"
        ]
        
        return actionable
    
    def _update_metrics(self, inference_time_ms: float, confidence: float) -> None:
        """Update performance tracking metrics.
        
        Args:
            inference_time_ms: Inference time in milliseconds
            confidence: Prediction confidence score
        """
        self.performance_metrics['total_predictions'] += 1
        
        # Update rolling average latency
        total_preds = self.performance_metrics['total_predictions']
        current_avg = self.performance_metrics['avg_latency_ms']
        self.performance_metrics['avg_latency_ms'] = (
            (current_avg * (total_preds - 1) + inference_time_ms) / total_preds
        )
        
        # Track high confidence predictions
        if confidence >= self.confidence_threshold:
            self.performance_metrics['high_confidence_predictions'] += 1
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance metrics summary.
        
        Returns:
            Dictionary containing performance statistics
        """
        total_preds = self.performance_metrics['total_predictions']
        
        if total_preds == 0:
            return {
                'total_predictions': 0,
                'avg_latency_ms': 0.0,
                'cache_hit_rate': 0.0,
                'high_confidence_rate': 0.0,
                'latency_target_met': True
            }
        
        cache_hit_rate = self.performance_metrics['cache_hits'] / total_preds
        high_conf_rate = self.performance_metrics['high_confidence_predictions'] / total_preds
        avg_latency = self.performance_metrics['avg_latency_ms']
        
        return {
            'total_predictions': total_preds,
            'avg_latency_ms': avg_latency,
            'cache_hit_rate': cache_hit_rate,
            'high_confidence_rate': high_conf_rate,
            'latency_target_met': avg_latency <= 10.0
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache to free memory."""
        self.prediction_cache.clear()
        self.performance_metrics['cache_hits'] = 0
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.performance_metrics = {
            'total_predictions': 0,
            'avg_latency_ms': 0.0,
            'cache_hits': 0,
            'high_confidence_predictions': 0
        }
        self.clear_cache()
    
    def validate_latency_requirement(self, num_samples: int = 100) -> Dict[str, float]:
        """Validate that inference meets sub-10ms latency requirement.
        
        Args:
            num_samples: Number of test predictions to run
            
        Returns:
            Dictionary with latency validation results
        """
        if not self.predictor.is_trained:
            raise ValueError("Model must be trained for latency validation")
        
        # Generate test features
        test_features = []
        for i in range(num_samples):
            features = FeatureVector(
                timestamp=datetime.now(),
                symbol=f"TEST_{i}",
                lag1_return=np.random.normal(0, 0.02),
                lag2_return=np.random.normal(0, 0.02),
                sma_5_ratio=np.random.normal(1.0, 0.1),
                rsi_14=np.random.uniform(20, 80),
                macd_hist=np.random.normal(0, 0.01),
                daily_sentiment=np.random.uniform(-0.5, 0.5)
            )
            test_features.append(features)
        
        # Measure latencies
        latencies = []
        for features in test_features:
            start_time = time.perf_counter()
            self.predictor.predict(features)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'max_latency_ms': float(np.max(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'target_met_rate': float(np.mean(latencies <= 10.0)),
            'num_samples': num_samples
        }
    
    def format_prediction_for_output(self, result: PredictionResult) -> Dict:
        """Format prediction result for external consumption.
        
        Args:
            result: PredictionResult to format
            
        Returns:
            Dictionary with formatted prediction data
        """
        return {
            'timestamp': result.timestamp.isoformat(),
            'symbol': result.symbol,
            'prediction': {
                'price': round(result.predicted_close, 2),
                'signal': result.signal,
                'confidence': round(result.confidence, 3)
            },
            'metadata': {
                'model_version': result.model_version,
                'features_count': len(result.features_used),
                'is_actionable': result.is_actionable(self.confidence_threshold)
            }
        }
    
    def load_model(self, model_path: str) -> None:
        """Load a different model into the inference engine.
        
        Args:
            model_path: Path to the model file
        """
        self.predictor.load_model(model_path)
        self.reset_metrics()
        
        # Re-warm the model
        if self.predictor.is_trained:
            self._warm_up_model()
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Update confidence threshold for actionable predictions.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold