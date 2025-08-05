# nifty_ml_pipeline/models/validator.py
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ..data.models import FeatureVector, PredictionResult
from .predictor import XGBoostPredictor


class ModelValidator:
    """Model validation class with time series cross-validation.
    
    Provides comprehensive validation metrics for time series models
    with proper chronological splitting to prevent look-ahead bias.
    """
    
    def __init__(self, n_splits: int = 5):
        """Initialize model validator.
        
        Args:
            n_splits: Number of time series splits for cross-validation
        """
        self.n_splits = n_splits
        self.validation_results = {}
        
    def validate_model(self, predictor: XGBoostPredictor, 
                      X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform comprehensive model validation with time series splits.
        
        Args:
            predictor: Trained XGBoost predictor
            X: Feature matrix with datetime index
            y: Target values
            
        Returns:
            Dictionary containing validation metrics
        """
        if not predictor.is_trained:
            raise ValueError("Predictor must be trained before validation")
        
        # Ensure chronological ordering
        if not X.index.is_monotonic_increasing:
            sort_idx = X.index.argsort()
            X = X.iloc[sort_idx]
            y = y.iloc[sort_idx]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        fold_metrics = []
        directional_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train fold-specific model
            fold_predictor = XGBoostPredictor()
            fold_predictor.hyperparameters = predictor.hyperparameters.copy()
            fold_predictor.train(X_train, y_train)
            
            # Generate predictions
            y_pred = fold_predictor.predict(X_val)
            
            # Calculate fold metrics
            fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            fold_mae = mean_absolute_error(y_val, y_pred)
            
            # Calculate directional accuracy
            directional_acc = self._calculate_directional_accuracy(y_val, y_pred)
            
            fold_metrics.append({
                'rmse': fold_rmse,
                'mae': fold_mae,
                'directional_accuracy': directional_acc,
                'n_samples': len(y_val)
            })
            
            directional_accuracies.append(directional_acc)
        
        # Aggregate results
        validation_metrics = {
            'cv_rmse_mean': np.mean([m['rmse'] for m in fold_metrics]),
            'cv_rmse_std': np.std([m['rmse'] for m in fold_metrics]),
            'cv_mae_mean': np.mean([m['mae'] for m in fold_metrics]),
            'cv_mae_std': np.std([m['mae'] for m in fold_metrics]),
            'cv_directional_accuracy_mean': np.mean(directional_accuracies),
            'cv_directional_accuracy_std': np.std(directional_accuracies),
            'n_folds': self.n_splits,
            'total_samples': len(X)
        }
        
        # Store results
        self.validation_results = validation_metrics
        
        return validation_metrics
    
    def _calculate_directional_accuracy(self, y_true: pd.Series, 
                                      y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy as percentage (0-100)
        """
        # Convert to direction (positive/negative)
        true_directions = np.sign(y_true.values)
        pred_directions = np.sign(y_pred)
        
        # Calculate accuracy
        correct_directions = np.sum(true_directions == pred_directions)
        total_predictions = len(y_true)
        
        return (correct_directions / total_predictions) * 100.0
    
    def validate_latency(self, predictor: XGBoostPredictor, 
                        num_samples: int = 1000) -> Dict[str, float]:
        """Validate inference latency requirements.
        
        Args:
            predictor: Trained predictor
            num_samples: Number of test samples
            
        Returns:
            Dictionary with latency statistics
        """
        if not predictor.is_trained:
            raise ValueError("Predictor must be trained for latency validation")
        
        # Generate test features
        test_features = []
        for i in range(num_samples):
            features = FeatureVector(
                timestamp=datetime.now() + timedelta(seconds=i),
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
            predictor.predict(features)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
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
    
    def validate_accuracy_threshold(self, directional_accuracy: float, 
                                  threshold: float = 80.0) -> Dict[str, bool]:
        """Validate if model meets accuracy threshold requirements.
        
        Args:
            directional_accuracy: Model's directional accuracy percentage
            threshold: Required accuracy threshold
            
        Returns:
            Dictionary with validation results
        """
        return {
            'meets_threshold': directional_accuracy >= threshold,
            'accuracy': directional_accuracy,
            'threshold': threshold,
            'margin': directional_accuracy - threshold
        }
    
    def get_validation_summary(self) -> Dict:
        """Get comprehensive validation summary.
        
        Returns:
            Dictionary with validation summary
        """
        if not self.validation_results:
            return {'status': 'No validation performed'}
        
        return {
            'status': 'Validation completed',
            'metrics': self.validation_results,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not self.validation_results:
            return ["Perform model validation first"]
        
        # Check directional accuracy
        dir_acc = self.validation_results.get('cv_directional_accuracy_mean', 0)
        if dir_acc < 75:
            recommendations.append("Directional accuracy below 75%. Consider feature engineering or model tuning.")
        elif dir_acc < 80:
            recommendations.append("Directional accuracy below target 80%. Minor improvements needed.")
        else:
            recommendations.append("Directional accuracy meets requirements.")
        
        # Check RMSE stability
        rmse_std = self.validation_results.get('cv_rmse_std', 0)
        rmse_mean = self.validation_results.get('cv_rmse_mean', 0)
        if rmse_mean > 0 and (rmse_std / rmse_mean) > 0.2:
            recommendations.append("High RMSE variance across folds. Model may be unstable.")
        
        return recommendations


class PerformanceTracker:
    """Performance tracking class for accuracy and latency monitoring.
    
    Tracks model performance over time and provides alerting capabilities
    for performance degradation.
    """
    
    def __init__(self, accuracy_threshold: float = 75.0):
        """Initialize performance tracker.
        
        Args:
            accuracy_threshold: Minimum accuracy threshold for alerts
        """
        self.accuracy_threshold = accuracy_threshold
        self.performance_history = []
        self.alerts = []
        
    def track_prediction(self, prediction: PredictionResult, 
                        actual_return: Optional[float] = None,
                        inference_time_ms: float = 0.0) -> None:
        """Track a single prediction performance.
        
        Args:
            prediction: Prediction result
            actual_return: Actual return (if available)
            inference_time_ms: Inference time in milliseconds
        """
        record = {
            'timestamp': prediction.timestamp,
            'symbol': prediction.symbol,
            'predicted_return': (prediction.predicted_close / 100.0) - 1,  # Assuming base price of 100
            'actual_return': actual_return,
            'confidence': prediction.confidence,
            'signal': prediction.signal,
            'inference_time_ms': inference_time_ms,
            'model_version': prediction.model_version
        }
        
        self.performance_history.append(record)
        
        # Check for performance issues
        self._check_performance_alerts()
    
    def calculate_directional_accuracy(self, lookback_days: int = 30) -> float:
        """Calculate directional accuracy over specified period.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Directional accuracy percentage
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        recent_records = [
            r for r in self.performance_history 
            if r['timestamp'] >= cutoff_date and r['actual_return'] is not None
        ]
        
        if not recent_records:
            return 0.0
        
        correct_directions = 0
        for record in recent_records:
            pred_direction = np.sign(record['predicted_return'])
            actual_direction = np.sign(record['actual_return'])
            
            if pred_direction == actual_direction:
                correct_directions += 1
        
        return (correct_directions / len(recent_records)) * 100.0
    
    def calculate_average_latency(self, lookback_days: int = 7) -> float:
        """Calculate average inference latency over specified period.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Average latency in milliseconds
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        recent_records = [
            r for r in self.performance_history 
            if r['timestamp'] >= cutoff_date and r['inference_time_ms'] > 0
        ]
        
        if not recent_records:
            return 0.0
        
        latencies = [r['inference_time_ms'] for r in recent_records]
        return np.mean(latencies)
    
    def get_performance_metrics(self, lookback_days: int = 30) -> Dict[str, float]:
        """Get comprehensive performance metrics.
        
        Args:
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary with performance metrics
        """
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        
        recent_records = [
            r for r in self.performance_history 
            if r['timestamp'] >= cutoff_date
        ]
        
        if not recent_records:
            return {
                'total_predictions': 0,
                'directional_accuracy': 0.0,
                'avg_latency_ms': 0.0,
                'avg_confidence': 0.0,
                'signal_distribution': {}
            }
        
        # Calculate metrics
        total_predictions = len(recent_records)
        directional_accuracy = self.calculate_directional_accuracy(lookback_days)
        avg_latency = self.calculate_average_latency(lookback_days)
        avg_confidence = np.mean([r['confidence'] for r in recent_records])
        
        # Signal distribution
        signals = [r['signal'] for r in recent_records]
        signal_counts = {signal: signals.count(signal) for signal in set(signals)}
        signal_distribution = {
            signal: (count / total_predictions) * 100 
            for signal, count in signal_counts.items()
        }
        
        return {
            'total_predictions': total_predictions,
            'directional_accuracy': directional_accuracy,
            'avg_latency_ms': avg_latency,
            'avg_confidence': avg_confidence,
            'signal_distribution': signal_distribution,
            'lookback_days': lookback_days
        }
    
    def _check_performance_alerts(self) -> None:
        """Check for performance degradation and generate alerts."""
        # Check recent accuracy
        recent_accuracy = self.calculate_directional_accuracy(lookback_days=7)
        
        # Check if we have enough data and accuracy is below threshold
        cutoff_date = datetime.now() - timedelta(days=7)
        recent_records_with_actual = [
            r for r in self.performance_history 
            if r['timestamp'] >= cutoff_date and r['actual_return'] is not None
        ]
        
        if len(recent_records_with_actual) >= 3 and recent_accuracy < self.accuracy_threshold:
            alert = {
                'timestamp': datetime.now(),
                'type': 'accuracy_degradation',
                'message': f"Directional accuracy ({recent_accuracy:.1f}%) below threshold ({self.accuracy_threshold}%)",
                'severity': 'high' if recent_accuracy < self.accuracy_threshold - 10 else 'medium'
            }
            self.alerts.append(alert)
        
        # Check recent latency
        recent_latency = self.calculate_average_latency(lookback_days=1)
        
        if recent_latency > 10.0:
            alert = {
                'timestamp': datetime.now(),
                'type': 'latency_degradation',
                'message': f"Average latency ({recent_latency:.2f}ms) exceeds 10ms target",
                'severity': 'medium' if recent_latency < 20.0 else 'high'
            }
            self.alerts.append(alert)
    
    def get_active_alerts(self, hours_back: int = 24) -> List[Dict]:
        """Get active alerts from specified time period.
        
        Args:
            hours_back: Number of hours to look back for alerts
            
        Returns:
            List of alert dictionaries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        return [
            alert for alert in self.alerts 
            if alert['timestamp'] >= cutoff_time
        ]
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
    
    def export_performance_data(self, filepath: str) -> None:
        """Export performance history to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        if not self.performance_history:
            raise ValueError("No performance data to export")
        
        df = pd.DataFrame(self.performance_history)
        df.to_csv(filepath, index=False)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.performance_history:
            return {'status': 'No performance data available'}
        
        metrics_7d = self.get_performance_metrics(7)
        metrics_30d = self.get_performance_metrics(30)
        active_alerts = self.get_active_alerts(24)
        
        return {
            'status': 'Performance tracking active',
            'total_predictions': len(self.performance_history),
            'metrics_7d': metrics_7d,
            'metrics_30d': metrics_30d,
            'active_alerts': len(active_alerts),
            'alert_details': active_alerts,
            'accuracy_threshold': self.accuracy_threshold
        }