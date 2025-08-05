# nifty_ml_pipeline/models/predictor.py
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ..data.models import FeatureVector, PredictionResult


class XGBoostPredictor:
    """CPU-optimized XGBoost predictor for NIFTY 50 index predictions.
    
    Configured specifically for CPU execution with optimized hyperparameters
    to achieve sub-10ms inference latency while maintaining high accuracy.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize XGBoost predictor with CPU-optimized configuration.
        
        Args:
            model_path: Optional path to load pre-trained model
        """
        self.model = None
        self.feature_names = [
            'lag1_return', 'lag2_return', 'sma_5_ratio', 
            'rsi_14', 'macd_hist', 'daily_sentiment'
        ]
        self.model_version = f"xgb_cpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_trained = False
        
        # CPU-optimized hyperparameters
        self.hyperparameters = {
            'n_jobs': 1,  # Single-threaded for CPU optimization
            'tree_method': 'exact',  # CPU-optimized tree construction
            'max_depth': 6,  # Balanced complexity
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'reg:squarederror',  # Regression for price prediction
            'eval_metric': 'rmse'
        }
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.2) -> Dict[str, float]:
        """Train XGBoost model with TimeSeriesSplit for chronological validation.
        
        Args:
            X: Feature matrix with datetime index
            y: Target values (future returns or prices)
            validation_split: Fraction of data for validation
            
        Returns:
            Dictionary containing training metrics
        """
        if len(X) != len(y):
            raise ValueError("Feature matrix and target must have same length")
        
        # Ensure chronological ordering
        if not X.index.is_monotonic_increasing:
            sort_idx = X.index.argsort()
            X = X.iloc[sort_idx]
            y = y.iloc[sort_idx]
        
        # Initialize model with CPU-optimized parameters
        self.model = xgb.XGBRegressor(**self.hyperparameters)
        
        # TimeSeriesSplit for chronological validation
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = []
        training_start = time.perf_counter()
        
        # Cross-validation with time series splits
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train on fold
            fold_model = xgb.XGBRegressor(**self.hyperparameters)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Validate on fold
            y_pred = fold_model.predict(X_val_fold)
            fold_score = np.sqrt(np.mean((y_val_fold - y_pred) ** 2))
            cv_scores.append(fold_score)
        
        # Train final model on all data
        self.model.fit(X, y)
        training_end = time.perf_counter()
        
        self.is_trained = True
        
        # Calculate training metrics
        y_pred_train = self.model.predict(X)
        train_rmse = np.sqrt(np.mean((y - y_pred_train) ** 2))
        
        metrics = {
            'train_rmse': train_rmse,
            'cv_rmse_mean': np.mean(cv_scores),
            'cv_rmse_std': np.std(cv_scores),
            'training_time_seconds': training_end - training_start,
            'n_features': X.shape[1],
            'n_samples': len(X)
        }
        
        return metrics
    
    def predict(self, X: Union[pd.DataFrame, FeatureVector]) -> Union[np.ndarray, float]:
        """Generate predictions with optimized inference for single samples.
        
        Args:
            X: Feature matrix or single FeatureVector
            
        Returns:
            Predictions as numpy array or single float
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        inference_start = time.perf_counter()
        
        # Handle single FeatureVector input
        if isinstance(X, FeatureVector):
            features = np.array(X.to_array()).reshape(1, -1)
            prediction = self.model.predict(features)[0]
            
        # Handle DataFrame input
        elif isinstance(X, pd.DataFrame):
            prediction = self.model.predict(X)
            
        else:
            raise ValueError("Input must be pandas DataFrame or FeatureVector")
        
        inference_end = time.perf_counter()
        inference_time_ms = (inference_end - inference_start) * 1000
        
        # Log performance for monitoring
        if inference_time_ms > 10.0:
            print(f"Warning: Inference time {inference_time_ms:.2f}ms exceeds 10ms target")
        
        return prediction
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, FeatureVector]) -> Tuple[float, float]:
        """Generate prediction with confidence score.
        
        Args:
            X: Feature matrix or single FeatureVector
            
        Returns:
            Tuple of (prediction, confidence_score)
        """
        prediction = self.predict(X)
        
        # Simple confidence based on feature importance and prediction magnitude
        # In production, this could be enhanced with prediction intervals
        if isinstance(X, FeatureVector):
            feature_values = np.array(X.to_array())
        else:
            feature_values = X.iloc[0].values if len(X) == 1 else X.mean().values
        
        # Normalize confidence based on feature stability
        confidence = min(0.95, max(0.5, 1.0 - np.std(feature_values) * 0.1))
        
        return float(prediction) if isinstance(prediction, np.ndarray) else prediction, confidence
    
    def generate_signal(self, current_price: float, predicted_price: float, 
                       confidence: float, threshold: float = 0.02) -> str:
        """Generate trading signal based on prediction and confidence.
        
        Args:
            current_price: Current market price
            predicted_price: Model prediction
            confidence: Prediction confidence
            threshold: Minimum percentage change for signal generation
            
        Returns:
            Trading signal: "Buy", "Sell", or "Hold"
        """
        if confidence < 0.6:  # Low confidence threshold
            return "Hold"
        
        price_change_pct = (predicted_price - current_price) / current_price
        
        if price_change_pct > threshold:
            return "Buy"
        elif price_change_pct < -threshold:
            return "Sell"
        else:
            return "Hold"
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_version': self.model_version,
            'hyperparameters': self.hyperparameters,
            'is_trained': self.is_trained
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_version = model_data['model_version']
        self.hyperparameters = model_data['hyperparameters']
        self.is_trained = model_data['is_trained']
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_scores = self.model.feature_importances_
        return dict(zip(self.feature_names, importance_scores))