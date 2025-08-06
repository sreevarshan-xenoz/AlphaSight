# Machine Learning API

## Overview

The machine learning module provides CPU-optimized XGBoost models for NIFTY 50 predictions with sub-10ms inference latency and comprehensive model management capabilities.

## XGBoostPredictor

### Class: `XGBoostPredictor`

CPU-optimized XGBoost predictor for NIFTY 50 index predictions.

#### Constructor

```python
XGBoostPredictor(model_path: Optional[str] = None)
```

**Parameters:**
- `model_path`: Optional path to load pre-trained model

**CPU-Optimized Hyperparameters:**
- `n_jobs`: 1 (single-threaded for CPU optimization)
- `tree_method`: 'exact' (CPU-optimized tree construction)
- `max_depth`: 6 (balanced complexity)
- `learning_rate`: 0.1
- `n_estimators`: 100

#### Methods

##### `train(X, y, validation_split)`

Train XGBoost model with TimeSeriesSplit for chronological validation.

```python
def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, float]
```

**Parameters:**
- `X`: Feature matrix with datetime index
- `y`: Target values (future returns or prices)
- `validation_split`: Fraction of data for validation (default: 0.2)

**Returns:**
- `Dict[str, float]`: Dictionary containing training metrics

**Example:**
```python
from nifty_ml_pipeline.models.predictor import XGBoostPredictor
import pandas as pd

predictor = XGBoostPredictor()

# Assuming X is feature matrix and y is target
metrics = predictor.train(X, y)
print(f"Training RMSE: {metrics['train_rmse']:.4f}")
print(f"CV RMSE: {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f}")
print(f"Training time: {metrics['training_time_seconds']:.2f}s")
```

##### `predict(X)`

Generate predictions with optimized inference for single samples.

```python
def predict(self, X: Union[pd.DataFrame, FeatureVector]) -> Union[np.ndarray, float]
```

**Parameters:**
- `X`: Feature matrix or single FeatureVector

**Returns:**
- `Union[np.ndarray, float]`: Predictions as numpy array or single float

**Example:**
```python
# Single prediction
from nifty_ml_pipeline.data.models import FeatureVector
from datetime import datetime

feature_vector = FeatureVector(
    timestamp=datetime.now(),
    symbol="NIFTY 50",
    lag1_return=0.01,
    lag2_return=-0.005,
    sma_5_ratio=1.02,
    rsi_14=65.5,
    macd_hist=0.15,
    daily_sentiment=0.3
)

prediction = predictor.predict(feature_vector)
print(f"Predicted price: {prediction:.2f}")
```

##### `predict_with_confidence(X)`

Generate prediction with confidence score.

```python
def predict_with_confidence(self, X: Union[pd.DataFrame, FeatureVector]) -> Tuple[float, float]
```

**Parameters:**
- `X`: Feature matrix or single FeatureVector

**Returns:**
- `Tuple[float, float]`: Tuple of (prediction, confidence_score)

**Example:**
```python
prediction, confidence = predictor.predict_with_confidence(feature_vector)
print(f"Prediction: {prediction:.2f}, Confidence: {confidence:.3f}")
```

##### `generate_signal(current_price, predicted_price, confidence, threshold)`

Generate trading signal based on prediction and confidence.

```python
def generate_signal(self, current_price: float, predicted_price: float, 
                   confidence: float, threshold: float = 0.02) -> str
```

**Parameters:**
- `current_price`: Current market price
- `predicted_price`: Model prediction
- `confidence`: Prediction confidence
- `threshold`: Minimum percentage change for signal generation (default: 0.02)

**Returns:**
- `str`: Trading signal: "Buy", "Sell", or "Hold"

**Example:**
```python
signal = predictor.generate_signal(
    current_price=18500.0,
    predicted_price=18650.0,
    confidence=0.85,
    threshold=0.02
)
print(f"Trading signal: {signal}")
```

##### `save_model(filepath)`

Save trained model to disk.

```python
def save_model(self, filepath: str) -> None
```

**Parameters:**
- `filepath`: Path to save the model

##### `load_model(filepath)`

Load trained model from disk.

```python
def load_model(self, filepath: str) -> None
```

**Parameters:**
- `filepath`: Path to the saved model

##### `get_feature_importance()`

Get feature importance scores from trained model.

```python
def get_feature_importance(self) -> Dict[str, float]
```

**Returns:**
- `Dict[str, float]`: Dictionary mapping feature names to importance scores

**Example:**
```python
importance = predictor.get_feature_importance()
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.4f}")
```

## InferenceEngine

### Class: `InferenceEngine`

High-performance inference engine targeting sub-10ms latency.

#### Constructor

```python
InferenceEngine(predictor: XGBoostPredictor)
```

**Parameters:**
- `predictor`: Trained XGBoostPredictor instance

#### Methods

##### `predict_single(features, symbol)`

Generate single prediction with performance monitoring.

```python
def predict_single(self, features: Dict[str, float], symbol: str) -> PredictionResult
```

**Parameters:**
- `features`: Dictionary of feature values
- `symbol`: Symbol being predicted

**Returns:**
- `PredictionResult`: Structured prediction result with metadata

**Example:**
```python
from nifty_ml_pipeline.models.inference_engine import InferenceEngine

engine = InferenceEngine(predictor)
features = {
    'lag1_return': 0.01,
    'lag2_return': -0.005,
    'sma_5_ratio': 1.02,
    'rsi_14': 65.5,
    'macd_hist': 0.15,
    'daily_sentiment': 0.3
}

result = engine.predict_single(features, "NIFTY 50")
print(f"Signal: {result.signal}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Inference time: {result.inference_time_ms:.2f}ms")
```

##### `batch_predict(feature_list, symbol)`

Generate predictions for multiple feature vectors.

```python
def batch_predict(self, feature_list: List[Dict[str, float]], symbol: str) -> List[PredictionResult]
```

**Parameters:**
- `feature_list`: List of feature dictionaries
- `symbol`: Symbol being predicted

**Returns:**
- `List[PredictionResult]`: List of prediction results

## ModelValidator

### Class: `ModelValidator`

Implements time series cross-validation and performance tracking.

#### Constructor

```python
ModelValidator()
```

#### Methods

##### `validate_model(model, X, y, n_splits)`

Perform time series cross-validation.

```python
def validate_model(self, model: XGBoostPredictor, X: pd.DataFrame, y: pd.Series, 
                  n_splits: int = 5) -> Dict[str, float]
```

**Parameters:**
- `model`: XGBoostPredictor to validate
- `X`: Feature matrix
- `y`: Target values
- `n_splits`: Number of time series splits (default: 5)

**Returns:**
- `Dict[str, float]`: Validation metrics

**Example:**
```python
from nifty_ml_pipeline.models.validator import ModelValidator

validator = ModelValidator()
metrics = validator.validate_model(predictor, X, y)
print(f"Validation RMSE: {metrics['rmse']:.4f}")
print(f"Directional Accuracy: {metrics['directional_accuracy']:.3f}")
```

##### `calculate_directional_accuracy(y_true, y_pred)`

Calculate directional accuracy for trading predictions.

```python
def calculate_directional_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float
```

**Parameters:**
- `y_true`: Actual values
- `y_pred`: Predicted values

**Returns:**
- `float`: Directional accuracy (0.0 to 1.0)

## Data Models

### PredictionResult

Structured output of the prediction pipeline.

```python
@dataclass
class PredictionResult:
    timestamp: datetime
    symbol: str
    predicted_close: float
    signal: str  # "Buy", "Hold", "Sell"
    confidence: float  # 0.0 to 1.0
    model_version: str
    features_used: List[str] = field(default_factory=list)
```

**Methods:**
- `to_dict() -> Dict`: Convert to dictionary for serialization
- `is_actionable(min_confidence: float = 0.7) -> bool`: Check if prediction confidence meets threshold

## Performance Specifications

### Inference Latency
- **Target**: < 10ms on standard CPU hardware
- **Optimization**: Single-threaded execution with exact tree method
- **Monitoring**: Automatic latency tracking and alerting

### Model Accuracy
- **Target**: 80%+ directional accuracy
- **Validation**: TimeSeriesSplit for chronological validation
- **Metrics**: RMSE, directional accuracy, precision, recall

### CPU Optimization
- **XGBoost Configuration**: Optimized for CPU execution
- **Memory Usage**: Minimal memory footprint
- **Threading**: Single-threaded for consistent performance

## Error Handling

### Model Training Errors
- **Data Validation**: Comprehensive input validation
- **Training Failures**: Graceful handling with detailed error messages
- **Recovery**: Automatic fallback to previous model versions

### Inference Errors
- **Input Validation**: Feature vector validation
- **Performance Monitoring**: Latency and accuracy tracking
- **Fallback**: Neutral predictions for failed inference

### Model Management
- **Serialization**: Robust model saving and loading
- **Versioning**: Automatic model versioning with metadata
- **Validation**: Model integrity checks on load

## Best Practices

### Training
- Use TimeSeriesSplit for chronological validation
- Monitor for overfitting with cross-validation
- Regular retraining with updated data

### Inference
- Validate feature vectors before prediction
- Monitor inference latency continuously
- Use confidence thresholds for actionable signals

### Model Management
- Version control for model artifacts
- Regular performance evaluation
- Automated model deployment and rollback