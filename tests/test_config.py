"""
Tests for configuration module.
"""
import pytest
from config.settings import get_config, XGBOOST_CONFIG, TECHNICAL_INDICATORS


class TestConfiguration:
    """Test configuration loading and validation."""
    
    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary."""
        config = get_config()
        assert isinstance(config, dict)
    
    def test_config_has_required_sections(self):
        """Test that configuration has all required sections."""
        config = get_config()
        required_sections = ["api", "data", "model", "performance", "scheduling", "logging", "paths"]
        
        for section in required_sections:
            assert section in config, f"Missing configuration section: {section}"
    
    def test_xgboost_config_has_required_params(self):
        """Test that XGBoost configuration has required parameters."""
        required_params = ["n_jobs", "tree_method", "max_depth", "learning_rate", "n_estimators"]
        
        for param in required_params:
            assert param in XGBOOST_CONFIG, f"Missing XGBoost parameter: {param}"
    
    def test_technical_indicators_config(self):
        """Test technical indicators configuration."""
        assert TECHNICAL_INDICATORS["RSI_PERIOD"] == 14
        assert TECHNICAL_INDICATORS["SMA_PERIOD"] == 5
        assert TECHNICAL_INDICATORS["MACD_FAST"] == 12
        assert TECHNICAL_INDICATORS["MACD_SLOW"] == 26
    
    def test_performance_thresholds_are_valid(self):
        """Test that performance thresholds are within valid ranges."""
        config = get_config()
        perf = config["performance"]
        
        assert 0 < perf["MIN_ACCURACY"] <= 1
        assert 0 < perf["TARGET_ACCURACY"] <= 1
        assert perf["MAX_INFERENCE_LATENCY_MS"] > 0
        assert perf["MAX_SENTIMENT_PROCESSING_TIME_S"] > 0