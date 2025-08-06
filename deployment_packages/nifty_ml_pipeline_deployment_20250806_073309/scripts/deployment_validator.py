#!/usr/bin/env python3
"""
Deployment validation script for NIFTY 50 ML Pipeline.

This script validates the production deployment by running comprehensive
health checks, performance tests, and system validation.
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from config.manager import get_environment_config
from nifty_ml_pipeline.orchestration.controller import PipelineController
from nifty_ml_pipeline.models.inference_engine import InferenceEngine
from nifty_ml_pipeline.models.predictor import XGBoostPredictor
from nifty_ml_pipeline.utils.alerting import AlertManager, AlertSeverity, AlertCategory


@dataclass
class ValidationResult:
    """Validation test result."""
    test_name: str
    passed: bool
    duration_ms: float
    message: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class DeploymentValidationReport:
    """Complete deployment validation report."""
    timestamp: datetime
    environment: str
    version: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_status: str
    validation_duration_ms: float
    results: List[ValidationResult]
    system_info: Dict[str, Any]
    recommendations: List[str]


class DeploymentValidator:
    """Comprehensive deployment validator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize deployment validator.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or get_environment_config()
        self.results: List[ValidationResult] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize alert manager if configured
        self.alert_manager = None
        if self.config.get('monitoring', {}).get('alerts'):
            try:
                self.alert_manager = AlertManager(self.config['monitoring']['alerts'])
            except Exception as e:
                self.logger.warning(f"Could not initialize alert manager: {e}")
        
        self.logger.info("Deployment validator initialized")
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> ValidationResult:
        """Run a validation test and record results.
        
        Args:
            test_name: Name of the test
            test_func: Test function to run
            *args: Test function arguments
            **kwargs: Test function keyword arguments
            
        Returns:
            ValidationResult object
        """
        self.logger.info(f"Running test: {test_name}")
        
        start_time = time.perf_counter()
        
        try:
            result = test_func(*args, **kwargs)
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            if isinstance(result, tuple):
                passed, message, details = result
            elif isinstance(result, bool):
                passed, message, details = result, "Test completed", None
            else:
                passed, message, details = True, str(result), None
            
            validation_result = ValidationResult(
                test_name=test_name,
                passed=passed,
                duration_ms=duration_ms,
                message=message,
                details=details
            )
            
        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            validation_result = ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                message=f"Test failed with exception: {str(e)}",
                error=str(e)
            )
            
            self.logger.error(f"Test {test_name} failed: {e}")
        
        self.results.append(validation_result)
        
        status = "PASSED" if validation_result.passed else "FAILED"
        self.logger.info(f"Test {test_name}: {status} ({duration_ms:.2f}ms)")
        
        return validation_result
    
    def test_environment_setup(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test environment setup and configuration."""
        details = {}
        issues = []
        
        # Check Python version
        python_version = sys.version_info
        details['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        
        if python_version < (3, 8):
            issues.append("Python version should be 3.8 or higher")
        
        # Check required environment variables
        required_vars = [
            'ECONOMIC_TIMES_API_KEY',
            'LOG_LEVEL',
            'ENVIRONMENT'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            issues.append(f"Missing environment variables: {', '.join(missing_vars)}")
        
        details['environment_variables'] = {
            var: "SET" if os.getenv(var) else "MISSING" 
            for var in required_vars
        }
        
        # Check directory structure
        required_dirs = [
            self.config.get('paths', {}).get('data', 'data'),
            self.config.get('paths', {}).get('models', 'models'),
            self.config.get('paths', {}).get('logs', 'logs')
        ]
        
        dir_status = {}
        for dir_path in required_dirs:
            path = Path(dir_path)
            exists = path.exists()
            writable = path.is_dir() and os.access(path, os.W_OK) if exists else False
            
            dir_status[str(path)] = {
                'exists': exists,
                'writable': writable
            }
            
            if not exists:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    dir_status[str(path)]['created'] = True
                except Exception as e:
                    issues.append(f"Cannot create directory {path}: {e}")
                    dir_status[str(path)]['error'] = str(e)
        
        details['directories'] = dir_status
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage('/')
            free_gb = free / (1024**3)
            details['disk_space_gb'] = {
                'total': total / (1024**3),
                'used': used / (1024**3),
                'free': free_gb
            }
            
            if free_gb < 1.0:  # Less than 1GB free
                issues.append(f"Low disk space: {free_gb:.2f}GB free")
                
        except Exception as e:
            issues.append(f"Could not check disk space: {e}")
        
        passed = len(issues) == 0
        message = "Environment setup validated" if passed else f"Issues found: {'; '.join(issues)}"
        
        return passed, message, details
    
    def test_dependencies(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test required dependencies are installed and working."""
        details = {}
        issues = []
        
        # Test core dependencies
        required_packages = [
            ('pandas', 'pd'),
            ('numpy', 'np'),
            ('scikit-learn', 'sklearn'),
            ('xgboost', 'xgb'),
            ('nltk', 'nltk'),
            ('requests', 'requests')
        ]
        
        package_status = {}
        for package_name, import_name in required_packages:
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                package_status[package_name] = {
                    'installed': True,
                    'version': version
                }
            except ImportError as e:
                package_status[package_name] = {
                    'installed': False,
                    'error': str(e)
                }
                issues.append(f"Missing package: {package_name}")
        
        details['packages'] = package_status
        
        # Test NLTK data
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            # Try to create analyzer (will download data if needed)
            analyzer = SentimentIntensityAnalyzer()
            test_score = analyzer.polarity_scores("This is a test sentence.")
            
            details['nltk_data'] = {
                'available': True,
                'test_score': test_score
            }
        except Exception as e:
            details['nltk_data'] = {
                'available': False,
                'error': str(e)
            }
            issues.append(f"NLTK data not available: {e}")
        
        passed = len(issues) == 0
        message = "All dependencies validated" if passed else f"Dependency issues: {'; '.join(issues)}"
        
        return passed, message, details
    
    def test_configuration_loading(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test configuration loading and validation."""
        details = {}
        issues = []
        
        try:
            # Test configuration loading
            config = get_environment_config()
            details['config_loaded'] = True
            details['environment'] = config.get('environment', 'unknown')
            
            # Validate critical configuration sections
            required_sections = ['api', 'data', 'model', 'performance', 'logging']
            missing_sections = []
            
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
                else:
                    details[f'{section}_config'] = 'present'
            
            if missing_sections:
                issues.append(f"Missing config sections: {', '.join(missing_sections)}")
            
            # Validate performance thresholds
            perf_config = config.get('performance', {})
            if 'MAX_INFERENCE_LATENCY_MS' not in perf_config:
                issues.append("Missing inference latency threshold")
            
            if 'TARGET_ACCURACY' not in perf_config:
                issues.append("Missing target accuracy threshold")
            
            details['performance_thresholds'] = {
                'latency_ms': perf_config.get('MAX_INFERENCE_LATENCY_MS'),
                'accuracy': perf_config.get('TARGET_ACCURACY')
            }
            
        except Exception as e:
            issues.append(f"Configuration loading failed: {e}")
            details['config_error'] = str(e)
        
        passed = len(issues) == 0
        message = "Configuration validated" if passed else f"Config issues: {'; '.join(issues)}"
        
        return passed, message, details
    
    def test_model_loading(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test model loading and basic functionality."""
        details = {}
        issues = []
        
        try:
            # Test XGBoost predictor initialization
            predictor = XGBoostPredictor()
            details['predictor_created'] = True
            
            # Test inference engine initialization
            inference_engine = InferenceEngine(predictor)
            details['inference_engine_created'] = True
            
            # Test with mock data if no trained model available
            if not predictor.is_trained:
                # Create mock model for testing
                from unittest.mock import Mock, patch
                
                with patch('xgboost.XGBRegressor') as mock_xgb:
                    mock_model = Mock()
                    mock_model.predict.return_value = np.array([0.02])
                    mock_xgb.return_value = mock_model
                    
                    # Test prediction
                    from nifty_ml_pipeline.data.models import FeatureVector
                    test_features = FeatureVector(
                        timestamp=datetime.now(),
                        symbol="TEST",
                        lag1_return=0.01,
                        lag2_return=0.005,
                        sma_5_ratio=1.02,
                        rsi_14=55.0,
                        macd_hist=0.3,
                        daily_sentiment=0.1
                    )
                    
                    prediction = inference_engine.predict_single(test_features, 100.0)
                    details['test_prediction'] = {
                        'signal': prediction.signal,
                        'confidence': prediction.confidence
                    }
            else:
                details['trained_model_available'] = True
            
        except Exception as e:
            issues.append(f"Model loading failed: {e}")
            details['model_error'] = str(e)
        
        passed = len(issues) == 0
        message = "Model loading validated" if passed else f"Model issues: {'; '.join(issues)}"
        
        return passed, message, details
    
    def test_inference_performance(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test inference performance meets requirements."""
        details = {}
        issues = []
        
        try:
            from unittest.mock import Mock, patch
            
            with patch('xgboost.XGBRegressor') as mock_xgb:
                # Setup fast mock model
                mock_model = Mock()
                mock_model.predict.return_value = np.array([0.02])
                mock_xgb.return_value = mock_model
                
                predictor = XGBoostPredictor()
                predictor.is_trained = True
                inference_engine = InferenceEngine(predictor)
                
                # Test inference latency
                from nifty_ml_pipeline.data.models import FeatureVector
                test_features = FeatureVector(
                    timestamp=datetime.now(),
                    symbol="PERF_TEST",
                    lag1_return=0.01,
                    lag2_return=0.005,
                    sma_5_ratio=1.02,
                    rsi_14=55.0,
                    macd_hist=0.3,
                    daily_sentiment=0.1
                )
                
                # Warm up
                inference_engine.predict_single(test_features, 100.0)
                
                # Measure latency over multiple runs
                latencies = []
                num_tests = 100
                
                for _ in range(num_tests):
                    start_time = time.perf_counter()
                    prediction = inference_engine.predict_single(test_features, 100.0)
                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)
                
                latencies = np.array(latencies)
                
                # Calculate statistics
                mean_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                max_latency = np.max(latencies)
                
                details['latency_stats'] = {
                    'mean_ms': float(mean_latency),
                    'p95_ms': float(p95_latency),
                    'max_ms': float(max_latency),
                    'samples': num_tests
                }
                
                # Check against thresholds
                target_latency = self.config.get('performance', {}).get('MAX_INFERENCE_LATENCY_MS', 10.0)
                
                if mean_latency > target_latency:
                    issues.append(f"Mean latency {mean_latency:.2f}ms exceeds target {target_latency}ms")
                
                if p95_latency > target_latency * 1.5:
                    issues.append(f"P95 latency {p95_latency:.2f}ms exceeds threshold")
                
                details['performance_targets'] = {
                    'target_latency_ms': target_latency,
                    'mean_meets_target': mean_latency <= target_latency,
                    'p95_acceptable': p95_latency <= target_latency * 1.5
                }
                
        except Exception as e:
            issues.append(f"Performance testing failed: {e}")
            details['performance_error'] = str(e)
        
        passed = len(issues) == 0
        message = "Performance requirements met" if passed else f"Performance issues: {'; '.join(issues)}"
        
        return passed, message, details
    
    def test_data_pipeline(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test data pipeline components."""
        details = {}
        issues = []
        
        try:
            # Test data collectors
            from nifty_ml_pipeline.data.collectors import NSEDataCollector, NewsDataCollector
            
            # Test NSE collector initialization
            nse_collector = NSEDataCollector()
            details['nse_collector_created'] = True
            
            # Test news collector initialization
            news_collector = NewsDataCollector(self.config)
            details['news_collector_created'] = True
            
            # Test feature engineering components
            from nifty_ml_pipeline.features.technical_indicators import TechnicalIndicatorCalculator
            from nifty_ml_pipeline.features.sentiment_analysis import SentimentAnalyzer
            from nifty_ml_pipeline.features.feature_normalizer import FeatureNormalizer
            
            tech_calc = TechnicalIndicatorCalculator()
            sentiment_analyzer = SentimentAnalyzer()
            normalizer = FeatureNormalizer()
            
            details['feature_components_created'] = True
            
            # Test with sample data
            sample_price_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=30, freq='D'),
                'open': np.random.uniform(95, 105, 30),
                'high': np.random.uniform(100, 110, 30),
                'low': np.random.uniform(90, 100, 30),
                'close': np.random.uniform(95, 105, 30),
                'volume': np.random.randint(500000, 2000000, 30)
            })
            
            # Test technical indicators
            price_with_indicators = tech_calc.calculate_all_indicators(sample_price_data)
            details['technical_indicators_computed'] = len(price_with_indicators.columns) > len(sample_price_data.columns)
            
            # Test sentiment analysis
            sample_news = pd.DataFrame({
                'headline': ['Market outlook positive', 'Economic concerns rise'],
                'timestamp': pd.date_range('2024-01-01', periods=2, freq='15D'),
                'source': ['test_source'] * 2,
                'url': ['http://test.com/1', 'http://test.com/2']
            })
            
            news_with_sentiment = sentiment_analyzer.analyze_dataframe(sample_news)
            details['sentiment_analysis_completed'] = 'sentiment_score' in news_with_sentiment.columns
            
            # Test feature normalization
            feature_vectors = normalizer.create_feature_vectors(price_with_indicators, news_with_sentiment)
            details['feature_vectors_created'] = len(feature_vectors) > 0
            
        except Exception as e:
            issues.append(f"Data pipeline testing failed: {e}")
            details['pipeline_error'] = str(e)
        
        passed = len(issues) == 0
        message = "Data pipeline validated" if passed else f"Pipeline issues: {'; '.join(issues)}"
        
        return passed, message, details
    
    def test_monitoring_and_alerting(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test monitoring and alerting systems."""
        details = {}
        issues = []
        
        try:
            # Test performance monitor
            from nifty_ml_pipeline.orchestration.performance_monitor import PerformanceMonitor
            
            monitor = PerformanceMonitor(self.config)
            details['performance_monitor_created'] = True
            
            # Test alert manager if available
            if self.alert_manager:
                # Test alert system (without actually sending alerts)
                test_results = {}
                
                # Mock the actual sending to avoid spam
                original_send = self.alert_manager.send_alert
                
                def mock_send(*args, **kwargs):
                    return True
                
                self.alert_manager.send_alert = mock_send
                
                try:
                    # Test different alert types
                    test_results['performance_alert'] = self.alert_manager.send_performance_alert(
                        'test_metric', 15.0, 10.0
                    )
                    
                    test_results['error_alert'] = self.alert_manager.send_error_alert(
                        'TestError', 'This is a test error message'
                    )
                    
                    test_results['system_alert'] = self.alert_manager.send_system_alert(
                        'cpu', 85.0, 'High CPU usage detected'
                    )
                    
                finally:
                    # Restore original send method
                    self.alert_manager.send_alert = original_send
                
                details['alert_system_tests'] = test_results
                details['alert_manager_available'] = True
            else:
                details['alert_manager_available'] = False
                issues.append("Alert manager not configured")
            
        except Exception as e:
            issues.append(f"Monitoring system testing failed: {e}")
            details['monitoring_error'] = str(e)
        
        passed = len(issues) == 0
        message = "Monitoring systems validated" if passed else f"Monitoring issues: {'; '.join(issues)}"
        
        return passed, message, details
    
    def test_security_configuration(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Test security configuration."""
        details = {}
        issues = []
        
        try:
            # Check environment variables are not hardcoded
            config_files = list(project_root.rglob("config/*.py"))
            
            hardcoded_secrets = []
            for config_file in config_files:
                if config_file.name == '__pycache__':
                    continue
                    
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                        
                        # Look for potential hardcoded secrets
                        import re
                        secret_patterns = [
                            r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']',
                            r'password\s*=\s*["\'][^"\']{5,}["\']',
                            r'secret\s*=\s*["\'][^"\']{10,}["\']'
                        ]
                        
                        for pattern in secret_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                hardcoded_secrets.append(str(config_file.relative_to(project_root)))
                                break
                                
                except Exception:
                    continue
            
            if hardcoded_secrets:
                issues.append(f"Potential hardcoded secrets in: {', '.join(hardcoded_secrets)}")
            
            details['hardcoded_secrets_check'] = {
                'files_checked': len(config_files),
                'issues_found': len(hardcoded_secrets),
                'problematic_files': hardcoded_secrets
            }
            
            # Check file permissions
            sensitive_files = ['.env', 'config/production.py']
            permission_issues = []
            
            for file_path in sensitive_files:
                full_path = project_root / file_path
                if full_path.exists():
                    stat_info = full_path.stat()
                    mode = stat_info.st_mode
                    
                    # Check if world-readable
                    if mode & 0o004:
                        permission_issues.append(f"{file_path} is world-readable")
                    
                    # Check if world-writable
                    if mode & 0o002:
                        permission_issues.append(f"{file_path} is world-writable")
            
            if permission_issues:
                issues.extend(permission_issues)
            
            details['file_permissions'] = {
                'files_checked': len(sensitive_files),
                'permission_issues': permission_issues
            }
            
            # Check SSL configuration
            ssl_config = self.config.get('security', {}).get('ssl', {})
            if not ssl_config.get('verify_certificates', True):
                issues.append("SSL certificate verification is disabled")
            
            details['ssl_configuration'] = ssl_config
            
        except Exception as e:
            issues.append(f"Security validation failed: {e}")
            details['security_error'] = str(e)
        
        passed = len(issues) == 0
        message = "Security configuration validated" if passed else f"Security issues: {'; '.join(issues)}"
        
        return passed, message, details
    
    def run_comprehensive_validation(self) -> DeploymentValidationReport:
        """Run comprehensive deployment validation.
        
        Returns:
            DeploymentValidationReport with all results
        """
        self.logger.info("Starting comprehensive deployment validation")
        
        start_time = time.perf_counter()
        
        # Clear previous results
        self.results.clear()
        
        # Run all validation tests
        test_functions = [
            ("Environment Setup", self.test_environment_setup),
            ("Dependencies", self.test_dependencies),
            ("Configuration Loading", self.test_configuration_loading),
            ("Model Loading", self.test_model_loading),
            ("Inference Performance", self.test_inference_performance),
            ("Data Pipeline", self.test_data_pipeline),
            ("Monitoring & Alerting", self.test_monitoring_and_alerting),
            ("Security Configuration", self.test_security_configuration)
        ]
        
        for test_name, test_func in test_functions:
            self.run_test(test_name, test_func)
        
        end_time = time.perf_counter()
        total_duration_ms = (end_time - start_time) * 1000
        
        # Calculate results
        passed_tests = sum(1 for result in self.results if result.passed)
        failed_tests = len(self.results) - passed_tests
        
        overall_status = "PASSED" if failed_tests == 0 else "FAILED"
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Get system information
        system_info = self._get_system_info()
        
        report = DeploymentValidationReport(
            timestamp=datetime.now(),
            environment=self.config.get('environment', 'unknown'),
            version=self.config.get('version', 'unknown'),
            total_tests=len(self.results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            overall_status=overall_status,
            validation_duration_ms=total_duration_ms,
            results=self.results,
            system_info=system_info,
            recommendations=recommendations
        )
        
        self.logger.info(f"Deployment validation completed: {overall_status} "
                        f"({passed_tests}/{len(self.results)} tests passed)")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        failed_results = [r for r in self.results if not r.passed]
        
        if not failed_results:
            recommendations.append("✅ All validation tests passed! Deployment is ready for production.")
            return recommendations
        
        # Categorize failures and provide specific recommendations
        for result in failed_results:
            if "Environment Setup" in result.test_name:
                recommendations.append(
                    "🔧 Fix environment setup issues before deployment. "
                    "Ensure all required directories exist and environment variables are set."
                )
            
            elif "Dependencies" in result.test_name:
                recommendations.append(
                    "📦 Install missing dependencies using: pip install -r requirements.txt"
                )
            
            elif "Configuration" in result.test_name:
                recommendations.append(
                    "⚙️ Review configuration files and ensure all required sections are present."
                )
            
            elif "Model Loading" in result.test_name:
                recommendations.append(
                    "🤖 Check model files and ensure XGBoost is properly configured."
                )
            
            elif "Performance" in result.test_name:
                recommendations.append(
                    "⚡ Optimize inference performance or adjust performance thresholds."
                )
            
            elif "Data Pipeline" in result.test_name:
                recommendations.append(
                    "📊 Fix data pipeline components before processing real data."
                )
            
            elif "Monitoring" in result.test_name:
                recommendations.append(
                    "📈 Configure monitoring and alerting systems for production visibility."
                )
            
            elif "Security" in result.test_name:
                recommendations.append(
                    "🔒 Address security issues immediately before production deployment."
                )
        
        # Add general recommendations
        if len(failed_results) > 3:
            recommendations.append(
                "⚠️ Multiple validation failures detected. "
                "Consider running tests in a staging environment first."
            )
        
        return recommendations
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for the report."""
        import platform
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            system_info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'timestamp': datetime.now().isoformat(),
                'note': 'psutil not available for detailed system info'
            }
        
        return system_info
    
    def save_report(self, report: DeploymentValidationReport, 
                   output_dir: str = "deployment_reports") -> str:
        """Save deployment validation report.
        
        Args:
            report: Validation report
            output_dir: Output directory
            
        Returns:
            Path to saved report file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"deployment_validation_{timestamp}.json"
        report_path = output_path / filename
        
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        self.logger.info(f"Deployment validation report saved to {report_path}")
        return str(report_path)
    
    def print_summary(self, report: DeploymentValidationReport) -> None:
        """Print validation summary to console."""
        print("\n" + "="*80)
        print("DEPLOYMENT VALIDATION SUMMARY")
        print("="*80)
        print(f"Environment: {report.environment}")
        print(f"Version: {report.version}")
        print(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {report.validation_duration_ms:.2f}ms")
        print(f"Overall Status: {report.overall_status}")
        print(f"Tests: {report.passed_tests}/{report.total_tests} passed")
        
        if report.failed_tests > 0:
            print(f"\n❌ FAILED TESTS ({report.failed_tests}):")
            for result in report.results:
                if not result.passed:
                    print(f"  • {result.test_name}: {result.message}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print(f"\n💻 SYSTEM INFO:")
        for key, value in report.system_info.items():
            if key != 'timestamp':
                print(f"  {key}: {value}")


def main():
    """Main function to run deployment validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate NIFTY 50 ML Pipeline deployment")
    parser.add_argument("--output-dir", default="deployment_reports",
                       help="Output directory for reports")
    parser.add_argument("--config-file", help="Custom configuration file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--send-alerts", action="store_true",
                       help="Send alerts for validation failures")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = None
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    # Create validator
    validator = DeploymentValidator(config)
    
    try:
        # Run validation
        report = validator.run_comprehensive_validation()
        
        # Save report
        report_path = validator.save_report(report, args.output_dir)
        
        # Print summary
        validator.print_summary(report)
        
        # Send alerts if requested and there are failures
        if args.send_alerts and report.failed_tests > 0 and validator.alert_manager:
            validator.alert_manager.send_alert(
                severity=AlertSeverity.ERROR,
                category=AlertCategory.SYSTEM,
                title="Deployment Validation Failed",
                message=f"Deployment validation failed with {report.failed_tests} failed tests",
                details={
                    'total_tests': report.total_tests,
                    'failed_tests': report.failed_tests,
                    'report_path': report_path
                },
                source="deployment_validator"
            )
        
        # Return appropriate exit code
        return 0 if report.overall_status == "PASSED" else 1
        
    except Exception as e:
        print(f"Deployment validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())