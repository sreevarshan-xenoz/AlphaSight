#!/usr/bin/env python3
"""
Performance profiler and optimization script for NIFTY 50 ML Pipeline.

This script profiles system performance, identifies bottlenecks, validates
memory usage and CPU utilization under load, and provides optimization
recommendations.
"""

import cProfile
import pstats
import io
import time
import psutil
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import tempfile
import tracemalloc
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from config.manager import get_environment_config
from nifty_ml_pipeline.orchestration.controller import PipelineController
from nifty_ml_pipeline.models.inference_engine import InferenceEngine
from nifty_ml_pipeline.models.predictor import XGBoostPredictor
from nifty_ml_pipeline.features.technical_indicators import TechnicalIndicatorCalculator
from nifty_ml_pipeline.features.sentiment_analysis import SentimentAnalyzer
from nifty_ml_pipeline.features.feature_normalizer import FeatureNormalizer


@dataclass
class PerformanceProfile:
    """Performance profiling results."""
    timestamp: datetime
    component: str
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    peak_memory_mb: Optional[float] = None
    memory_allocations: Optional[int] = None
    function_calls: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    disk_usage_percent: float
    process_memory_mb: float
    process_cpu_percent: float
    thread_count: int
    file_descriptors: int


class PerformanceProfiler:
    """Comprehensive performance profiler for the ML pipeline."""
    
    def __init__(self, output_dir: str = "performance_reports"):
        """Initialize profiler with output directory.
        
        Args:
            output_dir: Directory to save profiling reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.profiles: List[PerformanceProfile] = []
        self.system_metrics: List[SystemMetrics] = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Get process for monitoring
        self.process = psutil.Process(os.getpid())
        
        self.logger.info(f"Performance profiler initialized, output dir: {self.output_dir}")
    
    def profile_component(self, component_name: str, operation_name: str, 
                         func, *args, **kwargs) -> Any:
        """Profile a specific component operation.
        
        Args:
            component_name: Name of the component being profiled
            operation_name: Name of the operation being profiled
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Start memory tracking
        tracemalloc.start()
        
        # Get initial system state
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = self.process.cpu_percent()
        
        # Profile execution
        profiler = cProfile.Profile()
        start_time = time.perf_counter()
        
        profiler.enable()
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        # Get final system state
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = self.process.cpu_percent()
        
        # Get memory allocation stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Get function call stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        total_calls = stats.total_calls
        
        # Create performance profile
        profile = PerformanceProfile(
            timestamp=datetime.now(),
            component=component_name,
            operation=operation_name,
            duration_ms=duration_ms,
            memory_usage_mb=final_memory - initial_memory,
            cpu_percent=(initial_cpu + final_cpu) / 2,
            peak_memory_mb=peak / 1024 / 1024,
            memory_allocations=current,
            function_calls=total_calls,
            metadata={
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'peak_memory_mb': peak / 1024 / 1024
            }
        )
        
        self.profiles.append(profile)
        
        # Save detailed profiling stats
        stats_file = self.output_dir / f"{component_name}_{operation_name}_profile.txt"
        with open(stats_file, 'w') as f:
            stats.print_stats(file=f)
        
        self.logger.info(f"Profiled {component_name}.{operation_name}: "
                        f"{duration_ms:.2f}ms, {profile.memory_usage_mb:.2f}MB")
        
        return result
    
    def record_system_metrics(self) -> SystemMetrics:
        """Record current system metrics.
        
        Returns:
            SystemMetrics object
        """
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get process metrics
        process_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        process_cpu = self.process.cpu_percent()
        
        try:
            thread_count = self.process.num_threads()
            file_descriptors = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
        except (psutil.AccessDenied, AttributeError):
            thread_count = 0
            file_descriptors = 0
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / 1024**3,
            memory_used_gb=memory.used / 1024**3,
            disk_usage_percent=disk.percent,
            process_memory_mb=process_memory,
            process_cpu_percent=process_cpu,
            thread_count=thread_count,
            file_descriptors=file_descriptors
        )
        
        self.system_metrics.append(metrics)
        return metrics
    
    def profile_inference_latency(self, num_samples: int = 1000) -> Dict[str, float]:
        """Profile inference engine latency with multiple samples.
        
        Args:
            num_samples: Number of inference samples to test
            
        Returns:
            Latency statistics
        """
        self.logger.info(f"Profiling inference latency with {num_samples} samples")
        
        # Create mock predictor for testing
        from unittest.mock import Mock, patch
        
        with patch('xgboost.XGBRegressor') as mock_xgb:
            # Setup mock model
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.02])
            mock_xgb.return_value = mock_model
            
            predictor = XGBoostPredictor()
            # Set up the mock model directly
            predictor.model = mock_model
            predictor.is_trained = True  # Mark as trained for testing
            
            inference_engine = InferenceEngine(predictor)
            
            # Generate test data
            test_features = []
            for i in range(num_samples):
                from nifty_ml_pipeline.data.models import FeatureVector
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
            
            # Profile inference latency
            latencies = []
            
            def run_inference():
                for features in test_features:
                    start_time = time.perf_counter()
                    inference_engine.predict_single(features, 100.0)  # Mock current price
                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)
            
            # Profile the inference run
            self.profile_component("InferenceEngine", "batch_inference", run_inference)
            
            # Calculate statistics
            latencies = np.array(latencies)
            
            stats = {
                'mean_latency_ms': float(np.mean(latencies)),
                'median_latency_ms': float(np.median(latencies)),
                'p95_latency_ms': float(np.percentile(latencies, 95)),
                'p99_latency_ms': float(np.percentile(latencies, 99)),
                'max_latency_ms': float(np.max(latencies)),
                'min_latency_ms': float(np.min(latencies)),
                'std_latency_ms': float(np.std(latencies)),
                'samples_under_10ms': int(np.sum(latencies <= 10.0)),
                'target_compliance_rate': float(np.mean(latencies <= 10.0)),
                'num_samples': num_samples
            }
            
            self.logger.info(f"Inference latency stats: "
                           f"mean={stats['mean_latency_ms']:.2f}ms, "
                           f"p95={stats['p95_latency_ms']:.2f}ms, "
                           f"compliance={stats['target_compliance_rate']:.1%}")
            
            return stats
    
    def profile_feature_engineering(self) -> Dict[str, Any]:
        """Profile feature engineering components.
        
        Returns:
            Feature engineering performance statistics
        """
        self.logger.info("Profiling feature engineering components")
        
        # Generate realistic test data
        price_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=365, freq='D'),
            'open': np.random.uniform(95, 105, 365),
            'high': np.random.uniform(100, 110, 365),
            'low': np.random.uniform(90, 100, 365),
            'close': np.random.uniform(95, 105, 365),
            'volume': np.random.randint(500000, 2000000, 365)
        })
        
        news_data = pd.DataFrame({
            'headline': [f"Market analysis and outlook {i}" for i in range(100)],
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='3D'),
            'source': ['test_source'] * 100,
            'url': [f'http://test.com/{i}' for i in range(100)]
        })
        
        results = {}
        
        # Profile technical indicators
        tech_calc = TechnicalIndicatorCalculator()
        price_with_indicators = self.profile_component(
            "TechnicalIndicators", "calculate_all",
            tech_calc.calculate_all_indicators, price_data
        )
        
        # Profile sentiment analysis
        sentiment_analyzer = SentimentAnalyzer()
        news_with_sentiment = self.profile_component(
            "SentimentAnalysis", "analyze_dataframe",
            sentiment_analyzer.analyze_dataframe, news_data
        )
        
        # Profile feature normalization
        normalizer = FeatureNormalizer()
        feature_vectors = self.profile_component(
            "FeatureNormalizer", "create_feature_vectors",
            normalizer.create_feature_vectors, price_with_indicators, news_with_sentiment
        )
        
        # Calculate per-record processing times
        tech_profile = next(p for p in self.profiles if p.component == "TechnicalIndicators")
        sentiment_profile = next(p for p in self.profiles if p.component == "SentimentAnalysis")
        normalizer_profile = next(p for p in self.profiles if p.component == "FeatureNormalizer")
        
        results = {
            'technical_indicators': {
                'total_duration_ms': tech_profile.duration_ms,
                'per_record_ms': tech_profile.duration_ms / len(price_data),
                'memory_usage_mb': tech_profile.memory_usage_mb,
                'records_processed': len(price_data)
            },
            'sentiment_analysis': {
                'total_duration_ms': sentiment_profile.duration_ms,
                'per_record_ms': sentiment_profile.duration_ms / len(news_data),
                'memory_usage_mb': sentiment_profile.memory_usage_mb,
                'records_processed': len(news_data)
            },
            'feature_normalization': {
                'total_duration_ms': normalizer_profile.duration_ms,
                'per_record_ms': normalizer_profile.duration_ms / len(feature_vectors),
                'memory_usage_mb': normalizer_profile.memory_usage_mb,
                'records_processed': len(feature_vectors)
            }
        }
        
        self.logger.info("Feature engineering profiling completed")
        return results
    
    def profile_memory_usage(self, duration_seconds: int = 60) -> Dict[str, Any]:
        """Profile memory usage over time.
        
        Args:
            duration_seconds: How long to monitor memory usage
            
        Returns:
            Memory usage statistics
        """
        self.logger.info(f"Profiling memory usage for {duration_seconds} seconds")
        
        memory_samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            metrics = self.record_system_metrics()
            memory_samples.append(metrics.process_memory_mb)
            time.sleep(1)  # Sample every second
        
        memory_samples = np.array(memory_samples)
        
        stats = {
            'mean_memory_mb': float(np.mean(memory_samples)),
            'max_memory_mb': float(np.max(memory_samples)),
            'min_memory_mb': float(np.min(memory_samples)),
            'std_memory_mb': float(np.std(memory_samples)),
            'memory_growth_mb': float(memory_samples[-1] - memory_samples[0]),
            'peak_memory_mb': float(np.max(memory_samples)),
            'samples_collected': len(memory_samples),
            'duration_seconds': duration_seconds
        }
        
        self.logger.info(f"Memory usage stats: "
                        f"mean={stats['mean_memory_mb']:.1f}MB, "
                        f"peak={stats['peak_memory_mb']:.1f}MB, "
                        f"growth={stats['memory_growth_mb']:.1f}MB")
        
        return stats
    
    def run_comprehensive_profile(self) -> Dict[str, Any]:
        """Run comprehensive performance profiling.
        
        Returns:
            Complete profiling results
        """
        self.logger.info("Starting comprehensive performance profiling")
        
        # Clear any existing profiles
        self.profiles.clear()
        self.system_metrics.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Record initial system state
        initial_metrics = self.record_system_metrics()
        
        # Profile different components
        results = {
            'timestamp': datetime.now().isoformat(),
            'initial_system_metrics': asdict(initial_metrics),
            'inference_latency': self.profile_inference_latency(500),
            'feature_engineering': self.profile_feature_engineering(),
            'memory_usage': self.profile_memory_usage(30),
            'component_profiles': [asdict(p) for p in self.profiles],
            'system_metrics_history': [asdict(m) for m in self.system_metrics]
        }
        
        # Record final system state
        final_metrics = self.record_system_metrics()
        results['final_system_metrics'] = asdict(final_metrics)
        
        # Calculate overall statistics
        results['summary'] = self._calculate_summary_stats()
        
        self.logger.info("Comprehensive profiling completed")
        return results
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics from all profiles.
        
        Returns:
            Summary statistics
        """
        if not self.profiles:
            return {}
        
        total_duration = sum(p.duration_ms for p in self.profiles)
        total_memory = sum(p.memory_usage_mb for p in self.profiles if p.memory_usage_mb > 0)
        avg_cpu = np.mean([p.cpu_percent for p in self.profiles if p.cpu_percent > 0])
        
        # Find bottlenecks (slowest operations)
        slowest_operations = sorted(self.profiles, key=lambda p: p.duration_ms, reverse=True)[:5]
        
        # Find memory-intensive operations
        memory_intensive = sorted(
            [p for p in self.profiles if p.memory_usage_mb > 0], 
            key=lambda p: p.memory_usage_mb, reverse=True
        )[:5]
        
        return {
            'total_duration_ms': total_duration,
            'total_memory_usage_mb': total_memory,
            'average_cpu_percent': float(avg_cpu) if not np.isnan(avg_cpu) else 0.0,
            'num_operations_profiled': len(self.profiles),
            'slowest_operations': [
                {
                    'component': p.component,
                    'operation': p.operation,
                    'duration_ms': p.duration_ms
                } for p in slowest_operations
            ],
            'memory_intensive_operations': [
                {
                    'component': p.component,
                    'operation': p.operation,
                    'memory_usage_mb': p.memory_usage_mb
                } for p in memory_intensive
            ]
        }
    
    def generate_optimization_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on profiling results.
        
        Args:
            results: Profiling results
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Check inference latency
        inference_stats = results.get('inference_latency', {})
        if inference_stats.get('target_compliance_rate', 1.0) < 0.95:
            recommendations.append(
                f"Inference latency compliance is {inference_stats.get('target_compliance_rate', 0):.1%}. "
                "Consider optimizing XGBoost parameters or reducing feature count."
            )
        
        if inference_stats.get('p95_latency_ms', 0) > 15:
            recommendations.append(
                f"P95 inference latency is {inference_stats.get('p95_latency_ms', 0):.2f}ms. "
                "Consider model quantization or feature caching."
            )
        
        # Check memory usage
        memory_stats = results.get('memory_usage', {})
        if memory_stats.get('peak_memory_mb', 0) > 1000:  # 1GB threshold
            recommendations.append(
                f"Peak memory usage is {memory_stats.get('peak_memory_mb', 0):.1f}MB. "
                "Consider implementing data streaming or batch size reduction."
            )
        
        if memory_stats.get('memory_growth_mb', 0) > 100:
            recommendations.append(
                f"Memory growth during profiling was {memory_stats.get('memory_growth_mb', 0):.1f}MB. "
                "Check for memory leaks or implement garbage collection."
            )
        
        # Check feature engineering performance
        feature_stats = results.get('feature_engineering', {})
        
        tech_per_record = feature_stats.get('technical_indicators', {}).get('per_record_ms', 0)
        if tech_per_record > 1.0:
            recommendations.append(
                f"Technical indicator calculation takes {tech_per_record:.2f}ms per record. "
                "Consider vectorization or caching of intermediate results."
            )
        
        sentiment_per_record = feature_stats.get('sentiment_analysis', {}).get('per_record_ms', 0)
        if sentiment_per_record > 10.0:  # 0.01s requirement
            recommendations.append(
                f"Sentiment analysis takes {sentiment_per_record:.2f}ms per record. "
                "Consider batch processing or model optimization."
            )
        
        # Check system resource usage
        final_metrics = results.get('final_system_metrics', {})
        if final_metrics.get('cpu_percent', 0) > 80:
            recommendations.append(
                f"CPU usage is {final_metrics.get('cpu_percent', 0):.1f}%. "
                "Consider parallel processing or algorithm optimization."
            )
        
        if final_metrics.get('memory_percent', 0) > 85:
            recommendations.append(
                f"System memory usage is {final_metrics.get('memory_percent', 0):.1f}%. "
                "Consider reducing memory footprint or adding swap space."
            )
        
        # Check for bottlenecks
        summary = results.get('summary', {})
        slowest_ops = summary.get('slowest_operations', [])
        if slowest_ops:
            slowest = slowest_ops[0]
            if slowest['duration_ms'] > 1000:
                recommendations.append(
                    f"Slowest operation is {slowest['component']}.{slowest['operation']} "
                    f"at {slowest['duration_ms']:.2f}ms. Focus optimization efforts here."
                )
        
        if not recommendations:
            recommendations.append("Performance looks good! No major optimization recommendations.")
        
        return recommendations
    
    def save_report(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save profiling report to file.
        
        Args:
            results: Profiling results
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to saved report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        report_path = self.output_dir / filename
        
        # Add optimization recommendations
        results['optimization_recommendations'] = self.generate_optimization_recommendations(results)
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved to {report_path}")
        return str(report_path)


def main():
    """Main function to run performance profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile NIFTY 50 ML Pipeline performance")
    parser.add_argument("--output-dir", default="performance_reports",
                       help="Output directory for reports")
    parser.add_argument("--inference-samples", type=int, default=1000,
                       help="Number of inference samples to test")
    parser.add_argument("--memory-duration", type=int, default=60,
                       help="Memory profiling duration in seconds")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create profiler
    profiler = PerformanceProfiler(args.output_dir)
    
    try:
        # Run comprehensive profiling
        results = profiler.run_comprehensive_profile()
        
        # Save report
        report_path = profiler.save_report(results)
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE PROFILING SUMMARY")
        print("="*60)
        
        summary = results.get('summary', {})
        print(f"Total operations profiled: {summary.get('num_operations_profiled', 0)}")
        print(f"Total duration: {summary.get('total_duration_ms', 0):.2f}ms")
        print(f"Total memory usage: {summary.get('total_memory_usage_mb', 0):.2f}MB")
        print(f"Average CPU usage: {summary.get('average_cpu_percent', 0):.1f}%")
        
        # Print inference stats
        inference_stats = results.get('inference_latency', {})
        print(f"\nInference Performance:")
        print(f"  Mean latency: {inference_stats.get('mean_latency_ms', 0):.2f}ms")
        print(f"  P95 latency: {inference_stats.get('p95_latency_ms', 0):.2f}ms")
        print(f"  Target compliance: {inference_stats.get('target_compliance_rate', 0):.1%}")
        
        # Print recommendations
        recommendations = results.get('optimization_recommendations', [])
        print(f"\nOptimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print(f"\nDetailed report saved to: {report_path}")
        
    except Exception as e:
        print(f"Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())