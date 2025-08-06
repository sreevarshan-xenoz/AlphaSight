# nifty_ml_pipeline/orchestration/performance_monitor.py
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque

from ..data.models import PipelineStage


logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics being tracked."""
    LATENCY = "latency"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: datetime
    metric_type: MetricType
    stage: Optional[PipelineStage]
    value: float
    unit: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    """Performance alert information."""
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    stage: Optional[PipelineStage]
    message: str
    current_value: float
    threshold_value: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceSummary:
    """Summary of performance metrics over a time period."""
    start_time: datetime
    end_time: datetime
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_latency_ms: float
    average_accuracy: Optional[float]
    error_rate: float
    alerts_generated: int
    stage_performance: Dict[str, Dict[str, float]]


class PerformanceMonitor:
    """
    System health tracking and performance monitoring.
    
    Implements structured logging for latency, accuracy, and system metrics
    with configurable alerting for performance degradation scenarios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance monitor with configuration.
        
        Args:
            config: Configuration dictionary containing monitoring settings
        """
        self.config = config
        self.metrics_history: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.alerts_history: deque = deque(maxlen=1000)    # Keep last 1k alerts
        
        # Performance thresholds from config
        self.thresholds = config.get('performance', {})
        self.min_accuracy_threshold = self.thresholds.get('MIN_ACCURACY', 0.75)
        self.max_latency_threshold = self.thresholds.get('MAX_INFERENCE_LATENCY_MS', 10.0)
        self.max_error_rate_threshold = self.thresholds.get('MAX_ERROR_RATE', 0.1)
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Monitoring state
        self.monitoring_start_time = datetime.now()
        self.execution_count = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        logger.info("Performance monitor initialized with thresholds:")
        logger.info(f"  Min accuracy: {self.min_accuracy_threshold}")
        logger.info(f"  Max latency: {self.max_latency_threshold}ms")
        logger.info(f"  Max error rate: {self.max_error_rate_threshold}")
    
    def record_metric(self, metric_type: MetricType, value: float, unit: str,
                     stage: Optional[PipelineStage] = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a performance metric.
        
        Args:
            metric_type: Type of metric being recorded
            value: Metric value
            unit: Unit of measurement
            stage: Optional pipeline stage
            metadata: Optional additional metadata
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            stage=stage,
            value=value,
            unit=unit,
            metadata=metadata
        )
        
        self.metrics_history.append(metric)
        
        # Log structured metric
        self._log_structured_metric(metric)
        
        # Check for threshold violations
        self._check_thresholds(metric)
    
    def record_pipeline_execution(self, execution_id: str, success: bool, 
                                duration_ms: float, stage_results: List[Dict[str, Any]],
                                predictions: List[Dict[str, Any]]) -> None:
        """Record complete pipeline execution metrics.
        
        Args:
            execution_id: Unique execution identifier
            success: Whether execution was successful
            duration_ms: Total execution duration in milliseconds
            stage_results: Results from each pipeline stage
            predictions: Generated predictions
        """
        self.execution_count += 1
        
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        # Record overall latency
        self.record_metric(
            MetricType.LATENCY,
            duration_ms,
            "ms",
            metadata={
                'execution_id': execution_id,
                'success': success,
                'stage_count': len(stage_results)
            }
        )
        
        # Record stage-specific metrics
        for stage_result in stage_results:
            stage_name = stage_result.get('stage')
            stage_duration = stage_result.get('duration_ms', 0)
            stage_status = stage_result.get('status')
            
            if stage_name:
                try:
                    stage_enum = PipelineStage(stage_name)
                    self.record_metric(
                        MetricType.LATENCY,
                        stage_duration,
                        "ms",
                        stage=stage_enum,
                        metadata={
                            'execution_id': execution_id,
                            'status': stage_status,
                            'data_count': stage_result.get('data_count', 0)
                        }
                    )
                except ValueError:
                    logger.warning(f"Unknown stage name: {stage_name}")
        
        # Record accuracy if predictions available
        if predictions and success:
            # For now, record prediction confidence as a proxy for accuracy
            # In production, this would be calculated against actual outcomes
            avg_confidence = sum(p.get('confidence', 0) for p in predictions) / len(predictions)
            self.record_metric(
                MetricType.ACCURACY,
                avg_confidence,
                "ratio",
                metadata={
                    'execution_id': execution_id,
                    'prediction_count': len(predictions)
                }
            )
        
        # Record error rate
        current_error_rate = self.failed_executions / self.execution_count
        self.record_metric(
            MetricType.ERROR_RATE,
            current_error_rate,
            "ratio",
            metadata={
                'execution_id': execution_id,
                'total_executions': self.execution_count,
                'failed_executions': self.failed_executions
            }
        )
    
    def record_system_metrics(self) -> None:
        """Record system-level metrics (CPU, memory usage)."""
        try:
            import psutil
            
            # Record CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric(
                MetricType.CPU_USAGE,
                cpu_percent,
                "percent",
                metadata={'cores': psutil.cpu_count()}
            )
            
            # Record memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.record_metric(
                MetricType.MEMORY_USAGE,
                memory_percent,
                "percent",
                metadata={
                    'total_gb': round(memory.total / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2)
                }
            )
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
    
    def _log_structured_metric(self, metric: PerformanceMetric) -> None:
        """Log metric in structured format.
        
        Args:
            metric: Performance metric to log
        """
        log_data = {
            'event': 'performance_metric',
            'timestamp': metric.timestamp.isoformat(),
            'metric_type': metric.metric_type.value,
            'stage': metric.stage.value if metric.stage else None,
            'value': metric.value,
            'unit': metric.unit,
            'metadata': metric.metadata
        }
        
        logger.info(json.dumps(log_data))
    
    def _check_thresholds(self, metric: PerformanceMetric) -> None:
        """Check metric against configured thresholds and generate alerts.
        
        Args:
            metric: Performance metric to check
        """
        alert = None
        
        if metric.metric_type == MetricType.ACCURACY:
            if metric.value < self.min_accuracy_threshold:
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.ERROR,
                    metric_type=metric.metric_type,
                    stage=metric.stage,
                    message=f"Accuracy {metric.value:.3f} below threshold {self.min_accuracy_threshold}",
                    current_value=metric.value,
                    threshold_value=self.min_accuracy_threshold,
                    metadata=metric.metadata
                )
        
        elif metric.metric_type == MetricType.LATENCY:
            if metric.stage == PipelineStage.MODEL_INFERENCE and metric.value > self.max_latency_threshold:
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    metric_type=metric.metric_type,
                    stage=metric.stage,
                    message=f"Inference latency {metric.value:.2f}ms exceeds threshold {self.max_latency_threshold}ms",
                    current_value=metric.value,
                    threshold_value=self.max_latency_threshold,
                    metadata=metric.metadata
                )
        
        elif metric.metric_type == MetricType.ERROR_RATE:
            if metric.value > self.max_error_rate_threshold:
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.CRITICAL,
                    metric_type=metric.metric_type,
                    stage=metric.stage,
                    message=f"Error rate {metric.value:.3f} exceeds threshold {self.max_error_rate_threshold}",
                    current_value=metric.value,
                    threshold_value=self.max_error_rate_threshold,
                    metadata=metric.metadata
                )
        
        elif metric.metric_type == MetricType.CPU_USAGE:
            if metric.value > 90.0:  # High CPU usage threshold
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    metric_type=metric.metric_type,
                    stage=metric.stage,
                    message=f"High CPU usage: {metric.value:.1f}%",
                    current_value=metric.value,
                    threshold_value=90.0,
                    metadata=metric.metadata
                )
        
        elif metric.metric_type == MetricType.MEMORY_USAGE:
            if metric.value > 85.0:  # High memory usage threshold
                alert = Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    metric_type=metric.metric_type,
                    stage=metric.stage,
                    message=f"High memory usage: {metric.value:.1f}%",
                    current_value=metric.value,
                    threshold_value=85.0,
                    metadata=metric.metadata
                )
        
        if alert:
            self._handle_alert(alert)
    
    def _handle_alert(self, alert: Alert) -> None:
        """Handle generated alert.
        
        Args:
            alert: Alert to handle
        """
        self.alerts_history.append(alert)
        
        # Log structured alert
        self._log_structured_alert(alert)
        
        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _log_structured_alert(self, alert: Alert) -> None:
        """Log alert in structured format.
        
        Args:
            alert: Alert to log
        """
        log_data = {
            'event': 'performance_alert',
            'timestamp': alert.timestamp.isoformat(),
            'level': alert.level.value,
            'metric_type': alert.metric_type.value,
            'stage': alert.stage.value if alert.stage else None,
            'message': alert.message,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'metadata': alert.metadata
        }
        
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.INFO)
        
        logger.log(log_level, json.dumps(log_data))
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback function to be called when alerts are generated.
        
        Args:
            callback: Function to call with Alert object
        """
        self.alert_callbacks.append(callback)
        logger.info(f"Added alert callback: {callback.__name__}")
    
    def remove_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Remove alert callback function.
        
        Args:
            callback: Function to remove
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            logger.info(f"Removed alert callback: {callback.__name__}")
    
    def get_performance_summary(self, hours: int = 24) -> PerformanceSummary:
        """Get performance summary for the specified time period.
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            PerformanceSummary object
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Filter metrics within time range
        recent_metrics = [
            m for m in self.metrics_history 
            if start_time <= m.timestamp <= end_time
        ]
        
        if not recent_metrics:
            return PerformanceSummary(
                start_time=start_time,
                end_time=end_time,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                average_latency_ms=0.0,
                average_accuracy=None,
                error_rate=0.0,
                alerts_generated=0,
                stage_performance={}
            )
        
        # Calculate summary statistics
        latency_metrics = [m for m in recent_metrics if m.metric_type == MetricType.LATENCY]
        accuracy_metrics = [m for m in recent_metrics if m.metric_type == MetricType.ACCURACY]
        
        avg_latency = sum(m.value for m in latency_metrics) / len(latency_metrics) if latency_metrics else 0.0
        avg_accuracy = sum(m.value for m in accuracy_metrics) / len(accuracy_metrics) if accuracy_metrics else None
        
        # Calculate stage performance
        stage_performance = {}
        for stage in PipelineStage:
            stage_metrics = [m for m in latency_metrics if m.stage == stage]
            if stage_metrics:
                stage_performance[stage.value] = {
                    'average_latency_ms': sum(m.value for m in stage_metrics) / len(stage_metrics),
                    'min_latency_ms': min(m.value for m in stage_metrics),
                    'max_latency_ms': max(m.value for m in stage_metrics),
                    'execution_count': len(stage_metrics)
                }
        
        # Count recent alerts
        recent_alerts = [
            a for a in self.alerts_history 
            if start_time <= a.timestamp <= end_time
        ]
        
        return PerformanceSummary(
            start_time=start_time,
            end_time=end_time,
            total_executions=self.execution_count,
            successful_executions=self.successful_executions,
            failed_executions=self.failed_executions,
            average_latency_ms=avg_latency,
            average_accuracy=avg_accuracy,
            error_rate=self.failed_executions / self.execution_count if self.execution_count > 0 else 0.0,
            alerts_generated=len(recent_alerts),
            stage_performance=stage_performance
        )
    
    def get_recent_alerts(self, hours: int = 24, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get recent alerts within specified time period.
        
        Args:
            hours: Number of hours to look back
            level: Optional alert level filter
            
        Returns:
            List of Alert objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        alerts = [
            alert for alert in self.alerts_history 
            if alert.timestamp >= cutoff_time
        ]
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_metrics_by_type(self, metric_type: MetricType, hours: int = 24) -> List[PerformanceMetric]:
        """Get metrics of specific type within time period.
        
        Args:
            metric_type: Type of metrics to retrieve
            hours: Number of hours to look back
            
        Returns:
            List of PerformanceMetric objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metric for metric in self.metrics_history 
            if metric.metric_type == metric_type and metric.timestamp >= cutoff_time
        ]
    
    def clear_history(self) -> None:
        """Clear metrics and alerts history."""
        self.metrics_history.clear()
        self.alerts_history.clear()
        self.execution_count = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.monitoring_start_time = datetime.now()
        
        logger.info("Performance monitoring history cleared")
    
    def export_metrics(self, filepath: str, hours: int = 24) -> None:
        """Export metrics to JSON file.
        
        Args:
            filepath: Path to export file
            hours: Number of hours of data to export
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            asdict(metric) for metric in self.metrics_history 
            if metric.timestamp >= cutoff_time
        ]
        
        recent_alerts = [
            asdict(alert) for alert in self.alerts_history 
            if alert.timestamp >= cutoff_time
        ]
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'time_range_hours': hours,
            'summary': asdict(self.get_performance_summary(hours)),
            'metrics': recent_metrics,
            'alerts': recent_alerts
        }
        
        # Convert datetime objects and enums to JSON serializable format
        def convert_for_json(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (MetricType, AlertLevel, PipelineStage)):
                return obj.value
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        export_data = convert_for_json(export_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")