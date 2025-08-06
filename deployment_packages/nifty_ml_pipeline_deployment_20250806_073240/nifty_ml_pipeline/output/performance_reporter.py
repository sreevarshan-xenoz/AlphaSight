# nifty_ml_pipeline/output/performance_reporter.py
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import asdict

from ..data.models import PredictionResult
from .prediction_storage import PredictionStorage


logger = logging.getLogger(__name__)


class PerformanceReporter:
    """Generates performance reports and metrics for prediction accuracy tracking.
    
    Provides comprehensive performance analysis including accuracy metrics,
    trend analysis, and model drift detection capabilities.
    """
    
    def __init__(self, 
                 storage: PredictionStorage,
                 accuracy_threshold: float = 0.75,
                 drift_threshold: float = 0.1):
        """Initialize performance reporter.
        
        Args:
            storage: PredictionStorage instance for data retrieval
            accuracy_threshold: Minimum accuracy threshold for alerts
            drift_threshold: Maximum drift threshold before retraining trigger
        """
        self.storage = storage
        self.accuracy_threshold = accuracy_threshold
        self.drift_threshold = drift_threshold
        
        logger.info(f"Initialized PerformanceReporter with accuracy_threshold={accuracy_threshold}")
    
    def generate_performance_summary(self, 
                                   start_date: datetime, 
                                   end_date: datetime,
                                   symbol: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance summary for a date range.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            symbol: Optional symbol filter
            
        Returns:
            Dict: Performance summary with accuracy, precision, recall metrics
        """
        try:
            # Retrieve predictions for the period
            predictions = self.storage.retrieve_predictions(start_date, end_date, symbol)
            
            if not predictions:
                logger.warning(f"No predictions found for period {start_date} to {end_date}")
                return self._empty_summary()
            
            # Calculate basic metrics
            total_predictions = len(predictions)
            actionable_predictions = len([p for p in predictions if p.is_actionable()])
            
            # Signal distribution
            signal_distribution = self._calculate_signal_distribution(predictions)
            
            # Confidence statistics
            confidence_stats = self._calculate_confidence_stats(predictions)
            
            # Time-based analysis
            temporal_analysis = self._analyze_temporal_patterns(predictions)
            
            # Model version analysis
            model_analysis = self._analyze_model_versions(predictions)
            
            summary = {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'symbol_filter': symbol
                },
                'overview': {
                    'total_predictions': total_predictions,
                    'actionable_predictions': actionable_predictions,
                    'actionable_rate': round(actionable_predictions / total_predictions, 3) if total_predictions > 0 else 0.0
                },
                'signal_distribution': signal_distribution,
                'confidence_statistics': confidence_stats,
                'temporal_analysis': temporal_analysis,
                'model_analysis': model_analysis,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Generated performance summary for {total_predictions} predictions")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {str(e)}")
            return {'error': str(e), 'generated_at': datetime.now().isoformat()}
    
    def calculate_accuracy_metrics(self, 
                                 predictions: List[PredictionResult],
                                 actual_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate accuracy metrics by comparing predictions with actual outcomes.
        
        Args:
            predictions: List of prediction results
            actual_prices: Dictionary mapping (symbol, timestamp) to actual prices
            
        Returns:
            Dict: Accuracy metrics including directional accuracy, MAE, RMSE
        """
        try:
            if not predictions or not actual_prices:
                return self._empty_accuracy_metrics()
            
            correct_directions = 0
            total_directional = 0
            absolute_errors = []
            squared_errors = []
            
            for pred in predictions:
                # Create key for actual price lookup
                price_key = f"{pred.symbol}_{pred.timestamp.strftime('%Y-%m-%d_%H-%M')}"
                
                if price_key not in actual_prices:
                    continue
                
                actual_price = actual_prices[price_key]
                predicted_price = pred.predicted_close
                
                # Calculate absolute and squared errors
                abs_error = abs(predicted_price - actual_price)
                sq_error = (predicted_price - actual_price) ** 2
                
                absolute_errors.append(abs_error)
                squared_errors.append(sq_error)
                
                # For directional accuracy, we need a reference price (previous close)
                # For now, we'll use the predicted vs actual comparison
                if pred.signal != "Hold":
                    total_directional += 1
                    
                    # Simplified directional accuracy
                    if pred.signal == "Buy" and actual_price >= predicted_price * 0.98:
                        correct_directions += 1
                    elif pred.signal == "Sell" and actual_price <= predicted_price * 1.02:
                        correct_directions += 1
            
            # Calculate metrics
            directional_accuracy = correct_directions / total_directional if total_directional > 0 else 0.0
            mae = np.mean(absolute_errors) if absolute_errors else 0.0
            rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else 0.0
            mape = np.mean([abs_err / actual for abs_err, actual in 
                           zip(absolute_errors, [actual_prices[k] for k in actual_prices.keys()][:len(absolute_errors)])]) if absolute_errors else 0.0
            
            metrics = {
                'directional_accuracy': round(directional_accuracy, 3),
                'mean_absolute_error': round(mae, 2),
                'root_mean_squared_error': round(rmse, 2),
                'mean_absolute_percentage_error': round(mape * 100, 2),  # Convert to percentage
                'total_comparisons': len(absolute_errors),
                'directional_comparisons': total_directional
            }
            
            logger.info(f"Calculated accuracy metrics for {len(absolute_errors)} predictions")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate accuracy metrics: {str(e)}")
            return {'error': str(e)}
    
    def analyze_historical_performance(self, 
                                     days: int = 30,
                                     symbol: Optional[str] = None) -> Dict[str, Any]:
        """Analyze historical performance trends over specified period.
        
        Args:
            days: Number of days to analyze
            symbol: Optional symbol filter
            
        Returns:
            Dict: Historical performance analysis with trends
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            predictions = self.storage.retrieve_predictions(start_date, end_date, symbol)
            
            if not predictions:
                logger.warning(f"No predictions found for historical analysis")
                return {'error': 'No predictions found', 'period_days': days}
            
            # Group predictions by day for trend analysis
            daily_stats = self._calculate_daily_statistics(predictions)
            
            # Calculate trends
            trends = self._calculate_performance_trends(daily_stats)
            
            # Identify performance patterns
            patterns = self._identify_performance_patterns(daily_stats)
            
            analysis = {
                'period': {
                    'days': days,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'symbol_filter': symbol
                },
                'daily_statistics': daily_stats,
                'trends': trends,
                'patterns': patterns,
                'summary': {
                    'total_days_analyzed': len(daily_stats),
                    'avg_daily_predictions': round(np.mean([day['prediction_count'] for day in daily_stats.values()]), 1),
                    'avg_daily_confidence': round(np.mean([day['avg_confidence'] for day in daily_stats.values()]), 3)
                },
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Analyzed {days} days of historical performance")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze historical performance: {str(e)}")
            return {'error': str(e), 'period_days': days}
    
    def detect_model_drift(self, 
                          baseline_days: int = 30,
                          comparison_days: int = 7,
                          symbol: Optional[str] = None) -> Dict[str, Any]:
        """Detect model drift by comparing recent performance with baseline.
        
        Args:
            baseline_days: Number of days for baseline period
            comparison_days: Number of recent days to compare
            symbol: Optional symbol filter
            
        Returns:
            Dict: Drift detection results with retraining recommendations
        """
        try:
            end_date = datetime.now()
            
            # Define periods
            comparison_start = end_date - timedelta(days=comparison_days)
            baseline_start = end_date - timedelta(days=baseline_days)
            baseline_end = comparison_start
            
            # Get predictions for both periods
            baseline_predictions = self.storage.retrieve_predictions(baseline_start, baseline_end, symbol)
            recent_predictions = self.storage.retrieve_predictions(comparison_start, end_date, symbol)
            
            if not baseline_predictions or not recent_predictions:
                logger.warning("Insufficient data for drift detection")
                return {
                    'error': 'Insufficient data for drift detection',
                    'baseline_predictions': len(baseline_predictions) if baseline_predictions else 0,
                    'recent_predictions': len(recent_predictions) if recent_predictions else 0
                }
            
            # Calculate metrics for both periods
            baseline_metrics = self._calculate_period_metrics(baseline_predictions)
            recent_metrics = self._calculate_period_metrics(recent_predictions)
            
            # Calculate drift indicators
            drift_indicators = self._calculate_drift_indicators(baseline_metrics, recent_metrics)
            
            # Determine if retraining is needed
            retraining_needed = self._assess_retraining_need(drift_indicators)
            
            drift_analysis = {
                'periods': {
                    'baseline': {
                        'start_date': baseline_start.isoformat(),
                        'end_date': baseline_end.isoformat(),
                        'prediction_count': len(baseline_predictions)
                    },
                    'recent': {
                        'start_date': comparison_start.isoformat(),
                        'end_date': end_date.isoformat(),
                        'prediction_count': len(recent_predictions)
                    }
                },
                'baseline_metrics': baseline_metrics,
                'recent_metrics': recent_metrics,
                'drift_indicators': drift_indicators,
                'retraining_assessment': retraining_needed,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Completed drift detection analysis")
            return drift_analysis
            
        except Exception as e:
            logger.error(f"Failed to detect model drift: {str(e)}")
            return {'error': str(e)}
    
    def generate_dashboard_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Generate data for performance dashboard visualization.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dict: Dashboard data with charts and metrics
        """
        try:
            # Get data for different time periods
            end_date = datetime.now()
            
            # Last 7 days for recent performance
            week_start = end_date - timedelta(days=7)
            week_data = self.generate_performance_summary(week_start, end_date, symbol)
            
            # Last 30 days for trend analysis
            month_data = self.analyze_historical_performance(30, symbol)
            
            # Drift detection
            drift_data = self.detect_model_drift(symbol=symbol)
            
            # Recent predictions for charts
            recent_predictions = self.storage.get_latest_predictions(limit=50)
            chart_data = self._prepare_chart_data(recent_predictions)
            
            dashboard = {
                'overview': {
                    'last_updated': end_date.isoformat(),
                    'symbol_filter': symbol,
                    'data_freshness': self._assess_data_freshness(recent_predictions)
                },
                'weekly_performance': week_data,
                'monthly_trends': month_data,
                'drift_detection': drift_data,
                'charts': chart_data,
                'alerts': self._generate_alerts(week_data, drift_data),
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info("Generated dashboard data")
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {str(e)}")
            return {'error': str(e)}
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty performance summary structure."""
        return {
            'overview': {
                'total_predictions': 0,
                'actionable_predictions': 0,
                'actionable_rate': 0.0
            },
            'signal_distribution': {'Buy': 0, 'Sell': 0, 'Hold': 0},
            'confidence_statistics': {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def _empty_accuracy_metrics(self) -> Dict[str, float]:
        """Return empty accuracy metrics structure."""
        return {
            'directional_accuracy': 0.0,
            'mean_absolute_error': 0.0,
            'root_mean_squared_error': 0.0,
            'mean_absolute_percentage_error': 0.0,
            'total_comparisons': 0,
            'directional_comparisons': 0
        }
    
    def _calculate_signal_distribution(self, predictions: List[PredictionResult]) -> Dict[str, int]:
        """Calculate distribution of trading signals."""
        distribution = {'Buy': 0, 'Sell': 0, 'Hold': 0}
        
        for pred in predictions:
            if pred.signal in distribution:
                distribution[pred.signal] += 1
        
        return distribution
    
    def _calculate_confidence_stats(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        """Calculate confidence statistics."""
        confidences = [pred.confidence for pred in predictions]
        
        if not confidences:
            return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': round(np.mean(confidences), 3),
            'median': round(np.median(confidences), 3),
            'std': round(np.std(confidences), 3),
            'min': round(np.min(confidences), 3),
            'max': round(np.max(confidences), 3)
        }
    
    def _analyze_temporal_patterns(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Analyze temporal patterns in predictions."""
        if not predictions:
            return {}
        
        # Group by hour of day
        hourly_counts = {}
        for pred in predictions:
            hour = pred.timestamp.hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        
        # Group by day of week
        daily_counts = {}
        for pred in predictions:
            day = pred.timestamp.strftime('%A')
            daily_counts[day] = daily_counts.get(day, 0) + 1
        
        return {
            'hourly_distribution': hourly_counts,
            'daily_distribution': daily_counts,
            'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None,
            'peak_day': max(daily_counts.items(), key=lambda x: x[1])[0] if daily_counts else None
        }
    
    def _analyze_model_versions(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Analyze model version usage."""
        version_counts = {}
        version_confidence = {}
        
        for pred in predictions:
            version = pred.model_version
            version_counts[version] = version_counts.get(version, 0) + 1
            
            if version not in version_confidence:
                version_confidence[version] = []
            version_confidence[version].append(pred.confidence)
        
        # Calculate average confidence per version
        version_avg_confidence = {}
        for version, confidences in version_confidence.items():
            version_avg_confidence[version] = round(np.mean(confidences), 3)
        
        return {
            'version_distribution': version_counts,
            'version_avg_confidence': version_avg_confidence,
            'most_used_version': max(version_counts.items(), key=lambda x: x[1])[0] if version_counts else None
        }
    
    def _calculate_daily_statistics(self, predictions: List[PredictionResult]) -> Dict[str, Dict[str, Any]]:
        """Calculate daily statistics for trend analysis."""
        daily_stats = {}
        
        # Group predictions by date
        for pred in predictions:
            date_key = pred.timestamp.strftime('%Y-%m-%d')
            
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    'prediction_count': 0,
                    'confidences': [],
                    'signals': {'Buy': 0, 'Sell': 0, 'Hold': 0}
                }
            
            daily_stats[date_key]['prediction_count'] += 1
            daily_stats[date_key]['confidences'].append(pred.confidence)
            daily_stats[date_key]['signals'][pred.signal] += 1
        
        # Calculate aggregated metrics for each day
        for date_key, stats in daily_stats.items():
            confidences = stats['confidences']
            stats['avg_confidence'] = round(np.mean(confidences), 3) if confidences else 0.0
            stats['actionable_count'] = sum(1 for c in confidences if c >= 0.7)
            stats['actionable_rate'] = round(stats['actionable_count'] / len(confidences), 3) if confidences else 0.0
            
            # Remove raw confidences to reduce data size
            del stats['confidences']
        
        return daily_stats
    
    def _calculate_performance_trends(self, daily_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance trends from daily statistics."""
        if len(daily_stats) < 2:
            return {'error': 'Insufficient data for trend calculation'}
        
        dates = sorted(daily_stats.keys())
        
        # Extract time series data
        prediction_counts = [daily_stats[date]['prediction_count'] for date in dates]
        avg_confidences = [daily_stats[date]['avg_confidence'] for date in dates]
        actionable_rates = [daily_stats[date]['actionable_rate'] for date in dates]
        
        # Calculate trends (simple linear trend)
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return round(slope, 4)
        
        return {
            'prediction_count_trend': calculate_trend(prediction_counts),
            'confidence_trend': calculate_trend(avg_confidences),
            'actionable_rate_trend': calculate_trend(actionable_rates),
            'trend_period_days': len(dates)
        }
    
    def _identify_performance_patterns(self, daily_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns in performance data."""
        if not daily_stats:
            return {}
        
        dates = sorted(daily_stats.keys())
        
        # Find best and worst performing days
        best_day = max(dates, key=lambda d: daily_stats[d]['avg_confidence'])
        worst_day = min(dates, key=lambda d: daily_stats[d]['avg_confidence'])
        
        # Calculate consistency (coefficient of variation)
        confidences = [daily_stats[date]['avg_confidence'] for date in dates]
        consistency = 1 - (np.std(confidences) / np.mean(confidences)) if np.mean(confidences) > 0 else 0
        
        return {
            'best_performance_day': {
                'date': best_day,
                'avg_confidence': daily_stats[best_day]['avg_confidence']
            },
            'worst_performance_day': {
                'date': worst_day,
                'avg_confidence': daily_stats[worst_day]['avg_confidence']
            },
            'consistency_score': round(consistency, 3),
            'performance_volatility': round(np.std(confidences), 3)
        }
    
    def _calculate_period_metrics(self, predictions: List[PredictionResult]) -> Dict[str, float]:
        """Calculate metrics for a specific period."""
        if not predictions:
            return {}
        
        confidences = [pred.confidence for pred in predictions]
        signals = [pred.signal for pred in predictions]
        
        return {
            'avg_confidence': round(np.mean(confidences), 3),
            'confidence_std': round(np.std(confidences), 3),
            'prediction_count': len(predictions),
            'actionable_rate': round(sum(1 for c in confidences if c >= 0.7) / len(confidences), 3),
            'buy_rate': round(signals.count('Buy') / len(signals), 3),
            'sell_rate': round(signals.count('Sell') / len(signals), 3),
            'hold_rate': round(signals.count('Hold') / len(signals), 3)
        }
    
    def _calculate_drift_indicators(self, baseline: Dict[str, float], recent: Dict[str, float]) -> Dict[str, Any]:
        """Calculate drift indicators between baseline and recent periods."""
        indicators = {}
        
        for metric in ['avg_confidence', 'actionable_rate', 'buy_rate', 'sell_rate']:
            if metric in baseline and metric in recent:
                baseline_val = baseline[metric]
                recent_val = recent[metric]
                
                # Calculate absolute and relative drift
                abs_drift = abs(recent_val - baseline_val)
                rel_drift = abs_drift / baseline_val if baseline_val != 0 else 0
                
                indicators[f'{metric}_drift'] = {
                    'absolute': round(abs_drift, 4),
                    'relative': round(rel_drift, 4),
                    'direction': 'increase' if recent_val > baseline_val else 'decrease'
                }
        
        return indicators
    
    def _assess_retraining_need(self, drift_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if model retraining is needed based on drift indicators."""
        high_drift_metrics = []
        
        for metric, drift_data in drift_indicators.items():
            if isinstance(drift_data, dict) and 'relative' in drift_data:
                if drift_data['relative'] > self.drift_threshold:
                    high_drift_metrics.append(metric)
        
        retraining_needed = len(high_drift_metrics) > 0
        
        return {
            'retraining_recommended': retraining_needed,
            'high_drift_metrics': high_drift_metrics,
            'drift_threshold': self.drift_threshold,
            'assessment_reason': f"Found {len(high_drift_metrics)} metrics exceeding drift threshold" if retraining_needed else "All metrics within acceptable drift range"
        }
    
    def _prepare_chart_data(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Prepare data for dashboard charts."""
        if not predictions:
            return {}
        
        # Time series data for confidence over time
        time_series = []
        for pred in predictions:
            time_series.append({
                'timestamp': pred.timestamp.isoformat(),
                'confidence': pred.confidence,
                'signal': pred.signal,
                'symbol': pred.symbol
            })
        
        # Signal distribution for pie chart
        signal_counts = {'Buy': 0, 'Sell': 0, 'Hold': 0}
        for pred in predictions:
            signal_counts[pred.signal] += 1
        
        return {
            'confidence_time_series': time_series,
            'signal_distribution': signal_counts,
            'data_points': len(predictions)
        }
    
    def _assess_data_freshness(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Assess freshness of prediction data."""
        if not predictions:
            return {'status': 'no_data', 'last_prediction': None}
        
        latest_prediction = max(predictions, key=lambda p: p.timestamp)
        time_since_last = datetime.now() - latest_prediction.timestamp
        
        # Determine freshness status
        if time_since_last.total_seconds() < 3600:  # Less than 1 hour
            status = 'fresh'
        elif time_since_last.total_seconds() < 86400:  # Less than 1 day
            status = 'recent'
        else:
            status = 'stale'
        
        return {
            'status': status,
            'last_prediction': latest_prediction.timestamp.isoformat(),
            'hours_since_last': round(time_since_last.total_seconds() / 3600, 1)
        }
    
    def _generate_alerts(self, weekly_data: Dict[str, Any], drift_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on performance data."""
        alerts = []
        
        # Check for low accuracy
        if 'overview' in weekly_data and weekly_data['overview']['actionable_rate'] < self.accuracy_threshold:
            alerts.append({
                'type': 'low_accuracy',
                'severity': 'warning',
                'message': f"Actionable prediction rate ({weekly_data['overview']['actionable_rate']:.1%}) below threshold ({self.accuracy_threshold:.1%})",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for drift
        if 'retraining_assessment' in drift_data and drift_data['retraining_assessment']['retraining_recommended']:
            alerts.append({
                'type': 'model_drift',
                'severity': 'critical',
                'message': f"Model drift detected. Retraining recommended due to: {', '.join(drift_data['retraining_assessment']['high_drift_metrics'])}",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts