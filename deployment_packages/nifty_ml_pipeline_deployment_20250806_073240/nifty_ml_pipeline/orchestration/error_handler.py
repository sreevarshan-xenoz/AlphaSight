# nifty_ml_pipeline/orchestration/error_handler.py
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from ..data.models import PipelineStage


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = "retry"
    SKIP_STAGE = "skip_stage"
    USE_FALLBACK = "use_fallback"
    ABORT_PIPELINE = "abort_pipeline"
    CONTINUE_WITH_WARNING = "continue_with_warning"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    stage: PipelineStage
    error: Exception
    timestamp: datetime
    execution_id: str
    attempt_number: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryResult:
    """Result of error recovery attempt."""
    action_taken: RecoveryAction
    success: bool
    message: str
    data: Optional[Any] = None
    should_continue: bool = True


class ErrorHandler:
    """
    Comprehensive error handler with graceful failure recovery.
    
    Provides stage-specific error handling for data collection, feature engineering,
    and inference stages with configurable recovery strategies and detailed logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize error handler with configuration.
        
        Args:
            config: Configuration dictionary containing error handling settings
        """
        self.config = config
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Error handling configuration
        self.max_retries = config.get('error_handling', {}).get('max_retries', 3)
        self.retry_delay_seconds = config.get('error_handling', {}).get('retry_delay_seconds', 5)
        self.enable_fallback_data = config.get('error_handling', {}).get('enable_fallback_data', True)
        
        logger.info("Error handler initialized with recovery strategies")
    
    def _initialize_recovery_strategies(self) -> Dict[PipelineStage, Dict[type, RecoveryAction]]:
        """Initialize recovery strategies for different error types and stages.
        
        Returns:
            Dictionary mapping stages and error types to recovery actions
        """
        return {
            PipelineStage.DATA_COLLECTION: {
                ConnectionError: RecoveryAction.RETRY,
                TimeoutError: RecoveryAction.RETRY,
                ValueError: RecoveryAction.USE_FALLBACK,
                KeyError: RecoveryAction.USE_FALLBACK,
                Exception: RecoveryAction.SKIP_STAGE  # Generic fallback
            },
            PipelineStage.FEATURE_ENGINEERING: {
                ValueError: RecoveryAction.USE_FALLBACK,
                KeyError: RecoveryAction.USE_FALLBACK,
                ZeroDivisionError: RecoveryAction.USE_FALLBACK,
                MemoryError: RecoveryAction.ABORT_PIPELINE,
                Exception: RecoveryAction.CONTINUE_WITH_WARNING
            },
            PipelineStage.MODEL_INFERENCE: {
                ValueError: RecoveryAction.USE_FALLBACK,
                MemoryError: RecoveryAction.ABORT_PIPELINE,
                RuntimeError: RecoveryAction.RETRY,
                Exception: RecoveryAction.CONTINUE_WITH_WARNING
            }
        }
    
    def handle_error(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle error with appropriate recovery strategy.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            RecoveryResult indicating the action taken and outcome
        """
        # Record error in history
        self.error_history.append(error_context)
        
        # Log error details
        self._log_error_details(error_context)
        
        # Determine recovery strategy
        recovery_action = self._determine_recovery_action(error_context)
        
        # Execute recovery action
        recovery_result = self._execute_recovery_action(recovery_action, error_context)
        
        # Log recovery result
        self._log_recovery_result(error_context, recovery_result)
        
        return recovery_result
    
    def _determine_recovery_action(self, error_context: ErrorContext) -> RecoveryAction:
        """Determine the appropriate recovery action for the error.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            RecoveryAction to take
        """
        stage = error_context.stage
        error_type = type(error_context.error)
        
        # Get stage-specific strategies
        stage_strategies = self.recovery_strategies.get(stage, {})
        
        # Look for specific error type strategy
        if error_type in stage_strategies:
            return stage_strategies[error_type]
        
        # Look for parent class strategies
        for strategy_error_type, action in stage_strategies.items():
            if issubclass(error_type, strategy_error_type):
                return action
        
        # Default to generic Exception strategy
        return stage_strategies.get(Exception, RecoveryAction.ABORT_PIPELINE)
    
    def _execute_recovery_action(self, action: RecoveryAction, error_context: ErrorContext) -> RecoveryResult:
        """Execute the specified recovery action.
        
        Args:
            action: Recovery action to execute
            error_context: Context information about the error
            
        Returns:
            RecoveryResult with outcome details
        """
        try:
            if action == RecoveryAction.RETRY:
                return self._handle_retry(error_context)
            
            elif action == RecoveryAction.SKIP_STAGE:
                return self._handle_skip_stage(error_context)
            
            elif action == RecoveryAction.USE_FALLBACK:
                return self._handle_use_fallback(error_context)
            
            elif action == RecoveryAction.ABORT_PIPELINE:
                return self._handle_abort_pipeline(error_context)
            
            elif action == RecoveryAction.CONTINUE_WITH_WARNING:
                return self._handle_continue_with_warning(error_context)
            
            else:
                return RecoveryResult(
                    action_taken=action,
                    success=False,
                    message=f"Unknown recovery action: {action}",
                    should_continue=False
                )
        
        except Exception as recovery_error:
            logger.error(f"Recovery action {action} failed: {recovery_error}")
            return RecoveryResult(
                action_taken=action,
                success=False,
                message=f"Recovery action failed: {str(recovery_error)}",
                should_continue=False
            )
    
    def _handle_retry(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle retry recovery action.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            RecoveryResult indicating retry decision
        """
        if error_context.attempt_number >= self.max_retries:
            return RecoveryResult(
                action_taken=RecoveryAction.RETRY,
                success=False,
                message=f"Maximum retries ({self.max_retries}) exceeded for {error_context.stage.value}",
                should_continue=False
            )
        
        return RecoveryResult(
            action_taken=RecoveryAction.RETRY,
            success=True,
            message=f"Retrying {error_context.stage.value} (attempt {error_context.attempt_number + 1}/{self.max_retries})",
            should_continue=True
        )
    
    def _handle_skip_stage(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle skip stage recovery action.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            RecoveryResult indicating stage skip
        """
        return RecoveryResult(
            action_taken=RecoveryAction.SKIP_STAGE,
            success=True,
            message=f"Skipping {error_context.stage.value} due to error: {str(error_context.error)}",
            should_continue=True
        )
    
    def _handle_use_fallback(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle use fallback recovery action.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            RecoveryResult with fallback data if available
        """
        if not self.enable_fallback_data:
            return RecoveryResult(
                action_taken=RecoveryAction.USE_FALLBACK,
                success=False,
                message="Fallback data is disabled in configuration",
                should_continue=False
            )
        
        # Generate fallback data based on stage
        fallback_data = self._generate_fallback_data(error_context.stage)
        
        if fallback_data is not None:
            return RecoveryResult(
                action_taken=RecoveryAction.USE_FALLBACK,
                success=True,
                message=f"Using fallback data for {error_context.stage.value}",
                data=fallback_data,
                should_continue=True
            )
        else:
            return RecoveryResult(
                action_taken=RecoveryAction.USE_FALLBACK,
                success=False,
                message=f"No fallback data available for {error_context.stage.value}",
                should_continue=False
            )
    
    def _handle_abort_pipeline(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle abort pipeline recovery action.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            RecoveryResult indicating pipeline abort
        """
        return RecoveryResult(
            action_taken=RecoveryAction.ABORT_PIPELINE,
            success=False,
            message=f"Aborting pipeline due to critical error in {error_context.stage.value}: {str(error_context.error)}",
            should_continue=False
        )
    
    def _handle_continue_with_warning(self, error_context: ErrorContext) -> RecoveryResult:
        """Handle continue with warning recovery action.
        
        Args:
            error_context: Context information about the error
            
        Returns:
            RecoveryResult indicating continuation with warning
        """
        return RecoveryResult(
            action_taken=RecoveryAction.CONTINUE_WITH_WARNING,
            success=True,
            message=f"Continuing pipeline with warning for {error_context.stage.value}: {str(error_context.error)}",
            should_continue=True
        )
    
    def _generate_fallback_data(self, stage: PipelineStage) -> Optional[Any]:
        """Generate fallback data for the specified stage.
        
        Args:
            stage: Pipeline stage requiring fallback data
            
        Returns:
            Fallback data if available, None otherwise
        """
        import pandas as pd
        from datetime import datetime, timedelta
        
        if stage == PipelineStage.DATA_COLLECTION:
            # Generate minimal fallback price data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            fallback_price_data = pd.DataFrame({
                'open': [100.0] * 30,
                'high': [105.0] * 30,
                'low': [95.0] * 30,
                'close': [102.0] * 30,
                'volume': [1000000] * 30
            }, index=dates)
            
            # Generate minimal fallback news data
            fallback_news_data = pd.DataFrame({
                'headline': ['Market update: Normal trading activity'] * 5,
                'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='D'),
                'source': ['fallback'] * 5,
                'url': [None] * 5
            })
            
            return (fallback_price_data, fallback_news_data)
        
        elif stage == PipelineStage.FEATURE_ENGINEERING:
            # Generate minimal feature data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            fallback_features = pd.DataFrame({
                'lag1_return': [0.0] * 30,
                'lag2_return': [0.0] * 30,
                'sma_5_ratio': [1.0] * 30,
                'rsi_14': [50.0] * 30,
                'macd_hist': [0.0] * 30,
                'daily_sentiment': [0.0] * 30
            }, index=dates)
            
            return fallback_features
        
        elif stage == PipelineStage.MODEL_INFERENCE:
            # Generate neutral prediction
            fallback_prediction = {
                'timestamp': datetime.now(),
                'symbol': 'NIFTY 50',
                'predicted_direction': 'Hold',
                'confidence': 0.5
            }
            
            return [fallback_prediction]
        
        return None
    
    def _log_error_details(self, error_context: ErrorContext) -> None:
        """Log detailed error information.
        
        Args:
            error_context: Context information about the error
        """
        logger.error(f"Error in {error_context.stage.value} (execution: {error_context.execution_id})")
        logger.error(f"Attempt: {error_context.attempt_number}")
        logger.error(f"Error type: {type(error_context.error).__name__}")
        logger.error(f"Error message: {str(error_context.error)}")
        
        if error_context.metadata:
            logger.error(f"Error metadata: {error_context.metadata}")
        
        # Log stack trace for debugging
        logger.debug(f"Stack trace: {traceback.format_exc()}")
    
    def _log_recovery_result(self, error_context: ErrorContext, recovery_result: RecoveryResult) -> None:
        """Log recovery action result.
        
        Args:
            error_context: Context information about the error
            recovery_result: Result of recovery action
        """
        log_level = logging.INFO if recovery_result.success else logging.ERROR
        
        logger.log(log_level, f"Recovery action for {error_context.stage.value}: {recovery_result.action_taken.value}")
        logger.log(log_level, f"Recovery result: {recovery_result.message}")
        logger.log(log_level, f"Should continue: {recovery_result.should_continue}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered during pipeline execution.
        
        Returns:
            Dictionary containing error summary statistics
        """
        if not self.error_history:
            return {
                'total_errors': 0,
                'errors_by_stage': {},
                'errors_by_type': {},
                'recovery_actions': {}
            }
        
        # Count errors by stage
        errors_by_stage = {}
        for error_context in self.error_history:
            stage = error_context.stage.value
            errors_by_stage[stage] = errors_by_stage.get(stage, 0) + 1
        
        # Count errors by type
        errors_by_type = {}
        for error_context in self.error_history:
            error_type = type(error_context.error).__name__
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'errors_by_stage': errors_by_stage,
            'errors_by_type': errors_by_type,
            'first_error_time': self.error_history[0].timestamp.isoformat(),
            'last_error_time': self.error_history[-1].timestamp.isoformat()
        }
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history.clear()
        logger.info("Error history cleared")
    
    def should_abort_pipeline(self) -> bool:
        """Determine if pipeline should be aborted based on error history.
        
        Returns:
            True if pipeline should be aborted, False otherwise
        """
        if not self.error_history:
            return False
        
        # Check for critical errors in recent history
        recent_errors = [e for e in self.error_history[-5:]]  # Last 5 errors
        critical_error_count = sum(1 for e in recent_errors 
                                 if self._get_error_severity(e.error) == ErrorSeverity.CRITICAL)
        
        return critical_error_count >= 2  # Abort if 2+ critical errors in recent history
    
    def _get_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity level.
        
        Args:
            error: Exception to evaluate
            
        Returns:
            ErrorSeverity level
        """
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, KeyError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW