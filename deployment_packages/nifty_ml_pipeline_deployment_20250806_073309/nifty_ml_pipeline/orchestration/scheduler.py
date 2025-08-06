# nifty_ml_pipeline/orchestration/scheduler.py
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from threading import Thread, Event
import pytz

from .controller import PipelineController


logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Task scheduler for daily pipeline execution at 5:30 PM IST.
    
    Manages timing, execution coordination, and provides hooks for
    monitoring and error handling during scheduled runs.
    """
    
    def __init__(self, config: Dict[str, Any], pipeline_controller: Optional[PipelineController] = None):
        """Initialize task scheduler with configuration.
        
        Args:
            config: Configuration dictionary containing scheduling settings
            pipeline_controller: Optional pre-initialized pipeline controller
        """
        self.config = config
        self.pipeline_controller = pipeline_controller
        self.timezone = pytz.timezone(config['scheduling']['timezone'])
        self.execution_time = config['scheduling']['execution_time']  # "17:30"
        self.is_running = False
        self.stop_event = Event()
        self.scheduler_thread: Optional[Thread] = None
        
        # Parse execution time
        self.execution_hour, self.execution_minute = map(int, self.execution_time.split(':'))
        
        logger.info(f"Task scheduler initialized for daily execution at {self.execution_time} {self.timezone}")
    
    def start_scheduler(self) -> None:
        """Start the scheduler in a background thread."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        self.scheduler_thread = Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Task scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler gracefully."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("Stopping task scheduler...")
        self.stop_event.set()
        self.is_running = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
        
        logger.info("Task scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop that runs in background thread."""
        logger.info("Scheduler loop started")
        
        while not self.stop_event.is_set():
            try:
                # Calculate next execution time
                next_execution = self._get_next_execution_time()
                current_time = datetime.now(self.timezone)
                
                # Calculate wait time
                wait_seconds = (next_execution - current_time).total_seconds()
                
                if wait_seconds > 0:
                    logger.info(f"Next execution scheduled for {next_execution.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                    logger.info(f"Waiting {wait_seconds:.0f} seconds until next execution")
                    
                    # Wait with periodic checks for stop signal
                    if self._wait_with_interrupt(wait_seconds):
                        break  # Stop signal received
                
                # Execute pipeline if not stopped
                if not self.stop_event.is_set():
                    self._execute_scheduled_pipeline()
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                # Wait 60 seconds before retrying to avoid tight error loops
                if self._wait_with_interrupt(60):
                    break
        
        logger.info("Scheduler loop ended")
    
    def _get_next_execution_time(self) -> datetime:
        """Calculate the next execution time based on current time and schedule.
        
        Returns:
            Next execution datetime in the configured timezone
        """
        now = datetime.now(self.timezone)
        
        # Create today's execution time
        today_execution = now.replace(
            hour=self.execution_hour,
            minute=self.execution_minute,
            second=0,
            microsecond=0
        )
        
        # If today's execution time has passed, schedule for tomorrow
        if now >= today_execution:
            next_execution = today_execution + timedelta(days=1)
        else:
            next_execution = today_execution
        
        return next_execution
    
    def _wait_with_interrupt(self, wait_seconds: float) -> bool:
        """Wait for specified seconds with periodic interrupt checks.
        
        Args:
            wait_seconds: Number of seconds to wait
            
        Returns:
            True if interrupted by stop signal, False if wait completed
        """
        # Check every 10 seconds or at the end of wait period, whichever is shorter
        check_interval = min(10.0, wait_seconds)
        elapsed = 0.0
        
        while elapsed < wait_seconds and not self.stop_event.is_set():
            remaining = wait_seconds - elapsed
            sleep_time = min(check_interval, remaining)
            
            if self.stop_event.wait(timeout=sleep_time):
                return True  # Stop signal received
            
            elapsed += sleep_time
        
        return self.stop_event.is_set()
    
    def _execute_scheduled_pipeline(self) -> None:
        """Execute the pipeline during scheduled time."""
        execution_start = datetime.now(self.timezone)
        logger.info(f"Starting scheduled pipeline execution at {execution_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        try:
            # Initialize pipeline controller if not provided
            if self.pipeline_controller is None:
                self.pipeline_controller = PipelineController(self.config)
            
            # Execute pipeline
            result = self.pipeline_controller.execute_pipeline()
            
            # Log execution results
            execution_end = datetime.now(self.timezone)
            duration_minutes = (execution_end - execution_start).total_seconds() / 60
            
            if result.was_successful():
                logger.info(f"Scheduled pipeline execution completed successfully in {duration_minutes:.2f} minutes")
                logger.info(f"Execution ID: {result.execution_id}")
                logger.info(f"Generated {len(result.predictions)} predictions")
            else:
                logger.error(f"Scheduled pipeline execution failed: {result.error_message}")
                logger.error(f"Execution ID: {result.execution_id}")
            
            # Log stage performance
            for stage_result in result.stage_results:
                stage_name = stage_result.get('stage', 'unknown')
                stage_status = stage_result.get('status', 'unknown')
                stage_duration = stage_result.get('duration_ms', 0)
                
                logger.info(f"Stage {stage_name}: {stage_status} ({stage_duration:.2f}ms)")
            
        except Exception as e:
            logger.error(f"Scheduled pipeline execution failed with exception: {e}")
            logger.exception("Full exception details:")
    
    def execute_now(self, symbol: str = "NIFTY 50") -> Dict[str, Any]:
        """Execute pipeline immediately (for testing/manual execution).
        
        Args:
            symbol: Symbol to process
            
        Returns:
            Dictionary containing execution results
        """
        logger.info(f"Manual pipeline execution requested for {symbol}")
        
        try:
            # Initialize pipeline controller if not provided
            if self.pipeline_controller is None:
                self.pipeline_controller = PipelineController(self.config)
            
            # Execute pipeline
            result = self.pipeline_controller.execute_pipeline(symbol)
            
            logger.info(f"Manual execution completed with status: {result.status}")
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Manual pipeline execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now(self.timezone).isoformat()
            }
    
    def get_next_execution_info(self) -> Dict[str, Any]:
        """Get information about the next scheduled execution.
        
        Returns:
            Dictionary containing next execution details
        """
        if not self.is_running:
            return {
                "scheduler_running": False,
                "message": "Scheduler is not running"
            }
        
        next_execution = self._get_next_execution_time()
        current_time = datetime.now(self.timezone)
        time_until_execution = next_execution - current_time
        
        return {
            "scheduler_running": True,
            "next_execution_time": next_execution.isoformat(),
            "current_time": current_time.isoformat(),
            "time_until_execution_seconds": time_until_execution.total_seconds(),
            "time_until_execution_formatted": str(time_until_execution).split('.')[0],  # Remove microseconds
            "timezone": str(self.timezone),
            "execution_time": self.execution_time
        }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status and configuration.
        
        Returns:
            Dictionary containing scheduler status
        """
        return {
            "is_running": self.is_running,
            "execution_time": self.execution_time,
            "timezone": str(self.timezone),
            "thread_alive": self.scheduler_thread.is_alive() if self.scheduler_thread else False,
            "stop_event_set": self.stop_event.is_set(),
            "next_execution": self.get_next_execution_info()
        }