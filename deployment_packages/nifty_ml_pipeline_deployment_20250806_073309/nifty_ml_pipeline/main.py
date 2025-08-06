"""
Main entry point for the NIFTY 50 ML Pipeline.
"""
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.manager import get_environment_config
from config.settings import validate_config_at_startup


def setup_logging(config):
    """Set up logging configuration."""
    log_config = config['logging']
    
    logging.basicConfig(
        level=getattr(logging, log_config['level'].upper()),
        format=log_config['format'],
        datefmt=log_config.get('date_format'),
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_config['file'])
        ]
    )


def main():
    """Main entry point for the pipeline."""
    try:
        # Load and validate configuration
        config = get_environment_config()
        validate_config_at_startup()
        
        # Setup logging with configuration
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting NIFTY 50 ML Pipeline")
        logger.info(f"Environment: {config['environment']}")
        logger.info(f"Version: {config['version']}")
        logger.info(f"Debug mode: {config['debug']}")
        
        logger.info(f"Configuration loaded successfully")
        logger.info(f"Pipeline configured for {config['scheduling']['execution_time']} execution")
        
        # Log performance targets
        perf_config = config['performance']
        logger.info(f"Performance targets - Latency: {perf_config['MAX_INFERENCE_LATENCY_MS']}ms, "
                   f"Accuracy: {perf_config['TARGET_ACCURACY']}")
        
        # Pipeline execution will be implemented in future tasks
        logger.info("Pipeline setup complete - ready for implementation")
        
    except Exception as e:
        # Use basic logging if configuration fails
        logging.basicConfig(level=logging.ERROR)
        logger = logging.getLogger(__name__)
        logger.error(f"Pipeline initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()