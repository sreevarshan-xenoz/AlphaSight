"""
Main entry point for the NIFTY 50 ML Pipeline.
"""
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_config, LOG_LEVEL, LOG_FORMAT


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log")
        ]
    )


def main():
    """Main entry point for the pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting NIFTY 50 ML Pipeline")
    
    try:
        config = get_config()
        logger.info(f"Configuration loaded successfully")
        logger.info(f"Pipeline configured for {config['scheduling']['execution_time']} execution")
        
        # Pipeline execution will be implemented in future tasks
        logger.info("Pipeline setup complete - ready for implementation")
        
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()