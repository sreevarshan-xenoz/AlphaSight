"""
Configuration manager for environment-specific settings.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages environment-specific configurations and validation."""
    
    def __init__(self):
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.config_dir = Path(__file__).parent
        self.environments_dir = self.config_dir / 'environments'
        self._loaded_config: Optional[Dict[str, Any]] = None
    
    def load_environment_config(self) -> Dict[str, Any]:
        """
        Load configuration for the current environment.
        
        Returns:
            Dict containing environment-specific configuration
        """
        if self._loaded_config:
            return self._loaded_config
        
        # Load base configuration
        from .settings import get_config
        config = get_config()
        
        # Load environment-specific overrides
        env_file = self.environments_dir / f"{self.environment}.env"
        if env_file.exists():
            self._load_env_file(env_file)
            logger.info(f"Loaded environment configuration: {env_file}")
            
            # Reload configuration with environment overrides
            from .settings import reload_config
            config = reload_config()
        else:
            logger.warning(f"Environment configuration file not found: {env_file}")
        
        # Validate configuration
        self._validate_environment_config(config)
        
        self._loaded_config = config
        return config
    
    def _load_env_file(self, env_file: Path) -> None:
        """Load environment variables from file."""
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        except Exception as e:
            logger.error(f"Error loading environment file {env_file}: {e}")
    
    def _validate_environment_config(self, config: Dict[str, Any]) -> None:
        """Validate environment-specific configuration."""
        environment = config.get('environment', 'development')
        
        if environment == 'production':
            self._validate_production_config(config)
        elif environment == 'staging':
            self._validate_staging_config(config)
        elif environment == 'development':
            self._validate_development_config(config)
    
    def _validate_production_config(self, config: Dict[str, Any]) -> None:
        """Validate production-specific configuration."""
        # Ensure debug is disabled
        if config.get('debug', False):
            logger.warning("Debug mode is enabled in production environment")
        
        # Ensure strict performance requirements
        max_latency = config.get('performance', {}).get('MAX_INFERENCE_LATENCY_MS', 0)
        if max_latency > 10:
            logger.warning(f"Production inference latency target is high: {max_latency}ms")
        
        # Ensure monitoring is enabled
        if not config.get('monitoring', {}).get('enabled', False):
            logger.warning("Performance monitoring is disabled in production")
        
        # Check for required production settings
        alerts = config.get('monitoring', {}).get('alerts', {})
        if not alerts.get('email') and not alerts.get('slack_webhook'):
            logger.warning("No alert channels configured for production")
    
    def _validate_staging_config(self, config: Dict[str, Any]) -> None:
        """Validate staging-specific configuration."""
        # Ensure debug is disabled
        if config.get('debug', False):
            logger.warning("Debug mode is enabled in staging environment")
        
        # Check monitoring configuration
        if not config.get('monitoring', {}).get('enabled', False):
            logger.info("Performance monitoring is disabled in staging")
    
    def _validate_development_config(self, config: Dict[str, Any]) -> None:
        """Validate development-specific configuration."""
        # Development-specific validations
        if config.get('scheduling', {}).get('enabled', False):
            logger.info("Scheduler is enabled in development mode")
        
        # Check if mock data is available when enabled
        if config.get('development', {}).get('mock_data', False):
            mock_path = Path(config.get('development', {}).get('mock_data_path', 'tests/fixtures/'))
            if not mock_path.exists():
                logger.warning(f"Mock data path does not exist: {mock_path}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment."""
        return {
            'environment': self.environment,
            'config_dir': str(self.config_dir),
            'environments_dir': str(self.environments_dir),
            'available_environments': self._get_available_environments(),
            'current_config_file': str(self.environments_dir / f"{self.environment}.env")
        }
    
    def _get_available_environments(self) -> list:
        """Get list of available environment configurations."""
        if not self.environments_dir.exists():
            return []
        
        env_files = list(self.environments_dir.glob("*.env"))
        return [f.stem for f in env_files]
    
    def switch_environment(self, new_environment: str) -> Dict[str, Any]:
        """
        Switch to a different environment configuration.
        
        Args:
            new_environment: Name of the environment to switch to
            
        Returns:
            Dict containing the new configuration
        """
        if new_environment not in self._get_available_environments():
            raise ValueError(f"Environment '{new_environment}' not available. "
                           f"Available: {self._get_available_environments()}")
        
        # Update environment variable
        os.environ['ENVIRONMENT'] = new_environment
        self.environment = new_environment
        
        # Clear cached configuration
        self._loaded_config = None
        
        # Load new configuration
        config = self.load_environment_config()
        
        logger.info(f"Switched to environment: {new_environment}")
        return config
    
    def create_environment_config(self, environment_name: str, 
                                config_overrides: Dict[str, str]) -> None:
        """
        Create a new environment configuration file.
        
        Args:
            environment_name: Name of the new environment
            config_overrides: Dictionary of configuration overrides
        """
        env_file = self.environments_dir / f"{environment_name}.env"
        
        # Ensure environments directory exists
        self.environments_dir.mkdir(exist_ok=True)
        
        # Write configuration file
        with open(env_file, 'w') as f:
            f.write(f"# {environment_name.title()} Environment Configuration\n")
            f.write(f"# Auto-generated configuration file\n\n")
            f.write(f"ENVIRONMENT={environment_name}\n\n")
            
            for key, value in config_overrides.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Created environment configuration: {env_file}")
    
    def backup_current_config(self) -> str:
        """
        Create a backup of the current configuration.
        
        Returns:
            Path to the backup file
        """
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.config_dir / f"backup_config_{self.environment}_{timestamp}.env"
        
        # Get current environment variables
        env_vars = {}
        for key in os.environ:
            if any(key.startswith(prefix) for prefix in [
                'NIFTY_', 'API_', 'DATA_', 'MODEL_', 'LOG_', 'PERFORMANCE_',
                'XGBOOST_', 'RSI_', 'SMA_', 'MACD_', 'SENTIMENT_'
            ]):
                env_vars[key] = os.environ[key]
        
        # Write backup file
        with open(backup_file, 'w') as f:
            f.write(f"# Configuration backup for {self.environment}\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n\n")
            
            for key, value in sorted(env_vars.items()):
                f.write(f"{key}={value}\n")
        
        logger.info(f"Configuration backup created: {backup_file}")
        return str(backup_file)


# Global configuration manager instance
config_manager = ConfigManager()


def get_environment_config() -> Dict[str, Any]:
    """Get configuration for the current environment."""
    return config_manager.load_environment_config()


def switch_environment(environment: str) -> Dict[str, Any]:
    """Switch to a different environment."""
    return config_manager.switch_environment(environment)


def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment."""
    return config_manager.get_environment_info()