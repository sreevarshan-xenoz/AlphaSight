#!/usr/bin/env python3
"""
Configuration testing script for NIFTY 50 ML Pipeline.

This script tests configuration loading, validation, and environment switching.
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.manager import config_manager, get_environment_config
from config.validator import ConfigValidator
from config.settings import get_config, reload_config


def test_basic_config_loading():
    """Test basic configuration loading."""
    print("Testing basic configuration loading...")
    
    try:
        config = get_config()
        print(f"✓ Configuration loaded successfully")
        print(f"  Environment: {config.get('environment', 'unknown')}")
        print(f"  Version: {config.get('version', 'unknown')}")
        print(f"  Debug: {config.get('debug', False)}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_environment_config_loading():
    """Test environment-specific configuration loading."""
    print("\nTesting environment-specific configuration loading...")
    
    try:
        config = get_environment_config()
        print(f"✓ Environment configuration loaded successfully")
        print(f"  Environment: {config.get('environment', 'unknown')}")
        
        # Test environment info
        env_info = config_manager.get_environment_info()
        print(f"  Available environments: {env_info['available_environments']}")
        return True
    except Exception as e:
        print(f"✗ Environment configuration loading failed: {e}")
        return False


def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    try:
        config = get_config()
        validator = ConfigValidator()
        
        is_valid = validator.validate_config(config)
        summary = validator.get_validation_summary()
        
        if is_valid:
            print(f"✓ Configuration validation passed")
        else:
            print(f"✗ Configuration validation failed")
        
        print(f"  Errors: {summary['error_count']}")
        print(f"  Warnings: {summary['warning_count']}")
        
        # Print errors and warnings
        for error in summary['errors']:
            print(f"    ERROR: {error}")
        
        for warning in summary['warnings']:
            print(f"    WARNING: {warning}")
        
        return is_valid
    except Exception as e:
        print(f"✗ Configuration validation failed with exception: {e}")
        return False


def test_environment_switching():
    """Test environment switching functionality."""
    print("\nTesting environment switching...")
    
    original_env = os.getenv('ENVIRONMENT', 'development')
    
    try:
        # Get available environments
        env_info = config_manager.get_environment_info()
        available_envs = env_info['available_environments']
        
        if not available_envs:
            print("  No environment configurations found")
            return True
        
        # Test switching to each available environment
        for env_name in available_envs:
            if env_name != original_env:
                try:
                    print(f"  Switching to {env_name}...")
                    config = config_manager.switch_environment(env_name)
                    print(f"    ✓ Successfully switched to {env_name}")
                    print(f"    Environment: {config.get('environment')}")
                    print(f"    Debug: {config.get('debug')}")
                except Exception as e:
                    print(f"    ✗ Failed to switch to {env_name}: {e}")
                    return False
        
        # Switch back to original environment
        if original_env in available_envs:
            config_manager.switch_environment(original_env)
            print(f"  ✓ Switched back to original environment: {original_env}")
        
        return True
    except Exception as e:
        print(f"✗ Environment switching test failed: {e}")
        return False


def test_config_backup():
    """Test configuration backup functionality."""
    print("\nTesting configuration backup...")
    
    try:
        backup_path = config_manager.backup_current_config()
        backup_file = Path(backup_path)
        
        if backup_file.exists():
            print(f"✓ Configuration backup created: {backup_path}")
            
            # Check backup file content
            with open(backup_file, 'r') as f:
                content = f.read()
                if content.strip():
                    print(f"  Backup file contains {len(content.splitlines())} lines")
                else:
                    print(f"  ⚠ Backup file is empty")
            
            # Clean up backup file
            backup_file.unlink()
            print(f"  Backup file cleaned up")
            
            return True
        else:
            print(f"✗ Backup file was not created")
            return False
    except Exception as e:
        print(f"✗ Configuration backup test failed: {e}")
        return False


def test_specific_config_values():
    """Test specific configuration values."""
    print("\nTesting specific configuration values...")
    
    try:
        config = get_config()
        
        # Test critical configuration values
        tests = [
            ("performance.MAX_INFERENCE_LATENCY_MS", lambda x: isinstance(x, (int, float)) and x > 0),
            ("model.xgboost.n_jobs", lambda x: isinstance(x, int) and x >= 1),
            ("model.xgboost.tree_method", lambda x: x in ['exact', 'approx', 'hist']),
            ("data.retention_days", lambda x: isinstance(x, int) and x > 0),
            ("logging.level", lambda x: x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
        ]
        
        all_passed = True
        for config_path, validator_func in tests:
            try:
                value = _get_nested_value(config, config_path)
                if validator_func(value):
                    print(f"  ✓ {config_path}: {value}")
                else:
                    print(f"  ✗ {config_path}: {value} (validation failed)")
                    all_passed = False
            except Exception as e:
                print(f"  ✗ {config_path}: Error accessing value - {e}")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"✗ Specific configuration values test failed: {e}")
        return False


def _get_nested_value(config, key_path):
    """Get nested configuration value using dot notation."""
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(f"Key '{key}' not found in path '{key_path}'")
    
    return value


def test_environment_variables():
    """Test environment variable loading."""
    print("\nTesting environment variable loading...")
    
    try:
        # Set a test environment variable
        test_key = "TEST_CONFIG_VALUE"
        test_value = "test_value_123"
        os.environ[test_key] = test_value
        
        # Reload configuration
        config = reload_config()
        
        # Check if environment variable is accessible
        retrieved_value = os.getenv(test_key)
        if retrieved_value == test_value:
            print(f"✓ Environment variable loading works")
            print(f"  Set: {test_key}={test_value}")
            print(f"  Retrieved: {retrieved_value}")
        else:
            print(f"✗ Environment variable loading failed")
            print(f"  Expected: {test_value}")
            print(f"  Got: {retrieved_value}")
            return False
        
        # Clean up
        del os.environ[test_key]
        
        return True
    except Exception as e:
        print(f"✗ Environment variable test failed: {e}")
        return False


def main():
    """Run all configuration tests."""
    print("NIFTY 50 ML Pipeline - Configuration Testing")
    print("=" * 50)
    
    tests = [
        test_basic_config_loading,
        test_environment_config_loading,
        test_config_validation,
        test_environment_switching,
        test_config_backup,
        test_specific_config_values,
        test_environment_variables,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Configuration Tests Summary: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All configuration tests passed!")
        sys.exit(0)
    else:
        print(f"✗ {total - passed} configuration tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()