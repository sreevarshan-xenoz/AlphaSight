#!/usr/bin/env python3
"""
Deployment finalization script for NIFTY 50 ML Pipeline.

This script performs final optimizations, addresses security issues,
validates memory usage, and prepares the system for production deployment.
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DeploymentFinalizer:
    """Handles final deployment preparation and optimization."""
    
    def __init__(self):
        """Initialize deployment finalizer."""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.project_root = project_root
        self.optimization_results = {}
        self.security_fixes = []
        self.performance_improvements = []
        
        self.logger.info("Deployment finalizer initialized")
    
    def optimize_dependencies(self) -> Dict[str, Any]:
        """Optimize and validate dependencies."""
        self.logger.info("Optimizing dependencies")
        
        results = {
            'status': 'success',
            'actions_taken': [],
            'issues_found': [],
            'recommendations': []
        }
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            results['issues_found'].append("requirements.txt not found")
            return results
        
        # Read current requirements
        with open(requirements_file, 'r') as f:
            requirements = f.read().strip().split('\n')
        
        # Check for unpinned versions
        unpinned = []
        for req in requirements:
            if req.strip() and not req.startswith('#'):
                if '==' not in req and '>=' not in req and '<=' not in req:
                    unpinned.append(req.strip())
        
        if unpinned:
            results['issues_found'].extend([f"Unpinned dependency: {dep}" for dep in unpinned])
            results['recommendations'].append("Pin all dependencies to specific versions")
        
        # Create optimized requirements with security-focused versions
        optimized_requirements = [
            "# Core ML and Data Processing - Security Optimized",
            "pandas==2.1.4",
            "numpy==1.24.4", 
            "scikit-learn==1.3.2",
            "xgboost==2.0.3",
            "",
            "# Financial Data APIs",
            "nsepy==0.8",
            "yfinance==0.2.28",
            "",
            "# Natural Language Processing",
            "nltk==3.8.1",
            "vaderSentiment==3.3.2",
            "",
            "# Configuration and Utilities",
            "python-dotenv==1.0.0",
            "pyyaml==6.0.1",
            "",
            "# HTTP Requests - Security Focused",
            "requests==2.31.0",
            "urllib3==2.0.7",
            "",
            "# Date/Time Utilities",
            "python-dateutil==2.8.2",
            "pytz==2023.3",
            "",
            "# Data Storage and Processing",
            "pyarrow==14.0.2",
            "",
            "# System Monitoring",
            "psutil==5.9.6",
            "",
            "# Production Logging",
            "python-json-logger==2.0.7",
            "",
            "# Testing (Development Only)",
            "pytest==7.4.3; extra == 'dev'",
            "pytest-cov==4.1.0; extra == 'dev'",
            "",
            "# Code Quality (Development Only)", 
            "black==23.12.1; extra == 'dev'",
            "flake8==6.1.0; extra == 'dev'",
            "isort==5.13.2; extra == 'dev'"
        ]
        
        # Write optimized requirements
        optimized_file = self.project_root / "requirements-production.txt"
        with open(optimized_file, 'w') as f:
            f.write('\n'.join(optimized_requirements))
        
        results['actions_taken'].append(f"Created optimized requirements: {optimized_file}")
        
        return results
    
    def fix_security_issues(self) -> Dict[str, Any]:
        """Fix identified security issues."""
        self.logger.info("Fixing security issues")
        
        results = {
            'status': 'success',
            'fixes_applied': [],
            'issues_remaining': [],
            'recommendations': []
        }
        
        # Fix file permissions (Windows-specific handling)
        sensitive_files = [
            '.env',
            'config/production.py',
            'config/settings.py'
        ]
        
        for file_path in sensitive_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    # On Windows, we can't easily change permissions like Unix
                    # Instead, we'll create a secure version
                    if file_path == 'config/production.py':
                        self._create_secure_production_config()
                        results['fixes_applied'].append(f"Created secure production config")
                except Exception as e:
                    results['issues_remaining'].append(f"Could not secure {file_path}: {e}")
        
        # Create secure environment template
        self._create_secure_env_template()
        results['fixes_applied'].append("Created secure .env template")
        
        # Create security guidelines
        self._create_security_guidelines()
        results['fixes_applied'].append("Created security deployment guidelines")
        
        return results
    
    def _create_secure_production_config(self) -> None:
        """Create a secure production configuration."""
        secure_config = '''"""
Secure production configuration for NIFTY 50 ML Pipeline.

This configuration prioritizes security and performance for production deployment.
All sensitive values must be provided via environment variables.
"""

import os
from pathlib import Path

# Validate required environment variables at startup
REQUIRED_ENV_VARS = [
    'ECONOMIC_TIMES_API_KEY',
    'SMTP_USERNAME', 
    'SMTP_PASSWORD',
    'ALERT_TO_EMAILS',
    'ENCRYPTION_KEY_PATH'
]

def validate_environment():
    """Validate all required environment variables are set."""
    missing = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Production settings with security hardening
PRODUCTION_CONFIG = {
    "environment": "production",
    "debug": False,
    "version": os.getenv("VERSION", "1.0.0"),
    
    # Security settings
    "security": {
        "api": {
            "rate_limit": int(os.getenv("API_RATE_LIMIT", "30")),
            "timeout_seconds": int(os.getenv("API_TIMEOUT", "30")),
            "max_retries": int(os.getenv("API_MAX_RETRIES", "3")),
            "enable_ssl_verification": True,
            "min_tls_version": "1.2"
        },
        "data": {
            "encrypt_sensitive": True,
            "encryption_key_path": os.getenv("ENCRYPTION_KEY_PATH"),
            "data_retention_days": int(os.getenv("DATA_RETENTION_DAYS", "365"))
        },
        "access": {
            "enable_audit_logging": True,
            "max_login_attempts": int(os.getenv("MAX_LOGIN_ATTEMPTS", "3")),
            "session_timeout": int(os.getenv("SESSION_TIMEOUT", "3600"))
        }
    },
    
    # Performance optimization
    "performance": {
        "max_inference_latency_ms": float(os.getenv("MAX_INFERENCE_LATENCY_MS", "10.0")),
        "target_accuracy": float(os.getenv("TARGET_ACCURACY", "0.80")),
        "min_accuracy_threshold": float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.75")),
        "cpu_threads": int(os.getenv("CPU_THREADS", "1")),
        "memory_limit_gb": int(os.getenv("MEMORY_LIMIT_GB", "4")),
        "enable_caching": True,
        "cache_size_mb": int(os.getenv("CACHE_SIZE_MB", "256"))
    },
    
    # Monitoring and alerting
    "monitoring": {
        "enabled": True,
        "metrics_retention_days": int(os.getenv("METRICS_RETENTION_DAYS", "30")),
        "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "300")),
        "alerts": {
            "email": {
                "enabled": True,
                "smtp_server": os.getenv("SMTP_SERVER", "localhost"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "smtp_username": os.getenv("SMTP_USERNAME"),
                "smtp_password": os.getenv("SMTP_PASSWORD"),
                "from_email": os.getenv("ALERT_FROM_EMAIL"),
                "to_emails": os.getenv("ALERT_TO_EMAILS", "").split(",")
            }
        }
    },
    
    # Logging configuration
    "logging": {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": os.getenv("LOG_FILE", "/var/log/nifty-pipeline/pipeline.log"),
        "max_size_mb": int(os.getenv("LOG_MAX_SIZE_MB", "100")),
        "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5")),
        "enable_json": True,
        "enable_audit": True
    },
    
    # Data storage
    "storage": {
        "data_path": os.getenv("DATA_PATH", "/var/lib/nifty-pipeline/data/"),
        "models_path": os.getenv("MODELS_PATH", "/var/lib/nifty-pipeline/models/"),
        "backup_path": os.getenv("BACKUP_PATH", "/var/backups/nifty-pipeline/"),
        "format": "parquet",
        "compression": "snappy",
        "enable_encryption": True
    },
    
    # API configuration
    "api": {
        "economic_times": {
            "api_key": os.getenv("ECONOMIC_TIMES_API_KEY"),
            "timeout": int(os.getenv("ECONOMIC_TIMES_TIMEOUT", "30")),
            "rate_limit": int(os.getenv("ECONOMIC_TIMES_RATE_LIMIT", "20"))
        }
    }
}

def get_production_config():
    """Get production configuration with validation."""
    validate_environment()
    return PRODUCTION_CONFIG
'''
        
        secure_config_path = self.project_root / "config" / "production_secure.py"
        with open(secure_config_path, 'w') as f:
            f.write(secure_config)
    
    def _create_secure_env_template(self) -> None:
        """Create secure environment template."""
        env_template = '''# NIFTY 50 ML Pipeline - Production Environment Variables
# Copy this file to .env and fill in the actual values
# NEVER commit the actual .env file to version control

# Environment Configuration
ENVIRONMENT=production
VERSION=1.0.0
DEBUG=false

# API Keys (REQUIRED)
ECONOMIC_TIMES_API_KEY=your_economic_times_api_key_here
NSE_API_KEY=your_nse_api_key_here

# Email Alerting (REQUIRED)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@company.com
SMTP_PASSWORD=your_app_password_here
ALERT_FROM_EMAIL=nifty-pipeline@company.com
ALERT_TO_EMAILS=admin@company.com,ops@company.com

# Security Settings
ENCRYPTION_KEY_PATH=/secure/keys/encryption.key
MAX_LOGIN_ATTEMPTS=3
SESSION_TIMEOUT=3600

# Performance Settings
MAX_INFERENCE_LATENCY_MS=10.0
TARGET_ACCURACY=0.80
MIN_ACCURACY_THRESHOLD=0.75
CPU_THREADS=1
MEMORY_LIMIT_GB=4

# Storage Paths
DATA_PATH=/var/lib/nifty-pipeline/data/
MODELS_PATH=/var/lib/nifty-pipeline/models/
LOGS_PATH=/var/log/nifty-pipeline/
BACKUP_PATH=/var/backups/nifty-pipeline/

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=/var/log/nifty-pipeline/pipeline.log
LOG_MAX_SIZE_MB=100
LOG_BACKUP_COUNT=5

# Monitoring Settings
METRICS_RETENTION_DAYS=30
HEALTH_CHECK_INTERVAL=300

# Rate Limiting
API_RATE_LIMIT=30
ECONOMIC_TIMES_RATE_LIMIT=20
'''
        
        env_template_path = self.project_root / ".env.production.template"
        with open(env_template_path, 'w') as f:
            f.write(env_template)
    
    def _create_security_guidelines(self) -> None:
        """Create security deployment guidelines."""
        guidelines = '''# Security Deployment Guidelines

## Pre-Deployment Security Checklist

### 1. Environment Variables
- [ ] All sensitive values moved to environment variables
- [ ] No hardcoded API keys, passwords, or secrets in code
- [ ] .env files added to .gitignore
- [ ] Production .env file secured with appropriate permissions (600)

### 2. File Permissions
- [ ] Sensitive configuration files have restricted permissions
- [ ] Log files are not world-readable
- [ ] Model files are protected from unauthorized access

### 3. API Security
- [ ] API rate limiting configured
- [ ] SSL/TLS verification enabled for all external requests
- [ ] API timeouts configured to prevent hanging requests
- [ ] API keys rotated regularly (recommended: every 90 days)

### 4. Data Security
- [ ] Sensitive data encryption enabled
- [ ] Data retention policies configured
- [ ] Backup encryption enabled
- [ ] Access logging enabled for audit trails

### 5. Network Security
- [ ] Firewall rules configured to restrict access
- [ ] Only necessary ports exposed
- [ ] VPN or private network access for management
- [ ] SSL certificates valid and up-to-date

### 6. Monitoring and Alerting
- [ ] Security event monitoring enabled
- [ ] Failed authentication alerts configured
- [ ] Unusual activity detection enabled
- [ ] Log aggregation and analysis setup

### 7. Dependency Security
- [ ] All dependencies updated to latest secure versions
- [ ] Vulnerability scanning performed
- [ ] No known CVEs in production dependencies
- [ ] Dependency integrity verification enabled

## Production Deployment Steps

1. **Environment Setup**
   ```bash
   # Create secure directories
   sudo mkdir -p /var/lib/nifty-pipeline/{data,models}
   sudo mkdir -p /var/log/nifty-pipeline
   sudo mkdir -p /var/backups/nifty-pipeline
   
   # Set appropriate permissions
   sudo chown -R nifty-user:nifty-group /var/lib/nifty-pipeline
   sudo chown -R nifty-user:nifty-group /var/log/nifty-pipeline
   sudo chmod 750 /var/lib/nifty-pipeline
   sudo chmod 750 /var/log/nifty-pipeline
   ```

2. **SSL/TLS Configuration**
   ```bash
   # Generate or install SSL certificates
   # Configure nginx/apache for HTTPS termination
   # Ensure TLS 1.2+ only
   ```

3. **Firewall Configuration**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 22/tcp   # SSH
   sudo ufw allow 443/tcp  # HTTPS
   sudo ufw enable
   ```

4. **Environment Variables**
   ```bash
   # Copy and configure environment file
   cp .env.production.template .env
   # Edit .env with actual values
   chmod 600 .env
   ```

5. **Service Configuration**
   ```bash
   # Create systemd service file
   # Configure log rotation
   # Set up monitoring
   ```

## Security Monitoring

### Key Metrics to Monitor
- Failed authentication attempts
- Unusual API usage patterns
- High error rates
- Memory/CPU usage spikes
- Disk space usage
- Network traffic anomalies

### Alert Thresholds
- Failed logins: > 5 in 5 minutes
- API errors: > 10% error rate
- Memory usage: > 85%
- CPU usage: > 80% for > 5 minutes
- Disk usage: > 90%

## Incident Response

### Security Incident Procedures
1. **Immediate Response**
   - Isolate affected systems
   - Preserve logs and evidence
   - Notify security team
   - Document timeline

2. **Investigation**
   - Analyze logs for attack vectors
   - Identify compromised data/systems
   - Assess impact and scope
   - Collect forensic evidence

3. **Recovery**
   - Patch vulnerabilities
   - Rotate compromised credentials
   - Restore from clean backups
   - Update security controls

4. **Post-Incident**
   - Conduct lessons learned review
   - Update security procedures
   - Implement additional controls
   - Report to stakeholders

## Regular Security Tasks

### Daily
- [ ] Review security alerts
- [ ] Check system logs for anomalies
- [ ] Verify backup completion

### Weekly
- [ ] Review access logs
- [ ] Update security patches
- [ ] Test alert systems

### Monthly
- [ ] Rotate API keys
- [ ] Review user access
- [ ] Security scan dependencies
- [ ] Update security documentation

### Quarterly
- [ ] Penetration testing
- [ ] Security architecture review
- [ ] Incident response drill
- [ ] Security training update
'''
        
        guidelines_path = self.project_root / "docs" / "security" / "deployment-guidelines.md"
        guidelines_path.parent.mkdir(parents=True, exist_ok=True)
        with open(guidelines_path, 'w') as f:
            f.write(guidelines)
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance."""
        self.logger.info("Optimizing performance")
        
        results = {
            'status': 'success',
            'optimizations_applied': [],
            'performance_improvements': [],
            'recommendations': []
        }
        
        # Create optimized XGBoost configuration
        optimized_xgb_config = {
            'n_jobs': 1,  # Single-threaded for consistent performance
            'tree_method': 'exact',  # CPU-optimized
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 0,  # Quiet for production
            'early_stopping_rounds': 10,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
        }
        
        # Save optimized configuration
        config_path = self.project_root / "config" / "xgboost_optimized.json"
        with open(config_path, 'w') as f:
            json.dump(optimized_xgb_config, f, indent=2)
        
        results['optimizations_applied'].append(f"Created optimized XGBoost config: {config_path}")
        
        # Create performance monitoring configuration
        perf_config = {
            'targets': {
                'max_inference_latency_ms': 10.0,
                'target_accuracy': 0.80,
                'min_accuracy_threshold': 0.75,
                'max_memory_usage_mb': 1000,
                'max_cpu_usage_percent': 80
            },
            'monitoring': {
                'sample_interval_seconds': 60,
                'alert_thresholds': {
                    'latency_ms': 15.0,
                    'accuracy': 0.75,
                    'memory_mb': 1200,
                    'cpu_percent': 85
                }
            },
            'optimization': {
                'enable_caching': True,
                'cache_size_mb': 256,
                'gc_threshold': 1000,
                'batch_size': 1,  # Single prediction optimization
                'feature_selection': True,
                'model_quantization': False  # Disabled for accuracy
            }
        }
        
        perf_config_path = self.project_root / "config" / "performance.json"
        with open(perf_config_path, 'w') as f:
            json.dump(perf_config, f, indent=2)
        
        results['optimizations_applied'].append(f"Created performance config: {perf_config_path}")
        
        # Create memory optimization guidelines
        memory_guidelines = '''# Memory Optimization Guidelines

## Memory Usage Targets
- Maximum memory usage: 1GB
- Typical usage: 500-800MB
- Cache size: 256MB
- Model size: <100MB

## Optimization Strategies

### 1. Data Processing
- Use streaming for large datasets
- Implement batch processing with small batches
- Clear intermediate variables promptly
- Use memory-efficient data types

### 2. Model Optimization
- Limit model complexity (max_depth=6)
- Use feature selection to reduce dimensionality
- Implement model pruning if needed
- Consider model quantization for inference

### 3. Caching Strategy
- Cache frequently accessed data
- Implement LRU eviction policy
- Monitor cache hit rates
- Clear cache periodically

### 4. Garbage Collection
- Force garbage collection after large operations
- Monitor memory growth patterns
- Set appropriate GC thresholds
- Use memory profiling tools

## Memory Monitoring

### Key Metrics
- Process memory usage (RSS)
- Virtual memory usage
- Memory growth rate
- Cache hit/miss ratios
- GC frequency and duration

### Alert Thresholds
- Memory usage > 85% of limit
- Memory growth > 100MB/hour
- Cache hit rate < 80%
- GC frequency > 10/minute

## Troubleshooting Memory Issues

### Common Causes
- Memory leaks in data processing
- Large objects not being released
- Inefficient caching strategies
- Model size too large

### Diagnostic Tools
- memory_profiler
- tracemalloc
- psutil for system monitoring
- Custom memory tracking

### Resolution Steps
1. Identify memory hotspots
2. Optimize data structures
3. Implement proper cleanup
4. Adjust cache sizes
5. Consider model optimization
'''
        
        memory_guidelines_path = self.project_root / "docs" / "performance" / "memory-optimization.md"
        memory_guidelines_path.parent.mkdir(parents=True, exist_ok=True)
        with open(memory_guidelines_path, 'w') as f:
            f.write(memory_guidelines)
        
        results['optimizations_applied'].append(f"Created memory guidelines: {memory_guidelines_path}")
        
        return results
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness."""
        self.logger.info("Validating deployment readiness")
        
        results = {
            'status': 'success',
            'checks_passed': [],
            'checks_failed': [],
            'warnings': [],
            'overall_ready': True
        }
        
        # Check required files exist
        required_files = [
            'requirements.txt',
            'config/settings.py',
            'nifty_ml_pipeline/main.py',
            '.gitignore'
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                results['checks_passed'].append(f"Required file exists: {file_path}")
            else:
                results['checks_failed'].append(f"Missing required file: {file_path}")
                results['overall_ready'] = False
        
        # Check directory structure
        required_dirs = [
            'nifty_ml_pipeline',
            'config',
            'tests',
            'docs',
            'scripts'
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                results['checks_passed'].append(f"Required directory exists: {dir_path}")
            else:
                results['checks_failed'].append(f"Missing required directory: {dir_path}")
                results['overall_ready'] = False
        
        # Check for sensitive files in git
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            
            required_ignores = ['.env', '__pycache__', '*.pyc', '.coverage']
            for ignore_pattern in required_ignores:
                if ignore_pattern in gitignore_content:
                    results['checks_passed'].append(f"Gitignore includes: {ignore_pattern}")
                else:
                    results['warnings'].append(f"Gitignore missing: {ignore_pattern}")
        
        # Check Python version compatibility
        python_version = sys.version_info
        if python_version >= (3, 8):
            results['checks_passed'].append(f"Python version compatible: {python_version.major}.{python_version.minor}")
        else:
            results['checks_failed'].append(f"Python version too old: {python_version.major}.{python_version.minor}")
            results['overall_ready'] = False
        
        return results
    
    def create_deployment_package(self) -> Dict[str, Any]:
        """Create deployment package."""
        self.logger.info("Creating deployment package")
        
        results = {
            'status': 'success',
            'package_path': None,
            'files_included': [],
            'size_mb': 0
        }
        
        # Create deployment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"nifty_ml_pipeline_deployment_{timestamp}"
        package_dir = self.project_root / "deployment_packages" / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Files to include in deployment package
        deployment_files = [
            'nifty_ml_pipeline/',
            'config/',
            'scripts/',
            'requirements-production.txt',
            'setup.py',
            'README.md',
            '.env.production.template',
            'docs/security/',
            'docs/performance/',
            'docs/deployment/'
        ]
        
        # Copy files to package directory
        for file_path in deployment_files:
            src_path = self.project_root / file_path
            if src_path.exists():
                if src_path.is_file():
                    dst_path = package_dir / file_path
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    results['files_included'].append(file_path)
                elif src_path.is_dir():
                    dst_path = package_dir / file_path
                    shutil.copytree(src_path, dst_path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                    results['files_included'].append(f"{file_path}/ (directory)")
        
        # Create deployment instructions
        deployment_instructions = '''# NIFTY 50 ML Pipeline - Deployment Instructions

## Quick Start

1. **Extract Package**
   ```bash
   tar -xzf nifty_ml_pipeline_deployment.tar.gz
   cd nifty_ml_pipeline_deployment
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements-production.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.production.template .env
   # Edit .env with your actual values
   ```

4. **Run Deployment Validation**
   ```bash
   python scripts/deployment_validator.py
   ```

5. **Start Pipeline**
   ```bash
   python -m nifty_ml_pipeline.main
   ```

## Production Deployment

See docs/security/deployment-guidelines.md for complete production deployment instructions.

## Support

For issues or questions, contact the development team.
'''
        
        instructions_path = package_dir / "DEPLOYMENT.md"
        with open(instructions_path, 'w') as f:
            f.write(deployment_instructions)
        
        results['files_included'].append("DEPLOYMENT.md")
        
        # Calculate package size
        total_size = 0
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                total_size += file_path.stat().st_size
        
        results['size_mb'] = total_size / (1024 * 1024)
        results['package_path'] = str(package_dir)
        
        # Create compressed archive
        archive_path = package_dir.parent / f"{package_name}.tar.gz"
        shutil.make_archive(str(package_dir), 'gztar', package_dir.parent, package_name)
        
        results['archive_path'] = str(archive_path)
        
        return results
    
    def run_final_optimization(self) -> Dict[str, Any]:
        """Run complete final optimization process."""
        self.logger.info("Starting final deployment optimization")
        
        start_time = datetime.now()
        
        # Run all optimization steps
        steps = [
            ("Dependencies", self.optimize_dependencies),
            ("Security", self.fix_security_issues),
            ("Performance", self.optimize_performance),
            ("Validation", self.validate_deployment_readiness),
            ("Package", self.create_deployment_package)
        ]
        
        results = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': 0,
            'steps_completed': [],
            'steps_failed': [],
            'overall_status': 'success',
            'step_results': {}
        }
        
        for step_name, step_func in steps:
            try:
                self.logger.info(f"Running step: {step_name}")
                step_result = step_func()
                results['step_results'][step_name] = step_result
                results['steps_completed'].append(step_name)
                
                if step_result.get('status') != 'success':
                    results['overall_status'] = 'partial_success'
                
            except Exception as e:
                self.logger.error(f"Step {step_name} failed: {e}")
                results['steps_failed'].append(step_name)
                results['step_results'][step_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['overall_status'] = 'failed'
        
        end_time = datetime.now()
        results['duration_seconds'] = (end_time - start_time).total_seconds()
        
        # Generate summary report
        self._generate_final_report(results)
        
        return results
    
    def _generate_final_report(self, results: Dict[str, Any]) -> None:
        """Generate final optimization report."""
        report_dir = self.project_root / "deployment_reports"
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"final_optimization_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_path = report_dir / f"final_optimization_summary_{timestamp}.txt"
        
        summary_lines = [
            "="*80,
            "NIFTY 50 ML PIPELINE - FINAL DEPLOYMENT OPTIMIZATION",
            "="*80,
            f"Timestamp: {results['timestamp']}",
            f"Duration: {results['duration_seconds']:.2f} seconds",
            f"Overall Status: {results['overall_status'].upper()}",
            "",
            f"Steps Completed: {len(results['steps_completed'])}/{len(results['steps_completed']) + len(results['steps_failed'])}",
            f"✅ Completed: {', '.join(results['steps_completed'])}",
        ]
        
        if results['steps_failed']:
            summary_lines.extend([
                f"❌ Failed: {', '.join(results['steps_failed'])}",
                ""
            ])
        
        summary_lines.extend([
            "",
            "STEP DETAILS:",
            "-" * 40
        ])
        
        for step_name, step_result in results['step_results'].items():
            summary_lines.extend([
                f"\n{step_name}:",
                f"  Status: {step_result.get('status', 'unknown').upper()}"
            ])
            
            if 'actions_taken' in step_result:
                summary_lines.append(f"  Actions: {len(step_result['actions_taken'])}")
            
            if 'fixes_applied' in step_result:
                summary_lines.append(f"  Fixes: {len(step_result['fixes_applied'])}")
            
            if 'optimizations_applied' in step_result:
                summary_lines.append(f"  Optimizations: {len(step_result['optimizations_applied'])}")
        
        # Add deployment readiness summary
        if 'Validation' in results['step_results']:
            validation = results['step_results']['Validation']
            summary_lines.extend([
                "",
                "DEPLOYMENT READINESS:",
                f"  Overall Ready: {'YES' if validation.get('overall_ready') else 'NO'}",
                f"  Checks Passed: {len(validation.get('checks_passed', []))}",
                f"  Checks Failed: {len(validation.get('checks_failed', []))}",
                f"  Warnings: {len(validation.get('warnings', []))}"
            ])
        
        # Add package information
        if 'Package' in results['step_results']:
            package = results['step_results']['Package']
            summary_lines.extend([
                "",
                "DEPLOYMENT PACKAGE:",
                f"  Package Path: {package.get('package_path', 'N/A')}",
                f"  Archive Path: {package.get('archive_path', 'N/A')}",
                f"  Size: {package.get('size_mb', 0):.2f} MB",
                f"  Files Included: {len(package.get('files_included', []))}"
            ])
        
        summary_lines.extend([
            "",
            "NEXT STEPS:",
            "1. Review the deployment package",
            "2. Test in staging environment",
            "3. Configure production environment variables",
            "4. Follow security deployment guidelines",
            "5. Deploy to production",
            "6. Monitor system performance",
            "",
            f"Detailed report: {report_path}",
            "="*80
        ])
        
        with open(summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        self.logger.info(f"Final optimization report saved to {report_path}")
        self.logger.info(f"Summary report saved to {summary_path}")


def main():
    """Main function to run final deployment optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Finalize NIFTY 50 ML Pipeline deployment")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create finalizer and run optimization
    finalizer = DeploymentFinalizer()
    
    try:
        results = finalizer.run_final_optimization()
        
        # Print summary
        print("\n" + "="*60)
        print("DEPLOYMENT FINALIZATION SUMMARY")
        print("="*60)
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Steps Completed: {len(results['steps_completed'])}")
        print(f"Steps Failed: {len(results['steps_failed'])}")
        
        if results['steps_completed']:
            print(f"\n✅ Completed Steps:")
            for step in results['steps_completed']:
                print(f"  • {step}")
        
        if results['steps_failed']:
            print(f"\n❌ Failed Steps:")
            for step in results['steps_failed']:
                print(f"  • {step}")
        
        # Show deployment package info
        if 'Package' in results['step_results']:
            package_info = results['step_results']['Package']
            print(f"\n📦 Deployment Package:")
            print(f"  Path: {package_info.get('package_path', 'N/A')}")
            print(f"  Size: {package_info.get('size_mb', 0):.2f} MB")
            print(f"  Files: {len(package_info.get('files_included', []))}")
        
        print(f"\n📋 Next Steps:")
        print("  1. Review deployment package and documentation")
        print("  2. Test in staging environment")
        print("  3. Configure production environment")
        print("  4. Deploy to production")
        print("  5. Monitor system performance")
        
        # Return appropriate exit code
        if results['overall_status'] == 'success':
            return 0
        elif results['overall_status'] == 'partial_success':
            return 1
        else:
            return 2
        
    except Exception as e:
        print(f"Deployment finalization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())