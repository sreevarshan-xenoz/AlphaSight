# NIFTY 50 ML Pipeline Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the NIFTY 50 ML Pipeline in various environments, from local development to production deployment.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- 4GB+ RAM
- CPU with at least 2 cores (optimized for CPU-only execution)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd nifty-50-ml-pipeline
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up configuration:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the pipeline:**
```bash
python -m nifty_ml_pipeline.main
```

## Environment Setup

### Development Environment

#### System Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for data and models
- **CPU**: Multi-core processor (optimized for CPU execution)

#### Development Dependencies
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/ -v

# Generate coverage report
python -m pytest tests/ --cov=nifty_ml_pipeline --cov-report=html
```

### Production Environment

#### System Requirements
- **OS**: Linux (Ubuntu 20.04+ or CentOS 8+)
- **Python**: 3.9 or 3.10 (recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for data, models, and logs
- **CPU**: 4+ cores for optimal performance
- **Network**: Stable internet connection for data collection

#### Production Setup
```bash
# Create production user
sudo useradd -m -s /bin/bash nifty-pipeline
sudo su - nifty-pipeline

# Clone and setup
git clone <repository-url> nifty-ml-pipeline
cd nifty-ml-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install -r requirements.txt

# Set up configuration
cp .env.example .env
# Configure production settings in .env

# Create necessary directories
mkdir -p data/{cache,news,prices} logs models

# Set up log rotation
sudo cp deployment/logrotate.conf /etc/logrotate.d/nifty-pipeline
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
ECONOMIC_TIMES_API_KEY=your_api_key_here
NSE_API_TIMEOUT=30

# Data Configuration
DATA_RETENTION_DAYS=365
CACHE_ENABLED=true
CACHE_TTL_HOURS=24

# Model Configuration
MODEL_PATH=models/xgboost_model.pkl
MAX_INFERENCE_LATENCY_MS=10
MIN_CONFIDENCE_THRESHOLD=0.7

# Performance Configuration
CPU_THREADS=1
MEMORY_LIMIT_GB=4

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=pipeline.log
LOG_MAX_SIZE_MB=100
LOG_BACKUP_COUNT=5

# Scheduling Configuration
EXECUTION_TIME=17:30
TIMEZONE=Asia/Kolkata
ENABLE_SCHEDULER=true

# Monitoring Configuration
ENABLE_PERFORMANCE_MONITORING=true
ALERT_EMAIL=admin@example.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Configuration Files

#### `config/settings.py`

The main configuration file that loads environment variables and provides default values:

```python
import os
from typing import Dict, Any

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        'api': {
            'keys': {
                'ECONOMIC_TIMES_API_KEY': os.getenv('ECONOMIC_TIMES_API_KEY'),
            },
            'timeouts': {
                'NSE_API_TIMEOUT': int(os.getenv('NSE_API_TIMEOUT', '30')),
            }
        },
        'data': {
            'retention_days': int(os.getenv('DATA_RETENTION_DAYS', '365')),
            'storage_format': os.getenv('STORAGE_FORMAT', 'parquet'),
            'cache_enabled': os.getenv('CACHE_ENABLED', 'true').lower() == 'true',
            'cache_ttl_hours': int(os.getenv('CACHE_TTL_HOURS', '24')),
        },
        'model': {
            'path': os.getenv('MODEL_PATH', 'models/xgboost_model.pkl'),
            'max_inference_latency_ms': int(os.getenv('MAX_INFERENCE_LATENCY_MS', '10')),
            'min_confidence_threshold': float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.7')),
        },
        'performance': {
            'cpu_threads': int(os.getenv('CPU_THREADS', '1')),
            'memory_limit_gb': int(os.getenv('MEMORY_LIMIT_GB', '4')),
            'MAX_INFERENCE_LATENCY_MS': int(os.getenv('MAX_INFERENCE_LATENCY_MS', '10')),
        },
        'logging': {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'file': os.getenv('LOG_FILE', 'pipeline.log'),
            'max_size_mb': int(os.getenv('LOG_MAX_SIZE_MB', '100')),
            'backup_count': int(os.getenv('LOG_BACKUP_COUNT', '5')),
        },
        'scheduling': {
            'execution_time': os.getenv('EXECUTION_TIME', '17:30'),
            'timezone': os.getenv('TIMEZONE', 'Asia/Kolkata'),
            'enabled': os.getenv('ENABLE_SCHEDULER', 'true').lower() == 'true',
        },
        'monitoring': {
            'enabled': os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true',
            'alert_email': os.getenv('ALERT_EMAIL'),
            'slack_webhook': os.getenv('SLACK_WEBHOOK_URL'),
        },
        'paths': {
            'data': os.getenv('DATA_PATH', 'data'),
            'models': os.getenv('MODELS_PATH', 'models'),
            'logs': os.getenv('LOGS_PATH', 'logs'),
        }
    }
```

## Deployment Scenarios

### Local Development

For local development and testing:

```bash
# Set development environment
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export ENABLE_SCHEDULER=false

# Run pipeline manually
python -m nifty_ml_pipeline.main

# Run with specific symbol
python -c "
from nifty_ml_pipeline.orchestration.controller import PipelineController
from config.settings import get_config

controller = PipelineController(get_config())
result = controller.execute_pipeline('NIFTY 50')
print(f'Pipeline completed: {result.status}')
"
```

### Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 pipeline && chown -R pipeline:pipeline /app
USER pipeline

# Create necessary directories
RUN mkdir -p data/{cache,news,prices} logs models

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "nifty_ml_pipeline.main"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  nifty-pipeline:
    build: .
    container_name: nifty-ml-pipeline
    restart: unless-stopped
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - DATA_RETENTION_DAYS=365
      - MAX_INFERENCE_LATENCY_MS=10
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
      - ./.env:/app/.env
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Add monitoring services
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

### GitHub Actions Deployment

The pipeline includes automated deployment via GitHub Actions:

```yaml
name: Deploy NIFTY ML Pipeline

on:
  schedule:
    - cron: '30 12 * * *'  # Daily at 5:30 PM IST (12:00 UTC)
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run pipeline
      env:
        ECONOMIC_TIMES_API_KEY: ${{ secrets.ECONOMIC_TIMES_API_KEY }}
        LOG_LEVEL: INFO
        ENVIRONMENT: production
      run: |
        python -m nifty_ml_pipeline.main
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: pipeline-results
        path: |
          logs/
          data/predictions/
```

### Cloud Deployment

#### AWS EC2 Deployment

```bash
# Launch EC2 instance (t3.medium or larger)
# Connect to instance and run:

# Update system
sudo yum update -y
sudo yum install -y python3 python3-pip git

# Clone repository
git clone <repository-url> nifty-ml-pipeline
cd nifty-ml-pipeline

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with production settings

# Set up systemd service
sudo cp deployment/nifty-pipeline.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nifty-pipeline
sudo systemctl start nifty-pipeline

# Check status
sudo systemctl status nifty-pipeline
```

#### Systemd Service File

```ini
[Unit]
Description=NIFTY 50 ML Pipeline
After=network.target

[Service]
Type=simple
User=nifty-pipeline
WorkingDirectory=/home/nifty-pipeline/nifty-ml-pipeline
Environment=PATH=/home/nifty-pipeline/nifty-ml-pipeline/venv/bin
ExecStart=/home/nifty-pipeline/nifty-ml-pipeline/venv/bin/python -m nifty_ml_pipeline.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Monitoring and Maintenance

### Log Management

```bash
# View real-time logs
tail -f pipeline.log

# Search for errors
grep -i error pipeline.log

# Monitor performance
grep "inference_time" pipeline.log | tail -20
```

### Performance Monitoring

The pipeline includes built-in performance monitoring:

- **Inference Latency**: Tracks prediction response times
- **Accuracy Metrics**: Monitors model performance over time
- **Resource Usage**: CPU and memory utilization
- **Error Rates**: Tracks failures and recovery attempts

### Health Checks

```bash
# Check pipeline health
python -c "
from nifty_ml_pipeline.orchestration.controller import PipelineController
from config.settings import get_config
import sys

try:
    controller = PipelineController(get_config())
    print('Pipeline health: OK')
    sys.exit(0)
except Exception as e:
    print(f'Pipeline health: ERROR - {e}')
    sys.exit(1)
"
```

### Backup and Recovery

```bash
# Backup models and data
tar -czf backup-$(date +%Y%m%d).tar.gz models/ data/ logs/

# Restore from backup
tar -xzf backup-20240101.tar.gz
```

## Security Considerations

### API Keys
- Store API keys in environment variables or secure key management systems
- Rotate API keys regularly
- Use least-privilege access principles

### Network Security
- Use HTTPS for all external API calls
- Implement rate limiting for API requests
- Monitor for unusual network activity

### Data Security
- Encrypt sensitive data at rest
- Use secure file permissions (600 for config files)
- Implement data retention policies

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

## Performance Tuning

See [Performance Tuning Guide](performance-tuning.md) for optimization recommendations.