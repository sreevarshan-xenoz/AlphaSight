# NIFTY 50 ML Pipeline - Deployment Instructions

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
