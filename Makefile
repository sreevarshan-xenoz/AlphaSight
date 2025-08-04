.PHONY: help install install-dev test test-cov lint format clean run

help:
	@echo "Available commands:"
	@echo "  install     - Install production dependencies"
	@echo "  install-dev - Install development dependencies"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting (flake8)"
	@echo "  format      - Format code (black, isort)"
	@echo "  clean       - Clean build artifacts"
	@echo "  run         - Run the pipeline"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

test:
	pytest

test-cov:
	pytest --cov=nifty_ml_pipeline --cov-report=html --cov-report=term-missing

lint:
	flake8 nifty_ml_pipeline tests config
	black --check nifty_ml_pipeline tests config
	isort --check-only nifty_ml_pipeline tests config

format:
	black nifty_ml_pipeline tests config
	isort nifty_ml_pipeline tests config

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m nifty_ml_pipeline.main