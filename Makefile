.PHONY: help test test-verbose test-fast test-coverage test-hypothesis \
        lint format clean install install-dev all

help:
	@echo "BioStat Easy - Available Commands:"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run all tests"
	@echo "  make test-verbose      - Run tests with verbose output"
	@echo "  make test-fast         - Run tests excluding slow tests"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo "  make test-hypothesis   - Run only hypothesis tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint              - Check code with mypy"
	@echo "  make format            - Format code with black"
	@echo ""
	@echo "Setup:"
	@echo "  make install           - Install dependencies"
	@echo "  make install-dev       - Install dev dependencies"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean             - Clean temporary files"
	@echo ""
	@echo "Full:"
	@echo "  make all               - Clean, install-dev, and test-coverage"
	@echo ""

test:
	python -m pytest tests/ -v

test-verbose:
	python -m pytest tests/ -vv --tb=long

test-fast:
	python -m pytest tests/ -v -m "not slow"

test-coverage:
	python -m pytest tests/ --cov=modules --cov-report=html --cov-report=term-missing

test-hypothesis:
	python -m pytest tests/ -v -m "hypothesis"

test-markers:
	python -m pytest tests/ -v --markers

lint:
	python -m mypy modules/ --ignore-missing-imports

format:
	python -m black modules/ tests/ setup.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements.txt
	python -m pip install -r requirements-dev.txt

all: clean install-dev test-coverage
	@echo "✓ Full test suite completed"
