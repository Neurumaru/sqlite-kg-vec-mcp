# Makefile for SQLite KG Vec MCP project
.PHONY: help install install-dev lint format type-check test clean build

# Variables
PYTHON = uv run python
UV = uv

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  lint         - Run linting (black, isort, flake8)"
	@echo "  format       - Format code (black, isort)"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  clean        - Clean up build artifacts"
	@echo "  build        - Build the package"

# Installation
install:
	$(UV) sync --no-dev

install-dev:
	$(UV) sync --group dev

# Linting and formatting
lint: format type-check
	@echo "✅ All linting checks passed!"

format:
	@echo "🔧 Formatting code..."
	$(UV) run black src/ tests/ examples/
	$(UV) run isort src/ tests/ examples/
	@echo "✅ Code formatting complete!"

type-check:
	@echo "🔍 Running type checks..."
	$(UV) run mypy src/
	@echo "✅ Type checking complete!"

flake8:
	@echo "🔍 Running flake8..."
	$(UV) run flake8 src/ tests/ examples/
	@echo "✅ Flake8 checks complete!"

# Testing
test:
	@echo "🧪 Running tests..."
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

test-cov:
	@echo "🧪 Running tests with coverage..."
	$(UV) run coverage run -m unittest discover -s tests -p "test_*.py"
	$(UV) run coverage report
	$(UV) run coverage html

# Development utilities
clean:
	@echo "🧹 Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

build:
	@echo "📦 Building package..."
	$(UV) build
	@echo "✅ Build complete!"

# Development workflow
dev-setup: install-dev
	@echo "🚀 Development environment ready!"

check: lint test
	@echo "✅ All checks passed!"