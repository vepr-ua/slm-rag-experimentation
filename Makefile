.PHONY: help install install-dev setup test test-cov lint format clean docker-up docker-down run dev

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make setup        - Full project setup (recommended for first time)"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make lint         - Run linters (ruff, mypy)"
	@echo "  make format       - Format code (black, isort, ruff)"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo "  make docker-up    - Start Docker services (SurrealDB)"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make run          - Run the API server"
	@echo "  make dev          - Run API server in development mode (auto-reload)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Full setup
setup:
	bash scripts/setup.sh

# Testing
test:
	pytest

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

test-watch:
	pytest-watch

# Code quality
lint:
	@echo "Running ruff..."
	ruff check src tests
	@echo "Running mypy..."
	mypy src

format:
	@echo "Running isort..."
	isort src tests
	@echo "Running black..."
	black src tests
	@echo "Running ruff --fix..."
	ruff check --fix src tests

# Cleanup
clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	@echo "✅ Cleaned up build artifacts and cache"

# Docker services
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Run application
run:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

dev:
	python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Database
db-setup:
	python scripts/setup_db.py

db-reset:
	python scripts/reset_db.py

# Knowledge base
ingest-data:
	python scripts/ingest_data.py

build-graph:
	python scripts/build_graph.py

# Evaluation
evaluate:
	python scripts/evaluate.py

benchmark:
	python scripts/benchmark.py
