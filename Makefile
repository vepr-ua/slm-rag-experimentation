.PHONY: help install install-dev setup test test-cov lint format clean docker-up docker-down run dev

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup (requires uv package manager):"
	@echo "  make install-uv   - Install uv package manager"
	@echo "  make setup        - Full project setup (recommended for first time)"
	@echo "  make install      - Install production dependencies (requires uv)"
	@echo "  make install-dev  - Install development dependencies (requires uv)"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make test-config  - Test configuration loading from .env"
	@echo "  make lint         - Run linters (ruff, mypy)"
	@echo "  make format       - Format code (black, isort, ruff)"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo ""
	@echo "Data Collection (for model training):"
	@echo "  make collect-data     - Collect from all sources (slow, respects API limits)"
	@echo "  make collect-cv       - Collect from Cross Validated (100 questions, safe)"
	@echo "  make collect-cv-full  - Collect 1000+ questions (requires API key)"
	@echo "  make collect-arxiv    - Collect ArXiv papers metadata"
	@echo "  make collect-arxiv-pdfs - Download ArXiv PDFs (very slow)"
	@echo ""
	@echo "Synthetic Q&A Generation (requires Claude API key):"
	@echo "  make generate-qa-test - Generate from 10 papers (test, ~\$0.24)"
	@echo "  make generate-qa      - Generate from all papers (~\$4-5)"
	@echo ""
	@echo "Model Training (requires GPU with 6+ GB VRAM):"
	@echo "  make combine-datasets - Combine Cross Validated + ArXiv datasets"
	@echo "  make train            - Train Llama 3.2 3B with QLoRA (default config)"
	@echo "  make train-full       - Combine datasets and train (4-12 hours)"
	@echo "  make train-custom     - Train with custom config (CONFIG=path/to/config.json)"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make docker-up    - Start Docker services (SurrealDB)"
	@echo "  make docker-down  - Stop Docker services"
	@echo "  make run          - Run the API server"
	@echo "  make dev          - Run API server in development mode (auto-reload)"

# Installation (requires uv)
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

# Full setup (recommended - checks for uv and sets everything up)
setup:
	bash scripts/setup.sh

# Install uv (if not already installed)
install-uv:
	@echo "Installing uv package manager..."
	@if command -v uv &> /dev/null; then \
		echo "✅ uv is already installed"; \
	else \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "✅ uv installed. Restart your shell or run: source ~/.cargo/env"; \
	fi

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

# Configuration
test-config:
	@echo "Testing configuration loading from .env..."
	python scripts/test_config.py

debug-stackexchange:
	@echo "Testing StackExchange API connection..."
	python scripts/debug_stackexchange.py

# Data Collection (for model training)
# Note: Respects API rate limits. Add API keys to .env for higher limits.
collect-data:
	@echo "⚠️  This will collect from all sources. Estimated time: ~30+ minutes."
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	python scripts/collect_data.py --source all

collect-cv:
	@echo "Collecting from Cross Validated (max 100 questions, safe for no API key)"
	python scripts/collect_data.py --source cross-validated --max-questions 100

collect-cv-full:
	@echo "⚠️  Collecting 1000+ questions. Requires StackExchange API key in .env"
	python scripts/collect_data.py --source cross-validated --max-questions 1000

collect-arxiv:
	@echo "Collecting ArXiv papers (metadata only, no PDFs)"
	python scripts/collect_data.py --source arxiv

collect-arxiv-pdfs:
	@echo "⚠️  Downloading PDFs may take a long time. Press Ctrl+C to cancel..."
	@sleep 3
	python scripts/collect_data.py --source arxiv --download-pdfs

# Synthetic Q&A Generation (requires ANTHROPIC_API_KEY in .env)
generate-qa-test:
	@echo "Generating synthetic Q&A from 10 papers (test run)..."
	python scripts/generate_synthetic_qa.py --source arxiv --max-papers 10

generate-qa:
	@echo "⚠️  This will generate Q&A from ALL ArXiv papers. Estimated cost: ~\$4-5."
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	python scripts/generate_synthetic_qa.py --source arxiv

# Model Training (requires GPU with 6+ GB VRAM recommended)
combine-datasets:
	@echo "Combining Cross Validated and ArXiv synthetic datasets..."
	python scripts/train_model.py --combine-only

train:
	@echo "Training Llama 3.2 3B with QLoRA..."
	python scripts/train_model.py

train-full:
	@echo "⚠️  This will combine datasets and train the model. Estimated time: 4-12 hours."
	@echo "Requires: GPU with 6+ GB VRAM (RTX 3060+, M1 Max+)"
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	python scripts/train_model.py --combine-datasets

train-custom:
	@echo "Training with custom configuration..."
	@if [ -z "$(CONFIG)" ]; then \
		echo "❌ Please provide CONFIG path: make train-custom CONFIG=configs/my_config.json"; \
		exit 1; \
	fi
	python scripts/train_model.py --config $(CONFIG)

# Knowledge base (deprecated for now, focusing on model training)
ingest-data:
	python scripts/ingest_data.py

build-graph:
	python scripts/build_graph.py

# Evaluation
evaluate:
	python scripts/evaluate.py

benchmark:
	python scripts/benchmark.py
