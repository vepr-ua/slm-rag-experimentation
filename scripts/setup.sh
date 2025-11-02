#!/bin/bash
# Setup script for slm-rag-experimentation project

set -e

echo "🚀 Setting up slm-rag-experimentation..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed!"
    echo ""
    echo "Please install uv first:"
    echo "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Or use: make install-uv"
    echo ""
    exit 1
fi
echo "✅ uv is installed"

# Check Python version
echo "Checking Python version..."
if ! python3 --version &> /dev/null; then
    echo "❌ python3 not found!"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python: $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)" 2>/dev/null; then
    echo "⚠️  Warning: Python 3.13+ is recommended. Found: $PYTHON_VERSION"
    echo "Continuing with Python $PYTHON_VERSION..."
    PYTHON_CMD="python3"
else
    echo "✅ Python version OK"
    PYTHON_CMD="python3"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    # Try with specific version first, fall back to system python
    if ! uv venv .venv --python 3.13 2>/dev/null; then
        echo "Python 3.13 not found, using system python3..."
        uv venv .venv
    fi
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "❌ Virtual environment activation script not found!"
    exit 1
fi

# Install dependencies with uv
echo "Installing dependencies with uv..."
uv pip install -e ".[dev]"
echo "✅ Dependencies installed"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file (please configure it)"
else
    echo "✅ .env file already exists"
fi

# Create data directories
echo "Setting up data directories..."
mkdir -p data/raw data/processed data/embeddings data/db
echo "✅ Data directories ready"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Configure .env file with your settings"
echo "  3. Start SurrealDB: docker-compose up -d surrealdb"
echo "  4. Initialize database schema: python scripts/setup_db.py"
echo ""
