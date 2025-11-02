#!/bin/bash
# Setup script for slm-rag-experimentation project

set -e

echo "🚀 Setting up slm-rag-experimentation..."

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.13"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 13) else 1)"; then
    echo "❌ Python 3.13+ is required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python version: $PYTHON_VERSION"

# Check if uv is available, otherwise use pip
if command -v uv &> /dev/null; then
    echo "✅ Using uv for package management"
    PKG_MANAGER="uv pip"
else
    echo "ℹ️  uv not found, using pip. Consider installing uv: https://github.com/astral-sh/uv"
    PKG_MANAGER="pip"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
if [ "$PKG_MANAGER" = "uv pip" ]; then
    uv pip install -e ".[dev]"
else
    pip install -e ".[dev]"
fi
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
