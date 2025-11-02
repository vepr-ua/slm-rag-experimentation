# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental project building a Small Language Model (SLM) powered by Graph-based Retrieval Augmented Generation (GraphRAG) for answering questions about experimentation methodology, statistical analysis, and A/B testing best practices.

**Key Architecture Decisions:**
- GraphRAG over traditional RAG for better handling of interconnected statistical concepts and multi-hop reasoning
- SLM (3-7B parameters: Phi-3, Llama 3.2, or Qwen2.5) for cost efficiency and speed
- SurrealDB as both graph database and document store for native graph + vector capabilities
- Python 3.13+ with FastAPI for the API layer

## Current Status

This is a very early-stage project (initial commit phase). The repository currently contains:
- Basic Python project structure with pyproject.toml
- Comprehensive README.md documenting the complete vision and roadmap
- Directory structure set up for development
- No implementation code yet (main.py is a placeholder)

## Directory Structure

```
src/
├── api/          # FastAPI server (endpoints, models, middleware)
├── graph/        # SurrealDB client and graph traversal algorithms
├── rag/          # RAG pipeline (retrieval, query understanding, context assembly)
├── llm/          # SLM integration (Ollama/vLLM client, prompts, inference)
├── knowledge/    # Knowledge base construction (entity extraction, data loaders)
└── utils/        # Common utilities (logging, config, metrics)

tests/            # Mirror structure of src/ for unit tests

data/
├── raw/          # Raw source documents (textbooks, papers, blogs)
├── processed/    # Extracted entities and relationships
└── embeddings/   # Vector embeddings and model artifacts

config/           # Configuration files
scripts/          # Utility scripts (setup, ingestion, evaluation)
docs/             # Additional documentation
```

**Note**: Each major module has its own README.md explaining its purpose and planned components.

## Project Architecture

The planned system has these layers:

1. **Query Understanding Layer**: Extracts entities and identifies question types
2. **GraphRAG Pipeline**:
   - Entity extraction from user queries
   - Graph traversal in SurrealDB to find related concepts
   - Context assembly with relevance ranking
3. **SLM Layer**: Generates answers from retrieved context
4. **Response Formatter**: Returns answer with citations and related concepts

**Knowledge Graph Schema (planned):**
- `concept` table: Statistical concepts with names, definitions, categories, embeddings
- `document` table: Source content chunks with metadata
- `relates_to` relation: Concept relationships with types and strength weights

## Development Commands

### Initial Setup

**Automated setup (recommended):**
```bash
make setup
```

**Manual setup:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux

# Install dependencies with development tools
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Common Commands

All commands are available via `make`. Run `make help` to see all options.

**Development:**
```bash
make install-dev    # Install all dependencies
make dev            # Run API server with auto-reload
make test           # Run tests
make test-cov       # Run tests with coverage
make lint           # Run linters (ruff, mypy)
make format         # Format code (black, isort, ruff)
```

**Infrastructure:**
```bash
make docker-up      # Start SurrealDB
make docker-down    # Stop Docker services
make docker-logs    # View Docker logs
```

**Database:**
```bash
make db-setup       # Initialize database schema
make db-reset       # Reset database
```

**Data & Knowledge Base:**
```bash
make ingest-data    # Ingest source documents
make build-graph    # Build knowledge graph
```

**Evaluation:**
```bash
make evaluate       # Run evaluation framework
make benchmark      # Performance benchmarks
```

**Cleanup:**
```bash
make clean          # Remove build artifacts and cache
```

### Direct Commands (without Make)

```bash
# Run API server
python -m uvicorn src.api.main:app --reload

# Run tests
pytest

# Format code
black src tests
ruff check --fix src tests

# Start infrastructure
docker-compose up -d
```

## Development Phases

The project follows a 6-week Phase I roadmap:

**Weeks 1-2: Knowledge Base Construction**
- Data collection from textbooks, papers, blogs, Wikipedia
- Entity extraction for statistical concepts
- Relationship mapping between concepts
- SurrealDB schema design and implementation

**Weeks 3-4: GraphRAG Pipeline**
- Query understanding (entity recognition, intent classification)
- Graph traversal algorithms (BFS/DFS with weighted paths)
- Context assembly and ranking
- Retrieval evaluation framework

**Week 5: SLM Integration**
- Model selection and benchmarking
- Prompt engineering for different question types
- Inference pipeline setup (Ollama or vLLM)

**Week 6: API & Evaluation**
- FastAPI server implementation
- Evaluation framework with 100 test questions
- Documentation

## Tech Stack

- **Language**: Python 3.13+
- **Graph DB**: SurrealDB (native graph + document + vector)
- **Embeddings**: sentence-transformers (local)
- **LLM**: Phi-3 / Llama 3.2 / Qwen2.5 (3-4B params)
- **Inference**: Ollama
- **API**: FastAPI
- **Orchestration**: Docker Compose
- **Testing**: pytest

## Success Metrics (Phase I Targets)

- 85%+ accuracy on statistical definition questions
- 75%+ accuracy on methodology questions
- Retrieval latency < 200ms
- End-to-end latency < 2 seconds
- 90%+ citation accuracy
- Cost < $0.001 per query

## Data Sources

When implementing data collection:
- OpenStax Statistics Textbook (CC-BY)
- ArXiv experimentation papers
- Industry blogs: Netflix, Booking.com, Airbnb
- Wikipedia statistical concepts
- "Trustworthy Online Controlled Experiments" (Kohavi et al.)
- Evan Miller's statistics blog
