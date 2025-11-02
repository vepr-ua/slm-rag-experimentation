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
- No implementation code yet (main.py is a placeholder)

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

Since the project is in early stages, there are no build, test, or run commands yet.

**Project setup (when dependencies are added):**
```bash
# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux

# Install dependencies (when added to pyproject.toml)
uv pip install -e .
```

**Planned future commands (from README):**
```bash
# Start infrastructure (SurrealDB, etc.)
docker-compose up -d

# Start API server
python -m src.api.main

# Run tests
pytest
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
