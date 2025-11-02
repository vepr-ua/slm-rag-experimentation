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

**Phase I: Model Training Pipeline** (In Progress)

Completed:
- ✅ Project structure and dependency management with UV
- ✅ Data collection infrastructure (Cross Validated, ArXiv)
- ✅ Synthetic Q&A generation pipeline using Claude API
- ✅ QLoRA training infrastructure with HuggingFace TRL
- ✅ Dataset combination and preprocessing utilities

In Progress:
- 🔄 Generating synthetic Q&A from ArXiv papers (~172 papers → ~860 Q&A pairs)
- 🔄 Preparing combined training dataset (target: 5,000-10,000 examples)

Next:
- ⏭️ Fine-tune Llama 3.2 3B with QLoRA
- ⏭️ Build evaluation framework
- ⏭️ Deploy inference API

Postponed to Phase II:
- GraphRAG pipeline (SurrealDB, graph traversal)
- Knowledge graph construction

## Directory Structure

```
src/
├── api/                  # FastAPI server (endpoints, models, middleware)
├── data_collection/      # ✅ Data collection infrastructure
│   ├── collectors/       #    - Cross Validated, ArXiv collectors
│   ├── formatters/       #    - ChatML formatting for training
│   ├── generators/       #    - Synthetic Q&A generation with Claude
│   └── validators/       #    - Quality validation
├── llm/                  # ✅ LLM infrastructure
│   ├── claude_client.py  #    - Claude API wrapper
│   └── prompts.py        #    - Generation prompts
├── training/             # ✅ Model training infrastructure
│   ├── config.py         #    - Training configuration (QLoRA, hyperparameters)
│   ├── dataset.py        #    - Dataset loaders and combiners
│   └── trainer.py        #    - QLoRA trainer with HuggingFace TRL
├── graph/                # (Postponed) SurrealDB client and graph traversal
├── rag/                  # (Postponed) RAG pipeline
├── knowledge/            # (Postponed) Knowledge base construction
└── utils/                # Common utilities (logging, config, metrics)

tests/                    # Mirror structure of src/ for unit tests

data/
├── raw/                  # ✅ Raw data from collectors
│   ├── cross_validated/  #    - Cross Validated Q&A
│   └── arxiv/           #    - ArXiv papers and metadata
├── processed/            # ✅ Processed training data
│   ├── *_chatml.jsonl   #    - ChatML formatted for training
│   └── *_raw.json       #    - Raw structured data
└── db/                  # SurrealDB data (gitignored)

models/                   # Model checkpoints and artifacts
├── checkpoints/          # Training checkpoints
└── final/               # Final trained model

scripts/                  # ✅ Utility scripts
├── collect_data.py      #    - Data collection CLI
├── generate_synthetic_qa.py  # - Synthetic Q&A generation
└── train_model.py       #    - Model training CLI

docs/                     # Additional documentation
```

**Note**: ✅ indicates implemented modules. Each major module has its own README.md.

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

### Prerequisites

This project uses **uv** for fast, reliable package management.

**Install uv:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use make
make install-uv
```

### Initial Setup

**Automated setup (recommended):**
```bash
make setup
```

This will:
1. Check for uv installation
2. Verify Python 3.13+
3. Create virtual environment
4. Install all dependencies
5. Set up .env file

**Manual setup:**
```bash
# Install uv (if not already installed)
make install-uv

# Create virtual environment with uv
uv venv --python 3.13

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux

# Install dependencies with uv
uv pip install -e ".[dev]"

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

**Data Collection:**
```bash
make collect-cv          # Collect from Cross Validated (100 questions, safe)
make collect-cv-full     # Collect 1000+ questions (requires API key)
make collect-arxiv       # Collect ArXiv papers metadata
make collect-arxiv-pdfs  # Download ArXiv PDFs (slow)
make collect-data        # Collect from all sources (slow)
```

**Synthetic Q&A Generation:**
```bash
make generate-qa-test    # Generate from 10 papers (test, ~$0.24)
make generate-qa         # Generate from all papers (~$4-5)
```

**Model Training:**
```bash
make combine-datasets    # Combine Cross Validated + ArXiv datasets
make train              # Train Llama 3.2 3B with QLoRA (default config)
make train-full         # Combine datasets and train (4-12 hours)
make train-custom CONFIG=path/to/config.json  # Train with custom config
```

**Infrastructure (Postponed):**
```bash
make docker-up      # Start SurrealDB (not needed for training)
make docker-down    # Stop Docker services
```

**Evaluation (Coming Soon):**
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

**Phase I: Model Training (Current Focus)**
- ✅ Data collection infrastructure (Cross Validated, ArXiv)
- ✅ Synthetic Q&A generation with Claude API
- ✅ QLoRA training infrastructure
- 🔄 Generate 5,000-10,000 training examples
- ⏭️ Fine-tune Llama 3.2 3B with QLoRA
- ⏭️ Build evaluation framework
- ⏭️ Deploy inference API

**Phase II: GraphRAG Enhancement (Planned)**
- Knowledge graph construction from trained model outputs
- SurrealDB integration for graph + document + vector storage
- Graph traversal for multi-hop reasoning
- Enhanced retrieval with graph context
- Comparative evaluation: Fine-tuned SLM vs. SLM+GraphRAG

**Rationale for Phase Split:**
We're starting with fine-tuning because:
1. Faster time to usable model (~1 week vs. ~4 weeks)
2. Establishes baseline performance for comparison
3. GraphRAG can enhance an already-good model
4. Iterative approach reduces risk

## Tech Stack

- **Language**: Python 3.13+
- **Package Manager**: uv (10-100x faster than pip)
- **Graph DB**: SurrealDB (native graph + document + vector) - *Postponed*
- **Embeddings**: sentence-transformers (local)
- **LLM**: Llama 3.2 3B (strong reasoning, 128K context)
- **Training**: HuggingFace TRL + QLoRA (4-bit fine-tuning)
- **Inference**: Ollama
- **API**: FastAPI
- **Orchestration**: Docker Compose
- **Testing**: pytest

**Current Focus**: Model training pipeline (data collection → fine-tuning)

## Success Metrics (Phase I Targets)

- 85%+ accuracy on statistical definition questions
- 75%+ accuracy on methodology questions
- Retrieval latency < 200ms
- End-to-end latency < 2 seconds
- 90%+ citation accuracy
- Cost < $0.001 per query

## Data Sources

**Currently Implemented:**
- ✅ **Cross Validated (StackExchange)**: Real Q&A about experimentation and statistics
  - Tags: a-b-testing, experimental-design, sample-size, statistical-power, etc.
  - ~100-1,000 question-answer pairs
- ✅ **ArXiv Papers**: Research papers on experimentation methodology
  - ~172 papers collected
  - Synthetic Q&A generated from abstracts using Claude API
  - ~860 Q&A pairs from synthetic generation

**Future Data Sources:**
- OpenStax Statistics Textbook (CC-BY)
- Industry blogs: Netflix, Booking.com, Airbnb
- Wikipedia statistical concepts
- "Trustworthy Online Controlled Experiments" (Kohavi et al.)
- Evan Miller's statistics blog

**Training Dataset Target:**
- 5,000-10,000 total examples
- Mix of real Q&A and synthetic Q&A
- ChatML format for Llama 3.2 fine-tuning
