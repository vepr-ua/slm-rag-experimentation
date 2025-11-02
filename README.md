# Experimentation Intelligence: SLM + GraphRAG

A specialized Small Language Model (SLM) for answering questions about experimentation methodology, statistical analysis, and A/B testing best practices.

**Current Focus**: Fine-tuning Llama 3.2 3B with QLoRA on experimentation Q&A data.

**Future**: Graph-based Retrieval Augmented Generation (GraphRAG) for enhanced multi-hop reasoning.

## Project Status

**Phase I: Model Training** (In Progress)

- ✅ Data collection infrastructure (Cross Validated, ArXiv)
- ✅ Synthetic Q&A generation with Claude API
- ✅ QLoRA training infrastructure with HuggingFace TRL
- 🔄 Generating ~860 synthetic Q&A pairs from 172 ArXiv papers
- ⏭️ Fine-tune Llama 3.2 3B on 5,000-10,000 training examples
- ⏭️ Build evaluation framework
- ⏭️ Deploy inference API

**Phase II: GraphRAG Enhancement** (Planned)

- Knowledge graph construction
- SurrealDB integration
- Graph-enhanced retrieval

## Usage of LLMs

I will be leveraging Claude and other LLM tools to help me get to a proof of concept faster. I will continue to use my software engineering skills to direct, fix, and modify the project code. I'd like the code to be readable and exercising the best practices for security and performance.

## Project Vision

Build a domain-specific AI assistant that provides expert-level guidance on:

- **Statistical Concepts**: Confidence intervals, p-values, statistical significance, power analysis
- **Experimentation Design**: Sample size calculations, test duration, metric selection
- **Analysis Methods**: Hypothesis testing, variance reduction, multiple testing corrections
- **Best Practices**: Our internal experimentation processes and methodologies

## Why This Approach?

### GraphRAG Over Traditional RAG

- **Entities & Relationships**: Statistical concepts are interconnected (e.g., "power analysis" relates to "sample size", "effect size", "significance level")
- **Multi-hop Reasoning**: Questions like "How does variance affect my sample size calculation?" require traversing concept relationships
- **Context Preservation**: Graph structure maintains the semantic relationships between experimentation concepts
- **Better Retrieval**: Find relevant information through concept graphs, not just keyword matching

### SLM Over Large Models

- **Cost Efficiency**: 10-100x cheaper than GPT-4 for high-volume queries
- **Speed**: Sub-second responses for real-time guidance
- **Privacy**: All data stays in-house (important for proprietary methodologies)
- **Specialization**: Focused training on statistics/experimentation = better domain performance

### SurrealDB as Knowledge Graph

- **Native Graph + Document Store**: Store both structured relationships and unstructured content
- **GraphQL Support**: Flexible querying for graph traversal
- **Scalable**: Handles growing knowledge base as we add experiments
- **SQL-like Syntax**: Easy to work with, familiar to data scientists

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Question                        │
│   "What sample size do I need for 5% MDE at 80% power?" │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Query Understanding Layer                   │
│  - Extract entities (sample size, MDE, power)           │
│  - Identify question type (calculation vs explanation)  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 GraphRAG Pipeline                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 1. Entity Extraction                             │   │
│  │    - Identify concepts in query                  │   │
│  └─────────────────┬────────────────────────────────┘   │
│  ┌─────────────────▼────────────────────────────────┐   │
│  │ 2. Graph Traversal (SurrealDB)                   │   │
│  │    - Find related concepts                       │   │
│  │    - Retrieve connected documentation            │   │
│  └─────────────────┬────────────────────────────────┘   │
│  ┌─────────────────▼────────────────────────────────┐   │
│  │ 3. Context Assembly                              │   │
│  │    - Rank by relevance                           │   │
│  │    - Build knowledge subgraph                    │   │
│  └─────────────────┬────────────────────────────────┘   │
└────────────────────┼────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Small Language Model (3-7B)                 │
│  - Phi-3 / Llama 3.2 / Qwen2.5                          │
│  - Fine-tuned on statistics/experimentation             │
│  - Generates answer from retrieved context              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Response Formatter                      │
│  - Answer + reasoning                                    │
│  - Source citations (graph nodes)                        │
│  - Related concepts to explore                           │
└─────────────────────────────────────────────────────────┘
```

## Phase I: Model Training (Current, ~1-2 weeks)

**Rationale**: Start with fine-tuning to establish baseline performance before building GraphRAG infrastructure.

### Data Collection ✅

**Goal**: Gather training data from public sources

- [x] **Cross Validated (StackExchange)**
  - Real Q&A about experimentation and statistics
  - Tags: a-b-testing, experimental-design, sample-size, statistical-power, etc.
  - ~100-1,000 question-answer pairs

- [x] **ArXiv Papers**
  - Research papers on experimentation methodology
  - ~172 papers collected
  - Metadata and abstracts

### Synthetic Q&A Generation ✅

**Goal**: Generate training data from ArXiv papers using Claude API

- [x] **Claude API Integration**
  - Rate-limited wrapper with retry logic
  - Cost estimation and tracking

- [x] **Q&A Generation from Abstracts**
  - 5 diverse Q&A pairs per paper
  - Question types: definition, methodology, comparison, best practice
  - ~860 synthetic Q&A pairs from 172 papers

- [x] **Quality Validation**
  - Length validation (50-5000 chars)
  - Content quality checks
  - ChatML formatting for Llama 3.2

### Model Training (In Progress)

**Goal**: Fine-tune Llama 3.2 3B with QLoRA

- [x] **Training Infrastructure**
  - QLoRA configuration (4-bit quantization + LoRA adapters)
  - HuggingFace TRL SFTTrainer
  - Dataset combination and preprocessing

- [ ] **Training Execution**
  - Combine Cross Validated + ArXiv synthetic datasets
  - Train for 3 epochs on 5,000-10,000 examples
  - Evaluate on holdout test set

- [ ] **Model Artifacts**
  - Save LoRA adapters
  - Export merged model
  - Document hyperparameters

### Evaluation Framework (Next)

**Goal**: Measure model quality

- [ ] **Test Set Creation**
  - 100 curated questions covering:
    - Definitions (20%)
    - Calculations (30%)
    - Methodology (30%)
    - Best practices (20%)

- [ ] **Metrics**
  - Accuracy on classification questions
  - ROUGE/BLEU for generation quality
  - Human evaluation on critical questions

- [ ] **Target Performance**
  - 80%+ accuracy overall
  - 85%+ on definition questions
  - 75%+ on methodology questions

### Deployment (Final)

**Goal**: Serve the fine-tuned model

- [ ] **Inference API**
  - FastAPI server with model loading
  - `/query` endpoint for Q&A
  - Response streaming support

- [ ] **Optimization**
  - Quantization for faster inference
  - Request batching
  - Response caching
  - Metrics: Accuracy, Completeness, Citation quality
  - Latency benchmarks

- [ ] **Documentation**
  - API docs (auto-generated with FastAPI)
  - Usage examples
  - Deployment guide

## Phase II: GraphRAG Enhancement (Planned, ~4-6 weeks)

**Goal**: Add graph-based retrieval for multi-hop reasoning and enhanced context

### Knowledge Graph Construction

- [ ] **Entity Extraction**
  - Identify statistical concepts from trained model outputs
  - Extract definitions, formulas, relationships
  - Tag by category (statistical test, metric, methodology)

- [ ] **Relationship Mapping**
  - Map concept relationships: "power analysis REQUIRES sample_size, effect_size"
  - Build prerequisite chains: "understand p-value BEFORE FDR"
  - Connect related methods: "t-test SIMILAR_TO mann_whitney"

- [ ] **SurrealDB Integration**
  - Graph + document + vector database
  - Schema design for concepts and relationships
  - Efficient graph traversal queries

### GraphRAG Pipeline

- [ ] **Query Understanding**
  - Entity recognition in user questions
  - Intent classification (definition, calculation, methodology)

- [ ] **Graph Traversal**
  - BFS/DFS for concept exploration
  - Weighted path finding
  - Subgraph extraction around query entities

- [ ] **Hybrid Retrieval**
  - Combine fine-tuned model with graph context
  - Enhanced multi-hop reasoning
  - Improved citation tracking

### Evaluation & Comparison

- [ ] **Comparative Analysis**
  - Fine-tuned SLM alone (Phase I)
  - Fine-tuned SLM + GraphRAG (Phase II)
  - Measure improvement on complex multi-hop questions

## Future Enhancements

### Phase III: Production Hardening

- Monitoring and observability
- A/B testing framework for model changes
- User feedback collection
- Continuous improvement pipeline

### Phase IV: Advanced Features

- Multi-turn conversations
- Calculation tools integration
- Experiment analysis integration
- Internal knowledge base (company-specific)

## Success Metrics

**Phase I Targets (Fine-Tuned SLM):**

- 80%+ overall accuracy on test questions
- 85%+ accuracy on statistical definition questions
- 75%+ accuracy on methodology questions
- End-to-end latency < 2 seconds
- Cost < $0.001 per query (vs $0.01-0.03 for GPT-4)
- Training cost < $10 total (data collection + fine-tuning)

**Phase II Targets (SLM + GraphRAG):**

- 90%+ accuracy on multi-hop reasoning questions
- 95%+ citation accuracy (graph-backed answers)
- Retrieval latency < 200ms
- Improved context relevance through graph traversal

## Tech Stack

| Component           | Technology                  | Status    | Reasoning                           |
| ------------------- | --------------------------- | --------- | ----------------------------------- |
| **Language**        | Python 3.13+                | ✅ Active | ML ecosystem, rapid development     |
| **Package Mgr**     | uv                          | ✅ Active | 10-100x faster than pip             |
| **Data Collection** | ArXiv, StackExchange API    | ✅ Active | Public Q&A and research papers      |
| **Synthetic Data**  | Claude Sonnet 4.5           | ✅ Active | High-quality Q&A generation         |
| **LLM**             | Llama 3.2 3B Instruct       | ✅ Active | Strong reasoning, 128K context      |
| **Training**        | HuggingFace TRL + QLoRA     | ✅ Active | Efficient 4-bit fine-tuning         |
| **Dataset Mgr**     | HuggingFace Datasets        | ✅ Active | Standard dataset loading            |
| **Quantization**    | BitsAndBytes (4-bit)        | ✅ Active | Memory-efficient training           |
| **API Framework**   | FastAPI                     | ✅ Active | Async, auto-docs, type hints        |
| **Testing**         | pytest                      | ✅ Active | Standard Python testing             |
| **Graph DB**        | SurrealDB                   | Phase II  | Native graph + document store       |
| **Vector Store**    | SurrealDB (native)          | Phase II  | Single DB for graph + embeddings    |
| **Embeddings**      | sentence-transformers       | Phase II  | Local, fast, good quality           |
| **Inference**       | Ollama                      | Phase II  | Easy local deployment               |
| **Orchestration**   | Docker Compose              | Phase II  | Multi-service setup                 |

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- GPU with 6+ GB VRAM (recommended for training)
- API Keys:
  - StackExchange API key (optional, increases rate limit)
  - Anthropic API key (required for synthetic Q&A generation)

### Setup

```bash
# Clone repo
git clone <repo-url>
cd slm-rag-experimentation

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: make install-uv

# Run automated setup
make setup

# This will:
# - Create virtual environment
# - Install all dependencies
# - Set up .env file (add your API keys here)
```

### Workflow

**1. Collect Data**

```bash
# Collect from Cross Validated (safe without API key)
make collect-cv

# Collect ArXiv papers metadata
make collect-arxiv
```

**2. Generate Synthetic Q&A**

```bash
# Add your Anthropic API key to .env first
# ANTHROPIC_API_KEY=sk-ant-your-key-here

# Test with 10 papers (~$0.24)
make generate-qa-test

# Full generation from all papers (~$4-5)
make generate-qa
```

**3. Train Model**

```bash
# Combine datasets and train (4-12 hours on GPU)
make train-full

# Or step by step:
make combine-datasets
make train
```

**4. Deploy (Coming Soon)**

```bash
# Run inference API
make run

# Start infrastructure
docker-compose up -d

# Start API server
python -m uvicorn src.api.main:app --reload

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is statistical power?"}'
```

### Current Focus: Model Training

We're currently building the training data pipeline and SLM fine-tuning:

```bash
# Collect training data from Cross Validated
make collect-cv

# Collect ArXiv papers for synthetic generation
make collect-arxiv

# Test configuration
make test-config
```

## Key Resources

**Learning Materials:**

- [GraphRAG Paper](https://arxiv.org/abs/2404.16130) - Microsoft Research
- [LlamaIndex Knowledge Graphs](https://docs.llamaindex.ai/en/stable/examples/query_engine/knowledge_graph_query_engine/)
- [SurrealDB Graph Docs](https://surrealdb.com/docs/surrealql/statements/relate)

**Data Sources:**

- OpenStax Statistics Textbook (CC-BY)
- Trustworthy Online Controlled Experiments (Kohavi et al.)
- Evan Miller's blog on statistics
- Netflix, Booking.com, Airbnb tech blogs

## Contributing

This is a research/experimentation project. Key areas for contribution:

- New data sources for experimentation knowledge
- Improved entity extraction methods
- Better graph traversal algorithms
- Evaluation test cases
- Documentation improvements
