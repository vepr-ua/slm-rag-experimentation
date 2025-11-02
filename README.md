# Experimentation Intelligence: SLM + GraphRAG

A specialized Small Language Model (SLM) powered by Graph-based Retrieval Augmented Generation (GraphRAG) for answering questions about experimentation methodology, statistical analysis, and A/B testing best practices.

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

## Phase I: Core Experimentation SLM (4-6 weeks)

### Week 1-2: Knowledge Base Construction

**Goal**: Build a comprehensive graph of experimentation concepts

- [ ] **Data Collection**
  - Scrape public stats textbooks (OpenStax, etc.)
  - Fetch experimentation papers from ArXiv
  - Collect industry blog posts (Booking.com, Netflix, Airbnb experimentation blogs)
  - Curate Wikipedia articles on statistical concepts

- [ ] **Entity Extraction**
  - Identify core concepts: "p-value", "confidence interval", "A/B test", etc.
  - Extract definitions, formulas, examples
  - Tag by category (statistical test, metric, methodology, etc.)

- [ ] **Relationship Mapping**
  - Map concept relationships: "power analysis REQUIRES sample_size, effect_size, significance_level"
  - Build prerequisite chains: "understand p-value BEFORE understanding FDR"
  - Connect related methods: "t-test SIMILAR_TO mann_whitney"

- [ ] **SurrealDB Schema Design**

```sql
  DEFINE TABLE concept SCHEMAFULL;
  DEFINE FIELD name ON concept TYPE string;
  DEFINE FIELD definition ON concept TYPE string;
  DEFINE FIELD category ON concept TYPE string;
  DEFINE FIELD embedding ON concept TYPE array;

  DEFINE TABLE document SCHEMAFULL;
  DEFINE FIELD content ON document TYPE string;
  DEFINE FIELD source ON document TYPE string;
  DEFINE FIELD chunk_index ON document TYPE int;

  DEFINE TABLE relates_to TYPE RELATION;
  DEFINE FIELD relationship_type ON relates_to TYPE string;
  DEFINE FIELD strength ON relates_to TYPE float;
```

### Week 3-4: GraphRAG Pipeline

**Goal**: Implement intelligent retrieval using graph structure

- [ ] **Query Understanding**
  - Entity recognition in user questions
  - Intent classification (definition, calculation, methodology)
  - Ambiguity detection

- [ ] **Graph Traversal**
  - BFS/DFS for concept exploration
  - Weighted path finding (stronger relationships = higher priority)
  - Subgraph extraction around query entities

- [ ] **Context Assembly**
  - Combine graph nodes into coherent context
  - Rank by relevance (embedding similarity + graph centrality)
  - Limit context to model window (2K-4K tokens)

- [ ] **Retrieval Evaluation**
  - Create test set of 50-100 questions
  - Measure retrieval precision/recall
  - Compare to baseline (vector-only RAG)

### Week 5: SLM Integration

**Goal**: Connect retrieval to language model

- [ ] **Model Selection**
  - Benchmark Phi-3-mini (3.8B), Llama 3.2 (3B), Qwen2.5 (3B)
  - Test on statistics Q&A without fine-tuning
  - Select best base model

- [ ] **Prompt Engineering**
  - Design prompts for different question types
  - Include graph context effectively
  - Handle uncertainty (when to say "I don't know")

- [ ] **Inference Pipeline**
  - Set up Ollama or vLLM locally
  - Implement response streaming
  - Add caching for common queries

### Week 6: API & Evaluation

**Goal**: Production-ready API with quality metrics

- [ ] **FastAPI Server**
  - `/query` endpoint with streaming support
  - `/health` and `/metrics` endpoints
  - Request validation and error handling

- [ ] **Evaluation Framework**
  - 100 test questions covering:
    - Definitions (20%)
    - Calculations (30%)
    - Methodology (30%)
    - Best practices (20%)
  - Metrics: Accuracy, Completeness, Citation quality
  - Latency benchmarks

- [ ] **Documentation**
  - API docs (auto-generated with FastAPI)
  - Usage examples
  - Deployment guide

## Future Phases

### Phase II: Internal Knowledge Integration (Weeks 7-10)

- Ingest internal experiment documentation
- Add company-specific best practices
- Privacy-preserving graph (redact sensitive metrics)
- Access control for proprietary knowledge

### Phase III: Experiment Analysis Integration (Weeks 11-14)

- Connect to experiment data warehouse
- Real-time experiment status in RAG
- Historical experiment search
- Automated insights from past experiments

### Phase IV: Fine-Tuning & Optimization (Weeks 15-18)

- Fine-tune SLM on experimentation Q&A pairs
- LoRA adapters for company-specific knowledge
- Response quality improvements
- Performance optimization

## Success Metrics

**Phase I Targets:**

- ✅ 85%+ accuracy on statistical definition questions
- ✅ 75%+ accuracy on methodology questions
- ✅ Retrieval latency < 200ms
- ✅ End-to-end latency < 2 seconds
- ✅ 90%+ citation accuracy (retrieved context is actually relevant)
- ✅ Cost < $0.001 per query (vs $0.01-0.03 for GPT-4)

## Tech Stack

| Component         | Technology                  | Reasoning                           |
| ----------------- | --------------------------- | ----------------------------------- |
| **Language**      | Python 3.11+                | ML ecosystem, rapid development     |
| **Graph DB**      | SurrealDB                   | Native graph + document, easy setup |
| **Vector Store**  | SurrealDB (native)          | Single DB for graph + embeddings    |
| **Embeddings**    | sentence-transformers       | Local, fast, good quality           |
| **LLM**           | Phi-3 / Llama 3.2 / Qwen2.5 | 3-4B params, good stats reasoning   |
| **Inference**     | Ollama                      | Easy local deployment               |
| **API**           | FastAPI                     | Async, auto-docs, type hints        |
| **Orchestration** | Docker Compose              | Simple multi-service setup          |
| **Testing**       | pytest                      | Standard Python testing             |

## End Goal

```bash
# Clone repo
git clone <repo-url>
cd experimentation-slm

# Install dependencies
pip install -e .

# Start infrastructure
docker-compose up -d

# Start API server
python -m src.api.main

# Test query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is statistical power?"}'
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
