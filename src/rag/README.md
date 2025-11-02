# RAG Module

Retrieval Augmented Generation pipeline implementation.

## Components

- `retriever.py` - Main retrieval logic
- `query_understanding.py` - Entity recognition and intent classification
- `context_assembly.py` - Context ranking and assembly
- `embeddings.py` - Vector embedding generation and similarity

## Pipeline Flow

1. Query understanding (extract entities, classify intent)
2. Graph traversal (find related concepts)
3. Context assembly (rank by relevance, build context window)
4. Return context to LLM
