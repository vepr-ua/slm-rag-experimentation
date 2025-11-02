# Data Directory

Storage for knowledge base data, embeddings, and processed artifacts.

## Structure

### `raw/`
Raw source documents:
- Textbooks (PDF, HTML)
- Research papers
- Blog posts
- Wikipedia exports

### `processed/`
Processed knowledge:
- Extracted entities (JSON)
- Relationships (JSON)
- Chunked documents

### `embeddings/`
Vector embeddings:
- Concept embeddings
- Document chunk embeddings
- Model artifacts

### `db/`
SurrealDB database files:
- Database storage (when using Docker with local volume)
- Automatically created by Docker Compose
- Contains knowledge graph and document store

## Note

This directory is excluded from git (see .gitignore). Data should be sourced from original locations or stored separately.
