# Knowledge Module

Knowledge base construction and entity extraction.

## Components

- `extractors/` - Entity and relationship extractors
- `loaders/` - Data loaders for different sources (textbooks, papers, blogs)
- `processors/` - Content processors and chunkers
- `graph_builder.py` - Builds knowledge graph from extracted entities

## Data Sources

- OpenStax Statistics Textbook
- ArXiv experimentation papers
- Industry blogs (Netflix, Booking.com, Airbnb)
- Wikipedia statistical concepts

## Workflow

1. Load raw documents
2. Extract entities (concepts, definitions, formulas)
3. Identify relationships between concepts
4. Build graph structure in SurrealDB
