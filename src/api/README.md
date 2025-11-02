# API Module

FastAPI server implementation for the experimentation SLM.

## Components

- `main.py` - FastAPI application entry point
- `routes/` - API route handlers
- `models/` - Pydantic request/response models
- `middleware/` - Custom middleware (logging, auth, etc.)

## Endpoints (planned)

- `POST /query` - Main query endpoint with streaming support
- `GET /health` - Health check
- `GET /metrics` - Performance metrics
