# LLM Module

Small Language Model integration and inference.

## Components

- `client.py` - Ollama/vLLM client
- `prompts.py` - Prompt templates for different question types
- `inference.py` - Inference pipeline with streaming
- `cache.py` - Response caching for common queries

## Supported Models

- Phi-3-mini (3.8B)
- Llama 3.2 (3B)
- Qwen2.5 (3B)

## Responsibilities

- Model loading and initialization
- Prompt formatting with retrieved context
- Response generation and streaming
- Query caching
