# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeRAG is a semantic code search system that enables natural language queries over multi-language codebases. The system uses Weaviate vector database, Hugging Face embeddings, and LlamaIndex to provide semantic search capabilities for developers.

## Architecture

The system consists of three main components:
1. **Indexing System** (`build_index.py`, `multi_repo_index.py`): Processes multi-language code files and creates vector embeddings
2. **Search Interface** (`app.py`): Streamlit web application for querying indexed code
3. **MCP Server** (`mcp/code_rag_mcp.py`): Model Context Protocol server for IDE integration (e.g., Cursor)

## Multi-Language Support

The system supports semantic search across multiple programming languages:
- Python (.py)
- Go (.go)
- Shell scripts (.sh, .bash, .zsh)
- Dockerfiles (Dockerfile*)
- Makefiles (Makefile*)
- And other text-based code files

Code files are intelligently split based on language-specific structures:
- Python: Functions and classes
- Go: Functions
- Shell: Functions
- Dockerfile: Instruction blocks
- Makefile: Targets

## Key Components

### Weaviate Vector Database
- Stores code chunks with metadata (file path, chunk index, repository name)
- Uses hybrid search (keyword + vector) for improved results
- Configured via docker-compose.yaml

### Custom Retrieval System
- `customer_retriever.py` implements WeaviateHybridRetriever
- Supports context-aware and repository-filtered searches
- Implements boosting based on file path similarity

### MCP Integration
- Provides Model Context Protocol interface for IDEs
- Allows direct code retrieval within development environments
- Enables tight integration with tools like Cursor

## Development Commands

### Environment Setup
```bash
# Install dependencies with uv
uv sync

# Or install with pip from pyproject.toml
pip install -e .
```

### Start Infrastructure
```bash
# Start Weaviate vector database
docker compose up -d

# Verify Weaviate is running
curl http://localhost:8080/v1/.well-known/ready
```

### Index Code
```bash
# Index a single repository
CODE_DIR=/path/to/your/code/repository uv run python src/build_index.py

# Index multiple repositories into the same collection
uv run python src/multi_repo_index.py /path/to/repo1 /path/to/repo2 /path/to/repo3

# Index with custom configuration
export CODE_DIR=/path/to/your/code
export WEAVIATE_HOST=localhost
export WEAVIATE_PORT=8080
export WEAVIATE_CLASS_NAME=CodeIndex
export EMBED_MODEL_NAME=BAAI/bge-large-en-v1.5
export DEVICE=cpu  # or cuda
uv run python src/build_index.py
```

### Run Applications
```bash
# Run Streamlit web interface
uv run streamlit run src/app.py

# Run MCP server for IDE integration
HF_ENDPOINT=https://hf-mirror.com uv run python -m mcp.code_rag_mcp

# Run tests
uv run python src/test_build_index.py
```

### Environment Variables
Key configuration options:
- `WEAVIATE_HOST`: Weaviate server host (default: localhost)
- `WEAVIATE_PORT`: Weaviate server port (default: 8080)
- `WEAVIATE_CLASS_NAME`: Weaviate collection name (default: CodeIndex)
- `EMBED_MODEL_NAME`: Embedding model to use (default: BAAI/bge-large-en-v1.5)
- `DEVICE`: Device for embeddings (default: cuda if available, otherwise cpu)
- `CACHE_DIR`: Directory for model caching (default: ./model_cache)
- `TOP_K_INITIAL`: Initial retrieval count for hybrid search (default: 20)
- `TOP_N_RERANK`: Final results count after reranking (default: 5)
- `LLM_MODEL_NAME`: LLM model for response generation (default: qwen3:8b)
- `OLLAMA_BASE_URL`: Ollama server URL (default: http://localhost:11434)

### Development Workflow

1. **Start Weaviate**: `docker compose up -d`
2. **Index your code**: `CODE_DIR=/path/to/code uv run python src/build_index.py`
3. **Run the web interface**: `uv run streamlit run src/app.py`
4. **Or run MCP server for IDE integration**: `uv run python -m mcp.code_rag_mcp`

### Testing and Validation

```bash
# Test the indexing functionality
uv run python src/test_build_index.py

# Validate the index exists
uv run python src/verify_index.py  # if available
```

### MCP Server for IDE Integration

The MCP server enables integration with Cursor and other MCP-compatible IDEs:

```bash
# Run the MCP server
HF_ENDPOINT=https://hf-mirror.com uv run python -m mcp.code_rag_mcp

# Environment variables for MCP
export WEAVIATE_HOST=localhost
export WEAVIATE_PORT=8080
export WEAVIATE_CLASS_NAME=CodeIndex
export TOP_K_RESULTS=10
```

## Troubleshooting

- **Weaviate Connection Issues**: Verify `docker compose up -d` is running and accessible at http://localhost:8080
- **Empty Search Results**: Confirm your codebase has been indexed using `build_index.py`
- **Model Download Issues**: If experiencing issues with Hugging Face model downloads, try setting `HF_ENDPOINT=https://hf-mirror.com`
- **Memory Issues**: Large codebases may require adjusting the embedding model or chunk sizes
- **Ollama Connection**: Ensure Ollama is running at `OLLAMA_BASE_URL` for the web interface to generate responses