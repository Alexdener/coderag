# Code RAG MCP Module

This module provides code retrieval capabilities via Model Context Protocol (MCP), allowing IDEs like Cursor to access your indexed codebase directly.

## Overview

The Code RAG MCP module separates the retrieval of relevant code snippets from LLM processing, addressing local compute limitations while providing seamless IDE integration.

## Features

- **MCP Protocol Support**: Compatible with MCP-enabled IDEs
- **Multi-language Support**: Go, Python, Dockerfile, Makefile, Shell scripts
- **Fast Retrieval**: Direct access to indexed code without LLM processing overhead
- **Flexible Output**: Multiple formats for different use cases
- **IDE Integration**: Designed for tools like Cursor, VS Code

## Architecture

```
[IDE/Cursor] → [MCP Server] → [Code RAG System] → [Weaviate Vector DB]
     ↓              ↓              ↓
[Context] ← [Retrieved Code] ← [Search Results]
```

## Installation

The MCP module shares dependencies with the main codebase. Ensure you have:

- Weaviate running locally (or configured endpoint)
- Your codebase indexed (using `build_index.py`)
- Required Python packages (from main requirements)

## Configuration

The module uses the same environment variables as the main application:

```bash
export WEAVIATE_HOST=localhost
export WEAVIATE_PORT=8080
export WEAVIATE_CLASS_NAME=CodeIndex
export EMBED_MODEL_NAME=BAAI/bge-large-en-v1.5
export DEVICE=cpu  # or cuda
export TOP_K_RESULTS=10
```

## Usage

### Running the MCP Server

```bash
cd /path/to/codebase
HF_ENDPOINT=https://hf-mirror.com uv run python -m mcp.code_rag_mcp
```

### API Endpoints

The MCP module provides several retrieval methods:

- `search_code()`: General code search with natural language queries
- `get_references()`: Find references to specific symbols/functions
- `get_definitions()`: Find definitions of specific symbols/functions

### Example Usage

```python
from mcp.code_rag_mcp import CodeRAGMCP, CodeSearchRequest

# Initialize the server
mcp = CodeRAGMCP()
await mcp.initialize()

# Search for code
request = CodeSearchRequest(query="Find database connection functions", top_k=5)
response = await mcp.search_code(request)

# Access results
for snippet in response.results:
    print(f"File: {snippet.file_path}")
    print(f"Language: {snippet.language}")
    print(f"Content: {snippet.content[:200]}...")
    print(f"Score: {snippet.score}")
```

## Integration with IDEs

### Cursor
1. Configure Cursor to use the MCP server endpoint
2. The server will provide code context directly to Cursor's AI assistant
3. No need for local LLM processing - just retrieval

### VS Code
1. Install MCP-compatible extension
2. Configure to point to your MCP server
3. Access code context within the editor

## Benefits

1. **Performance**: Offloads compute-intensive retrieval to dedicated server
2. **IDE Integration**: Seamless context access within development environment  
3. **Scalability**: Can handle multiple concurrent requests
4. **Flexibility**: Can be used with any LLM (local or cloud)
5. **Caching**: Results can be cached for faster subsequent queries

## Non-Disruptive Design

- MCP module runs independently from Streamlit app
- Uses same Weaviate database and indexing logic
- No changes required to existing `build_index.py`
- Can coexist with current Streamlit interface

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| WEAVIATE_HOST | localhost | Weaviate server host |
| WEAVIATE_PORT | 8080 | Weaviate server port |
| WEAVIATE_CLASS_NAME | CodeIndex | Weaviate collection name |
| EMBED_MODEL_NAME | BAAI/bge-large-en-v1.5 | Embedding model to use |
| DEVICE | cpu | Device for embeddings (cpu/cuda) |
| TOP_K_RESULTS | 10 | Default number of results to return |
| CACHE_DIR | ./model_cache | Directory for model caching |

## Troubleshooting

### Common Issues

1. **Connection to Weaviate fails**: Ensure Weaviate is running and accessible
2. **No results returned**: Verify that your codebase has been indexed using `build_index.py`
3. **Performance issues**: Check that the embedding model is properly cached

### Logging

The module uses standard Python logging. Set `LOG_LEVEL=DEBUG` for detailed logs.