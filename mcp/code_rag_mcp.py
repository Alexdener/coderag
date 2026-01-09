#!/usr/bin/env python3
"""
Code RAG MCP Server
Provides code retrieval capabilities via Model Context Protocol (MCP)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Fallback: implement basic JSON-RPC for MCP protocol
    class TextContent:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text

import weaviate
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.storage import StorageContext
from llama_index.core.schema import QueryBundle


# Configuration
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "CodeIndex")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
DEVICE = os.getenv("DEVICE", "cpu")
CACHE_DIR = os.getenv("CACHE_DIR", "./model_cache")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))


class CodeSearchRequest(BaseModel):
    query: str = Field(..., description="The search query for code")
    top_k: int = Field(default=TOP_K_RESULTS, description="Number of results to return")
    file_types: Optional[List[str]] = Field(default=None, description="Filter by file extensions")


class CodeSnippet(BaseModel):
    content: str
    file_path: str
    chunk_index: int
    score: float
    language: str


class CodeSearchResponse(BaseModel):
    query: str
    results: List[CodeSnippet]
    total_results: int


class CodeRAGMCP:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.index = None
        self.embed_model = None
        
    async def initialize(self):
        """Initialize the MCP server with Weaviate connection and embeddings"""
        self.logger.info("Initializing Code RAG MCP server...")
        
        # Connect to Weaviate
        self.client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT
        )
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
            device=DEVICE,
            cache_folder=CACHE_DIR,
            normalize=True,
            max_length=512
        )
        
        # Initialize vector store and index
        vector_store = WeaviateVectorStore(
            weaviate_client=self.client,
            index_name=WEAVIATE_CLASS_NAME,
            text_key="content"
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model
        )
        
        self.logger.info("Code RAG MCP server initialized successfully")
    
    async def close(self):
        """Close the MCP server and clean up resources"""
        if self.client:
            self.client.close()
        self.logger.info("Code RAG MCP server closed")
    
    async def search_code(self, request: CodeSearchRequest) -> CodeSearchResponse:
        """Search for code based on the query"""
        self.logger.info(f"Searching for: {request.query}")
        
        try:
            # Create retriever and perform search
            retriever = self.index.as_retriever(similarity_top_k=request.top_k)
            nodes = retriever.retrieve(request.query)
            
            # Convert nodes to response format
            results = []
            for node in nodes:
                content = node.node.get_content()
                file_path = node.node.metadata.get('file_path', 'Unknown')
                chunk_index = node.node.metadata.get('chunk_index', -1)
                score = node.score or 0.0
                
                # Determine language from file extension
                file_ext = os.path.splitext(file_path)[1].lower()
                language_map = {
                    '.py': 'python',
                    '.go': 'go',
                    '.js': 'javascript',
                    '.ts': 'typescript',
                    '.jsx': 'javascript',
                    '.tsx': 'typescript',
                    '.java': 'java',
                    '.cpp': 'cpp',
                    '.c': 'c',
                    '.h': 'c',
                    '.hpp': 'cpp',
                    '.cs': 'csharp',
                    '.rb': 'ruby',
                    '.php': 'php',
                    '.html': 'html',
                    '.css': 'css',
                    '.json': 'json',
                    '.yaml': 'yaml',
                    '.yml': 'yaml',
                    '.xml': 'xml',
                    '.sql': 'sql',
                    '.sh': 'bash',
                    '.bash': 'bash',
                    '.zsh': 'bash',
                    '.dockerfile': 'dockerfile',
                    '.mk': 'makefile',
                }
                language = language_map.get(file_ext, 'text')
                
                if file_path.lower().endswith('dockerfile'):
                    language = 'dockerfile'
                elif file_path.lower().endswith('makefile'):
                    language = 'makefile'
                
                # Apply file type filter if specified
                if request.file_types is None or language in request.file_types or file_ext in request.file_types:
                    snippet = CodeSnippet(
                        content=content,
                        file_path=file_path,
                        chunk_index=chunk_index,
                        score=score,
                        language=language
                    )
                    results.append(snippet)
            
            response = CodeSearchResponse(
                query=request.query,
                results=results,
                total_results=len(results)
            )
            
            self.logger.info(f"Found {len(results)} results for query: {request.query}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error searching code: {e}")
            raise
    
    async def get_references(self, symbol: str) -> CodeSearchResponse:
        """Find references to a specific symbol/function"""
        query = f"Find references to {symbol}"
        request = CodeSearchRequest(query=query, top_k=TOP_K_RESULTS)
        return await self.search_code(request)
    
    async def get_definitions(self, symbol: str) -> CodeSearchResponse:
        """Find definitions of a specific symbol/function"""
        query = f"Find definition of {symbol}"
        request = CodeSearchRequest(query=query, top_k=TOP_K_RESULTS)
        return await self.search_code(request)


# Global instance
code_rag = CodeRAGMCP()


async def handle_search_code(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle code search tool call"""
    try:
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", TOP_K_RESULTS)
        file_types = arguments.get("file_types")
        
        request = CodeSearchRequest(query=query, top_k=top_k, file_types=file_types)
        response = await code_rag.search_code(request)
        
        # Format results as text content
        result_text = f"Found {response.total_results} results for query: {response.query}\n\n"
        for i, snippet in enumerate(response.results, 1):
            result_text += f"--- Result {i} ---\n"
            result_text += f"File: {snippet.file_path}\n"
            result_text += f"Language: {snippet.language}\n"
            result_text += f"Score: {snippet.score:.4f}\n"
            result_text += f"Content:\n```{snippet.language}\n{snippet.content}\n```\n\n"
        
        return [TextContent(type="text", text=result_text)]
    except Exception as e:
        code_rag.logger.error(f"Error in search_code: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main function to run the MCP server"""
    # Configure logging to stderr (MCP protocol requirement)
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s:%(name)s:%(message)s',
        stream=sys.stderr
    )
    
    # Suppress third-party library INFO logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.warning("Initializing Code RAG MCP server...")
    
    # Initialize the code RAG system
    await code_rag.initialize()
    logger.warning("Code RAG MCP server initialized successfully")
    
    if MCP_AVAILABLE:
        # Use official MCP SDK
        server = Server("codebase-rag-mcp")
        
        @server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="search_code",
                    description="Search for code snippets in the codebase using semantic search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for code"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": TOP_K_RESULTS
                            },
                            "file_types": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by file extensions (e.g., ['python', 'go'])"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]
        
        @server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "search_code":
                return await handle_search_code(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
        
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
    else:
        # Fallback: Basic JSON-RPC implementation for MCP protocol
        logger.warning("MCP SDK not available, using basic JSON-RPC implementation")
        
        async def process_request(line: str) -> Optional[Dict[str, Any]]:
            """Process a JSON-RPC request"""
            try:
                request = json.loads(line.strip())
                method = request.get("method")
                params = request.get("params", {})
                request_id = request.get("id")
                
                if method == "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "search_code",
                                    "description": "Search for code snippets in the codebase",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string"},
                                            "top_k": {"type": "integer", "default": TOP_K_RESULTS},
                                            "file_types": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["query"]
                                    }
                                }
                            ]
                        }
                    }
                    return response
                
                elif method == "tools/call":
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    
                    if tool_name == "search_code":
                        contents = await handle_search_code(arguments)
                        response = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "result": {
                                "content": [{"type": "text", "text": c.text} for c in contents]
                            }
                        }
                        return response
                
                elif method == "initialize":
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {}
                            },
                            "serverInfo": {
                                "name": "codebase-rag-mcp",
                                "version": "0.1.0"
                            }
                        }
                    }
                    return response
                
                return None
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if 'request_id' in locals() else None,
                    "error": {"code": -32603, "message": str(e)}
                }
        
        # Read from stdin and write to stdout (async)
        async def read_stdin():
            loop = asyncio.get_event_loop()
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            await loop.connect_read_pipe(lambda: protocol, sys.stdin)
            
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8').strip()
                    if not line_str:
                        continue
                    response = await process_request(line_str)
                    if response:
                        print(json.dumps(response), flush=True)
            except Exception as e:
                logger.error(f"Error reading stdin: {e}")
            finally:
                await code_rag.close()
        
        try:
            await read_stdin()
        except KeyboardInterrupt:
            logger.warning("Shutting down MCP server...")
        finally:
            await code_rag.close()


if __name__ == "__main__":
    asyncio.run(main())