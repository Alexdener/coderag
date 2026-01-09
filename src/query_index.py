#!/usr/bin/env python3
"""
Simple query script to test the code index
This script allows you to perform queries against the indexed code
"""
import os
import logging
import weaviate
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import torch

# --- Configuration ---
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "CodeIndex")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = os.getenv("CACHE_DIR", "./model_cache")

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def query_index(query_text, top_k=5):
    """Query the index with the given text"""
    logger.info(f"Connecting to Weaviate...")
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT
    )
    
    try:
        # Initialize vector store
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=WEAVIATE_CLASS_NAME,
            text_key="content"
        )
        
        # Configure the same embedding model as used in build_index
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
            device=DEVICE,
            cache_folder=CACHE_DIR,
            normalize=True,
            max_length=512
        )
        
        # Set the global embed model to avoid external API calls
        Settings.embed_model = embed_model

        # Explicitly set LLM to None to avoid external API calls
        Settings.llm = None

        # Create index from vector store
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=embed_model  # Use the same embedding model as build_index)
        )

        # Create query engine and perform query
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query_text)

        return response
        
    finally:
        client.close()

def main():
    """Main function to run queries"""
    print("Code Index Query Tool")
    print("=====================")
    print(f"Connected to Weaviate: {WEAVIATE_HOST}:{WEAVIATE_PORT}")
    print(f"Collection: {WEAVIATE_CLASS_NAME}")
    print()
    
    # Predefined test queries
    test_queries = [
        "Find functions related to configuration",
        "Show me main functions",
        "Find code related to file operations",
        "Show me import statements",
        "Find functions that handle errors"
    ]
    
    print("Running predefined test queries...")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test Query {i}: {query}")
        print("-" * 50)
        
        try:
            response = query_index(query, top_k=3)
            print(f"Response: {response}")
            
            # Show source nodes if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\nSource code snippets ({len(response.source_nodes)} found):")
                for j, node in enumerate(response.source_nodes, 1):
                    content_preview = node.node.get_content()[:200] + "..." if len(node.node.get_content()) > 200 else node.node.get_content()
                    file_path = node.node.metadata.get('file_path', 'Unknown')
                    chunk_index = node.node.metadata.get('chunk_index', 'Unknown')
                    score = node.score if node.score else 0
                    print(f"  [{j}] File: {file_path} (Chunk: {chunk_index}, Score: {score:.3f})")
                    print(f"      Content: {content_preview}")
                    print()
            else:
                print("No source nodes found.")
                
        except Exception as e:
            print(f"Error querying: {e}")
        
        print("=" * 80)
        print()
    
    # Interactive mode
    print("Interactive Query Mode")
    print("Enter your own queries (type 'quit' to exit):")
    print()
    
    while True:
        user_query = input("Enter query: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_query:
            print(f"Query: {user_query}")
            print("-" * 50)
            
            try:
                response = query_index(user_query, top_k=5)
                print(f"Response: {response}")
                
                # Show source nodes if available
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print(f"\nSource code snippets ({len(response.source_nodes)} found):")
                    for j, node in enumerate(response.source_nodes, 1):
                        content_preview = node.node.get_content()[:200] + "..." if len(node.node.get_content()) > 200 else node.node.get_content()
                        file_path = node.node.metadata.get('file_path', 'Unknown')
                        chunk_index = node.node.metadata.get('chunk_index', 'Unknown')
                        score = node.score if node.score else 0
                        print(f"  [{j}] File: {file_path} (Chunk: {chunk_index}, Score: {score:.3f})")
                        print(f"      Content: {content_preview}")
                        print()
                else:
                    print("No source nodes found.")
                    
            except Exception as e:
                print(f"Error querying: {e}")
        
        print("=" * 80)
        print()

if __name__ == "__main__":
    main()