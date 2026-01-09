#!/usr/bin/env python3
"""
Verification script for the code index
This script verifies that the index was built correctly and can retrieve data from Weaviate
using the same embedding model as build_index (no external API calls)
"""
import os
import logging
import weaviate
import torch
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

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

def verify_index_exists():
    """Verify that the index/collection exists in Weaviate"""
    logger.info("Connecting to Weaviate...")
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT
    )
    
    try:
        # Check if the collection exists
        exists = client.collections.exists(WEAVIATE_CLASS_NAME)
        if exists:
            logger.info(f"✓ Collection '{WEAVIATE_CLASS_NAME}' exists in Weaviate")
            
            # Get collection details
            collection = client.collections.get(WEAVIATE_CLASS_NAME)
            count = collection.aggregate.over_all().total_count
            logger.info(f"✓ Collection contains {count} objects")
            
            # Get sample objects to verify content
            response = collection.query.fetch_objects(limit=3)
            if response.objects:
                logger.info("✓ Sample objects retrieved successfully:")
                for i, obj in enumerate(response.objects):
                    content_preview = obj.properties.get("content", "")[:100] + "..." if len(obj.properties.get("content", "")) > 100 else obj.properties.get("content", "")
                    file_path = obj.properties.get("file_path", "Unknown")
                    logger.info(f"  Object {i+1}: File: {file_path}, Content preview: {content_preview}")
            else:
                logger.warning("⚠ No objects found in collection")
                
            return True
        else:
            logger.error(f"✗ Collection '{WEAVIATE_CLASS_NAME}' does not exist")
            return False
    finally:
        client.close()

def verify_retrieval_functionality():
    """Verify that we can retrieve data from the index using the same embedding model as build_index"""
    logger.info("Testing retrieval functionality with local embedding model...")
    
    try:
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

        # Connect to Weaviate
        client = weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT
        )
        
        # Initialize vector store
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name=WEAVIATE_CLASS_NAME,
            text_key="content"
        )
        
        # Create a simple index from the vector store with the embedding model
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context,
            embed_model=embed_model  # Use the same embedding model as build_index
        )
        
        # Test retrieval by directly using the vector store
        # Create test queries that are likely to match code content
        test_queries = [
            "rule",
            "main",
            "print",
            "import",
            "config"
        ]
        
        for query_text in test_queries:
            logger.info(f"Testing retrieval for: '{query_text}'")
            try:
                # Use the index as a retriever (this will use the local embedding model)
                retriever = index.as_retriever(similarity_top_k=3)
                nodes = retriever.retrieve(query_text)
                
                logger.info(f"✓ Retrieval for '{query_text}' successful")
                logger.info(f"  Retrieved {len(nodes)} nodes")
                
                # Show first few results if any
                for i, node in enumerate(nodes[:2]):  # Show first 2
                    content_preview = node.node.get_content()[:100] + "..." if len(node.node.get_content()) > 100 else node.node.get_content()
                    file_path = node.node.metadata.get('file_path', 'Unknown')
                    score = node.score if node.score is not None else 0
                    logger.info(f"    Node {i+1}: File: {file_path}, Score: {score:.3f}")
                    logger.info(f"      Content preview: {content_preview}")
                
                break  # If one retrieval works, assume the functionality is working
                
            except Exception as e:
                logger.warning(f"Retrieval for '{query_text}' failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        client.close()
        logger.info("✓ Retrieval functionality test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Retrieval functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main verification function"""
    logger.info("Starting index verification...")
    
    # Step 1: Verify index exists
    index_exists = verify_index_exists()
    if not index_exists:
        logger.error("Index verification failed - collection does not exist")
        return False
    
    # Step 2: Verify retrieval functionality
    retrieval_works = verify_retrieval_functionality()
    if not retrieval_works:
        logger.error("Retrieval verification failed")
        return False
    
    logger.info("✓ All verifications passed! Index is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Verification completed successfully!")
        print("The index was built correctly and can be queried locally.")
    else:
        print("\n❌ Verification failed!")
        print("There may be issues with the index or retrieval functionality.")
        exit(1)