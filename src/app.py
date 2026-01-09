#!/usr/bin/env python3
"""
Streamlit App for Multi-Language Code Semantic Search
Provides a web interface for querying indexed code using hybrid search with local LLM.
"""
import os
import logging
import streamlit as st
import weaviate
import torch
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import PromptTemplate
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from customer_retriever import WeaviateHybridRetriever

# --- Configuration ---
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "CodeIndex")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-large")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = os.getenv("CACHE_DIR", "./model_cache")
TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", "20"))
TOP_N_RERANK = int(os.getenv("TOP_N_RERANK", "5"))
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.5"))

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource  # Streamlit cache to avoid reloading
def load_components():
    """Load and configure all necessary components for the application."""
    logger.info("Loading components via st.cache_resource...")

    # 1. Configure LlamaIndex Settings (Embedding, LLM)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        device=DEVICE,
        cache_folder=CACHE_DIR,
        normalize=True,
        max_length=512
    )

    Settings.llm = Ollama(
        model=LLM_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        request_timeout=180.0
    )

    # 2. Initialize Weaviate client and VectorStore
    client = weaviate.connect_to_local(
        host=WEAVIATE_HOST,
        port=WEAVIATE_PORT
    )

    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=WEAVIATE_CLASS_NAME,
        text_key="content"
    )

    # 3. Initialize custom WeaviateHybridRetriever
    hybrid_retriever = WeaviateHybridRetriever(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
        similarity_top_k=TOP_K_INITIAL,
        alpha=0.7,  # Hybrid weight favoring semantic search
        search_properties=["content"],
        # Add context-aware features (these could be dynamic based on user's current file)
        # current_file_path=None,  # Would be set dynamically if known
        # repo_filter=None,      # Would be set if searching within specific repo
        # file_extension_filter=None  # Would be set for language-specific search
    )

    # 4. Define Prompt Template and Response Synthesizer
    custom_prompt_tmpl_str = (
        "You are a helpful assistant that answers questions about code. "
        "Use the following context to answer the question. "
        "If the context doesn't contain enough information, say so.\n\n"
        "Context:\n{context_str}\n\n"
        "Question: {query_str}\n\n"
        "Answer: "
    )
    custom_prompt_tmpl = PromptTemplate(custom_prompt_tmpl_str)

    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=custom_prompt_tmpl,
        # Enable verbose to see the actual prompts
        verbose=True
    )

    # 5. Create RetrieverQueryEngine with custom retriever
    query_engine = RetrieverQueryEngine(
        retriever=hybrid_retriever,
        response_synthesizer=response_synthesizer
        # Note: We're not using reranker here to avoid import issues
    )

    logger.info("Query engine created with custom retriever and prompt.")
    return query_engine, client


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Multi-Language Code Semantic Search",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Multi-Language Code Semantic Search")
    st.markdown("Search through indexed code (Go, Python, Dockerfile, Makefile, Shell) using natural language queries.")
    
    # Load components
    try:
        query_engine, client = load_components()
        st.success("✅ Successfully connected to the code index!")
    except Exception as e:
        st.error(f"❌ Error loading components: {e}")
        st.stop()
    
    # User query input
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input(
            "Enter your query about the code:",
            placeholder="e.g., 'Find functions that handle file operations', 'Show me the main entry points', 'How is configuration loaded?'",
            key="query_input"
        )
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        search_button = st.button("🔍 Search", type="primary")
    
    if search_button and user_query:
        st.subheader("Query Results")

        # First, get the retrieved nodes to show the context before sending to LLM
        try:
            query_engine, _ = load_components()

            # Get the retriever from the query engine to retrieve nodes first
            retriever = query_engine.retriever

            # Retrieve the nodes first
            from llama_index.core.schema import QueryBundle
            query_bundle = QueryBundle(query_str=user_query)
            retrieved_nodes = retriever.retrieve(query_bundle)

            # Show the context that will be sent to the LLM
            with st.expander("🔍 Debug: Show LLM Prompt"):
                if retrieved_nodes:
                    context_str = "\n\n".join([node.get_content() for node in retrieved_nodes])
                    full_prompt = f"You are a helpful assistant that answers questions about code. Use the following context to answer the question. If the context doesn't contain enough information, say so.\n\nContext:\n{context_str}\n\nQuestion: {user_query}\n\nAnswer: "
                    st.text_area("Full prompt sent to LLM:", value=full_prompt, height=300)
                    st.info(f"Context contains {len(retrieved_nodes)} code snippets")
                else:
                    st.info("No context was retrieved for this query.")

        except Exception as e:
            st.error(f"❌ Error retrieving context: {e}")
            logger.error(f"Context retrieval error: {e}")

        with st.spinner("🔍 Searching and generating response..."):
            try:
                # Now execute the full query with the response synthesizer
                response = query_engine.query(user_query)

                # Display the response
                st.write("### 🧠 AI Answer")
                st.write(str(response))

                # Display source code snippets
                st.write("### 📄 Relevant Code Snippets")

                # Get source nodes from response
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for i, node in enumerate(response.source_nodes):
                        with st.expander(f"Code snippet {i+1} (Score: {node.score:.3f})", expanded=True):
                            # Determine language for syntax highlighting
                            file_path = node.node.metadata.get('file_path', 'Unknown')
                            file_ext = os.path.splitext(file_path)[1].lower()

                            # Map file extensions to language names for syntax highlighting
                            lang_map = {
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
                                'makefile': 'makefile',
                            }

                            language = lang_map.get(file_ext, 'text')
                            if file_path.lower().endswith('dockerfile'):
                                language = 'dockerfile'
                            elif file_path.lower().endswith('makefile'):
                                language = 'makefile'

                            st.code(node.node.get_content(), language=language)
                            st.text(f"📁 File: {file_path}")
                            st.text(f"🔢 Chunk: {node.node.metadata.get('chunk_index', 'Unknown')}")
                else:
                    st.info("ℹ️ No source nodes available for this response.")

            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                logger.error(f"Query processing error: {e}")

                # Even if there's an error, the context was already shown above
                st.info("ℹ️ The context that was retrieved is shown in the 'Debug: Show LLM Prompt' section above.")
    
    # Add some information about the system
    with st.sidebar:
        st.header("⚙️ System Information")
        st.write(f"**Embedding Model:** {EMBED_MODEL_NAME}")
        st.write(f"**LLM Model:** {LLM_MODEL_NAME}")
        st.write(f"**Rerank Model:** {RERANK_MODEL_NAME}")
        st.write(f"**Weaviate Collection:** {WEAVIATE_CLASS_NAME}")
        st.write(f"**Device:** {DEVICE}")
        
        st.header("📚 Supported Languages")
        st.write("• Python (.py)")
        st.write("• Go (.go)")
        st.write("• Shell scripts (.sh, .bash, .zsh)")
        st.write("• Dockerfiles (Dockerfile*)")
        st.write("• Makefiles (Makefile*)")
        
        if st.button("🔄 Clear Cache"):
            st.cache_resource.clear()
            st.success("✅ Cache cleared!")


if __name__ == "__main__":
    main()