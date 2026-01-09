"""
Custom Weaviate Hybrid Retriever
Implements hybrid search (keyword + vector) for code retrieval using Weaviate v4 client.
Supports context-aware and repository-aware search.
"""
import logging
import os
from typing import List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from llama_index.core.base.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)

class WeaviateHybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_store: WeaviateVectorStore,
        embed_model: BaseEmbedding,
        similarity_top_k: int = 10,
        alpha: float = 0.5,  # 0 = pure keyword, 1 = pure vector
        search_properties: Optional[List[str]] = None,
        current_file_path: Optional[str] = None,  # For context-aware search
        repo_filter: Optional[str] = None,  # For repository-specific search
        file_extension_filter: Optional[List[str]] = None  # For language-specific search
    ):
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._similarity_top_k = similarity_top_k
        self._alpha = alpha
        self._search_properties = search_properties or ["content"]
        self._current_file_path = current_file_path
        self._repo_filter = repo_filter
        self._file_extension_filter = file_extension_filter
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not query_bundle or not query_bundle.query_str:
            return []

        logger.info(f"Performing hybrid search (alpha={self._alpha}) for query: '{query_bundle.query_str}'")
        if self._current_file_path:
            logger.info(f"Context-aware search for file: {self._current_file_path}")
        if self._repo_filter:
            logger.info(f"Repository filter: {self._repo_filter}")
        if self._file_extension_filter:
            logger.info(f"File extension filter: {self._file_extension_filter}")

        # Get query embedding
        try:
            query_embedding = self._embed_model.get_query_embedding(query_bundle.query_str)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

        # Execute Weaviate hybrid query using v4 client API
        try:
            client_v4 = self._vector_store.client
            collection = client_v4.collections.get(self._vector_store.index_name)
            
            # Build filters based on context
            filters = []
            
            # Repository filter
            if self._repo_filter:
                filters.append(collection.filter.by_property("repo_name").equal(self._repo_filter))
            
            # File extension filter
            if self._file_extension_filter:
                ext_conditions = []
                for ext in self._file_extension_filter:
                    ext_conditions.append(collection.filter.by_property("file_path").like(f"*{ext}"))
                
                if ext_conditions:
                    if len(ext_conditions) == 1:
                        filters.append(ext_conditions[0])
                    else:
                        filters.append(ext_conditions[0].or_(ext_conditions[1:]))
            
            # Combine filters
            combined_filter = None
            for f in filters:
                if combined_filter is None:
                    combined_filter = f
                else:
                    combined_filter = combined_filter.and_(f)
            
            # Perform hybrid search with filters
            response = collection.query.hybrid(
                query=query_bundle.query_str,
                vector=query_embedding,
                limit=self._similarity_top_k * 2,  # Get more results for post-processing
                alpha=self._alpha,
                query_properties=self._search_properties,
                filters=combined_filter,
                return_metadata=["score", "distance"],  # v4 way to get metadata
                return_properties=["content", "file_path", "chunk_index", "relationships"]  # Properties stored during indexing
            )
            
            raw_objects = response.objects if response else []
            
            # Post-process results for context awareness
            processed_objects = self._post_process_results(raw_objects, query_bundle)
            
        except Exception as e:
            logger.error(f"Error during Weaviate hybrid retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

        nodes_with_scores = []
        for obj in processed_objects:
            # Extract data from Weaviate v4 QueryObject
            text_content = obj.properties.get("content", "")
            metadata_dict = {
                "file_path": obj.properties.get("file_path", "N/A"),
                "chunk_index": obj.properties.get("chunk_index", -1),
                "repo_name": obj.properties.get("repo_name", "unknown"),
            }
            
            # Extract score from metadata
            score = 0.0
            if obj.metadata and hasattr(obj.metadata, 'score'):
                score = obj.metadata.score or 0.0
            elif obj.metadata and hasattr(obj.metadata, 'distance'):
                # Convert distance to similarity score (if needed)
                distance = obj.metadata.distance or 0.0
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity
            else:
                # Use the score from the response if available
                score = getattr(obj, 'score', 0.0)
            
            node_id = str(obj.uuid) if hasattr(obj, 'uuid') and obj.uuid else None

            if text_content:  # Ensure there's text content
                node = TextNode(
                    id_=node_id,
                    text=text_content,
                    metadata=metadata_dict
                )
                nodes_with_scores.append(NodeWithScore(node=node, score=score))

        # Limit to requested number after post-processing
        nodes_with_scores = nodes_with_scores[:self._similarity_top_k]
        
        logger.info(f"Hybrid search retrieved {len(nodes_with_scores)} nodes.")
        return nodes_with_scores

    def _post_process_results(self, raw_objects, query_bundle):
        """
        Post-process results to boost relevance based on context
        """
        if not raw_objects or not self._current_file_path:
            # Sort by original score if no context
            raw_objects.sort(
                key=lambda x: (x.metadata.score if x.metadata and hasattr(x.metadata, 'score') else 0),
                reverse=True
            )
            return raw_objects
        
        # Calculate relevance scores based on file path similarity
        processed_results = []
        current_dir = os.path.dirname(self._current_file_path)
        
        for obj in raw_objects:
            file_path = obj.properties.get("file_path", "")
            
            # Calculate context relevance score
            relevance_score = 0
            
            # Same directory boost
            if os.path.dirname(file_path) == current_dir:
                relevance_score += 3.0
            
            # Same repository boost
            repo_name = obj.properties.get("repo_name", "unknown")
            if hasattr(self, '_current_repo') and repo_name == self._current_repo:
                relevance_score += 2.0
            
            # File extension match boost
            current_ext = os.path.splitext(self._current_file_path)[1]
            obj_ext = os.path.splitext(file_path)[1]
            if current_ext == obj_ext:
                relevance_score += 1.0
            
            # Path similarity boost
            if current_dir in file_path or file_path in current_dir:
                relevance_score += 1.5
            
            # Store the relevance score for sorting
            obj._context_relevance = relevance_score
            processed_results.append(obj)
        
        # Sort by combined score (original score + context relevance)
        processed_results.sort(
            key=lambda x: (x.metadata.score if x.metadata and hasattr(x.metadata, 'score') else 0) + getattr(x, '_context_relevance', 0),
            reverse=True
        )
        
        return processed_results[:self._similarity_top_k]