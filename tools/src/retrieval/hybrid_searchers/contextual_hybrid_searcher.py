"""
Contextual Hybrid Searcher

This module provides a ContextualHybridSearcher that implements Anthropic's "Contextual Retrieval" technique,
which conditions embeddings on query context to improve retrieval performance.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import numpy as np

from .base_hybrid_searcher import BaseHybridSearcher

# Configure logging
logger = logging.getLogger(__name__)


class ContextualHybridSearcher(BaseHybridSearcher):
    """
    Contextual Hybrid Searcher that implements Anthropic's "Contextual Retrieval" technique.
    
    This searcher conditions embeddings on query context, creating query-specific embeddings
    rather than static document embeddings. It can be combined with traditional retrieval methods
    for improved results.
    
    Key features:
    - Query-conditioned embeddings that adapt to the specific information need
    - Combination with standard retrieval methods (vector, BM25)
    - Support for multiple fusion strategies (linear combination, reciprocal rank fusion)
    
    Attributes:
        vector_search_func (Callable): Function for standard vector similarity search
        embedding_model (Callable): Function to generate embeddings for text
        fusion_method (str): Method to fuse results ("linear", "rrf", "max", "min")
        config (Dict[str, Any]): Configuration for the hybrid searcher
    """
    
    def __init__(self, 
                 vector_search_func: Callable,
                 embedding_model: Callable,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ContextualHybridSearcher.
        
        Args:
            vector_search_func: Function for standard vector similarity search.
                Should accept (query, limit, **kwargs) and return a list of dicts with 'id' and 'score' keys.
            embedding_model: Function to generate embeddings for text.
                Should accept (text) and return an embedding vector.
            config: Optional configuration dictionary with the following keys:
                - static_weight: Weight for standard vector search results (default: 0.5)
                - contextual_weight: Weight for contextual search results (default: 0.5)
                - fusion_method: Method to combine results (default: 'linear')
                  Options: 'linear', 'rrf' (reciprocal rank fusion), 'max', 'min'
                - rrf_k: Constant k for RRF formula (default: 60)
                - min_score_threshold: Minimum score for results to be included (default: 0.0)
                - normalize_scores: Whether to normalize scores before combining (default: True)
                - query_template: Template for conditioning embeddings (default: "Query: {query}")
                - expand_results: Whether to include all results from both methods (default: True)
        """
        super().__init__(config or {})
        
        self.vector_search_func = vector_search_func
        self.embedding_model = embedding_model
        
        # Set default configuration
        default_config = {
            'static_weight': 0.5,
            'contextual_weight': 0.5,
            'fusion_method': 'linear',
            'rrf_k': 60,
            'min_score_threshold': 0.0,
            'normalize_scores': True,
            'query_template': "Query: {query}",
            'expand_results': True,
        }
        
        # Update config with default values for missing keys
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        logger.debug(f"Initialized ContextualHybridSearcher with config: {self.config}")
    
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for documents using contextual retrieval combined with standard vector search.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results, each containing at least:
                - 'id': Document ID
                - 'score': Combined relevance score
                - Additional metadata from the source document
        """
        logger.info(f"Performing contextual hybrid search for query: '{query}'")
        
        # Expand limit to ensure we have enough results for combination
        expanded_limit = limit * 2 if self.config['expand_results'] else limit
        
        # Get standard vector search results
        standard_results = self.vector_search_func(query, expanded_limit, **kwargs)
        logger.debug(f"Standard vector search returned {len(standard_results)} results")
        
        # Generate query embedding using the contextual template
        query_context = self.config['query_template'].format(query=query)
        query_embedding = self.embedding_model(query_context)
        
        # Perform contextual search by comparing query embedding with document embeddings
        contextual_results = self._perform_contextual_search(query_embedding, standard_results, expanded_limit)
        logger.debug(f"Contextual search returned {len(contextual_results)} results")
        
        # Combine results based on the configured fusion method
        fusion_method = self.config['fusion_method'].lower()
        if fusion_method == 'linear':
            combined_results = self._combine_linear(standard_results, contextual_results, limit)
        elif fusion_method == 'rrf':
            combined_results = self._combine_reciprocal_rank_fusion(standard_results, contextual_results, limit)
        elif fusion_method == 'max':
            combined_results = self._combine_max(standard_results, contextual_results, limit)
        elif fusion_method == 'min':
            combined_results = self._combine_min(standard_results, contextual_results, limit)
        else:
            logger.warning(f"Unknown fusion method '{fusion_method}', falling back to linear combination")
            combined_results = self._combine_linear(standard_results, contextual_results, limit)
        
        logger.info(f"Returned {len(combined_results)} combined results")
        
        return combined_results
    
    def _perform_contextual_search(self, 
                                  query_embedding: np.ndarray, 
                                  documents: List[Dict[str, Any]],
                                  limit: int) -> List[Dict[str, Any]]:
        """
        Perform contextual search by comparing the query embedding with document embeddings.
        
        Args:
            query_embedding: The embedding of the query with context
            documents: List of documents to search within
            limit: Maximum number of results to return
            
        Returns:
            List of search results with scores based on contextual similarity
        """
        results = []
        
        for doc in documents:
            # Get the document's content and embed it with the query context
            doc_content = doc.get('content', '')
            if not doc_content:
                logger.warning(f"Document {doc['id']} has no content, skipping contextual scoring")
                continue
            
            # Generate document embedding
            doc_embedding = self.embedding_model(doc_content)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            # Create result entry
            result = {
                'id': doc['id'],
                'score': float(similarity),
                'content': doc_content
            }
            
            # Copy metadata if present
            if 'metadata' in doc:
                result['metadata'] = doc['metadata']
            
            results.append(result)
        
        # Sort by score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply limit
        results = results[:limit]
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _combine_linear(self, 
                       standard_results: List[Dict[str, Any]], 
                       contextual_results: List[Dict[str, Any]], 
                       limit: int) -> List[Dict[str, Any]]:
        """
        Combine results using linear combination of scores.
        
        Args:
            standard_results: Results from standard vector search
            contextual_results: Results from contextual search
            limit: Maximum number of results to return
        
        Returns:
            Combined results
        """
        # Normalize scores if configured
        if self.config['normalize_scores']:
            standard_results = self._normalize_scores(standard_results)
            contextual_results = self._normalize_scores(contextual_results)
        
        # Create dictionaries for quick lookup
        standard_dict = {result['id']: result for result in standard_results}
        contextual_dict = {result['id']: result for result in contextual_results}
        
        # Combine all document IDs
        all_doc_ids = set(standard_dict.keys()) | set(contextual_dict.keys())
        
        # Calculate combined scores
        combined_results = []
        for doc_id in all_doc_ids:
            standard_score = standard_dict.get(doc_id, {}).get('score', 0.0)
            contextual_score = contextual_dict.get(doc_id, {}).get('score', 0.0)
            
            # Calculate weighted score
            combined_score = (
                self.config['static_weight'] * standard_score +
                self.config['contextual_weight'] * contextual_score
            )
            
            # Skip if below threshold
            if combined_score < self.config['min_score_threshold']:
                continue
            
            # Get document data
            doc_data = standard_dict.get(doc_id) or contextual_dict.get(doc_id)
            
            # Create result entry
            result = {
                'id': doc_id,
                'score': combined_score,
                'standard_score': standard_score,
                'contextual_score': contextual_score
            }
            
            # Copy content and metadata if present
            if 'content' in doc_data:
                result['content'] = doc_data['content']
            if 'metadata' in doc_data:
                result['metadata'] = doc_data['metadata']
            
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply limit
        combined_results = combined_results[:limit]
        
        return combined_results
    
    def _combine_reciprocal_rank_fusion(self, 
                                       standard_results: List[Dict[str, Any]], 
                                       contextual_results: List[Dict[str, Any]], 
                                       limit: int) -> List[Dict[str, Any]]:
        """
        Combine results using reciprocal rank fusion.
        
        Args:
            standard_results: Results from standard vector search
            contextual_results: Results from contextual search
            limit: Maximum number of results to return
        
        Returns:
            Combined results
        """
        # Create dictionaries to store ranks
        standard_ranks = {}
        contextual_ranks = {}
        
        # Constant k for RRF formula
        k = self.config['rrf_k']
        
        # Process standard results to get ranks
        for i, result in enumerate(standard_results):
            doc_id = result['id']
            standard_ranks[doc_id] = i + 1  # 1-based ranking
        
        # Process contextual results to get ranks
        for i, result in enumerate(contextual_results):
            doc_id = result['id']
            contextual_ranks[doc_id] = i + 1  # 1-based ranking
        
        # Create dictionaries for quick lookup
        standard_dict = {result['id']: result for result in standard_results}
        contextual_dict = {result['id']: result for result in contextual_results}
        
        # Combine all document IDs
        all_doc_ids = set(standard_ranks.keys()) | set(contextual_ranks.keys())
        
        # Calculate RRF scores
        combined_results = []
        for doc_id in all_doc_ids:
            # Get ranks, using a large value if not present in a result set
            standard_rank = standard_ranks.get(doc_id, len(standard_results) + 1)
            contextual_rank = contextual_ranks.get(doc_id, len(contextual_results) + 1)
            
            # Calculate RRF score
            standard_rrf = 1.0 / (k + standard_rank)
            contextual_rrf = 1.0 / (k + contextual_rank)
            
            # Apply weights
            weighted_rrf = (
                self.config['static_weight'] * standard_rrf +
                self.config['contextual_weight'] * contextual_rrf
            )
            
            # Skip if below threshold
            if weighted_rrf < self.config['min_score_threshold']:
                continue
            
            # Get document data
            doc_data = standard_dict.get(doc_id) or contextual_dict.get(doc_id)
            
            # Create result entry
            result = {
                'id': doc_id,
                'score': weighted_rrf,
                'standard_rank': standard_rank,
                'contextual_rank': contextual_rank
            }
            
            # Copy content and metadata if present
            if 'content' in doc_data:
                result['content'] = doc_data['content']
            if 'metadata' in doc_data:
                result['metadata'] = doc_data['metadata']
            
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply limit
        combined_results = combined_results[:limit]
        
        return combined_results
    
    def _combine_max(self, 
                    standard_results: List[Dict[str, Any]], 
                    contextual_results: List[Dict[str, Any]], 
                    limit: int) -> List[Dict[str, Any]]:
        """
        Combine results by taking the maximum score from either method.
        
        Args:
            standard_results: Results from standard vector search
            contextual_results: Results from contextual search
            limit: Maximum number of results to return
        
        Returns:
            Combined results
        """
        # Normalize scores if configured
        if self.config['normalize_scores']:
            standard_results = self._normalize_scores(standard_results)
            contextual_results = self._normalize_scores(contextual_results)
        
        # Create dictionaries for quick lookup
        standard_dict = {result['id']: result for result in standard_results}
        contextual_dict = {result['id']: result for result in contextual_results}
        
        # Combine all document IDs
        all_doc_ids = set(standard_dict.keys()) | set(contextual_dict.keys())
        
        # Calculate max scores
        combined_results = []
        for doc_id in all_doc_ids:
            standard_score = standard_dict.get(doc_id, {}).get('score', 0.0)
            contextual_score = contextual_dict.get(doc_id, {}).get('score', 0.0)
            
            # Take the maximum score
            max_score = max(standard_score, contextual_score)
            
            # Skip if below threshold
            if max_score < self.config['min_score_threshold']:
                continue
            
            # Get document data
            doc_data = standard_dict.get(doc_id) or contextual_dict.get(doc_id)
            
            # Create result entry
            result = {
                'id': doc_id,
                'score': max_score,
                'standard_score': standard_score,
                'contextual_score': contextual_score,
                'selected_method': 'standard' if standard_score >= contextual_score else 'contextual'
            }
            
            # Copy content and metadata if present
            if 'content' in doc_data:
                result['content'] = doc_data['content']
            if 'metadata' in doc_data:
                result['metadata'] = doc_data['metadata']
            
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply limit
        combined_results = combined_results[:limit]
        
        return combined_results
    
    def _combine_min(self, 
                    standard_results: List[Dict[str, Any]], 
                    contextual_results: List[Dict[str, Any]], 
                    limit: int) -> List[Dict[str, Any]]:
        """
        Combine results by taking the minimum score from both methods.
        This approach emphasizes results that score well in both methods.
        
        Args:
            standard_results: Results from standard vector search
            contextual_results: Results from contextual search
            limit: Maximum number of results to return
        
        Returns:
            Combined results
        """
        # Normalize scores if configured
        if self.config['normalize_scores']:
            standard_results = self._normalize_scores(standard_results)
            contextual_results = self._normalize_scores(contextual_results)
        
        # Create dictionaries for quick lookup
        standard_dict = {result['id']: result for result in standard_results}
        contextual_dict = {result['id']: result for result in contextual_results}
        
        # Find document IDs present in both result sets
        common_doc_ids = set(standard_dict.keys()) & set(contextual_dict.keys())
        
        # Calculate min scores
        combined_results = []
        for doc_id in common_doc_ids:
            standard_score = standard_dict[doc_id]['score']
            contextual_score = contextual_dict[doc_id]['score']
            
            # Take the minimum score
            min_score = min(standard_score, contextual_score)
            
            # Skip if below threshold
            if min_score < self.config['min_score_threshold']:
                continue
            
            # Get document data
            doc_data = standard_dict[doc_id]
            
            # Create result entry
            result = {
                'id': doc_id,
                'score': min_score,
                'standard_score': standard_score,
                'contextual_score': contextual_score
            }
            
            # Copy content and metadata if present
            if 'content' in doc_data:
                result['content'] = doc_data['content']
            if 'metadata' in doc_data:
                result['metadata'] = doc_data['metadata']
            
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply limit
        combined_results = combined_results[:limit]
        
        return combined_results
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores to the range [0, 1].
        
        Args:
            results: List of search results
            
        Returns:
            List of search results with normalized scores
        """
        if not results:
            return results
        
        # Find min and max scores
        scores = [result['score'] for result in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # If all scores are the same, return as-is
        if max_score == min_score:
            return results
        
        # Normalize scores
        normalized_results = []
        for result in results:
            normalized_result = result.copy()
            normalized_result['score'] = (result['score'] - min_score) / (max_score - min_score)
            normalized_results.append(normalized_result)
        
        return normalized_results
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the searcher configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        logger.info(f"Updated configuration: {config}")
    
    @property
    def supported_backends(self) -> List[str]:
        """
        Get the list of supported search backends.
        
        Returns:
            List of supported backend names
        """
        return ['vector', 'contextual']
    
    def get_component_weights(self) -> Dict[str, float]:
        """
        Get the weights assigned to each search component.
        
        Returns:
            Dictionary mapping component names to their weights
        """
        return {
            'static': self.config['static_weight'],
            'contextual': self.config['contextual_weight']
        }
    
    def set_component_weights(self, weights: Dict[str, float]) -> None:
        """
        Set the weights for search components.
        
        Args:
            weights: Dictionary mapping component names to their weights
        """
        if 'static' in weights:
            self.config['static_weight'] = weights['static']
        
        if 'contextual' in weights:
            self.config['contextual_weight'] = weights['contextual']
        
        logger.info(f"Updated component weights: {weights}")
    
    def explain_search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Explain how the contextual hybrid search results were generated.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            Explanation of the search process
        """
        # Perform the search
        results = self.search(query, limit, **kwargs)
        
        # Create explanation
        explanation = {
            "query": query,
            "results": results,
            "strategy": f"Contextual hybrid search using {self.config['fusion_method']} fusion",
            "weights": self.get_component_weights(),
            "config": self.config
        }
        
        return explanation 