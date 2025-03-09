"""
BM25 Vector Hybrid Searcher

This module provides the BM25VectorHybridSearcher class that combines
BM25 keyword search with vector similarity search to improve retrieval performance.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from rank_bm25 import BM25Okapi, BM25Plus, BM25L

from .base_hybrid_searcher import BaseHybridSearcher

# Configure logging
logger = logging.getLogger(__name__)


class BM25VectorHybridSearcher(BaseHybridSearcher):
    """
    Hybrid searcher that combines BM25 keyword search with vector similarity search.
    
    This searcher leverages the strengths of both BM25 (a sophisticated keyword search algorithm)
    and vector search excels at semantic understanding. BM25 is particularly good at
    exact keyword matching with proper term frequency normalization and document length
    considerations, while vector search excels at semantic understanding.
    
    Attributes:
        vector_search_func (Callable): Function for vector similarity search
        bm25_search_func (Callable): Function for BM25 search (if provided externally)
        bm25_variant (str): BM25 variant to use ('okapi', 'plus', or 'l')
        bm25_index (Optional[Union[BM25Okapi, BM25Plus, BM25L]]): BM25 index if built internally
        corpus (Optional[List[str]]): Document corpus for internal BM25 index
        doc_ids (Optional[List[str]]): Document IDs corresponding to corpus
        config (Dict[str, Any]): Configuration for the hybrid searcher
    """
    
    def __init__(self, 
                 vector_search_func: Callable,
                 bm25_search_func: Optional[Callable] = None,
                 corpus: Optional[List[str]] = None,
                 doc_ids: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BM25VectorHybridSearcher.
        
        Args:
            vector_search_func: Function for vector similarity search.
                Should accept (query, limit, **kwargs) and return a list of dicts with 'id' and 'score' keys.
            bm25_search_func: Optional function for BM25 search.
                Should accept (query, limit, **kwargs) and return a list of dicts with 'id' and 'score' keys.
                If not provided, an internal BM25 index will be built using the corpus.
            corpus: Optional list of document texts for building an internal BM25 index.
                Required if bm25_search_func is not provided.
            doc_ids: Optional list of document IDs corresponding to the corpus.
                Required if corpus is provided. Must be the same length as corpus.
            config: Optional configuration dictionary with the following keys:
                - vector_weight: Weight for vector search results (default: 0.5)
                - bm25_weight: Weight for BM25 search results (default: 0.5)
                - combination_method: Method to combine results (default: 'linear_combination')
                - min_score_threshold: Minimum score for results to be included (default: 0.0)
                - normalize_scores: Whether to normalize scores before combining (default: True)
                - expand_results: Whether to include all results from both methods (default: True)
                - bm25_variant: BM25 variant to use ('okapi', 'plus', or 'l') (default: 'plus')
                - bm25_k1: BM25 k1 parameter (default: 1.5)
                - bm25_b: BM25 b parameter (default: 0.75)
                - bm25_delta: BM25L delta parameter (default: 0.5)
        
        Raises:
            ValueError: If neither bm25_search_func nor (corpus and doc_ids) are provided.
            ValueError: If corpus is provided but doc_ids is not, or vice versa.
            ValueError: If corpus and doc_ids have different lengths.
        """
        super().__init__(config)
        
        self.vector_search_func = vector_search_func
        self.bm25_search_func = bm25_search_func
        self.bm25_index = None
        self.corpus = None
        self.doc_ids = None
        self.bm25_variant = self.config.get('bm25_variant', 'plus')
        
        # If no external BM25 search function is provided, build an internal BM25 index
        if bm25_search_func is None:
            if corpus is None or doc_ids is None:
                raise ValueError("Either bm25_search_func or both corpus and doc_ids must be provided")
            
            if len(corpus) != len(doc_ids):
                raise ValueError("corpus and doc_ids must have the same length")
            
            self.corpus = corpus
            self.doc_ids = doc_ids
            self._build_bm25_index()
        
        # Set default configuration
        default_config = {
            'vector_weight': 0.5,
            'bm25_weight': 0.5,
            'combination_method': 'linear_combination',
            'min_score_threshold': 0.0,
            'normalize_scores': True,
            'expand_results': True,
            'bm25_variant': 'plus',
            'bm25_k1': 1.5,
            'bm25_b': 0.75,
            'bm25_delta': 0.5
        }
        
        # Update with user-provided config
        if config:
            default_config.update(config)
        
        self.config = default_config
    
    def _build_bm25_index(self) -> None:
        """
        Build an internal BM25 index using the provided corpus.
        
        This method tokenizes the corpus and builds a BM25 index using the specified variant.
        """
        # Tokenize the corpus
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        
        # Build the BM25 index based on the specified variant
        if self.bm25_variant == 'okapi':
            self.bm25_index = BM25Okapi(
                tokenized_corpus, 
                k1=self.config.get('bm25_k1', 1.5), 
                b=self.config.get('bm25_b', 0.75)
            )
        elif self.bm25_variant == 'l':
            self.bm25_index = BM25L(
                tokenized_corpus, 
                k1=self.config.get('bm25_k1', 1.5), 
                b=self.config.get('bm25_b', 0.75), 
                delta=self.config.get('bm25_delta', 0.5)
            )
        else:  # default to 'plus'
            self.bm25_index = BM25Plus(
                tokenized_corpus, 
                k1=self.config.get('bm25_k1', 1.5), 
                b=self.config.get('bm25_b', 0.75), 
                delta=self.config.get('bm25_delta', 0.5)
            )
        
        logger.info(f"Built internal BM25 index ({self.bm25_variant}) with {len(self.corpus)} documents")
    
    def _internal_bm25_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform BM25 search using the internal BM25 index.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
        
        Returns:
            List of dictionaries with 'id' and 'score' keys
        """
        # Tokenize the query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Create a list of (doc_id, score) tuples
        scored_results = [(self.doc_ids[i], scores[i]) for i in range(len(scores))]
        
        # Sort by score in descending order and take top 'limit' results
        scored_results.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_results[:limit]
        
        # Convert to the expected format
        results = [{'id': doc_id, 'score': score} for doc_id, score in top_results]
        
        return results
    
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional arguments to pass to the search functions
        
        Returns:
            List of dictionaries with search results
        """
        # Get vector search results
        vector_results = self.vector_search_func(query, limit=limit if not self.config['expand_results'] else limit * 2, **kwargs)
        
        # Get BM25 search results
        if self.bm25_search_func:
            bm25_results = self.bm25_search_func(query, limit=limit if not self.config['expand_results'] else limit * 2, **kwargs)
        else:
            bm25_results = self._internal_bm25_search(query, limit=limit if not self.config['expand_results'] else limit * 2)
        
        # Normalize scores if configured to do so
        if self.config['normalize_scores']:
            vector_results = self._normalize_scores(vector_results)
            bm25_results = self._normalize_scores(bm25_results)
        
        # Combine results based on the specified method
        if self.config['combination_method'] == 'reciprocal_rank_fusion':
            combined_results = self._combine_by_reciprocal_rank_fusion(
                vector_results, bm25_results, limit
            )
        else:  # default to linear_combination
            combined_results = self._combine_by_linear_weights(
                vector_results, bm25_results, limit
            )
        
        # Filter results below the minimum score threshold
        if self.config['min_score_threshold'] > 0:
            combined_results = [
                result for result in combined_results 
                if result.get('score', 0) >= self.config['min_score_threshold']
            ]
        
        return combined_results
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration of the hybrid searcher.
        
        Args:
            config: New configuration dictionary
        """
        # Validate the new configuration
        if not self.validate_config(config):
            raise ValueError("Invalid configuration")
        
        # Update the configuration
        self.config.update(config)
        
        # If BM25 variant or parameters changed and we're using an internal index, rebuild it
        if (self.bm25_index is not None and 
            ('bm25_variant' in config or 'bm25_k1' in config or 'bm25_b' in config or 'bm25_delta' in config)):
            self.bm25_variant = config.get('bm25_variant', self.bm25_variant)
            self._build_bm25_index()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration dictionary.
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        # Check combination method
        if 'combination_method' in config and config['combination_method'] not in ['linear_combination', 'reciprocal_rank_fusion']:
            logger.error(f"Invalid combination method: {config['combination_method']}")
            return False
        
        # Check weights
        if 'vector_weight' in config and (not isinstance(config['vector_weight'], (int, float)) or config['vector_weight'] < 0):
            logger.error(f"Invalid vector weight: {config['vector_weight']}")
            return False
        
        if 'bm25_weight' in config and (not isinstance(config['bm25_weight'], (int, float)) or config['bm25_weight'] < 0):
            logger.error(f"Invalid BM25 weight: {config['bm25_weight']}")
            return False
        
        # Check BM25 variant
        if 'bm25_variant' in config and config['bm25_variant'] not in ['okapi', 'plus', 'l']:
            logger.error(f"Invalid BM25 variant: {config['bm25_variant']}")
            return False
        
        # Check BM25 parameters
        if 'bm25_k1' in config and (not isinstance(config['bm25_k1'], (int, float)) or config['bm25_k1'] < 0):
            logger.error(f"Invalid BM25 k1 parameter: {config['bm25_k1']}")
            return False
        
        if 'bm25_b' in config and (not isinstance(config['bm25_b'], (int, float)) or config['bm25_b'] < 0 or config['bm25_b'] > 1):
            logger.error(f"Invalid BM25 b parameter: {config['bm25_b']}")
            return False
        
        if 'bm25_delta' in config and (not isinstance(config['bm25_delta'], (int, float)) or config['bm25_delta'] < 0):
            logger.error(f"Invalid BM25 delta parameter: {config['bm25_delta']}")
            return False
        
        return True
    
    @property
    def supported_backends(self) -> List[str]:
        """
        Get the list of supported backend types.
        
        Returns:
            List of supported backend types
        """
        return ['generic', 'chroma', 'qdrant', 'pinecone', 'weaviate', 'milvus', 'pgvector']
    
    def get_component_weights(self) -> Dict[str, float]:
        """
        Get the current weights for each search component.
        
        Returns:
            Dictionary with component names as keys and weights as values
        """
        return {
            'vector': self.config['vector_weight'],
            'bm25': self.config['bm25_weight']
        }
    
    def set_component_weights(self, weights: Dict[str, float]) -> None:
        """
        Set the weights for each search component.
        
        Args:
            weights: Dictionary with component names as keys and weights as values
        
        Raises:
            ValueError: If the weights dictionary contains invalid keys or values
        """
        valid_components = {'vector', 'bm25'}
        for component, weight in weights.items():
            if component not in valid_components:
                raise ValueError(f"Invalid component: {component}. Valid components are: {valid_components}")
            
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"Invalid weight for {component}: {weight}. Weight must be a non-negative number.")
        
        # Update the weights in the configuration
        if 'vector' in weights:
            self.config['vector_weight'] = weights['vector']
        
        if 'bm25' in weights:
            self.config['bm25_weight'] = weights['bm25']
    
    def explain_search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Explain the hybrid search process and results.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional arguments to pass to the search functions
        
        Returns:
            Dictionary with explanation of the search process and results
        """
        # Get vector search results
        vector_results = self.vector_search_func(query, limit=limit if not self.config['expand_results'] else limit * 2, **kwargs)
        
        # Get BM25 search results
        if self.bm25_search_func:
            bm25_results = self.bm25_search_func(query, limit=limit if not self.config['expand_results'] else limit * 2, **kwargs)
        else:
            bm25_results = self._internal_bm25_search(query, limit=limit if not self.config['expand_results'] else limit * 2)
        
        # Normalize scores if configured to do so
        if self.config['normalize_scores']:
            vector_results = self._normalize_scores(vector_results.copy())
            bm25_results = self._normalize_scores(bm25_results.copy())
        
        # Combine results based on the specified method
        if self.config['combination_method'] == 'reciprocal_rank_fusion':
            combined_results = self._combine_by_reciprocal_rank_fusion(
                vector_results.copy(), bm25_results.copy(), limit
            )
        else:  # default to linear_combination
            combined_results = self._combine_by_linear_weights(
                vector_results.copy(), bm25_results.copy(), limit
            )
        
        # Filter results below the minimum score threshold
        if self.config['min_score_threshold'] > 0:
            combined_results = [
                result for result in combined_results 
                if result.get('score', 0) >= self.config['min_score_threshold']
            ]
        
        # Create the explanation
        explanation = {
            'query': query,
            'results': combined_results,
            'search_strategy': 'BM25 Vector Hybrid Search',
            'description': 'Combined BM25 keyword search with vector similarity search',
            'components': {
                'vector': {
                    'weight': self.config['vector_weight'],
                    'results': vector_results
                },
                'bm25': {
                    'weight': self.config['bm25_weight'],
                    'variant': self.bm25_variant,
                    'results': bm25_results
                }
            },
            'configuration': {
                'combination_method': self.config['combination_method'],
                'normalize_scores': self.config['normalize_scores'],
                'min_score_threshold': self.config['min_score_threshold'],
                'expand_results': self.config['expand_results'],
                'bm25_parameters': {
                    'variant': self.bm25_variant,
                    'k1': self.config.get('bm25_k1', 1.5),
                    'b': self.config.get('bm25_b', 0.75),
                    'delta': self.config.get('bm25_delta', 0.5)
                }
            }
        }
        
        return explanation
    
    def _combine_by_linear_weights(self, 
                                  vector_results: List[Dict[str, Any]], 
                                  bm25_results: List[Dict[str, Any]], 
                                  limit: int) -> List[Dict[str, Any]]:
        """
        Combine results using linear weights.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            limit: Maximum number of results to return
        
        Returns:
            Combined results
        """
        # Create a dictionary to store combined scores
        combined_scores = {}
        
        # Process vector results
        for result in vector_results:
            doc_id = result['id']
            score = result.get('score', 0.0)
            
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'id': doc_id,
                    'vector_score': score,
                    'bm25_score': 0.0,
                    'combined_score': 0.0,
                    'metadata': result.get('metadata', {}),
                    'content': result.get('content', '')
                }
            else:
                combined_scores[doc_id]['vector_score'] = score
        
        # Process BM25 results
        for result in bm25_results:
            doc_id = result['id']
            score = result.get('score', 0.0)
            
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {
                    'id': doc_id,
                    'vector_score': 0.0,
                    'bm25_score': score,
                    'combined_score': 0.0,
                    'metadata': result.get('metadata', {}),
                    'content': result.get('content', '')
                }
            else:
                combined_scores[doc_id]['bm25_score'] = score
                combined_scores[doc_id]['metadata'] = combined_scores[doc_id]['metadata'] or result.get('metadata', {})
                combined_scores[doc_id]['content'] = combined_scores[doc_id]['content'] or result.get('content', '')
        
        # Calculate combined scores
        for doc_id, data in combined_scores.items():
            data['combined_score'] = (
                self.config['vector_weight'] * data['vector_score'] +
                self.config['bm25_weight'] * data['bm25_score']
            )
        
        # Convert to list and sort by combined score
        combined_results = list(combined_scores.values())
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Take top 'limit' results
        combined_results = combined_results[:limit]
        
        # Rename 'combined_score' to 'score' for consistency
        for result in combined_results:
            result['score'] = result.pop('combined_score')
        
        return combined_results
    
    def _combine_by_reciprocal_rank_fusion(self, 
                                          vector_results: List[Dict[str, Any]], 
                                          bm25_results: List[Dict[str, Any]], 
                                          limit: int) -> List[Dict[str, Any]]:
        """
        Combine results using reciprocal rank fusion.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            limit: Maximum number of results to return
        
        Returns:
            Combined results
        """
        # Create dictionaries to store ranks
        vector_ranks = {}
        bm25_ranks = {}
        
        # Constant k for RRF formula
        k = 60  # Standard value used in RRF
        
        # Process vector results to get ranks
        for i, result in enumerate(vector_results):
            doc_id = result['id']
            vector_ranks[doc_id] = i + 1  # 1-based ranking
        
        # Process BM25 results to get ranks
        for i, result in enumerate(bm25_results):
            doc_id = result['id']
            bm25_ranks[doc_id] = i + 1  # 1-based ranking
        
        # Create a dictionary to store combined scores and document data
        combined_scores = {}
        
        # Process all document IDs from both result sets
        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        for doc_id in all_doc_ids:
            # Get ranks, using a large value if not present in a result set
            vector_rank = vector_ranks.get(doc_id, len(vector_results) + 1)
            bm25_rank = bm25_ranks.get(doc_id, len(bm25_results) + 1)
            
            # Calculate RRF score
            vector_rrf = 1.0 / (k + vector_rank)
            bm25_rrf = 1.0 / (k + bm25_rank)
            
            # Apply weights
            weighted_rrf = (
                self.config['vector_weight'] * vector_rrf +
                self.config['bm25_weight'] * bm25_rrf
            )
            
            # Find the document data from the result sets
            doc_data = None
            for result in vector_results:
                if result['id'] == doc_id:
                    doc_data = result
                    break
            
            if doc_data is None:
                for result in bm25_results:
                    if result['id'] == doc_id:
                        doc_data = result
                        break
            
            # Store the combined score and document data
            combined_scores[doc_id] = {
                'id': doc_id,
                'score': weighted_rrf,
                'vector_rank': vector_rank,
                'bm25_rank': bm25_rank,
                'metadata': doc_data.get('metadata', {}) if doc_data else {},
                'content': doc_data.get('content', '') if doc_data else ''
            }
        
        # Convert to list and sort by score
        combined_results = list(combined_scores.values())
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 'limit' results
        combined_results = combined_results[:limit]
        
        return combined_results
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores in the results to be between 0 and 1.
        
        Args:
            results: List of search results
        
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        # Extract scores
        scores = [result.get('score', 0.0) for result in results]
        
        # Find min and max scores
        min_score = min(scores)
        max_score = max(scores)
        
        # If all scores are the same, return as is
        if max_score == min_score:
            return results
        
        # Normalize scores
        for i, result in enumerate(results):
            normalized_score = (result.get('score', 0.0) - min_score) / (max_score - min_score)
            results[i]['score'] = normalized_score
        
        return results 