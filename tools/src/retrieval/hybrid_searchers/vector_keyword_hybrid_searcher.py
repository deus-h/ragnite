"""
Vector Keyword Hybrid Searcher

This module provides the VectorKeywordHybridSearcher class that combines
vector similarity search with keyword search to improve retrieval performance.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from .base_hybrid_searcher import BaseHybridSearcher

# Configure logging
logger = logging.getLogger(__name__)


class VectorKeywordHybridSearcher(BaseHybridSearcher):
    """
    Hybrid searcher that combines vector similarity search with keyword search.
    
    This searcher runs both vector similarity search and keyword search,
    then combines the results based on configurable weights to produce a
    final ranked list of documents.
    """
    
    def __init__(self, 
                 vector_search_func: Callable,
                 keyword_search_func: Callable,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the vector keyword hybrid searcher.
        
        Args:
            vector_search_func: Function that performs vector similarity search
                - It should take: (query: str, limit: int, **kwargs) as arguments
                - It should return: List[Dict[str, Any]] with each dict having at least 'id' and 'score' keys
            keyword_search_func: Function that performs keyword search
                - It should take: (query: str, limit: int, **kwargs) as arguments
                - It should return: List[Dict[str, Any]] with each dict having at least 'id' and 'score' keys
            config: Configuration dictionary with the following optional keys:
                - 'vector_weight': Weight for vector search results (default: 0.7)
                - 'keyword_weight': Weight for keyword search results (default: 0.3)
                - 'combination_method': Method to combine results ('linear_combination' or 'reciprocal_rank_fusion')
                - 'min_score_threshold': Minimum score for results to be included (default: 0.0)
                - 'normalize_scores': Whether to normalize scores before combining (default: True)
                - 'expand_results': Whether to include all results from both methods (default: True)
        """
        super().__init__(config or {})
        
        self.vector_search_func = vector_search_func
        self.keyword_search_func = keyword_search_func
        
        # Set default configuration if not provided
        self.config.setdefault("vector_weight", 0.7)
        self.config.setdefault("keyword_weight", 0.3)
        self.config.setdefault("combination_method", "linear_combination")
        self.config.setdefault("min_score_threshold", 0.0)
        self.config.setdefault("normalize_scores", True)
        self.config.setdefault("expand_results", True)
        
        # Verify weights sum to 1.0
        total_weight = self.config["vector_weight"] + self.config["keyword_weight"]
        if not np.isclose(total_weight, 1.0, rtol=1e-5):
            logger.warning(f"Weights do not sum to 1.0 (sum: {total_weight}). Normalizing weights.")
            self.config["vector_weight"] /= total_weight
            self.config["keyword_weight"] /= total_weight
    
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for documents using a combination of vector and keyword search.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters to pass to the underlying search functions
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing at least:
                - 'id': Document ID
                - 'score': Combined relevance score
                - Additional metadata from the source document
        """
        # Set expanded limit to ensure we get enough results to combine
        expanded_limit = limit * 2 if self.config["expand_results"] else limit
        
        # Get vector search results
        vector_results = self.vector_search_func(query, expanded_limit, **kwargs)
        
        # Get keyword search results
        keyword_results = self.keyword_search_func(query, expanded_limit, **kwargs)
        
        # Combine results based on the chosen method
        if self.config["combination_method"] == "reciprocal_rank_fusion":
            combined_results = self._combine_by_reciprocal_rank_fusion(
                vector_results, keyword_results, limit)
        else:  # Default to linear combination
            combined_results = self._combine_by_linear_weights(
                vector_results, keyword_results, limit)
        
        # Apply minimum score threshold
        min_score = self.config["min_score_threshold"]
        if min_score > 0:
            combined_results = [r for r in combined_results if r['score'] >= min_score]
        
        return combined_results[:limit]
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the searcher configuration.
        
        Args:
            config: Configuration dictionary
        """
        if self.validate_config(config):
            self.config.update(config)
            
            # Normalize weights if needed
            total_weight = self.config.get("vector_weight", 0) + self.config.get("keyword_weight", 0)
            if total_weight > 0 and not np.isclose(total_weight, 1.0, rtol=1e-5):
                self.config["vector_weight"] /= total_weight
                self.config["keyword_weight"] /= total_weight
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: True if the configuration is valid, False otherwise
        """
        if not super().validate_config(config):
            return False
        
        # Check that weights are non-negative
        if config.get("vector_weight", 0) < 0 or config.get("keyword_weight", 0) < 0:
            logger.error("Weights must be non-negative.")
            return False
        
        # Check that combination method is supported
        if config.get("combination_method") not in [None, "linear_combination", "reciprocal_rank_fusion"]:
            logger.error(f"Unsupported combination method: {config.get('combination_method')}")
            return False
        
        return True
    
    @property
    def supported_backends(self) -> List[str]:
        """
        Get the list of supported search backends.
        
        Returns:
            List[str]: List of supported backend names
        """
        return ["vector", "keyword"]
    
    def get_component_weights(self) -> Dict[str, float]:
        """
        Get the weights assigned to each search component.
        
        Returns:
            Dict[str, float]: Dictionary mapping component names to their weights
        """
        return {
            "vector": self.config["vector_weight"],
            "keyword": self.config["keyword_weight"]
        }
    
    def set_component_weights(self, weights: Dict[str, float]) -> None:
        """
        Set the weights for search components.
        
        Args:
            weights: Dictionary mapping component names to their weights
        """
        if "vector" in weights:
            self.config["vector_weight"] = weights["vector"]
        if "keyword" in weights:
            self.config["keyword_weight"] = weights["keyword"]
            
        # Normalize weights
        total_weight = self.config["vector_weight"] + self.config["keyword_weight"]
        if total_weight > 0 and not np.isclose(total_weight, 1.0, rtol=1e-5):
            self.config["vector_weight"] /= total_weight
            self.config["keyword_weight"] /= total_weight
    
    def explain_search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Explain how the hybrid search results were generated.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            Dict[str, Any]: Explanation of the search process
        """
        # Set expanded limit to ensure we get enough results to combine
        expanded_limit = limit * 2 if self.config["expand_results"] else limit
        
        # Get vector search results
        vector_results = self.vector_search_func(query, expanded_limit, **kwargs)
        
        # Get keyword search results
        keyword_results = self.keyword_search_func(query, expanded_limit, **kwargs)
        
        # Combine results based on the chosen method
        if self.config["combination_method"] == "reciprocal_rank_fusion":
            combined_results = self._combine_by_reciprocal_rank_fusion(
                vector_results, keyword_results, limit)
            combination_method = "Reciprocal Rank Fusion"
        else:  # Default to linear combination
            combined_results = self._combine_by_linear_weights(
                vector_results, keyword_results, limit)
            combination_method = "Linear Combination"
        
        # Apply minimum score threshold
        min_score = self.config["min_score_threshold"]
        if min_score > 0:
            combined_results = [r for r in combined_results if r['score'] >= min_score]
        
        return {
            "query": query,
            "results": combined_results[:limit],
            "strategy": f"Vector + Keyword Hybrid Search using {combination_method}",
            "component_results": {
                "vector": vector_results,
                "keyword": keyword_results
            },
            "weights": {
                "vector": self.config["vector_weight"],
                "keyword": self.config["keyword_weight"]
            },
            "configuration": {
                "normalize_scores": self.config["normalize_scores"],
                "expand_results": self.config["expand_results"],
                "min_score_threshold": self.config["min_score_threshold"]
            }
        }
    
    def _combine_by_linear_weights(self, 
                                  vector_results: List[Dict[str, Any]], 
                                  keyword_results: List[Dict[str, Any]], 
                                  limit: int) -> List[Dict[str, Any]]:
        """
        Combine results using linear combination of scores.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Combined results
        """
        # Create a mapping of document ID to result
        all_docs = {}
        
        # Normalize scores if configured to do so
        if self.config["normalize_scores"]:
            vector_results = self._normalize_scores(vector_results)
            keyword_results = self._normalize_scores(keyword_results)
        
        # Process vector results
        for result in vector_results:
            doc_id = result["id"]
            all_docs[doc_id] = {
                "id": doc_id,
                "vector_score": result["score"],
                "keyword_score": 0.0,
                "score": self.config["vector_weight"] * result["score"],
                **{k: v for k, v in result.items() if k not in ["id", "score"]}
            }
        
        # Process keyword results
        for result in keyword_results:
            doc_id = result["id"]
            if doc_id in all_docs:
                # Document already exists from vector results
                all_docs[doc_id]["keyword_score"] = result["score"]
                all_docs[doc_id]["score"] += self.config["keyword_weight"] * result["score"]
            else:
                # New document from keyword results
                all_docs[doc_id] = {
                    "id": doc_id,
                    "vector_score": 0.0,
                    "keyword_score": result["score"],
                    "score": self.config["keyword_weight"] * result["score"],
                    **{k: v for k, v in result.items() if k not in ["id", "score"]}
                }
        
        # Sort results by score in descending order
        combined_results = list(all_docs.values())
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results[:limit]
    
    def _combine_by_reciprocal_rank_fusion(self, 
                                          vector_results: List[Dict[str, Any]], 
                                          keyword_results: List[Dict[str, Any]], 
                                          limit: int) -> List[Dict[str, Any]]:
        """
        Combine results using reciprocal rank fusion.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Combined results
        """
        k = 60  # Constant to control balance between ranks
        
        # Create a mapping of document ID to result
        all_docs = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result["id"]
            vector_rank = rank + 1  # Convert to 1-based rank
            rrf_score_vector = 1 / (k + vector_rank)
            
            all_docs[doc_id] = {
                "id": doc_id,
                "vector_score": result["score"],
                "vector_rank": vector_rank,
                "keyword_score": 0.0,
                "keyword_rank": float('inf'),
                "rrf_score": self.config["vector_weight"] * rrf_score_vector,
                **{k: v for k, v in result.items() if k not in ["id", "score"]}
            }
        
        # Process keyword results
        for rank, result in enumerate(keyword_results):
            doc_id = result["id"]
            keyword_rank = rank + 1  # Convert to 1-based rank
            rrf_score_keyword = 1 / (k + keyword_rank)
            
            if doc_id in all_docs:
                # Document already exists from vector results
                all_docs[doc_id]["keyword_score"] = result["score"]
                all_docs[doc_id]["keyword_rank"] = keyword_rank
                all_docs[doc_id]["rrf_score"] += self.config["keyword_weight"] * rrf_score_keyword
            else:
                # New document from keyword results
                all_docs[doc_id] = {
                    "id": doc_id,
                    "vector_score": 0.0,
                    "vector_rank": float('inf'),
                    "keyword_score": result["score"],
                    "keyword_rank": keyword_rank,
                    "rrf_score": self.config["keyword_weight"] * rrf_score_keyword,
                    **{k: v for k, v in result.items() if k not in ["id", "score"]}
                }
        
        # Sort results by RRF score in descending order
        combined_results = list(all_docs.values())
        for result in combined_results:
            result["score"] = result.pop("rrf_score")  # Rename rrf_score to score
        
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results[:limit]
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores to the range [0, 1].
        
        Args:
            results: List of search results
            
        Returns:
            List[Dict[str, Any]]: Results with normalized scores
        """
        if not results:
            return results
        
        # Find min and max scores
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # If all scores are the same, return the original results
        if max_score - min_score < 1e-10:
            return results
        
        # Normalize scores
        normalized_results = []
        for result in results:
            result_copy = result.copy()
            result_copy["score"] = (result["score"] - min_score) / (max_score - min_score)
            normalized_results.append(result_copy)
        
        return normalized_results 