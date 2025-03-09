"""
Base Reranker

This module provides the BaseReranker abstract base class that all rerankers inherit from.
Rerankers improve retrieval performance by re-scoring and re-ordering initial retrieval results.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.
    
    Rerankers take an initial set of retrieval results and improve them by re-scoring
    and potentially re-ordering them based on more sophisticated relevance models.
    Different rerankers may use different approaches such as cross-encoders, monoT5,
    LLMs, or ensemble methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reranker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def rerank(self, query: str, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Rerank the retrieval results.
        
        Args:
            query: The search query
            results: Initial retrieval results, each as a dictionary containing at least:
                     - 'id': Document ID
                     - 'content': Document content
                     - 'score': Initial retrieval score
            **kwargs: Additional arguments for specific implementations
        
        Returns:
            Reranked results with updated scores
        """
        pass
    
    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration of the reranker.
        
        Args:
            config: New configuration dictionary
        """
        pass
    
    def validate_results(self, results: List[Dict[str, Any]]) -> bool:
        """
        Validate that the input results have the required fields.
        
        Args:
            results: List of retrieval result dictionaries
        
        Returns:
            True if results are valid, False otherwise
        """
        if not results:
            logger.warning("Empty results list provided")
            return True  # Empty list is technically valid
        
        for i, result in enumerate(results):
            if 'id' not in result:
                logger.error(f"Result at index {i} is missing 'id' field")
                return False
            
            if 'content' not in result and 'text' not in result:
                logger.error(f"Result at index {i} is missing both 'content' and 'text' fields")
                return False
        
        return True
    
    def explain_reranking(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Provide an explanation of the reranking process.
        
        This method should be overridden by concrete classes but provides a basic
        implementation that tracks original vs. reranked scores.
        
        Args:
            query: The search query
            results: Initial retrieval results
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with explanation of the reranking process and results
        """
        # Store original scores
        for i, result in enumerate(results):
            result['original_score'] = result.get('score', 0.0)
            result['original_rank'] = i
        
        # Perform reranking
        reranked_results = self.rerank(query, results, **kwargs)
        
        # Create basic explanation
        explanation = {
            'query': query,
            'reranking_method': self.__class__.__name__,
            'config': self.config,
            'results': reranked_results,
            'result_changes': []
        }
        
        # Track changes in rank
        for i, result in enumerate(reranked_results):
            original_rank = result.get('original_rank', -1)
            rank_change = original_rank - i if original_rank >= 0 else None
            
            explanation['result_changes'].append({
                'id': result.get('id', ''),
                'original_rank': original_rank,
                'new_rank': i,
                'rank_change': rank_change,
                'original_score': result.get('original_score', 0.0),
                'new_score': result.get('score', 0.0)
            })
        
        return explanation 