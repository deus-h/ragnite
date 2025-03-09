"""
Base Hybrid Searcher

This module provides the BaseHybridSearcher abstract base class that all hybrid searchers inherit from.
Hybrid searchers combine multiple search strategies to improve retrieval performance.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)


class BaseHybridSearcher(ABC):
    """
    Abstract base class for hybrid searchers.
    
    Hybrid searchers combine multiple search strategies (such as vector similarity
    and keyword search) to improve retrieval performance. Different hybrid searchers
    use different strategies and combination methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hybrid searcher.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    @abstractmethod
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for documents using a hybrid approach.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing at least:
                - 'id': Document ID
                - 'score': Combined relevance score
                - Additional metadata from the source document
        """
        pass
    
    @abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the searcher configuration.
        
        Args:
            config: Configuration dictionary
        """
        pass
    
    @property
    def supported_backends(self) -> List[str]:
        """
        Get the list of supported search backends.
        
        Returns:
            List[str]: List of supported backend names
        """
        return []
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: True if the configuration is valid, False otherwise
        """
        # Base implementation just checks if config is a dict
        return isinstance(config, dict)
    
    def explain_search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Explain how the hybrid search results were generated.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            Dict[str, Any]: Explanation of the search process, containing:
                - 'query': Original query
                - 'results': Search results
                - 'strategy': Description of the hybrid search strategy used
                - 'component_results': Results from individual search components
                - Additional explanation details specific to the hybrid searcher
        """
        results = self.search(query, limit, **kwargs)
        
        return {
            "query": query,
            "results": results,
            "strategy": "Base hybrid search strategy",
            "component_results": []
        }
    
    def get_component_weights(self) -> Dict[str, float]:
        """
        Get the weights assigned to each search component.
        
        Returns:
            Dict[str, float]: Dictionary mapping component names to their weights
        """
        return {}
    
    def set_component_weights(self, weights: Dict[str, float]) -> None:
        """
        Set the weights for search components.
        
        Args:
            weights: Dictionary mapping component names to their weights
        """
        pass 