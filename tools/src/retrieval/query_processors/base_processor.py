"""
Base Query Processor

This module provides the abstract base class for query processors.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union


class BaseQueryProcessor(ABC):
    """
    Abstract base class for query processors.
    
    This class defines the interface that all query processors must implement.
    Query processors transform or enhance the original query to improve retrieval performance.
    """
    
    @abstractmethod
    def process_query(self, query: str, **kwargs) -> Union[str, List[str]]:
        """
        Process a query to improve retrieval performance.
        
        Args:
            query: The original query string
            **kwargs: Additional parameters for query processing
            
        Returns:
            Union[str, List[str]]: Processed query or list of processed queries
        """
        pass

    def __call__(self, query: str, **kwargs) -> Union[str, List[str]]:
        """
        Process a query using the __call__ method.
        
        Args:
            query: The original query string
            **kwargs: Additional parameters for query processing
            
        Returns:
            Union[str, List[str]]: Processed query or list of processed queries
        """
        return self.process_query(query, **kwargs) 