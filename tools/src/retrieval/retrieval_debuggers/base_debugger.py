"""
Base Retrieval Debugger

This module provides the abstract base class for retrieval debuggers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple


class BaseRetrievalDebugger(ABC):
    """
    Abstract base class for retrieval debuggers.
    
    This class defines the interface that all retrieval debuggers must implement.
    Retrieval debuggers analyze and diagnose retrieval results to help improve performance.
    """
    
    @abstractmethod
    def analyze(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Analyze retrieval results for a given query.
        
        Args:
            query: The query string that produced the results
            results: List of retrieval results, each containing at least id, score, and content
            **kwargs: Additional parameters for analysis
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        pass
    
    @abstractmethod
    def compare(
        self,
        query: str,
        results_sets: List[List[Dict[str, Any]]],
        names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple sets of retrieval results for the same query.
        
        Args:
            query: The query string that produced the results
            results_sets: List of result sets to compare
            names: Optional names for each result set
            **kwargs: Additional parameters for comparison
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ground_truth: Union[List[str], List[Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results against ground truth.
        
        Args:
            query: The query string that produced the results
            results: List of retrieval results
            ground_truth: Ground truth relevant documents or IDs
            **kwargs: Additional parameters for evaluation
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_insights(self, query: str, results: List[Dict[str, Any]], **kwargs) -> List[str]:
        """
        Get actionable insights about retrieval results.
        
        Args:
            query: The query string that produced the results
            results: List of retrieval results
            **kwargs: Additional parameters for insight generation
            
        Returns:
            List[str]: List of insights or suggestions
        """
        pass
    
    def __call__(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Analyze retrieval results using the __call__ method.
        
        Args:
            query: The query string that produced the results
            results: List of retrieval results
            **kwargs: Additional parameters for analysis
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        return self.analyze(query, results, **kwargs) 