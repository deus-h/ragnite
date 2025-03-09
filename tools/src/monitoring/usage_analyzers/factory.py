#!/usr/bin/env python3
"""
Factory function for usage analyzers in RAG systems.

This module provides a factory function for creating usage analyzer instances
based on the specified type and configuration.
"""

from typing import Dict, Any, Optional
from .base import BaseUsageAnalyzer
from .query_analyzer import QueryAnalyzer
from .user_session_analyzer import UserSessionAnalyzer
from .feature_usage_analyzer import FeatureUsageAnalyzer
from .error_analyzer import ErrorAnalyzer


def get_usage_analyzer(
    analyzer_type: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> BaseUsageAnalyzer:
    """
    Create a usage analyzer instance based on the specified type and configuration.
    
    Args:
        analyzer_type (str): Type of usage analyzer to create.
            Valid values: 'query', 'user_session', 'feature_usage', 'error'.
        name (Optional[str]): Name of the analyzer.
            If not provided, a default name will be used based on the analyzer type.
        data_dir (Optional[str]): Directory to store analysis data.
            Defaults to './usage_data'.
        config (Optional[Dict[str, Any]]): Configuration options for the analyzer.
            Defaults to an empty dictionary.
            
    Returns:
        BaseUsageAnalyzer: Usage analyzer instance.
        
    Raises:
        ValueError: If an unsupported analyzer type is specified.
    """
    # Normalize analyzer type
    analyzer_type = analyzer_type.lower().strip()
    
    # Define mapping of analyzer types to classes
    analyzer_classes = {
        "query": QueryAnalyzer,
        "user_session": UserSessionAnalyzer,
        "feature_usage": FeatureUsageAnalyzer,
        "error": ErrorAnalyzer
    }
    
    # Check if analyzer type is supported
    if analyzer_type not in analyzer_classes:
        valid_types = ", ".join(f"'{t}'" for t in analyzer_classes.keys())
        raise ValueError(
            f"Unsupported analyzer type: '{analyzer_type}'. "
            f"Valid types are: {valid_types}."
        )
    
    # Get analyzer class
    analyzer_class = analyzer_classes[analyzer_type]
    
    # Set default name if not provided
    if name is None:
        name = f"{analyzer_type}_analyzer"
    
    # Create and return analyzer instance
    return analyzer_class(name=name, data_dir=data_dir, config=config) 