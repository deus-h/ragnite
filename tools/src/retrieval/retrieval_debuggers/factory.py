"""
Retrieval Debugger Factory

This module provides a factory function to get a retrieval debugger based on the debugger type.
"""

import logging
from typing import Dict, Any, Optional, Union, Callable

from .base_debugger import BaseRetrievalDebugger
from .retrieval_inspector import RetrievalInspector
from .query_analyzer import QueryAnalyzer
from .context_analyzer import ContextAnalyzer
from .retrieval_visualizer import RetrievalVisualizer

# Configure logging
logger = logging.getLogger(__name__)

def get_retrieval_debugger(
    debugger_type: str,
    **kwargs
) -> BaseRetrievalDebugger:
    """
    Get a retrieval debugger based on the debugger type.
    
    Args:
        debugger_type: Type of retrieval debugger ("inspector", "query_analyzer", "context_analyzer", "visualizer")
        **kwargs: Additional parameters to pass to the debugger constructor
        
    Returns:
        BaseRetrievalDebugger: Retrieval debugger
        
    Raises:
        ValueError: If the debugger type is not supported
    """
    # Normalize debugger type
    debugger_type = debugger_type.lower().strip()
    
    # Create debugger based on type
    if debugger_type in ["inspector", "retrieval_inspector", "retrievalinspector"]:
        debugger = RetrievalInspector(**kwargs)
    
    elif debugger_type in ["query_analyzer", "queryanalyzer"]:
        debugger = QueryAnalyzer(**kwargs)
    
    elif debugger_type in ["context_analyzer", "contextanalyzer"]:
        debugger = ContextAnalyzer(**kwargs)
    
    elif debugger_type in ["visualizer", "retrieval_visualizer", "retrievalvisualizer"]:
        debugger = RetrievalVisualizer(**kwargs)
    
    # Note: Other debuggers will be added here as they are implemented
    
    else:
        supported_debuggers = ["inspector", "query_analyzer", "context_analyzer", "visualizer"]
        raise ValueError(f"Unsupported debugger type: {debugger_type}. Supported types: {supported_debuggers}")
    
    return debugger 