"""
Query Processor Factory

This module provides a factory function to get a query processor based on the processor type.
"""

import logging
from typing import Dict, Any, Optional, Union

from .base_processor import BaseQueryProcessor
from .query_expander import QueryExpander
from .query_rewriter import QueryRewriter
from .query_decomposer import QueryDecomposer
from .query_translator import QueryTranslator

# Configure logging
logger = logging.getLogger(__name__)

def get_query_processor(
    processor_type: str,
    **kwargs
) -> BaseQueryProcessor:
    """
    Get a query processor based on the processor type.
    
    Args:
        processor_type: Type of query processor ("expander", "rewriter", "decomposer", "translator")
        **kwargs: Additional parameters to pass to the processor constructor
        
    Returns:
        BaseQueryProcessor: Query processor
        
    Raises:
        ValueError: If the processor type is not supported
    """
    # Normalize processor type
    processor_type = processor_type.lower().strip()
    
    # Create processor based on type
    if processor_type in ["expander", "query_expander", "queryexpander"]:
        processor = QueryExpander(**kwargs)
    
    elif processor_type in ["rewriter", "query_rewriter", "queryrewriter"]:
        processor = QueryRewriter(**kwargs)
    
    elif processor_type in ["decomposer", "query_decomposer", "querydecomposer"]:
        processor = QueryDecomposer(**kwargs)
    
    elif processor_type in ["translator", "query_translator", "querytranslator"]:
        processor = QueryTranslator(**kwargs)
    
    else:
        supported_processors = ["expander", "rewriter", "decomposer", "translator"]
        raise ValueError(f"Unsupported processor type: {processor_type}. Supported types: {supported_processors}")
    
    return processor 