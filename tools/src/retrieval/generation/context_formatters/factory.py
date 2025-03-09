"""
Context Formatter Factory

This module provides a factory function for creating context formatters.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable

from .base_context_formatter import BaseContextFormatter
from .basic_context_formatter import BasicContextFormatter
from .metadata_enriched_formatter import MetadataEnrichedFormatter
from .source_attribution_formatter import SourceAttributionFormatter
from .hierarchical_context_formatter import HierarchicalContextFormatter

# Configure logging
logger = logging.getLogger(__name__)


def get_context_formatter(
    formatter_type: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseContextFormatter:
    """
    Factory function to create context formatters.
    
    Args:
        formatter_type: Type of context formatter ('basic', 'metadata', 'source_attribution', 'hierarchical').
        config: Configuration options for the context formatter.
        **kwargs: Additional arguments passed to the specific context formatter.
    
    Returns:
        BaseContextFormatter: An instance of the requested context formatter.
    
    Raises:
        ValueError: If the formatter_type is not supported.
    """
    formatter_type = formatter_type.lower()
    config = config or {}
    
    # Update config with kwargs
    for key, value in kwargs.items():
        config[key] = value
    
    if formatter_type == "basic":
        return BasicContextFormatter(config=config)
    
    elif formatter_type == "metadata" or formatter_type == "metadata_enriched":
        return MetadataEnrichedFormatter(config=config)
    
    elif formatter_type == "source_attribution" or formatter_type == "attribution":
        return SourceAttributionFormatter(config=config)
    
    elif formatter_type == "hierarchical":
        return HierarchicalContextFormatter(config=config)
    
    else:
        logger.error(f"Unsupported context formatter type: {formatter_type}")
        raise ValueError(f"Unsupported context formatter type: {formatter_type}. "
                        f"Supported types: basic, metadata, source_attribution, hierarchical") 