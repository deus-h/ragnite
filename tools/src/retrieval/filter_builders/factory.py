"""
Filter Builder Factory

This module provides a factory function to get a filter builder based on the builder type.
"""

import logging
from typing import Dict, Any, Optional, Union

from .base_filter_builder import BaseFilterBuilder
from .metadata_filter_builder import MetadataFilterBuilder
from .date_filter_builder import DateFilterBuilder
from .numeric_filter_builder import NumericFilterBuilder
from .composite_filter_builder import CompositeFilterBuilder

# Configure logging
logger = logging.getLogger(__name__)

def get_filter_builder(
    builder_type: str,
    **kwargs
) -> BaseFilterBuilder:
    """
    Get a filter builder based on the builder type.
    
    Args:
        builder_type: Type of filter builder ("metadata", "date", "numeric", "composite")
        **kwargs: Additional parameters to pass to the builder constructor
        
    Returns:
        BaseFilterBuilder: Filter builder
        
    Raises:
        ValueError: If the builder type is not supported
    """
    # Normalize builder type
    builder_type = builder_type.lower().strip()
    
    # Create builder based on type
    if builder_type in ["metadata", "metadata_filter", "metadatafilter"]:
        builder = MetadataFilterBuilder()
    
    elif builder_type in ["date", "date_filter", "datefilter"]:
        builder = DateFilterBuilder()
    
    elif builder_type in ["numeric", "numeric_filter", "numericfilter", "number", "number_filter", "numberfilter"]:
        builder = NumericFilterBuilder()
    
    elif builder_type in ["composite", "composite_filter", "compositefilter", "complex", "complex_filter", "complexfilter"]:
        builder = CompositeFilterBuilder()
    
    else:
        supported_builders = ["metadata", "date", "numeric", "composite"]
        raise ValueError(f"Unsupported builder type: {builder_type}. Supported types: {supported_builders}")
    
    # Set target format if specified
    if "target_format" in kwargs:
        builder.target_format = kwargs["target_format"]
    
    return builder 