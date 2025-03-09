"""
Base Filter Builder

This module provides the BaseFilterBuilder abstract base class that all filter builders inherit from.
Filter builders are used to construct filters for vector database queries.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable

# Configure logging
logger = logging.getLogger(__name__)

class BaseFilterBuilder(ABC):
    """
    Abstract base class for filter builders.
    
    Filter builders help construct filter expressions for vector database queries.
    Different vector databases have different filter syntaxes, so filter builders
    abstract away these differences and provide a unified interface.
    """
    
    def __init__(self):
        """Initialize the filter builder."""
        self._filter = {}
        self._target_format = "generic"
    
    @property
    def filter(self) -> Dict[str, Any]:
        """Get the current filter."""
        return self._filter
    
    @property
    def target_format(self) -> str:
        """Get the target format for the filter."""
        return self._target_format
    
    @target_format.setter
    def target_format(self, value: str):
        """Set the target format for the filter."""
        supported_formats = ["generic", "chroma", "qdrant", "pinecone", "weaviate", "milvus", "pgvector"]
        if value not in supported_formats:
            logger.warning(f"Unsupported target format: {value}. Using 'generic' instead.")
            self._target_format = "generic"
        else:
            self._target_format = value
    
    def reset(self):
        """Reset the filter to empty state."""
        self._filter = {}
        return self
    
    @abstractmethod
    def build(self) -> Dict[str, Any]:
        """
        Build and return the filter in the specified format.
        
        Returns:
            Dict[str, Any]: Filter in the specified format
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the filter to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the filter
        """
        return self._filter
    
    def from_dict(self, filter_dict: Dict[str, Any]):
        """
        Load a filter from a dictionary representation.
        
        Args:
            filter_dict: Dictionary representation of the filter
        
        Returns:
            self: For method chaining
        """
        self._filter = filter_dict
        return self
    
    def _format_for_target(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a generic filter for the specified target database.
        
        Args:
            generic_filter: Generic filter representation
            
        Returns:
            Dict[str, Any]: Filter formatted for the target database
        """
        if self._target_format == "generic":
            return generic_filter
            
        # Each subclass should implement specific formatting for supported target formats
        logger.warning(f"No specific formatting implemented for {self._target_format}. Using generic format.")
        return generic_filter 