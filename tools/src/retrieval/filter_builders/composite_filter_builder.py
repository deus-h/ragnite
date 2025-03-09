"""
Composite Filter Builder

This module provides the CompositeFilterBuilder class for constructing complex filters
that combine different filter types (metadata, date, numeric) for vector database queries.
"""

import logging
import copy
from typing import Dict, Any, List, Optional, Union, Tuple, Set

from .base_filter_builder import BaseFilterBuilder
from .metadata_filter_builder import MetadataFilterBuilder
from .date_filter_builder import DateFilterBuilder
from .numeric_filter_builder import NumericFilterBuilder

# Configure logging
logger = logging.getLogger(__name__)


class CompositeFilterBuilder(BaseFilterBuilder):
    """
    Filter builder for constructing complex filters that combine different filter types.
    
    This builder allows for creating composite filters that combine metadata, date, and
    numeric filters with logical operators. It supports various vector database formats
    and provides a unified interface for complex filtering needs.
    """
    
    def __init__(self):
        """Initialize the composite filter builder."""
        super().__init__()
        self._conditions = []
    
    def add_filter(self, filter_builder: BaseFilterBuilder):
        """
        Add a filter builder to the composite filter.
        
        Args:
            filter_builder: Filter builder to add (MetadataFilterBuilder, DateFilterBuilder, NumericFilterBuilder, or another CompositeFilterBuilder)
            
        Returns:
            self: For method chaining
        """
        if not isinstance(filter_builder, BaseFilterBuilder):
            logger.warning(f"Invalid filter builder type: {type(filter_builder)}. Ignoring.")
            return self
            
        # Extract the filter from the builder
        if hasattr(filter_builder, "build") and callable(filter_builder.build):
            filter_dict = filter_builder.build()
            if filter_dict:  # Only add non-empty filters
                self._conditions.append({
                    "type": "filter",
                    "filter": filter_dict
                })
        
        return self
    
    def and_filters(self, *filter_builders: BaseFilterBuilder):
        """
        Combine multiple filter builders with AND logic.
        
        Args:
            *filter_builders: Filter builders to combine with AND logic
            
        Returns:
            self: For method chaining
        """
        valid_filters = []
        
        for builder in filter_builders:
            if isinstance(builder, BaseFilterBuilder):
                filter_dict = builder.build()
                if filter_dict:  # Only add non-empty filters
                    valid_filters.append(filter_dict)
        
        if valid_filters:
            self._conditions.append({
                "type": "and",
                "filters": valid_filters
            })
        
        return self
    
    def or_filters(self, *filter_builders: BaseFilterBuilder):
        """
        Combine multiple filter builders with OR logic.
        
        Args:
            *filter_builders: Filter builders to combine with OR logic
            
        Returns:
            self: For method chaining
        """
        valid_filters = []
        
        for builder in filter_builders:
            if isinstance(builder, BaseFilterBuilder):
                filter_dict = builder.build()
                if filter_dict:  # Only add non-empty filters
                    valid_filters.append(filter_dict)
        
        if valid_filters:
            self._conditions.append({
                "type": "or",
                "filters": valid_filters
            })
        
        return self
    
    def not_filter(self, filter_builder: BaseFilterBuilder):
        """
        Negate a filter builder.
        
        Args:
            filter_builder: Filter builder to negate
            
        Returns:
            self: For method chaining
        """
        if isinstance(filter_builder, BaseFilterBuilder):
            filter_dict = filter_builder.build()
            if filter_dict:  # Only add non-empty filters
                self._conditions.append({
                    "type": "not",
                    "filter": filter_dict
                })
        
        return self
    
    def metadata_filter(self):
        """
        Create and return a metadata filter builder.
        
        Returns:
            MetadataFilterBuilder: A new metadata filter builder
        """
        return MetadataFilterBuilder()
    
    def date_filter(self):
        """
        Create and return a date filter builder.
        
        Returns:
            DateFilterBuilder: A new date filter builder
        """
        return DateFilterBuilder()
    
    def numeric_filter(self):
        """
        Create and return a numeric filter builder.
        
        Returns:
            NumericFilterBuilder: A new numeric filter builder
        """
        return NumericFilterBuilder()
    
    def combine_with_and(self, other_filter: BaseFilterBuilder):
        """
        Combine this filter with another filter using AND logic.
        
        Args:
            other_filter: Filter to combine with this filter
            
        Returns:
            self: For method chaining
        """
        if isinstance(other_filter, BaseFilterBuilder):
            # Create a new composite filter with both filters
            this_filter = self.build()
            other_filter_dict = other_filter.build()
            
            if this_filter and other_filter_dict:
                # Create a new condition with both filters using AND
                self._conditions = [{
                    "type": "and",
                    "filters": [this_filter, other_filter_dict]
                }]
        
        return self
    
    def combine_with_or(self, other_filter: BaseFilterBuilder):
        """
        Combine this filter with another filter using OR logic.
        
        Args:
            other_filter: Filter to combine with this filter
            
        Returns:
            self: For method chaining
        """
        if isinstance(other_filter, BaseFilterBuilder):
            # Create a new composite filter with both filters
            this_filter = self.build()
            other_filter_dict = other_filter.build()
            
            if this_filter and other_filter_dict:
                # Create a new condition with both filters using OR
                self._conditions = [{
                    "type": "or",
                    "filters": [this_filter, other_filter_dict]
                }]
        
        return self
    
    def reset(self):
        """Reset the filter to empty state."""
        self._conditions = []
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build and return the filter in the specified format.
        
        Returns:
            Dict[str, Any]: Filter in the specified format
        """
        if not self._conditions:
            return {}
        
        # Build a generic filter structure
        generic_filter = {
            "type": "composite_filter",
            "conditions": copy.deepcopy(self._conditions)
        }
        
        # Format the filter for the target database
        return self._format_for_target(generic_filter)
    
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
        
        # Format for specific target databases
        if self._target_format == "chroma":
            return self._format_for_chroma(generic_filter)
        elif self._target_format == "qdrant":
            return self._format_for_qdrant(generic_filter)
        elif self._target_format == "pinecone":
            return self._format_for_pinecone(generic_filter)
        elif self._target_format == "weaviate":
            return self._format_for_weaviate(generic_filter)
        elif self._target_format == "milvus":
            return self._format_for_milvus(generic_filter)
        elif self._target_format == "pgvector":
            return self._format_for_pgvector(generic_filter)
        else:
            logger.warning(f"No specific formatting implemented for {self._target_format}. Using generic format.")
            return generic_filter
    
    def _format_for_chroma(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Chroma DB."""
        # Implementation will depend on Chroma's specific filter syntax
        return generic_filter
    
    def _format_for_qdrant(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Qdrant."""
        # Implementation will depend on Qdrant's specific filter syntax
        return generic_filter
    
    def _format_for_pinecone(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Pinecone."""
        # Implementation will depend on Pinecone's specific filter syntax
        return generic_filter
    
    def _format_for_weaviate(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Weaviate."""
        # Implementation will depend on Weaviate's specific filter syntax
        return generic_filter
    
    def _format_for_milvus(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Milvus."""
        # Implementation will depend on Milvus's specific filter syntax
        return generic_filter
    
    def _format_for_pgvector(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for pgvector."""
        # Implementation will depend on pgvector's specific filter syntax
        return generic_filter 