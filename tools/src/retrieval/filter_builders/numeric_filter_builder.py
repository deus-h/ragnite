"""
Numeric Filter Builder

This module provides the NumericFilterBuilder class for constructing filters
based on numeric fields for vector database queries.
"""

import logging
import copy
import math
from typing import Dict, Any, List, Optional, Union, Tuple, Set

from .base_filter_builder import BaseFilterBuilder

# Configure logging
logger = logging.getLogger(__name__)


class NumericFilterBuilder(BaseFilterBuilder):
    """
    Filter builder for constructing filters based on numeric fields.
    
    This builder allows for creating filters that match specific numeric values,
    ranges, and more complex numeric conditions. It supports various vector
    database formats and provides a unified interface.
    """
    
    def __init__(self):
        """Initialize the numeric filter builder."""
        super().__init__()
        self._conditions = []
    
    def equals(self, field: str, value: Union[int, float]):
        """
        Add an equals condition to the filter.
        
        Args:
            field: Numeric field name
            value: Value to match
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "equals", 
            "field": field, 
            "value": value
        })
        return self
    
    def not_equals(self, field: str, value: Union[int, float]):
        """
        Add a not equals condition to the filter.
        
        Args:
            field: Numeric field name
            value: Value to not match
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "not_equals", 
            "field": field, 
            "value": value
        })
        return self
    
    def greater_than(self, field: str, value: Union[int, float]):
        """
        Add a greater than condition to the filter.
        
        Args:
            field: Numeric field name
            value: Value to compare against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "gt", 
            "field": field, 
            "value": value
        })
        return self
    
    def greater_than_or_equal(self, field: str, value: Union[int, float]):
        """
        Add a greater than or equal condition to the filter.
        
        Args:
            field: Numeric field name
            value: Value to compare against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "gte", 
            "field": field, 
            "value": value
        })
        return self
    
    def less_than(self, field: str, value: Union[int, float]):
        """
        Add a less than condition to the filter.
        
        Args:
            field: Numeric field name
            value: Value to compare against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "lt", 
            "field": field, 
            "value": value
        })
        return self
    
    def less_than_or_equal(self, field: str, value: Union[int, float]):
        """
        Add a less than or equal condition to the filter.
        
        Args:
            field: Numeric field name
            value: Value to compare against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "lte", 
            "field": field, 
            "value": value
        })
        return self
    
    def between(self, field: str, min_value: Union[int, float], max_value: Union[int, float], inclusive: bool = True):
        """
        Add a between condition to the filter.
        
        Args:
            field: Numeric field name
            min_value: Minimum value
            max_value: Maximum value
            inclusive: Whether the range is inclusive or exclusive
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "between",
            "field": field,
            "min_value": min_value,
            "max_value": max_value,
            "inclusive": inclusive
        })
        return self
    
    def not_between(self, field: str, min_value: Union[int, float], max_value: Union[int, float], inclusive: bool = True):
        """
        Add a not between condition to the filter.
        
        Args:
            field: Numeric field name
            min_value: Minimum value
            max_value: Maximum value
            inclusive: Whether the range is inclusive or exclusive
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "not_between",
            "field": field,
            "min_value": min_value,
            "max_value": max_value,
            "inclusive": inclusive
        })
        return self
    
    def in_list(self, field: str, values: List[Union[int, float]]):
        """
        Add an in list condition to the filter.
        
        Args:
            field: Numeric field name
            values: List of values to match against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "in", 
            "field": field, 
            "values": values
        })
        return self
    
    def not_in_list(self, field: str, values: List[Union[int, float]]):
        """
        Add a not in list condition to the filter.
        
        Args:
            field: Numeric field name
            values: List of values to not match against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "not_in", 
            "field": field, 
            "values": values
        })
        return self
    
    def is_integer(self, field: str):
        """
        Add a condition to filter for integer values.
        
        Args:
            field: Numeric field name
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "is_integer",
            "field": field
        })
        return self
    
    def is_decimal(self, field: str):
        """
        Add a condition to filter for decimal values.
        
        Args:
            field: Numeric field name
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "is_decimal",
            "field": field
        })
        return self
    
    def divisible_by(self, field: str, divisor: Union[int, float]):
        """
        Add a condition to filter for values divisible by the given divisor.
        
        Args:
            field: Numeric field name
            divisor: Divisor to check against
            
        Returns:
            self: For method chaining
        """
        if divisor == 0:
            logger.warning("Cannot filter for divisibility by zero. Ignoring condition.")
            return self
            
        self._conditions.append({
            "type": "divisible_by",
            "field": field,
            "divisor": divisor
        })
        return self
    
    def is_positive(self, field: str, include_zero: bool = False):
        """
        Add a condition to filter for positive values.
        
        Args:
            field: Numeric field name
            include_zero: Whether to include zero as a positive value
            
        Returns:
            self: For method chaining
        """
        if include_zero:
            self._conditions.append({
                "type": "gte",
                "field": field,
                "value": 0
            })
        else:
            self._conditions.append({
                "type": "gt",
                "field": field,
                "value": 0
            })
        return self
    
    def is_negative(self, field: str, include_zero: bool = False):
        """
        Add a condition to filter for negative values.
        
        Args:
            field: Numeric field name
            include_zero: Whether to include zero as a negative value
            
        Returns:
            self: For method chaining
        """
        if include_zero:
            self._conditions.append({
                "type": "lte",
                "field": field,
                "value": 0
            })
        else:
            self._conditions.append({
                "type": "lt",
                "field": field,
                "value": 0
            })
        return self
    
    def near(self, field: str, target: Union[int, float], tolerance: Union[int, float]):
        """
        Add a condition to filter for values near the target within the given tolerance.
        
        Args:
            field: Numeric field name
            target: Target value
            tolerance: Tolerance range (target ± tolerance)
            
        Returns:
            self: For method chaining
        """
        min_value = target - tolerance
        max_value = target + tolerance
        
        self._conditions.append({
            "type": "between",
            "field": field,
            "min_value": min_value,
            "max_value": max_value,
            "inclusive": True
        })
        return self
    
    def exists(self, field: str):
        """
        Add an exists condition to the filter.
        
        Args:
            field: Numeric field name
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "exists", 
            "field": field
        })
        return self
    
    def not_exists(self, field: str):
        """
        Add a not exists condition to the filter.
        
        Args:
            field: Numeric field name
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "not_exists", 
            "field": field
        })
        return self
    
    def and_operator(self, *filter_builders):
        """
        Combine multiple filter builders with AND logic.
        
        Args:
            *filter_builders: Filter builders to combine
            
        Returns:
            self: For method chaining
        """
        combined_conditions = []
        for builder in filter_builders:
            if isinstance(builder, NumericFilterBuilder):
                combined_conditions.extend(builder._conditions)
        
        if combined_conditions:
            self._conditions.append({
                "type": "and",
                "conditions": combined_conditions
            })
        
        return self
    
    def or_operator(self, *filter_builders):
        """
        Combine multiple filter builders with OR logic.
        
        Args:
            *filter_builders: Filter builders to combine
            
        Returns:
            self: For method chaining
        """
        combined_conditions = []
        for builder in filter_builders:
            if isinstance(builder, NumericFilterBuilder):
                combined_conditions.extend(builder._conditions)
        
        if combined_conditions:
            self._conditions.append({
                "type": "or",
                "conditions": combined_conditions
            })
        
        return self
    
    def not_operator(self, filter_builder):
        """
        Negate a filter builder.
        
        Args:
            filter_builder: Filter builder to negate
            
        Returns:
            self: For method chaining
        """
        if isinstance(filter_builder, NumericFilterBuilder) and filter_builder._conditions:
            self._conditions.append({
                "type": "not",
                "condition": filter_builder._conditions
            })
        
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
            "type": "numeric_filter",
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