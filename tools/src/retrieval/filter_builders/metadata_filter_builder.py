"""
Metadata Filter Builder

This module provides the MetadataFilterBuilder class for constructing filters
based on document metadata fields for vector database queries.
"""

import logging
import copy
from typing import Dict, Any, List, Optional, Union, Tuple, Set

from .base_filter_builder import BaseFilterBuilder

# Configure logging
logger = logging.getLogger(__name__)

class MetadataFilterBuilder(BaseFilterBuilder):
    """
    Filter builder for constructing filters based on document metadata fields.
    
    This builder allows for creating filters that match exact values, ranges,
    lists, and more complex conditions on metadata fields. It supports
    various vector database formats and provides a unified interface.
    """
    
    def __init__(self):
        """Initialize the metadata filter builder."""
        super().__init__()
        self._conditions = []
    
    def equals(self, field: str, value: Any):
        """
        Add an equals condition to the filter.
        
        Args:
            field: Metadata field name
            value: Value to match
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "equals", "field": field, "value": value})
        return self
    
    def not_equals(self, field: str, value: Any):
        """
        Add a not equals condition to the filter.
        
        Args:
            field: Metadata field name
            value: Value to not match
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "not_equals", "field": field, "value": value})
        return self
    
    def in_list(self, field: str, values: List[Any]):
        """
        Add an in list condition to the filter.
        
        Args:
            field: Metadata field name
            values: List of values to match against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "in", "field": field, "values": values})
        return self
    
    def not_in_list(self, field: str, values: List[Any]):
        """
        Add a not in list condition to the filter.
        
        Args:
            field: Metadata field name
            values: List of values to not match against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "not_in", "field": field, "values": values})
        return self
    
    def greater_than(self, field: str, value: Union[int, float]):
        """
        Add a greater than condition to the filter.
        
        Args:
            field: Metadata field name
            value: Value to compare against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "gt", "field": field, "value": value})
        return self
    
    def greater_than_or_equal(self, field: str, value: Union[int, float]):
        """
        Add a greater than or equal condition to the filter.
        
        Args:
            field: Metadata field name
            value: Value to compare against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "gte", "field": field, "value": value})
        return self
    
    def less_than(self, field: str, value: Union[int, float]):
        """
        Add a less than condition to the filter.
        
        Args:
            field: Metadata field name
            value: Value to compare against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "lt", "field": field, "value": value})
        return self
    
    def less_than_or_equal(self, field: str, value: Union[int, float]):
        """
        Add a less than or equal condition to the filter.
        
        Args:
            field: Metadata field name
            value: Value to compare against
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "lte", "field": field, "value": value})
        return self
    
    def between(self, field: str, min_value: Union[int, float], max_value: Union[int, float], inclusive: bool = True):
        """
        Add a between condition to the filter.
        
        Args:
            field: Metadata field name
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
    
    def contains(self, field: str, value: str, case_sensitive: bool = False):
        """
        Add a contains condition to the filter for string fields.
        
        Args:
            field: Metadata field name
            value: Substring to check for
            case_sensitive: Whether the comparison is case sensitive
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "contains",
            "field": field,
            "value": value,
            "case_sensitive": case_sensitive
        })
        return self
    
    def starts_with(self, field: str, value: str, case_sensitive: bool = False):
        """
        Add a starts with condition to the filter for string fields.
        
        Args:
            field: Metadata field name
            value: Prefix to check for
            case_sensitive: Whether the comparison is case sensitive
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "starts_with",
            "field": field,
            "value": value,
            "case_sensitive": case_sensitive
        })
        return self
    
    def ends_with(self, field: str, value: str, case_sensitive: bool = False):
        """
        Add an ends with condition to the filter for string fields.
        
        Args:
            field: Metadata field name
            value: Suffix to check for
            case_sensitive: Whether the comparison is case sensitive
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({
            "type": "ends_with",
            "field": field,
            "value": value,
            "case_sensitive": case_sensitive
        })
        return self
    
    def exists(self, field: str):
        """
        Add an exists condition to the filter.
        
        Args:
            field: Metadata field name
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "exists", "field": field})
        return self
    
    def not_exists(self, field: str):
        """
        Add a not exists condition to the filter.
        
        Args:
            field: Metadata field name
            
        Returns:
            self: For method chaining
        """
        self._conditions.append({"type": "not_exists", "field": field})
        return self
    
    def and_operator(self, *filter_builders):
        """
        Combine multiple filter builders with AND logic.
        
        Args:
            *filter_builders: Filter builders to combine
            
        Returns:
            self: For method chaining
        """
        sub_conditions = []
        for builder in filter_builders:
            if isinstance(builder, BaseFilterBuilder):
                if hasattr(builder, "_conditions"):
                    # Extract conditions if it's a MetadataFilterBuilder
                    sub_conditions.extend(copy.deepcopy(builder._conditions))
                else:
                    # Otherwise just add the built filter
                    sub_conditions.append(builder.build())
            else:
                logger.warning(f"Ignoring non-FilterBuilder object in and_operator: {builder}")
        
        if sub_conditions:
            self._conditions.append({"type": "and", "conditions": sub_conditions})
        
        return self
    
    def or_operator(self, *filter_builders):
        """
        Combine multiple filter builders with OR logic.
        
        Args:
            *filter_builders: Filter builders to combine
            
        Returns:
            self: For method chaining
        """
        sub_conditions = []
        for builder in filter_builders:
            if isinstance(builder, BaseFilterBuilder):
                if hasattr(builder, "_conditions"):
                    # Extract conditions if it's a MetadataFilterBuilder
                    sub_conditions.extend(copy.deepcopy(builder._conditions))
                else:
                    # Otherwise just add the built filter
                    sub_conditions.append(builder.build())
            else:
                logger.warning(f"Ignoring non-FilterBuilder object in or_operator: {builder}")
        
        if sub_conditions:
            self._conditions.append({"type": "or", "conditions": sub_conditions})
        
        return self
    
    def not_operator(self, filter_builder):
        """
        Negate a filter builder.
        
        Args:
            filter_builder: Filter builder to negate
            
        Returns:
            self: For method chaining
        """
        if isinstance(filter_builder, BaseFilterBuilder):
            if hasattr(filter_builder, "_conditions"):
                # Extract conditions if it's a MetadataFilterBuilder
                sub_conditions = copy.deepcopy(filter_builder._conditions)
                if sub_conditions:
                    self._conditions.append({"type": "not", "conditions": sub_conditions})
            else:
                # Otherwise just add the built filter
                self._conditions.append({"type": "not", "condition": filter_builder.build()})
        else:
            logger.warning(f"Ignoring non-FilterBuilder object in not_operator: {filter_builder}")
        
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build and return the filter in the specified format.
        
        Returns:
            Dict[str, Any]: Filter in the specified format
        """
        if not self._conditions:
            return {}
        
        # Build generic filter format
        generic_filter = {"$and": [self._build_condition(cond) for cond in self._conditions]}
        
        # Format for specific target
        return self._format_for_target(generic_filter)
    
    def _build_condition(self, condition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a single condition in the generic format.
        
        Args:
            condition: Condition to build
            
        Returns:
            Dict[str, Any]: Condition in generic format
        """
        condition_type = condition.get("type")
        
        if condition_type == "equals":
            return {condition["field"]: {"$eq": condition["value"]}}
        
        elif condition_type == "not_equals":
            return {condition["field"]: {"$ne": condition["value"]}}
        
        elif condition_type == "in":
            return {condition["field"]: {"$in": condition["values"]}}
        
        elif condition_type == "not_in":
            return {condition["field"]: {"$nin": condition["values"]}}
        
        elif condition_type == "gt":
            return {condition["field"]: {"$gt": condition["value"]}}
        
        elif condition_type == "gte":
            return {condition["field"]: {"$gte": condition["value"]}}
        
        elif condition_type == "lt":
            return {condition["field"]: {"$lt": condition["value"]}}
        
        elif condition_type == "lte":
            return {condition["field"]: {"$lte": condition["value"]}}
        
        elif condition_type == "between":
            if condition["inclusive"]:
                return {
                    "$and": [
                        {condition["field"]: {"$gte": condition["min_value"]}},
                        {condition["field"]: {"$lte": condition["max_value"]}}
                    ]
                }
            else:
                return {
                    "$and": [
                        {condition["field"]: {"$gt": condition["min_value"]}},
                        {condition["field"]: {"$lt": condition["max_value"]}}
                    ]
                }
        
        elif condition_type == "contains":
            return {condition["field"]: {"$contains": condition["value"], "$case_sensitive": condition["case_sensitive"]}}
        
        elif condition_type == "starts_with":
            return {condition["field"]: {"$starts_with": condition["value"], "$case_sensitive": condition["case_sensitive"]}}
        
        elif condition_type == "ends_with":
            return {condition["field"]: {"$ends_with": condition["value"], "$case_sensitive": condition["case_sensitive"]}}
        
        elif condition_type == "exists":
            return {condition["field"]: {"$exists": True}}
        
        elif condition_type == "not_exists":
            return {condition["field"]: {"$exists": False}}
        
        elif condition_type == "and":
            sub_conditions = condition.get("conditions", [])
            return {"$and": [self._build_condition(cond) for cond in sub_conditions]}
        
        elif condition_type == "or":
            sub_conditions = condition.get("conditions", [])
            return {"$or": [self._build_condition(cond) for cond in sub_conditions]}
        
        elif condition_type == "not":
            sub_conditions = condition.get("conditions", [])
            if sub_conditions:
                return {"$not": {"$and": [self._build_condition(cond) for cond in sub_conditions]}}
            else:
                return {"$not": condition.get("condition", {})}
        
        else:
            logger.warning(f"Unknown condition type: {condition_type}")
            return {}
    
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
            
        # Format for different database systems
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
        # Chroma uses a similar format to the generic one
        return generic_filter
    
    def _format_for_qdrant(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Qdrant."""
        # Qdrant uses a different structure
        # This is a simplification - actual implementation would be more complex
        qdrant_filter = self._convert_generic_to_qdrant(generic_filter)
        return qdrant_filter
    
    def _convert_generic_to_qdrant(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert generic filter to Qdrant format."""
        # This is a simplification - actual implementation would need to handle all operators
        result = {}
        
        if "$and" in filter_dict:
            result["must"] = [self._convert_generic_to_qdrant(cond) for cond in filter_dict["$and"]]
            
        elif "$or" in filter_dict:
            result["should"] = [self._convert_generic_to_qdrant(cond) for cond in filter_dict["$or"]]
            
        elif "$not" in filter_dict:
            result["must_not"] = self._convert_generic_to_qdrant(filter_dict["$not"])
            
        else:
            # Handle leaf conditions
            for field, condition in filter_dict.items():
                if isinstance(condition, dict):
                    if "$eq" in condition:
                        result["must"] = [{"key": field, "match": {"value": condition["$eq"]}}]
                    elif "$gt" in condition:
                        result["must"] = [{"key": field, "range": {"gt": condition["$gt"]}}]
                    # ... and so on for other operators
        
        return result
    
    def _format_for_pinecone(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Pinecone."""
        # Simplified implementation
        return generic_filter
    
    def _format_for_weaviate(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Weaviate."""
        # Simplified implementation
        return generic_filter
    
    def _format_for_milvus(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for Milvus."""
        # Simplified implementation
        return generic_filter
    
    def _format_for_pgvector(self, generic_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format for pgvector."""
        # Simplified implementation - for pgvector this would likely convert to SQL WHERE clauses
        return generic_filter 