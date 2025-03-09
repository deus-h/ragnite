"""
Vector Database Schema Manager

This module provides tools for managing schemas in vector databases, including
schema validation, creation, migration, and compatibility checking.
"""

import os
import logging
import json
import copy
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class FieldType(str, Enum):
    """Enumeration of field types for schema validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    OBJECT = "object"
    ARRAY = "array"
    VECTOR = "vector"
    ANY = "any"

@dataclass
class FieldSchema:
    """Schema definition for a single field in a collection."""
    name: str
    type: FieldType
    required: bool = False
    description: Optional[str] = None
    default: Optional[Any] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    enum_values: Optional[List[Any]] = None
    array_item_type: Optional[FieldType] = None
    nested_fields: Optional[List['FieldSchema']] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "name": self.name,
            "type": self.type.value,
            "required": self.required
        }
        
        if self.description:
            result["description"] = self.description
        
        if self.default is not None:
            result["default"] = self.default
        
        if self.min_value is not None:
            result["min_value"] = self.min_value
        
        if self.max_value is not None:
            result["max_value"] = self.max_value
        
        if self.enum_values:
            result["enum_values"] = self.enum_values
        
        if self.array_item_type:
            result["array_item_type"] = self.array_item_type.value
        
        if self.nested_fields:
            result["nested_fields"] = [f.to_dict() for f in self.nested_fields]
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldSchema':
        """Create from dictionary representation."""
        nested_fields = None
        if "nested_fields" in data:
            nested_fields = [FieldSchema.from_dict(f) for f in data["nested_fields"]]
        
        return cls(
            name=data["name"],
            type=FieldType(data["type"]),
            required=data.get("required", False),
            description=data.get("description"),
            default=data.get("default"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            enum_values=data.get("enum_values"),
            array_item_type=FieldType(data["array_item_type"]) if "array_item_type" in data else None,
            nested_fields=nested_fields
        )

@dataclass
class CollectionSchema:
    """Schema definition for a vector database collection."""
    name: str
    dimension: int
    distance_metric: str = "cosine"
    metadata_fields: List[FieldSchema] = field(default_factory=list)
    description: Optional[str] = None
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
            "metadata_fields": [f.to_dict() for f in self.metadata_fields],
            "description": self.description,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CollectionSchema':
        """Create from dictionary representation."""
        metadata_fields = []
        if "metadata_fields" in data:
            metadata_fields = [FieldSchema.from_dict(f) for f in data["metadata_fields"]]
        
        return cls(
            name=data["name"],
            dimension=data["dimension"],
            distance_metric=data.get("distance_metric", "cosine"),
            metadata_fields=metadata_fields,
            description=data.get("description"),
            version=data.get("version", "1.0.0")
        )
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'CollectionSchema':
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate metadata against the schema.
        
        Args:
            metadata: Metadata to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field in self.metadata_fields:
            if field.required and (field.name not in metadata or metadata[field.name] is None):
                errors.append(f"Required field '{field.name}' is missing")
        
        # Validate field types and constraints
        for field in self.metadata_fields:
            if field.name in metadata and metadata[field.name] is not None:
                value = metadata[field.name]
                
                # Validate type
                if field.type == FieldType.STRING:
                    if not isinstance(value, str):
                        errors.append(f"Field '{field.name}' should be a string")
                
                elif field.type == FieldType.INTEGER:
                    if not isinstance(value, int) or isinstance(value, bool):
                        errors.append(f"Field '{field.name}' should be an integer")
                
                elif field.type == FieldType.FLOAT:
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        errors.append(f"Field '{field.name}' should be a number")
                
                elif field.type == FieldType.BOOLEAN:
                    if not isinstance(value, bool):
                        errors.append(f"Field '{field.name}' should be a boolean")
                
                elif field.type == FieldType.DATE:
                    # Simple check - could be enhanced with actual date parsing
                    if not isinstance(value, str):
                        errors.append(f"Field '{field.name}' should be a date string")
                
                elif field.type == FieldType.ARRAY:
                    if not isinstance(value, list):
                        errors.append(f"Field '{field.name}' should be an array")
                    elif field.array_item_type and value:
                        # Check array item types
                        for i, item in enumerate(value):
                            if field.array_item_type == FieldType.STRING and not isinstance(item, str):
                                errors.append(f"Item {i} in '{field.name}' should be a string")
                            elif field.array_item_type == FieldType.INTEGER and (not isinstance(item, int) or isinstance(item, bool)):
                                errors.append(f"Item {i} in '{field.name}' should be an integer")
                            elif field.array_item_type == FieldType.FLOAT and (not isinstance(item, (int, float)) or isinstance(item, bool)):
                                errors.append(f"Item {i} in '{field.name}' should be a number")
                            elif field.array_item_type == FieldType.BOOLEAN and not isinstance(item, bool):
                                errors.append(f"Item {i} in '{field.name}' should be a boolean")
                
                elif field.type == FieldType.OBJECT:
                    if not isinstance(value, dict):
                        errors.append(f"Field '{field.name}' should be an object")
                    elif field.nested_fields:
                        # Validate nested fields
                        is_valid, nested_errors = self._validate_nested_fields(field.nested_fields, value)
                        if not is_valid:
                            errors.extend([f"{field.name}.{err}" for err in nested_errors])
                
                elif field.type == FieldType.VECTOR:
                    if not isinstance(value, list):
                        errors.append(f"Field '{field.name}' should be a vector (array of numbers)")
                    else:
                        # Check if all items are numbers
                        for i, item in enumerate(value):
                            if not isinstance(item, (int, float)):
                                errors.append(f"Item {i} in vector '{field.name}' should be a number")
                
                # Validate constraints
                if field.enum_values is not None and value not in field.enum_values:
                    enum_str = ", ".join(str(v) for v in field.enum_values)
                    errors.append(f"Value of '{field.name}' should be one of: {enum_str}")
                
                if field.min_value is not None and isinstance(value, (int, float)):
                    if value < field.min_value:
                        errors.append(f"Value of '{field.name}' should be >= {field.min_value}")
                
                if field.max_value is not None and isinstance(value, (int, float)):
                    if value > field.max_value:
                        errors.append(f"Value of '{field.name}' should be <= {field.max_value}")
        
        return len(errors) == 0, errors
    
    def _validate_nested_fields(
        self, 
        nested_fields: List[FieldSchema], 
        data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate nested fields in an object.
        
        Args:
            nested_fields: List of field schemas for nested fields
            data: Object data to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        for field in nested_fields:
            if field.required and (field.name not in data or data[field.name] is None):
                errors.append(f"Required field '{field.name}' is missing")
        
        # Validate field types and constraints
        for field in nested_fields:
            if field.name in data and data[field.name] is not None:
                value = data[field.name]
                
                # Validate type (simplified for brevity - would repeat the logic from validate_metadata)
                if field.type == FieldType.STRING and not isinstance(value, str):
                    errors.append(f"Field '{field.name}' should be a string")
                elif field.type == FieldType.INTEGER and (not isinstance(value, int) or isinstance(value, bool)):
                    errors.append(f"Field '{field.name}' should be an integer")
                elif field.type == FieldType.FLOAT and (not isinstance(value, (int, float)) or isinstance(value, bool)):
                    errors.append(f"Field '{field.name}' should be a number")
                elif field.type == FieldType.BOOLEAN and not isinstance(value, bool):
                    errors.append(f"Field '{field.name}' should be a boolean")
                elif field.type == FieldType.OBJECT and not isinstance(value, dict):
                    errors.append(f"Field '{field.name}' should be an object")
                elif field.type == FieldType.ARRAY and not isinstance(value, list):
                    errors.append(f"Field '{field.name}' should be an array")
                
                # Handle nested objects recursively
                if field.type == FieldType.OBJECT and field.nested_fields and isinstance(value, dict):
                    is_valid, nested_errors = self._validate_nested_fields(field.nested_fields, value)
                    if not is_valid:
                        errors.extend([f"{field.name}.{err}" for err in nested_errors])
        
        return len(errors) == 0, errors
    
    def add_field(self, field: FieldSchema) -> 'CollectionSchema':
        """
        Add a field to the schema.
        
        Args:
            field: Field schema to add
            
        Returns:
            CollectionSchema: Updated schema (self)
        """
        for existing_field in self.metadata_fields:
            if existing_field.name == field.name:
                # Update existing field
                self.metadata_fields.remove(existing_field)
                break
        
        self.metadata_fields.append(field)
        return self
    
    def remove_field(self, field_name: str) -> 'CollectionSchema':
        """
        Remove a field from the schema.
        
        Args:
            field_name: Name of the field to remove
            
        Returns:
            CollectionSchema: Updated schema (self)
        """
        self.metadata_fields = [f for f in self.metadata_fields if f.name != field_name]
        return self
    
    def get_field(self, field_name: str) -> Optional[FieldSchema]:
        """
        Get a field schema by name.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Optional[FieldSchema]: Field schema if found, None otherwise
        """
        for field in self.metadata_fields:
            if field.name == field_name:
                return field
        return None
    
    def is_compatible_with(self, other: 'CollectionSchema') -> Tuple[bool, List[str]]:
        """
        Check if this schema is compatible with another schema.
        
        Args:
            other: Other schema to check compatibility with
            
        Returns:
            Tuple[bool, List[str]]: (is_compatible, incompatibility_reasons)
        """
        incompatibilities = []
        
        # Check dimension
        if self.dimension != other.dimension:
            incompatibilities.append(f"Dimension mismatch: {self.dimension} vs {other.dimension}")
        
        # Check distance metric
        if self.distance_metric != other.distance_metric:
            incompatibilities.append(f"Distance metric mismatch: {self.distance_metric} vs {other.distance_metric}")
        
        # Check required fields
        our_required_fields = {f.name: f for f in self.metadata_fields if f.required}
        their_required_fields = {f.name: f for f in other.metadata_fields if f.required}
        
        # Check if all our required fields exist in the other schema
        for name, field in our_required_fields.items():
            if name not in their_required_fields:
                incompatibilities.append(f"Required field '{name}' missing in the other schema")
            else:
                # Check type compatibility
                other_field = their_required_fields[name]
                if field.type != other_field.type:
                    incompatibilities.append(f"Field '{name}' type mismatch: {field.type.value} vs {other_field.type.value}")
        
        return len(incompatibilities) == 0, incompatibilities

@dataclass
class ValidationResult:
    """Result of a schema validation operation."""
    is_valid: bool
    errors: List[str]
    schema_name: str
    document_count: int = 0
    
    def __str__(self) -> str:
        """String representation of validation result."""
        if self.is_valid:
            return f"Schema '{self.schema_name}' validated successfully. {self.document_count} documents checked."
        else:
            error_list = "\n  - ".join(self.errors)
            return f"Schema '{self.schema_name}' validation failed with {len(self.errors)} errors:\n  - {error_list}"

class BaseSchemaManager(ABC):
    """
    Base abstract class for vector database schema managers.
    
    Schema managers provide tools to manage schemas in vector databases,
    including validation, creation, migration, and compatibility checking.
    """
    
    def __init__(
        self,
        db_connector: Any,
        schema_registry: Optional[Dict[str, CollectionSchema]] = None
    ):
        """
        Initialize BaseSchemaManager.
        
        Args:
            db_connector: Vector database connector instance
            schema_registry: Optional dictionary of known schemas
        """
        self.db_connector = db_connector
        self.schema_registry = schema_registry or {}
    
    @abstractmethod
    def get_collection_schema(self, collection_name: str) -> CollectionSchema:
        """
        Get the schema for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionSchema: Collection schema
            
        Raises:
            ValueError: If the collection does not exist or schema cannot be determined
        """
        pass
    
    @abstractmethod
    def apply_schema(
        self, 
        schema: CollectionSchema,
        create_if_not_exists: bool = True
    ) -> bool:
        """
        Apply a schema to a collection.
        
        Args:
            schema: Schema to apply
            create_if_not_exists: Whether to create the collection if it does not exist
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_collection(
        self,
        collection_name: str,
        schema: Optional[CollectionSchema] = None,
        sample_size: int = 100
    ) -> ValidationResult:
        """
        Validate a collection against a schema.
        
        Args:
            collection_name: Name of the collection to validate
            schema: Schema to validate against (if None, use registered schema)
            sample_size: Number of documents to sample for validation
            
        Returns:
            ValidationResult: Validation result
        """
        pass
    
    def register_schema(self, schema: CollectionSchema) -> None:
        """
        Register a schema in the schema registry.
        
        Args:
            schema: Schema to register
        """
        self.schema_registry[schema.name] = schema
    
    def get_registered_schema(self, collection_name: str) -> Optional[CollectionSchema]:
        """
        Get a registered schema by collection name.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Optional[CollectionSchema]: Registered schema if found, None otherwise
        """
        return self.schema_registry.get(collection_name)
    
    def is_collection_compatible(
        self,
        collection_name: str,
        schema: CollectionSchema
    ) -> Tuple[bool, List[str]]:
        """
        Check if a collection is compatible with a schema.
        
        Args:
            collection_name: Name of the collection
            schema: Schema to check compatibility with
            
        Returns:
            Tuple[bool, List[str]]: (is_compatible, incompatibility_reasons)
        """
        try:
            current_schema = self.get_collection_schema(collection_name)
            return schema.is_compatible_with(current_schema)
        except ValueError:
            return False, ["Collection does not exist"]
    
    def _check_db_connector(self):
        """
        Check if the database connector is provided and connected.
        
        Raises:
            ConnectionError: If the database connector is not connected
        """
        if self.db_connector is None:
            raise ConnectionError("Database connector not provided.")
        
        if not hasattr(self.db_connector, 'is_connected') or not self.db_connector.is_connected():
            raise ConnectionError("Database connector is not connected.")

class SchemaManager(BaseSchemaManager):
    """
    Generic schema manager for vector databases.
    
    This manager provides a common implementation that works with
    most vector databases through the common connector interface.
    """
    
    def get_collection_schema(self, collection_name: str) -> CollectionSchema:
        """
        Get the schema for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            CollectionSchema: Collection schema
            
        Raises:
            ValueError: If the collection does not exist or schema cannot be determined
        """
        self._check_db_connector()
        
        # Check if collection exists
        if not self.db_connector.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        # Check if we have a registered schema
        if collection_name in self.schema_registry:
            return self.schema_registry[collection_name]
        
        # Try to derive schema from collection info
        collection_info = self.db_connector.get_collection_info(collection_name)
        
        # Extract basic info
        dimension = collection_info.get("dimension")
        if not dimension:
            raise ValueError(f"Could not determine dimension for collection '{collection_name}'")
        
        distance_metric = collection_info.get("distance_metric", "cosine")
        
        # Create a basic schema without metadata fields
        # We can't reliably infer metadata schema without examining documents
        schema = CollectionSchema(
            name=collection_name,
            dimension=dimension,
            distance_metric=distance_metric,
            description=f"Auto-generated schema for {collection_name}"
        )
        
        # Try to infer metadata fields if the connector supports it
        if hasattr(self.db_connector, "get_collection_metadata_schema"):
            try:
                metadata_fields = self.db_connector.get_collection_metadata_schema(collection_name)
                if metadata_fields:
                    schema.metadata_fields = metadata_fields
            except:
                pass
        
        # Register the schema for future use
        self.register_schema(schema)
        
        return schema
    
    def apply_schema(
        self, 
        schema: CollectionSchema,
        create_if_not_exists: bool = True
    ) -> bool:
        """
        Apply a schema to a collection.
        
        Args:
            schema: Schema to apply
            create_if_not_exists: Whether to create the collection if it does not exist
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._check_db_connector()
        
        try:
            collection_name = schema.name
            
            # Check if collection exists
            collection_exists = self.db_connector.collection_exists(collection_name)
            
            if not collection_exists:
                if create_if_not_exists:
                    # Create new collection
                    logger.info(f"Creating collection '{collection_name}' with schema")
                    self.db_connector.create_collection(
                        collection_name=collection_name,
                        dimension=schema.dimension,
                        distance_metric=schema.distance_metric,
                        metadata={"schema_version": schema.version}
                    )
                else:
                    logger.error(f"Collection '{collection_name}' does not exist and create_if_not_exists=False")
                    return False
            else:
                # Check compatibility with existing collection
                is_compatible, reasons = self.is_collection_compatible(collection_name, schema)
                if not is_compatible:
                    reasons_str = "\n  - ".join(reasons)
                    logger.error(f"Schema is not compatible with existing collection:\n  - {reasons_str}")
                    return False
            
            # Register the schema
            self.register_schema(schema)
            
            return True
        
        except Exception as e:
            logger.error(f"Error applying schema: {str(e)}")
            return False
    
    def validate_collection(
        self,
        collection_name: str,
        schema: Optional[CollectionSchema] = None,
        sample_size: int = 100
    ) -> ValidationResult:
        """
        Validate a collection against a schema.
        
        Args:
            collection_name: Name of the collection to validate
            schema: Schema to validate against (if None, use registered schema)
            sample_size: Number of documents to sample for validation
            
        Returns:
            ValidationResult: Validation result
        """
        self._check_db_connector()
        
        # Get schema to validate against
        if schema is None:
            try:
                schema = self.get_collection_schema(collection_name)
            except ValueError as e:
                return ValidationResult(
                    is_valid=False,
                    errors=[str(e)],
                    schema_name=collection_name
                )
        
        try:
            # Check if collection exists
            if not self.db_connector.collection_exists(collection_name):
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Collection '{collection_name}' does not exist"],
                    schema_name=schema.name
                )
            
            # Get document count
            doc_count = self.db_connector.count_vectors(collection_name)
            
            if doc_count == 0:
                return ValidationResult(
                    is_valid=True,
                    errors=[],
                    schema_name=schema.name,
                    document_count=0
                )
            
            # Sample documents for validation
            sample_size = min(sample_size, doc_count)
            all_errors = []
            validated_count = 0
            
            # Try bulk sampling if supported
            if hasattr(self.db_connector, "get_collection_vectors_batch"):
                try:
                    batch_vectors, batch_ids, batch_metadata, batch_documents = self.db_connector.get_collection_vectors_batch(
                        collection_name=collection_name,
                        offset=0,
                        limit=sample_size
                    )
                    
                    validated_count = len(batch_ids)
                    
                    # Validate each document
                    for i, metadata in enumerate(batch_metadata):
                        if metadata:
                            is_valid, errors = schema.validate_metadata(metadata)
                            if not is_valid:
                                for error in errors:
                                    all_errors.append(f"Document {batch_ids[i]}: {error}")
                
                except Exception as e:
                    logger.warning(f"Bulk sampling failed, falling back to individual validation: {str(e)}")
                    # Fall through to individual validation
            
            # Fall back to individual sampling
            if validated_count == 0 and hasattr(self.db_connector, "get_all_vector_ids"):
                try:
                    all_ids = self.db_connector.get_all_vector_ids(collection_name)
                    
                    # Sample randomly or take the first N
                    import random
                    if len(all_ids) <= sample_size:
                        sample_ids = all_ids
                    else:
                        sample_ids = random.sample(all_ids, sample_size)
                    
                    # Validate each document
                    for id_str in sample_ids:
                        vector, metadata = self.db_connector.get_vector(collection_name, id_str)
                        validated_count += 1
                        
                        if metadata:
                            is_valid, errors = schema.validate_metadata(metadata)
                            if not is_valid:
                                for error in errors:
                                    all_errors.append(f"Document {id_str}: {error}")
                
                except Exception as e:
                    logger.error(f"Error during validation: {str(e)}")
                    all_errors.append(f"Validation error: {str(e)}")
            
            return ValidationResult(
                is_valid=len(all_errors) == 0,
                errors=all_errors,
                schema_name=schema.name,
                document_count=validated_count
            )
        
        except Exception as e:
            logger.error(f"Error validating collection: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                schema_name=schema.name
            )
    
    def create_index_for_field(
        self,
        collection_name: str,
        field_name: str,
        index_type: str = "default"
    ) -> bool:
        """
        Create an index for a field in a collection.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field to index
            index_type: Type of index to create
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._check_db_connector()
        
        try:
            # Check if the connector supports creating indexes
            if hasattr(self.db_connector, "create_metadata_index"):
                return self.db_connector.create_metadata_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    index_type=index_type
                )
            else:
                logger.warning(f"Database connector does not support creating metadata indexes")
                return False
        
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False
    
    def get_database_supported_field_types(self) -> Dict[str, List[FieldType]]:
        """
        Get the field types supported by the database.
        
        Returns:
            Dict[str, List[FieldType]]: Map of collection types to supported field types
        """
        # Generic implementation - override for specific databases
        return {
            "metadata": [
                FieldType.STRING,
                FieldType.INTEGER,
                FieldType.FLOAT,
                FieldType.BOOLEAN,
                FieldType.DATE,
                FieldType.OBJECT,
                FieldType.ARRAY,
                FieldType.ANY
            ],
            "vector": [
                FieldType.VECTOR
            ]
        }
    
    def export_schema(self, collection_name: str, file_path: str) -> bool:
        """
        Export a collection schema to a file.
        
        Args:
            collection_name: Name of the collection
            file_path: Path to export the schema to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            schema = self.get_collection_schema(collection_name)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Write schema to file
            with open(file_path, 'w') as f:
                f.write(schema.to_json())
            
            logger.info(f"Schema for collection '{collection_name}' exported to '{file_path}'")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting schema: {str(e)}")
            return False
    
    def import_schema(self, file_path: str, apply: bool = False) -> Optional[CollectionSchema]:
        """
        Import a collection schema from a file.
        
        Args:
            file_path: Path to import the schema from
            apply: Whether to apply the schema to the collection
            
        Returns:
            Optional[CollectionSchema]: Imported schema if successful, None otherwise
        """
        try:
            # Read schema from file
            with open(file_path, 'r') as f:
                schema_json = f.read()
            
            # Parse schema
            schema = CollectionSchema.from_json(schema_json)
            
            # Register schema
            self.register_schema(schema)
            
            # Apply schema if requested
            if apply:
                self.apply_schema(schema)
            
            logger.info(f"Schema imported from '{file_path}'")
            return schema
        
        except Exception as e:
            logger.error(f"Error importing schema: {str(e)}")
            return None

def get_schema_manager(
    db_connector: Any,
    schema_registry: Optional[Dict[str, CollectionSchema]] = None
) -> BaseSchemaManager:
    """
    Get a schema manager for a vector database.
    
    Args:
        db_connector: Vector database connector instance
        schema_registry: Optional dictionary of known schemas
        
    Returns:
        BaseSchemaManager: Schema manager
    """
    return SchemaManager(db_connector, schema_registry)

def validate_schema(
    db_connector: Any,
    collection_name: str,
    schema: Optional[CollectionSchema] = None,
    sample_size: int = 100
) -> ValidationResult:
    """
    Validate a collection against a schema.
    
    Args:
        db_connector: Vector database connector instance
        collection_name: Name of the collection to validate
        schema: Schema to validate against (if None, try to derive from collection)
        sample_size: Number of documents to sample for validation
        
    Returns:
        ValidationResult: Validation result
    """
    manager = get_schema_manager(db_connector)
    return manager.validate_collection(collection_name, schema, sample_size) 