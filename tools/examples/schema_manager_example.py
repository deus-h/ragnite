#!/usr/bin/env python
"""
Schema Manager Example

This script demonstrates how to use the schema manager tools in the
RAG Utility Tools package to manage and validate schemas for vector database collections.
"""

import os
import sys
import logging
import argparse
import tempfile
from typing import List, Dict, Any, Optional

# Add the parent directory to the path to import the package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vector_db import (
    get_database_connector,
    BaseVectorDBConnector
)

from src.vector_db.schema_manager import (
    FieldType,
    FieldSchema,
    CollectionSchema,
    ValidationResult,
    get_schema_manager,
    validate_schema
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_example_schema() -> CollectionSchema:
    """
    Create an example schema for demonstration purposes.
    
    Returns:
        CollectionSchema: Example schema
    """
    # Define fields
    fields = [
        FieldSchema(
            name="title",
            type=FieldType.STRING,
            required=True,
            description="Document title"
        ),
        FieldSchema(
            name="content",
            type=FieldType.STRING,
            description="Document content"
        ),
        FieldSchema(
            name="created_at",
            type=FieldType.DATE,
            description="Document creation date"
        ),
        FieldSchema(
            name="page_count",
            type=FieldType.INTEGER,
            min_value=1,
            max_value=1000,
            description="Number of pages in the document"
        ),
        FieldSchema(
            name="tags",
            type=FieldType.ARRAY,
            array_item_type=FieldType.STRING,
            description="Document tags"
        ),
        FieldSchema(
            name="status",
            type=FieldType.STRING,
            enum_values=["draft", "published", "archived"],
            description="Document status"
        ),
        FieldSchema(
            name="metadata",
            type=FieldType.OBJECT,
            description="Additional metadata",
            nested_fields=[
                FieldSchema(
                    name="author",
                    type=FieldType.STRING,
                    description="Document author"
                ),
                FieldSchema(
                    name="version",
                    type=FieldType.FLOAT,
                    description="Document version"
                )
            ]
        )
    ]
    
    # Create schema
    schema = CollectionSchema(
        name="documents",
        dimension=384,  # Example dimension for text embeddings
        distance_metric="cosine",
        metadata_fields=fields,
        description="Schema for document embeddings",
        version="1.0.0"
    )
    
    return schema

def setup_test_collection(
    connector: BaseVectorDBConnector,
    collection_name: str,
    dimension: int = 384
) -> bool:
    """
    Set up a test collection for schema validation.
    
    Args:
        connector: Vector database connector
        collection_name: Name of the collection to create
        dimension: Dimensionality of vectors
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Create collection
        logger.info(f"Creating test collection '{collection_name}'")
        connector.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            distance_metric="cosine",
            metadata={"description": "Test collection for schema validation"}
        )
        
        # Generate random vectors
        import numpy as np
        vector1 = np.random.rand(dimension).astype(np.float32).tolist()
        vector2 = np.random.rand(dimension).astype(np.float32).tolist()
        vector3 = np.random.rand(dimension).astype(np.float32).tolist()
        
        # Add vectors with valid metadata
        logger.info(f"Adding valid documents to '{collection_name}'")
        
        # Valid document 1
        connector.add_vectors(
            collection_name=collection_name,
            vectors=[vector1],
            ids=["doc1"],
            metadata=[{
                "title": "Example Document 1",
                "content": "This is an example document with valid metadata.",
                "created_at": "2023-01-01T12:00:00Z",
                "page_count": 10,
                "tags": ["example", "valid"],
                "status": "published",
                "metadata": {
                    "author": "Alice",
                    "version": 1.0
                }
            }]
        )
        
        # Add vectors with invalid metadata
        logger.info(f"Adding invalid documents to '{collection_name}'")
        
        # Invalid document 2 (missing required title)
        connector.add_vectors(
            collection_name=collection_name,
            vectors=[vector2],
            ids=["doc2"],
            metadata=[{
                "content": "This document is missing the required title field.",
                "page_count": 5,
                "status": "draft"
            }]
        )
        
        # Invalid document 3 (invalid page count and status)
        connector.add_vectors(
            collection_name=collection_name,
            vectors=[vector3],
            ids=["doc3"],
            metadata=[{
                "title": "Example Document 3",
                "page_count": 0,  # Below min_value
                "status": "pending",  # Not in enum_values
                "tags": ["example", 123]  # Invalid tag type
            }]
        )
        
        return True
    
    except Exception as e:
        logger.error(f"Error setting up test collection: {str(e)}")
        return False

def example_schema_creation_and_serialization():
    """
    Example of creating a schema and serializing it to JSON.
    """
    logger.info("=== Schema Creation and Serialization Example ===")
    
    # Create example schema
    schema = create_example_schema()
    
    # Print schema as JSON
    logger.info(f"Schema JSON:\n{schema.to_json()}")
    
    # Serialize to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_file.write(schema.to_json())
        temp_path = temp_file.name
    
    logger.info(f"Schema saved to temporary file: {temp_path}")
    
    # Deserialize from file
    with open(temp_path, 'r') as f:
        schema_json = f.read()
    
    deserialized_schema = CollectionSchema.from_json(schema_json)
    logger.info(f"Deserialized schema name: {deserialized_schema.name}")
    logger.info(f"Deserialized schema dimension: {deserialized_schema.dimension}")
    logger.info(f"Deserialized schema has {len(deserialized_schema.metadata_fields)} metadata fields")
    
    # Clean up
    os.unlink(temp_path)
    logger.info(f"Temporary file removed")

def example_schema_validation(db_connector: BaseVectorDBConnector):
    """
    Example of validating documents against a schema.
    
    Args:
        db_connector: Database connector
    """
    logger.info("=== Schema Validation Example ===")
    
    # Create test collection
    collection_name = "schema_validation_test"
    if not setup_test_collection(db_connector, collection_name):
        logger.error("Failed to set up test collection. Aborting.")
        return
    
    try:
        # Create schema manager
        manager = get_schema_manager(db_connector)
        
        # Create example schema
        schema = create_example_schema()
        schema.name = collection_name  # Update name to match collection
        
        # Register schema
        manager.register_schema(schema)
        
        # Validate collection against schema
        logger.info(f"Validating collection '{collection_name}' against schema")
        validation_result = manager.validate_collection(collection_name)
        
        # Display validation results
        logger.info(f"Validation result:\n{validation_result}")
        
        # Clean up
        db_connector.delete_collection(collection_name)
        
    except Exception as e:
        logger.error(f"Error in schema validation example: {str(e)}")
        # Clean up on error
        try:
            db_connector.delete_collection(collection_name)
        except:
            pass

def example_schema_export_import(db_connector: BaseVectorDBConnector):
    """
    Example of exporting and importing schemas.
    
    Args:
        db_connector: Database connector
    """
    logger.info("=== Schema Export and Import Example ===")
    
    # Create test collection
    collection_name = "schema_export_test"
    
    try:
        # Create collection
        db_connector.create_collection(
            collection_name=collection_name,
            dimension=384,
            distance_metric="cosine"
        )
        
        # Create schema manager
        manager = get_schema_manager(db_connector)
        
        # Create example schema
        schema = create_example_schema()
        schema.name = collection_name  # Update name to match collection
        
        # Register and apply schema
        logger.info(f"Applying schema to collection '{collection_name}'")
        if manager.apply_schema(schema):
            logger.info("Schema applied successfully")
        else:
            logger.error("Failed to apply schema")
            return
        
        # Export schema to file
        temp_dir = tempfile.mkdtemp()
        export_path = os.path.join(temp_dir, "exported_schema.json")
        
        logger.info(f"Exporting schema to {export_path}")
        if manager.export_schema(collection_name, export_path):
            logger.info("Schema exported successfully")
        else:
            logger.error("Failed to export schema")
            return
        
        # Create new schema manager with empty registry
        new_manager = get_schema_manager(db_connector, {})
        
        # Import schema from file
        logger.info(f"Importing schema from {export_path}")
        imported_schema = new_manager.import_schema(export_path)
        
        if imported_schema:
            logger.info(f"Schema imported successfully: {imported_schema.name}")
            logger.info(f"Imported schema has {len(imported_schema.metadata_fields)} metadata fields")
        else:
            logger.error("Failed to import schema")
        
        # Clean up
        db_connector.delete_collection(collection_name)
        os.unlink(export_path)
        os.rmdir(temp_dir)
        
    except Exception as e:
        logger.error(f"Error in schema export/import example: {str(e)}")
        # Clean up on error
        try:
            db_connector.delete_collection(collection_name)
        except:
            pass

def example_schema_compatibility(db_connector: BaseVectorDBConnector):
    """
    Example of checking schema compatibility.
    
    Args:
        db_connector: Database connector
    """
    logger.info("=== Schema Compatibility Example ===")
    
    # Create test collection
    collection_name = "schema_compatibility_test"
    
    try:
        # Create collection
        db_connector.create_collection(
            collection_name=collection_name,
            dimension=384,
            distance_metric="cosine"
        )
        
        # Create schema manager
        manager = get_schema_manager(db_connector)
        
        # Create first schema version
        schema_v1 = CollectionSchema(
            name=collection_name,
            dimension=384,
            metadata_fields=[
                FieldSchema(
                    name="title",
                    type=FieldType.STRING,
                    required=True
                ),
                FieldSchema(
                    name="description",
                    type=FieldType.STRING
                )
            ],
            version="1.0.0"
        )
        
        # Apply first schema version
        logger.info("Applying schema v1")
        manager.apply_schema(schema_v1)
        
        # Create compatible schema version
        schema_v2 = CollectionSchema(
            name=collection_name,
            dimension=384,
            metadata_fields=[
                FieldSchema(
                    name="title",
                    type=FieldType.STRING,
                    required=True
                ),
                FieldSchema(
                    name="description",
                    type=FieldType.STRING
                ),
                FieldSchema(
                    name="tags",
                    type=FieldType.ARRAY,
                    array_item_type=FieldType.STRING
                )
            ],
            version="2.0.0"
        )
        
        # Check compatibility of v2
        logger.info("Checking compatibility of schema v2")
        is_compatible, reasons = manager.is_collection_compatible(collection_name, schema_v2)
        logger.info(f"Schema v2 is compatible: {is_compatible}")
        
        # Create incompatible schema version (different dimension)
        schema_v3 = CollectionSchema(
            name=collection_name,
            dimension=768,  # Different dimension
            metadata_fields=[
                FieldSchema(
                    name="title",
                    type=FieldType.STRING,
                    required=True
                )
            ],
            version="3.0.0"
        )
        
        # Check compatibility of v3
        logger.info("Checking compatibility of schema v3")
        is_compatible, reasons = manager.is_collection_compatible(collection_name, schema_v3)
        logger.info(f"Schema v3 is compatible: {is_compatible}")
        if not is_compatible:
            logger.info(f"Incompatibility reasons:")
            for reason in reasons:
                logger.info(f"  - {reason}")
        
        # Clean up
        db_connector.delete_collection(collection_name)
        
    except Exception as e:
        logger.error(f"Error in schema compatibility example: {str(e)}")
        # Clean up on error
        try:
            db_connector.delete_collection(collection_name)
        except:
            pass

def example_helper_function(db_connector: BaseVectorDBConnector):
    """
    Example of using the helper function validate_schema.
    
    Args:
        db_connector: Database connector
    """
    logger.info("=== Helper Function Example ===")
    
    # Create test collection
    collection_name = "schema_helper_test"
    if not setup_test_collection(db_connector, collection_name):
        logger.error("Failed to set up test collection. Aborting.")
        return
    
    try:
        # Create example schema
        schema = create_example_schema()
        schema.name = collection_name  # Update name to match collection
        
        # Use helper function to validate
        logger.info(f"Validating collection '{collection_name}' using helper function")
        validation_result = validate_schema(
            db_connector=db_connector,
            collection_name=collection_name,
            schema=schema,
            sample_size=10
        )
        
        # Display validation results
        logger.info(f"Validation result:\n{validation_result}")
        
        # Clean up
        db_connector.delete_collection(collection_name)
        
    except Exception as e:
        logger.error(f"Error in helper function example: {str(e)}")
        # Clean up on error
        try:
            db_connector.delete_collection(collection_name)
        except:
            pass

def run_examples(db_type: str):
    """
    Run schema manager examples on the specified database.
    
    Args:
        db_type: Database type
    """
    logger.info(f"Running examples on database: {db_type}")
    
    # Get connection parameters based on database type
    connection_params = {}
    if db_type == "chromadb":
        connection_params = {
            "host": "localhost",
            "port": 8000,
            "in_memory": True
        }
    elif db_type == "postgres":
        connection_params = {
            "host": "localhost",
            "port": 5432,
            "database": "postgres",
            "user": "postgres",
            "password": "postgres"
        }
    elif db_type == "qdrant":
        connection_params = {
            "host": "localhost",
            "port": 6333
        }
    
    try:
        # Create database connector
        db_connector = get_database_connector(
            db_type=db_type,
            connection_params=connection_params
        )
        
        # Run standalone example
        example_schema_creation_and_serialization()
        
        # Run examples that require database connection
        example_schema_validation(db_connector)
        example_schema_export_import(db_connector)
        example_schema_compatibility(db_connector)
        example_helper_function(db_connector)
        
        # Disconnect
        db_connector.disconnect()
    
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Schema Manager Examples')
    parser.add_argument('--db', type=str, choices=['chromadb', 'postgres', 'qdrant'],
                      default='chromadb', help='Database to run examples on')
    
    args = parser.parse_args()
    
    run_examples(args.db)

if __name__ == '__main__':
    main() 