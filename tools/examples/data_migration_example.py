#!/usr/bin/env python
"""
Data Migration Example

This script demonstrates how to use the data migration tools in the
RAG Utility Tools package to migrate data between different vector databases.
"""

import os
import sys
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Optional

# Add the parent directory to the path to import the package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vector_db import (
    get_database_connector,
    BaseVectorDBConnector
)

from src.vector_db.data_migration import (
    MigrationConfig,
    MigrationResult,
    get_migrator,
    migrate_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample data for the example
SAMPLE_TEXTS = [
    "Retrieval-Augmented Generation (RAG) combines retrieval mechanisms with generative models.",
    "Vector databases are optimized for storing and searching high-dimensional vectors.",
    "Embeddings are numerical representations of text, images, or other data types.",
    "Cosine similarity is a common metric used to measure similarity between vectors.",
    "HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest neighbor search.",
    "PostgreSQL with pgvector enables efficient vector similarity search in a relational database.",
    "ChromaDB is an open-source vector database designed for storing and searching embeddings.",
    "Qdrant is a vector similarity search engine that provides a production-ready service.",
    "The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces.",
    "Semantic search uses embeddings to find results based on meaning rather than keywords.",
    "Data migration involves transferring data from one system to another while preserving its integrity."
]

def generate_random_embeddings(num_vectors: int, dimension: int) -> List[List[float]]:
    """
    Generate random embeddings for testing.
    
    Args:
        num_vectors: Number of embeddings to generate
        dimension: Dimensionality of embeddings
        
    Returns:
        List[List[float]]: List of random embeddings
    """
    # Generate random embeddings
    embeddings = np.random.rand(num_vectors, dimension).astype(np.float32)
    
    # Normalize embeddings (for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    
    return normalized_embeddings.tolist()

def setup_source_collection(
    connector: BaseVectorDBConnector,
    collection_name: str,
    dimension: int = 128,
    num_vectors: int = 100
) -> bool:
    """
    Set up a source collection with random vectors for migration.
    
    Args:
        connector: Vector database connector
        collection_name: Name of the collection to create
        dimension: Dimensionality of vectors
        num_vectors: Number of vectors to add
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        # Create collection
        logger.info(f"Creating source collection '{collection_name}'")
        connector.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            distance_metric="cosine",
            metadata={"description": "Source collection for migration testing"}
        )
        
        # Generate sample data
        vectors = generate_random_embeddings(num_vectors, dimension)
        ids = [f"doc_{i}" for i in range(num_vectors)]
        metadata = [
            {"source": "example", "index": i % 5, "type": "document" if i % 2 == 0 else "image"} 
            for i in range(num_vectors)
        ]
        
        # Use sample texts with repetition to fill all documents
        documents = []
        for i in range(num_vectors):
            if i % 2 == 0:  # Only add documents for even indices
                documents.append(SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)])
            else:
                documents.append(None)
        
        # Add vectors
        logger.info(f"Adding {num_vectors} vectors to collection '{collection_name}'")
        connector.add_vectors(
            collection_name=collection_name,
            vectors=vectors,
            ids=ids,
            metadata=metadata,
            documents=documents
        )
        
        return True
    
    except Exception as e:
        logger.error(f"Error setting up source collection: {str(e)}")
        return False

def example_basic_migration(source_db: BaseVectorDBConnector, target_db: BaseVectorDBConnector):
    """
    Basic example of migrating data from one database to another.
    
    Args:
        source_db: Source database connector
        target_db: Target database connector
    """
    logger.info("=== Basic Migration Example ===")
    
    try:
        # Set up source collection
        source_collection = "migration_source"
        if not setup_source_collection(source_db, source_collection, dimension=128, num_vectors=50):
            logger.error("Failed to set up source collection. Aborting.")
            return
        
        # Create basic configuration
        config = MigrationConfig(
            batch_size=10,
            include_metadata=True,
            include_documents=True,
            progress_bar=True
        )
        
        # Migrate data
        logger.info(f"Migrating data from {source_db.__class__.__name__} to {target_db.__class__.__name__}")
        result = migrate_data(
            source_db=source_db,
            target_db=target_db,
            collection_names=[source_collection],
            config=config
        )
        
        # Display results
        logger.info(f"Migration result:\n{result}")
        
        # Verify migration
        if result.success:
            target_collection = source_collection  # Same name
            
            # Check collection exists in target
            if target_db.collection_exists(target_collection):
                # Get count of vectors
                count = target_db.count_vectors(target_collection)
                logger.info(f"Verified {count} vectors in target collection '{target_collection}'")
                
                # Get some sample vectors
                if count > 0:
                    sample_id = "doc_0"
                    vector, metadata = target_db.get_vector(target_collection, sample_id)
                    logger.info(f"Sample vector dimension: {len(vector)}")
                    logger.info(f"Sample metadata: {metadata}")
            else:
                logger.error(f"Target collection '{target_collection}' does not exist after migration")
        
        # Clean up
        logger.info("Cleaning up source collection")
        source_db.delete_collection(source_collection)
        
        if result.success:
            logger.info("Cleaning up target collection")
            target_db.delete_collection(target_collection)
    
    except Exception as e:
        logger.error(f"Error in basic migration example: {str(e)}")

def example_collection_mapping(source_db: BaseVectorDBConnector, target_db: BaseVectorDBConnector):
    """
    Example of migrating data with collection name mapping.
    
    Args:
        source_db: Source database connector
        target_db: Target database connector
    """
    logger.info("=== Collection Mapping Example ===")
    
    try:
        # Set up multiple source collections
        source_collections = ["source_collection_1", "source_collection_2"]
        
        for i, collection in enumerate(source_collections):
            if not setup_source_collection(source_db, collection, dimension=128, num_vectors=20):
                logger.error(f"Failed to set up source collection '{collection}'. Skipping.")
                continue
        
        # Create configuration with collection mapping
        config = MigrationConfig(
            batch_size=10,
            include_metadata=True,
            include_documents=True,
            progress_bar=True,
            collection_mapping={
                "source_collection_1": "target_renamed_1",
                "source_collection_2": "target_renamed_2"
            }
        )
        
        # Migrate data
        logger.info(f"Migrating data with collection mapping")
        result = migrate_data(
            source_db=source_db,
            target_db=target_db,
            collection_names=source_collections,
            config=config
        )
        
        # Display results
        logger.info(f"Migration result:\n{result}")
        
        # Verify target collections
        target_collections = ["target_renamed_1", "target_renamed_2"]
        for collection in target_collections:
            if target_db.collection_exists(collection):
                count = target_db.count_vectors(collection)
                logger.info(f"Verified {count} vectors in target collection '{collection}'")
            else:
                logger.error(f"Target collection '{collection}' does not exist after migration")
        
        # Clean up
        for collection in source_collections:
            logger.info(f"Cleaning up source collection '{collection}'")
            source_db.delete_collection(collection)
        
        for collection in target_collections:
            if target_db.collection_exists(collection):
                logger.info(f"Cleaning up target collection '{collection}'")
                target_db.delete_collection(collection)
    
    except Exception as e:
        logger.error(f"Error in collection mapping example: {str(e)}")

def example_data_transformation(source_db: BaseVectorDBConnector, target_db: BaseVectorDBConnector):
    """
    Example of migrating data with vector and metadata transformation.
    
    Args:
        source_db: Source database connector
        target_db: Target database connector
    """
    logger.info("=== Data Transformation Example ===")
    
    try:
        # Set up source collection
        source_collection = "transform_source"
        if not setup_source_collection(source_db, source_collection, dimension=128, num_vectors=30):
            logger.error("Failed to set up source collection. Aborting.")
            return
        
        # Define vector transformation function (truncate to smaller dimension)
        def transform_vector(vec: List[float]) -> List[float]:
            # Truncate to first 64 dimensions
            truncated = vec[:64]
            # Normalize
            magnitude = sum(x**2 for x in truncated) ** 0.5
            return [x/magnitude for x in truncated]
        
        # Define metadata transformation function
        def transform_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
            if not meta:
                return {}
            
            # Create transformed metadata
            transformed = {
                "migrated": True,
                "original_source": meta.get("source", "unknown"),
                "category": "article" if meta.get("type") == "document" else "visual"
            }
            
            return transformed
        
        # Create target collection with the new dimension
        target_collection = "transform_target"
        target_db.create_collection(
            collection_name=target_collection,
            dimension=64,  # New dimension after transformation
            distance_metric="cosine",
            metadata={"description": "Target collection with transformed data"}
        )
        
        # Create configuration with transformations
        config = MigrationConfig(
            batch_size=10,
            include_metadata=True,
            include_documents=True,
            progress_bar=True,
            transform_vector_fn=transform_vector,
            transform_metadata_fn=transform_metadata
        )
        
        # Migrate data
        logger.info(f"Migrating data with transformations")
        result = migrate_data(
            source_db=source_db,
            target_db=target_db,
            collection_names=[source_collection],
            config=config
        )
        
        # Display results
        logger.info(f"Migration result:\n{result}")
        
        # Verify transformation
        if result.success and target_db.collection_exists(target_collection):
            # Get target collection info to verify dimension
            info = target_db.get_collection_info(target_collection)
            logger.info(f"Target collection dimension: {info.get('dimension')}")
            
            # Check transformed vectors
            vector, metadata = target_db.get_vector(target_collection, "doc_0")
            logger.info(f"Transformed vector dimension: {len(vector)}")
            logger.info(f"Transformed metadata: {metadata}")
        
        # Clean up
        logger.info("Cleaning up source collection")
        source_db.delete_collection(source_collection)
        
        if target_db.collection_exists(target_collection):
            logger.info("Cleaning up target collection")
            target_db.delete_collection(target_collection)
    
    except Exception as e:
        logger.error(f"Error in data transformation example: {str(e)}")

def example_filtering(source_db: BaseVectorDBConnector, target_db: BaseVectorDBConnector):
    """
    Example of migrating data with filtering.
    
    Args:
        source_db: Source database connector
        target_db: Target database connector
    """
    logger.info("=== Filtered Migration Example ===")
    
    try:
        # Set up source collection
        source_collection = "filter_source"
        if not setup_source_collection(source_db, source_collection, dimension=128, num_vectors=40):
            logger.error("Failed to set up source collection. Aborting.")
            return
        
        # Create configuration with filter
        config = MigrationConfig(
            batch_size=10,
            include_metadata=True,
            include_documents=True,
            progress_bar=True,
            source_filter={"type": "document"}  # Only migrate documents
        )
        
        # Migrate data
        logger.info(f"Migrating data with filter (only documents)")
        result = migrate_data(
            source_db=source_db,
            target_db=target_db,
            collection_names=[source_collection],
            config=config
        )
        
        # Display results
        logger.info(f"Migration result:\n{result}")
        
        # Verify filtering
        if result.success:
            target_collection = source_collection  # Same name
            
            # Check collection exists in target
            if target_db.collection_exists(target_collection):
                # Get count of vectors
                count = target_db.count_vectors(target_collection)
                logger.info(f"Verified {count} vectors in target collection '{target_collection}'")
                logger.info(f"Should be approximately half of source vectors due to filtering")
            else:
                logger.error(f"Target collection '{target_collection}' does not exist after migration")
        
        # Clean up
        logger.info("Cleaning up source collection")
        source_db.delete_collection(source_collection)
        
        if result.success:
            logger.info("Cleaning up target collection")
            target_db.delete_collection(target_collection)
    
    except Exception as e:
        logger.error(f"Error in filtered migration example: {str(e)}")

def run_examples(source_db_type: str, target_db_type: str):
    """
    Run data migration examples between specified database types.
    
    Args:
        source_db_type: Type of source database
        target_db_type: Type of target database
    """
    logger.info(f"Running examples with source: {source_db_type}, target: {target_db_type}")
    
    # Get connection parameters based on database types
    def get_connection_params(db_type):
        if db_type == "chromadb":
            return {
                "host": "localhost",
                "port": 8000,
                "in_memory": True
            }
        elif db_type == "postgres":
            return {
                "host": "localhost",
                "port": 5432,
                "database": "postgres",
                "user": "postgres",
                "password": "postgres"
            }
        elif db_type == "qdrant":
            return {
                "host": "localhost",
                "port": 6333
            }
        else:
            return {}
    
    try:
        # Create database connectors
        source_db = get_database_connector(
            db_type=source_db_type,
            connection_params=get_connection_params(source_db_type)
        )
        
        target_db = get_database_connector(
            db_type=target_db_type,
            connection_params=get_connection_params(target_db_type)
        )
        
        # Run examples
        example_basic_migration(source_db, target_db)
        example_collection_mapping(source_db, target_db)
        example_data_transformation(source_db, target_db)
        example_filtering(source_db, target_db)
        
        # Disconnect
        source_db.disconnect()
        target_db.disconnect()
    
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Data Migration Examples')
    parser.add_argument('--source', type=str, choices=['chromadb', 'postgres', 'qdrant'],
                      default='chromadb', help='Source database type')
    parser.add_argument('--target', type=str, choices=['chromadb', 'postgres', 'qdrant'],
                      default='qdrant', help='Target database type')
    
    args = parser.parse_args()
    
    run_examples(args.source, args.target)

if __name__ == '__main__':
    main() 