#!/usr/bin/env python
"""
Vector Database Example

This script demonstrates how to use the vector database connectors in the
RAG Utility Tools package.
"""

import os
import sys
import logging
import argparse
import numpy as np
from typing import List, Dict, Any

# Add the parent directory to the path to import the package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vector_db import (
    get_database_connector,
    BaseVectorDBConnector,
    ChromaDBConnector,
    PostgresVectorConnector,
    QdrantConnector
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
    "Semantic search uses embeddings to find results based on meaning rather than keywords."
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

def example_chromadb():
    """
    Example usage of ChromaDBConnector.
    """
    logger.info("=== ChromaDB Example ===")
    
    # Create connector
    connector = get_database_connector(
        db_type="chromadb",
        connection_params={
            "host": "localhost",
            "port": 8000,
            "in_memory": True  # Use in-memory database for example
        }
    )
    
    run_example(connector)

def example_postgres():
    """
    Example usage of PostgresVectorConnector.
    """
    logger.info("=== PostgreSQL with pgvector Example ===")
    
    # Create connector
    connector = get_database_connector(
        db_type="postgres",
        connection_params={
            "host": "localhost",
            "port": 5432,
            "database": "postgres",
            "user": "postgres",
            "password": "postgres",
            "schema": "vector_store"
        }
    )
    
    run_example(connector)

def example_qdrant():
    """
    Example usage of QdrantConnector.
    """
    logger.info("=== Qdrant Example ===")
    
    # Create connector
    connector = get_database_connector(
        db_type="qdrant",
        connection_params={
            "host": "localhost",
            "port": 6333,
            "prefer_grpc": False
        }
    )
    
    run_example(connector)

def run_example(connector: BaseVectorDBConnector):
    """
    Run a common example workflow with the provided connector.
    
    Args:
        connector: Vector database connector
    """
    try:
        # Check connection
        if not connector.is_connected():
            logger.error("Failed to connect to database")
            return
        
        # Collection parameters
        collection_name = "example_collection"
        dimension = 128
        
        # Create collection
        logger.info(f"Creating collection '{collection_name}'")
        connector.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            distance_metric="cosine",
            metadata={"description": "Example collection for testing"}
        )
        
        # List collections
        collections = connector.list_collections()
        logger.info(f"Collections: {collections}")
        
        # Generate sample data
        num_vectors = len(SAMPLE_TEXTS)
        vectors = generate_random_embeddings(num_vectors, dimension)
        ids = [f"doc_{i}" for i in range(num_vectors)]
        metadata = [{"source": "example", "index": i} for i in range(num_vectors)]
        
        # Add vectors
        logger.info(f"Adding {num_vectors} vectors to collection '{collection_name}'")
        connector.add_vectors(
            collection_name=collection_name,
            vectors=vectors,
            ids=ids,
            metadata=metadata,
            documents=SAMPLE_TEXTS
        )
        
        # Get collection info
        info = connector.get_collection_info(collection_name)
        logger.info(f"Collection info: {info}")
        
        # Get a vector
        logger.info(f"Getting vector 'doc_0'")
        vector, meta = connector.get_vector(collection_name, "doc_0")
        logger.info(f"Vector length: {len(vector)}, Metadata: {meta}")
        
        # Search vectors
        logger.info("Searching for similar vectors")
        query_vector = generate_random_embeddings(1, dimension)[0]
        results = connector.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=3
        )
        
        logger.info("Search results:")
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. ID: {result['id']}, Score: {result['score']:.4f}")
            if "document" in result:
                logger.info(f"     Document: {result['document']}")
        
        # Count vectors
        count = connector.count_vectors(collection_name)
        logger.info(f"Vector count: {count}")
        
        # Count with filter
        filter_count = connector.count_vectors(
            collection_name=collection_name,
            filter={"source": "example"}
        )
        logger.info(f"Vector count with filter: {filter_count}")
        
        # Delete a vector
        logger.info("Deleting vector 'doc_1'")
        connector.delete_vectors(collection_name, ["doc_1"])
        
        # Count after deletion
        count_after = connector.count_vectors(collection_name)
        logger.info(f"Vector count after deletion: {count_after}")
        
        # Delete collection
        logger.info(f"Deleting collection '{collection_name}'")
        connector.delete_collection(collection_name)
        
        # Disconnect
        logger.info("Disconnecting from database")
        connector.disconnect()
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Vector Database Examples')
    parser.add_argument('--db', type=str, choices=['chromadb', 'postgres', 'qdrant', 'all'],
                      default='all', help='Database to run example on')
    
    args = parser.parse_args()
    
    try:
        if args.db == 'chromadb' or args.db == 'all':
            example_chromadb()
        
        if args.db == 'postgres' or args.db == 'all':
            example_postgres()
        
        if args.db == 'qdrant' or args.db == 'all':
            example_qdrant()
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")

if __name__ == '__main__':
    main() 