#!/usr/bin/env python
"""
Index Optimizer Example

This script demonstrates how to use the index optimizers in the
RAG Utility Tools package.
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

from src.vector_db.index_optimizers import (
    get_index_optimizer,
    optimize_index,
    HNSWOptimizer,
    IVFFlatOptimizer
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

def setup_test_collection(
    connector: BaseVectorDBConnector,
    collection_name: str,
    dimension: int = 128,
    num_vectors: int = 100
) -> bool:
    """
    Set up a test collection with random vectors.
    
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
        logger.info(f"Creating collection '{collection_name}'")
        connector.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            distance_metric="cosine",
            metadata={"description": "Test collection for index optimization"}
        )
        
        # Generate sample data
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
            documents=SAMPLE_TEXTS[:min(len(SAMPLE_TEXTS), num_vectors)]
        )
        
        return True
    
    except Exception as e:
        logger.error(f"Error setting up test collection: {str(e)}")
        return False

def example_hnsw_optimizer(db_connector: BaseVectorDBConnector, collection_name: str):
    """
    Example usage of HNSWOptimizer.
    
    Args:
        db_connector: Database connector
        collection_name: Name of the collection to optimize
    """
    logger.info("=== HNSW Optimizer Example ===")
    
    try:
        # Create optimizer
        optimizer = HNSWOptimizer(db_connector=db_connector)
        
        # Analyze index
        logger.info(f"Analyzing index for collection '{collection_name}'")
        analysis = optimizer.analyze_index(collection_name, stats_format="dict")
        logger.info(f"Analysis results:")
        logger.info(f"  Vector count: {analysis.get('vector_count', 'N/A')}")
        logger.info(f"  Dimension: {analysis.get('dimension', 'N/A')}")
        
        if "recommendations" in analysis:
            logger.info(f"  Recommendations:")
            for param, value in analysis["recommendations"].items():
                logger.info(f"    {param}: {value}")
        
        # Estimate memory usage
        logger.info(f"Estimating memory usage for collection '{collection_name}'")
        memory_usage = optimizer.estimate_memory_usage(collection_name)
        
        if "estimated_memory" in memory_usage:
            mem = memory_usage["estimated_memory"]
            logger.info(f"  Estimated memory usage: {mem.get('total_mb', 'N/A'):.2f} MB")
        
        # Benchmark index (current configuration)
        logger.info(f"Benchmarking index for collection '{collection_name}'")
        benchmark = optimizer.benchmark_index(
            collection_name=collection_name,
            num_queries=10,
            top_k=5
        )
        
        if "latency_ms" in benchmark:
            lat = benchmark["latency_ms"]
            logger.info(f"  Latency (ms): avg={lat.get('avg', 'N/A'):.2f}, median={lat.get('median', 'N/A'):.2f}")
            logger.info(f"  Throughput: {benchmark.get('throughput_qps', 'N/A'):.2f} queries/second")
        
        # Optimize index (dry run)
        logger.info(f"Optimizing index for collection '{collection_name}' (dry run)")
        
        # Custom parameters
        custom_params = {
            "ef_construction": 128,
            "ef_search": 64,
            "m": 12
        }
        
        optimization = optimizer.optimize_index(
            collection_name=collection_name,
            parameters=custom_params,
            dry_run=True
        )
        
        logger.info(f"  Optimization parameters:")
        for param, value in optimization["parameters"].items():
            logger.info(f"    {param}: {value}")
    
    except Exception as e:
        logger.error(f"Error in HNSW optimizer example: {str(e)}")

def example_ivfflat_optimizer(db_connector: BaseVectorDBConnector, collection_name: str):
    """
    Example usage of IVFFlatOptimizer.
    
    Args:
        db_connector: Database connector
        collection_name: Name of the collection to optimize
    """
    logger.info("=== IVFFlat Optimizer Example ===")
    
    try:
        # Create optimizer
        optimizer = IVFFlatOptimizer(db_connector=db_connector)
        
        # Analyze index
        logger.info(f"Analyzing index for collection '{collection_name}'")
        analysis = optimizer.analyze_index(collection_name, stats_format="dict")
        logger.info(f"Analysis results:")
        logger.info(f"  Vector count: {analysis.get('vector_count', 'N/A')}")
        logger.info(f"  Dimension: {analysis.get('dimension', 'N/A')}")
        
        if "recommendations" in analysis:
            logger.info(f"  Recommendations:")
            for param, value in analysis["recommendations"].items():
                logger.info(f"    {param}: {value}")
        
        # Estimate memory usage
        logger.info(f"Estimating memory usage for collection '{collection_name}'")
        memory_usage = optimizer.estimate_memory_usage(collection_name)
        
        if "estimated_memory" in memory_usage:
            mem = memory_usage["estimated_memory"]
            logger.info(f"  Estimated memory usage: {mem.get('total_mb', 'N/A'):.2f} MB")
        
        # Benchmark index (current configuration)
        logger.info(f"Benchmarking index for collection '{collection_name}'")
        benchmark = optimizer.benchmark_index(
            collection_name=collection_name,
            num_queries=10,
            top_k=5
        )
        
        if "latency_ms" in benchmark:
            lat = benchmark["latency_ms"]
            logger.info(f"  Latency (ms): avg={lat.get('avg', 'N/A'):.2f}, median={lat.get('median', 'N/A'):.2f}")
            logger.info(f"  Throughput: {benchmark.get('throughput_qps', 'N/A'):.2f} queries/second")
        
        # Optimize index (dry run)
        logger.info(f"Optimizing index for collection '{collection_name}' (dry run)")
        
        # Custom parameters
        custom_params = {
            "nlist": 50,
            "nprobe": 5
        }
        
        optimization = optimizer.optimize_index(
            collection_name=collection_name,
            parameters=custom_params,
            dry_run=True
        )
        
        logger.info(f"  Optimization parameters:")
        for param, value in optimization["parameters"].items():
            logger.info(f"    {param}: {value}")
    
    except Exception as e:
        logger.error(f"Error in IVFFlat optimizer example: {str(e)}")

def example_auto_optimizer(db_connector: BaseVectorDBConnector, collection_name: str):
    """
    Example usage of auto-detection for optimizers.
    
    Args:
        db_connector: Database connector
        collection_name: Name of the collection to optimize
    """
    logger.info("=== Auto Optimizer Example ===")
    
    try:
        # Use the helper function with auto-detection
        optimization = optimize_index(
            db_connector=db_connector,
            collection_name=collection_name,
            optimizer_type="auto",
            dry_run=True
        )
        
        logger.info(f"Auto-selected optimizer: {optimization.get('optimizer', 'N/A')}")
        logger.info(f"Optimization parameters:")
        for param, value in optimization.get("parameters", {}).items():
            logger.info(f"  {param}: {value}")
    
    except Exception as e:
        logger.error(f"Error in auto optimizer example: {str(e)}")

def run_examples(db_type: str, collection_name: str):
    """
    Run index optimizer examples on the specified database.
    
    Args:
        db_type: Database type
        collection_name: Name of the collection
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
        
        # Set up test collection
        if not setup_test_collection(
            connector=db_connector,
            collection_name=collection_name,
            dimension=128,
            num_vectors=1000
        ):
            logger.error(f"Failed to set up test collection. Aborting.")
            return
        
        # Run examples
        example_hnsw_optimizer(db_connector, collection_name)
        example_ivfflat_optimizer(db_connector, collection_name)
        example_auto_optimizer(db_connector, collection_name)
        
        # Clean up
        logger.info(f"Cleaning up: deleting collection '{collection_name}'")
        db_connector.delete_collection(collection_name)
        
        # Disconnect
        db_connector.disconnect()
    
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Index Optimizer Examples')
    parser.add_argument('--db', type=str, choices=['chromadb', 'postgres', 'qdrant'],
                      default='chromadb', help='Database to run examples on')
    parser.add_argument('--collection', type=str, default='test_optimization',
                      help='Collection name to use for tests')
    
    args = parser.parse_args()
    
    run_examples(args.db, args.collection)

if __name__ == '__main__':
    main() 