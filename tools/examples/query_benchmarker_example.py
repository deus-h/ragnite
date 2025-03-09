#!/usr/bin/env python
"""
Query Benchmarker Example

This script demonstrates how to use the query benchmarkers in the
RAG Utility Tools package.
"""

import os
import sys
import logging
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
import time

# Add the parent directory to the path to import the package
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vector_db import (
    get_database_connector,
    BaseVectorDBConnector
)

from src.vector_db.query_benchmarker import (
    get_query_benchmarker,
    run_benchmark,
    BenchmarkResult,
    LatencyBenchmarker,
    ThroughputBenchmarker,
    RecallBenchmarker,
    PrecisionBenchmarker
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
    "Precision measures the fraction of retrieved items that are relevant.",
    "Recall measures the fraction of relevant items that are retrieved.",
    "Latency measures how quickly the system responds to queries.",
    "Throughput measures how many queries the system can handle per unit of time.",
    "Benchmarking is the process of evaluating system performance using standardized tests."
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
    num_vectors: int = 1000
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
            metadata={"description": "Test collection for benchmarking"}
        )
        
        # Generate sample data
        vectors = generate_random_embeddings(num_vectors, dimension)
        ids = [f"doc_{i}" for i in range(num_vectors)]
        metadata = [{"source": "example", "index": i % 5} for i in range(num_vectors)]
        
        # Split into batches for better progress visibility
        batch_size = 200
        num_batches = (num_vectors + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_vectors)
            
            batch_vectors = vectors[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            batch_metadata = metadata[start_idx:end_idx]
            batch_documents = SAMPLE_TEXTS * (1 + (end_idx - start_idx) // len(SAMPLE_TEXTS))
            batch_documents = batch_documents[:end_idx - start_idx]
            
            logger.info(f"Adding batch {i+1}/{num_batches} ({len(batch_vectors)} vectors)")
            connector.add_vectors(
                collection_name=collection_name,
                vectors=batch_vectors,
                ids=batch_ids,
                metadata=batch_metadata,
                documents=batch_documents
            )
        
        return True
    
    except Exception as e:
        logger.error(f"Error setting up test collection: {str(e)}")
        return False

def example_latency_benchmarker(db_connector: BaseVectorDBConnector, collection_name: str):
    """
    Example usage of LatencyBenchmarker.
    
    Args:
        db_connector: Database connector
        collection_name: Name of the collection to benchmark
    """
    logger.info("=== Latency Benchmarker Example ===")
    
    try:
        # Create benchmarker
        benchmarker = LatencyBenchmarker(db_connector=db_connector)
        
        # Run benchmark
        logger.info(f"Running latency benchmark on collection '{collection_name}'")
        result = benchmarker.benchmark(
            collection_name=collection_name,
            num_queries=20,
            top_k=10,
            warmup_runs=3,
            include_percentiles=True
        )
        
        # Display results
        logger.info(f"Benchmark results:\n{result}")
        
        # Example with filter
        logger.info(f"Running latency benchmark with filter")
        result_with_filter = benchmarker.benchmark(
            collection_name=collection_name,
            num_queries=20,
            top_k=10,
            warmup_runs=3,
            include_percentiles=True,
            filter={"source": "example"}
        )
        
        # Display results
        logger.info(f"Benchmark results with filter:\n{result_with_filter}")
    
    except Exception as e:
        logger.error(f"Error in latency benchmarker example: {str(e)}")

def example_throughput_benchmarker(db_connector: BaseVectorDBConnector, collection_name: str):
    """
    Example usage of ThroughputBenchmarker.
    
    Args:
        db_connector: Database connector
        collection_name: Name of the collection to benchmark
    """
    logger.info("=== Throughput Benchmarker Example ===")
    
    try:
        # Create benchmarker
        benchmarker = ThroughputBenchmarker(db_connector=db_connector)
        
        # Run query-based benchmark
        logger.info(f"Running query-based throughput benchmark on collection '{collection_name}'")
        result = benchmarker.benchmark(
            collection_name=collection_name,
            num_queries=50,
            top_k=10,
            warmup_runs=3,
            concurrent_queries=2,
            time_based=False
        )
        
        # Display results
        logger.info(f"Query-based benchmark results:\n{result}")
        
        # Run time-based benchmark
        logger.info(f"Running time-based throughput benchmark")
        result_time_based = benchmarker.benchmark(
            collection_name=collection_name,
            top_k=10,
            warmup_runs=3,
            concurrent_queries=2,
            time_based=True,
            duration_seconds=5
        )
        
        # Display results
        logger.info(f"Time-based benchmark results:\n{result_time_based}")
    
    except Exception as e:
        logger.error(f"Error in throughput benchmarker example: {str(e)}")

def example_recall_benchmarker(db_connector: BaseVectorDBConnector, collection_name: str):
    """
    Example usage of RecallBenchmarker.
    
    Args:
        db_connector: Database connector
        collection_name: Name of the collection to benchmark
    """
    logger.info("=== Recall Benchmarker Example ===")
    
    try:
        # Create benchmarker
        benchmarker = RecallBenchmarker(db_connector=db_connector)
        
        # Generate test query vectors
        collection_info = db_connector.get_collection_info(collection_name)
        dimension = collection_info.get("dimension", 128)
        query_vectors = generate_random_embeddings(10, dimension)
        
        # Run benchmark
        logger.info(f"Running recall benchmark on collection '{collection_name}'")
        result = benchmarker.benchmark(
            collection_name=collection_name,
            query_vectors=query_vectors,
            num_queries=10,
            k_values=[1, 5, 10],
            ground_truth_k=50
        )
        
        # Display results
        logger.info(f"Recall benchmark results:\n{result}")
    
    except Exception as e:
        logger.error(f"Error in recall benchmarker example: {str(e)}")

def example_precision_benchmarker(db_connector: BaseVectorDBConnector, collection_name: str):
    """
    Example usage of PrecisionBenchmarker.
    
    Args:
        db_connector: Database connector
        collection_name: Name of the collection to benchmark
    """
    logger.info("=== Precision Benchmarker Example ===")
    
    try:
        # Create benchmarker
        benchmarker = PrecisionBenchmarker(db_connector=db_connector)
        
        # Generate test query vectors
        collection_info = db_connector.get_collection_info(collection_name)
        dimension = collection_info.get("dimension", 128)
        query_vectors = generate_random_embeddings(10, dimension)
        
        # Custom relevance function
        def custom_relevance_fn(query_vec, result):
            # Example: consider results with score above 0.8 as relevant
            return result["score"] >= 0.8
        
        # Run benchmark
        logger.info(f"Running precision benchmark on collection '{collection_name}'")
        result = benchmarker.benchmark(
            collection_name=collection_name,
            query_vectors=query_vectors,
            num_queries=10,
            k_values=[1, 5, 10],
            relevance_threshold=0.8
        )
        
        # Display results
        logger.info(f"Precision benchmark results:\n{result}")
        
        # Run benchmark with custom relevance function
        logger.info(f"Running precision benchmark with custom relevance function")
        result_custom = benchmarker.benchmark(
            collection_name=collection_name,
            query_vectors=query_vectors,
            num_queries=10,
            k_values=[1, 5, 10],
            relevance_fn=custom_relevance_fn
        )
        
        # Display results
        logger.info(f"Precision benchmark results with custom function:\n{result_custom}")
    
    except Exception as e:
        logger.error(f"Error in precision benchmarker example: {str(e)}")

def example_helper_function(db_connector: BaseVectorDBConnector, collection_name: str):
    """
    Example usage of the run_benchmark helper function.
    
    Args:
        db_connector: Database connector
        collection_name: Name of the collection to benchmark
    """
    logger.info("=== Helper Function Example ===")
    
    try:
        # Run latency benchmark using helper function
        logger.info(f"Running latency benchmark with helper function")
        result = run_benchmark(
            db_connector=db_connector,
            collection_name=collection_name,
            benchmarker_type="latency",
            params={
                "num_queries": 10,
                "top_k": 5
            }
        )
        
        # Display results
        logger.info(f"Latency benchmark results:\n{result}")
        
        # Run throughput benchmark using helper function
        logger.info(f"Running throughput benchmark with helper function")
        result = run_benchmark(
            db_connector=db_connector,
            collection_name=collection_name,
            benchmarker_type="throughput",
            params={
                "num_queries": 10,
                "top_k": 5,
                "concurrent_queries": 2
            }
        )
        
        # Display results
        logger.info(f"Throughput benchmark results:\n{result}")
    
    except Exception as e:
        logger.error(f"Error in helper function example: {str(e)}")

def run_examples(db_type: str, collection_name: str):
    """
    Run query benchmarker examples on the specified database.
    
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
        example_latency_benchmarker(db_connector, collection_name)
        example_throughput_benchmarker(db_connector, collection_name)
        example_recall_benchmarker(db_connector, collection_name)
        example_precision_benchmarker(db_connector, collection_name)
        example_helper_function(db_connector, collection_name)
        
        # Clean up
        logger.info(f"Cleaning up: deleting collection '{collection_name}'")
        db_connector.delete_collection(collection_name)
        
        # Disconnect
        db_connector.disconnect()
    
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Query Benchmarker Examples')
    parser.add_argument('--db', type=str, choices=['chromadb', 'postgres', 'qdrant'],
                      default='chromadb', help='Database to run examples on')
    parser.add_argument('--collection', type=str, default='test_benchmarking',
                      help='Collection name to use for tests')
    
    args = parser.parse_args()
    
    run_examples(args.db, args.collection)

if __name__ == '__main__':
    main() 