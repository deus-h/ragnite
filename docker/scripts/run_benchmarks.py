#!/usr/bin/env python3
"""
run_benchmarks.py - Benchmarks vector database performance
"""
import argparse
import os
import sys
import logging
import time
import numpy as np
import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
NUM_TESTS = 5
NUM_QUERIES = 10
VECTOR_DIM = 384
TOP_K = 5
WARMUP_RUNS = 3

def generate_query_vectors(count: int = 10, dim: int = 384) -> List[List[float]]:
    """
    Generate random query vectors for benchmarking.
    
    Args:
        count: Number of query vectors to generate
        dim: Dimensionality of vectors
        
    Returns:
        List of query vectors
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate random queries
    queries = []
    for i in range(count):
        # Create a random embedding
        query = np.random.randn(dim)
        # Normalize
        query = query / np.linalg.norm(query)
        queries.append(query.tolist())
    
    return queries

def benchmark_chromadb():
    """Benchmark ChromaDB performance"""
    try:
        import chromadb
        
        logger.info("Benchmarking ChromaDB...")
        
        # Connect to ChromaDB
        client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Get test collection
        collection = client.get_collection("test_collection")
        
        # Generate query vectors
        query_vectors = generate_query_vectors(NUM_QUERIES)
        
        # Warm-up runs
        logger.info("Performing warm-up queries...")
        for i in range(WARMUP_RUNS):
            collection.query(
                query_embeddings=[query_vectors[0]],
                n_results=TOP_K
            )
        
        # Benchmark latency
        latencies = []
        
        for test in range(NUM_TESTS):
            test_latencies = []
            
            for query in query_vectors:
                start_time = time.time()
                
                results = collection.query(
                    query_embeddings=[query],
                    n_results=TOP_K
                )
                
                end_time = time.time()
                query_time = (end_time - start_time) * 1000  # Convert to ms
                test_latencies.append(query_time)
            
            avg_latency = sum(test_latencies) / len(test_latencies)
            latencies.append(avg_latency)
            logger.info(f"ChromaDB Test {test+1}/{NUM_TESTS}: Avg Latency = {avg_latency:.2f}ms")
        
        # Calculate overall statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        result = {
            "database": "ChromaDB",
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "test_runs": NUM_TESTS,
            "queries_per_run": NUM_QUERIES,
            "top_k": TOP_K
        }
        
        return result
        
    except ImportError:
        logger.error("chromadb package not installed. Install with: pip install chromadb")
        return {"database": "ChromaDB", "error": "Package not installed"}
        
    except Exception as e:
        logger.error(f"Error benchmarking ChromaDB: {str(e)}")
        return {"database": "ChromaDB", "error": str(e)}

def benchmark_postgres():
    """Benchmark PostgreSQL with pgvector performance"""
    try:
        import psycopg2
        import psycopg2.extras
        
        logger.info("Benchmarking PostgreSQL/pgvector...")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            dbname="vectordb",
            user="postgres",
            password="postgres"
        )
        conn.autocommit = True
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Generate query vectors
        query_vectors = generate_query_vectors(NUM_QUERIES)
        
        # Warm-up runs
        logger.info("Performing warm-up queries...")
        for i in range(WARMUP_RUNS):
            cursor.execute(
                """
                SELECT id, vector <=> %s as distance
                FROM vector_store.test_collection
                ORDER BY distance
                LIMIT %s
                """,
                (query_vectors[0], TOP_K)
            )
            cursor.fetchall()
        
        # Benchmark latency
        latencies = []
        
        for test in range(NUM_TESTS):
            test_latencies = []
            
            for query in query_vectors:
                start_time = time.time()
                
                cursor.execute(
                    """
                    SELECT id, vector <=> %s as distance
                    FROM vector_store.test_collection
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (query, TOP_K)
                )
                results = cursor.fetchall()
                
                end_time = time.time()
                query_time = (end_time - start_time) * 1000  # Convert to ms
                test_latencies.append(query_time)
            
            avg_latency = sum(test_latencies) / len(test_latencies)
            latencies.append(avg_latency)
            logger.info(f"PostgreSQL Test {test+1}/{NUM_TESTS}: Avg Latency = {avg_latency:.2f}ms")
        
        # Close connection
        cursor.close()
        conn.close()
        
        # Calculate overall statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        result = {
            "database": "PostgreSQL/pgvector",
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "test_runs": NUM_TESTS,
            "queries_per_run": NUM_QUERIES,
            "top_k": TOP_K
        }
        
        return result
        
    except ImportError:
        logger.error("psycopg2 package not installed. Install with: pip install psycopg2-binary")
        return {"database": "PostgreSQL/pgvector", "error": "Package not installed"}
        
    except Exception as e:
        logger.error(f"Error benchmarking PostgreSQL: {str(e)}")
        return {"database": "PostgreSQL/pgvector", "error": str(e)}

def benchmark_qdrant():
    """Benchmark Qdrant performance"""
    try:
        from qdrant_client import QdrantClient
        
        logger.info("Benchmarking Qdrant...")
        
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)
        
        # Generate query vectors
        query_vectors = generate_query_vectors(NUM_QUERIES)
        
        # Warm-up runs
        logger.info("Performing warm-up queries...")
        for i in range(WARMUP_RUNS):
            client.search(
                collection_name="test_collection",
                query_vector=query_vectors[0],
                limit=TOP_K
            )
        
        # Benchmark latency
        latencies = []
        
        for test in range(NUM_TESTS):
            test_latencies = []
            
            for query in query_vectors:
                start_time = time.time()
                
                results = client.search(
                    collection_name="test_collection",
                    query_vector=query,
                    limit=TOP_K
                )
                
                end_time = time.time()
                query_time = (end_time - start_time) * 1000  # Convert to ms
                test_latencies.append(query_time)
            
            avg_latency = sum(test_latencies) / len(test_latencies)
            latencies.append(avg_latency)
            logger.info(f"Qdrant Test {test+1}/{NUM_TESTS}: Avg Latency = {avg_latency:.2f}ms")
        
        # Calculate overall statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        result = {
            "database": "Qdrant",
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "test_runs": NUM_TESTS,
            "queries_per_run": NUM_QUERIES,
            "top_k": TOP_K
        }
        
        return result
        
    except ImportError:
        logger.error("qdrant_client package not installed. Install with: pip install qdrant-client")
        return {"database": "Qdrant", "error": "Package not installed"}
        
    except Exception as e:
        logger.error(f"Error benchmarking Qdrant: {str(e)}")
        return {"database": "Qdrant", "error": str(e)}

def plot_results(results):
    """
    Plot benchmark results.
    
    Args:
        results: List of benchmark results
    """
    # Filter out results with errors
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        logger.error("No valid benchmark results to plot.")
        return
    
    # Extract data for plotting
    databases = [r["database"] for r in valid_results]
    avg_latencies = [r["avg_latency_ms"] for r in valid_results]
    min_latencies = [r["min_latency_ms"] for r in valid_results]
    max_latencies = [r["max_latency_ms"] for r in valid_results]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar positions
    bar_width = 0.25
    r1 = np.arange(len(databases))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    plt.bar(r1, avg_latencies, width=bar_width, label='Avg Latency', color='skyblue')
    plt.bar(r2, min_latencies, width=bar_width, label='Min Latency', color='lightgreen')
    plt.bar(r3, max_latencies, width=bar_width, label='Max Latency', color='salmon')
    
    # Add labels and title
    plt.xlabel('Database')
    plt.ylabel('Latency (ms)')
    plt.title('Vector Database Query Performance')
    plt.xticks([r + bar_width for r in range(len(databases))], databases)
    plt.legend()
    
    # Save the plot
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/benchmark_results.png")
    logger.info("Saved benchmark plot to results/benchmark_results.png")
    
    # Show the plot
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Benchmark vector database performance")
    parser.add_argument("--db", choices=["chromadb", "postgres", "qdrant", "all"], 
                       default="all", help="Which database to benchmark")
    parser.add_argument("--plot", action="store_true", help="Generate a plot of results")
    
    args = parser.parse_args()
    
    results = []
    
    if args.db in ["chromadb", "all"]:
        chromadb_result = benchmark_chromadb()
        results.append(chromadb_result)
    
    if args.db in ["postgres", "all"]:
        postgres_result = benchmark_postgres()
        results.append(postgres_result)
    
    if args.db in ["qdrant", "all"]:
        qdrant_result = benchmark_qdrant()
        results.append(qdrant_result)
    
    # Print summary
    logger.info("Benchmark Results Summary:")
    for result in results:
        if "error" in result:
            logger.info(f"{result['database']}: Error - {result['error']}")
        else:
            logger.info(f"{result['database']}: Avg Latency = {result['avg_latency_ms']:.2f}ms")
    
    # Save results to file
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Saved benchmark results to results/benchmark_results.json")
    
    # Plot results if requested
    if args.plot:
        try:
            plot_results(results)
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")

if __name__ == "__main__":
    main() 