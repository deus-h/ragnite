"""
Vector Database Query Benchmarker

This module provides tools for benchmarking vector database query performance,
including latency, throughput, recall, and precision.
"""

import time
import logging
import json
import statistics
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import concurrent.futures

# Configure logging
logger = logging.getLogger(__name__)

class BenchmarkResult:
    """
    Class to hold and format benchmark results.
    """
    
    def __init__(self, benchmark_type: str, results: Dict[str, Any]):
        """
        Initialize BenchmarkResult.
        
        Args:
            benchmark_type: Type of benchmark ('latency', 'throughput', 'recall', 'precision')
            results: Benchmark results
        """
        self.benchmark_type = benchmark_type
        self.results = results
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "benchmark_type": self.benchmark_type,
            "timestamp": self.timestamp,
            "results": self.results
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.
        
        Args:
            indent: JSON indentation level
            
        Returns:
            str: JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def __str__(self) -> str:
        """
        String representation of benchmark results.
        
        Returns:
            str: Formatted benchmark results
        """
        lines = [f"Benchmark Type: {self.benchmark_type}"]
        
        for key, value in self.results.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, float):
                        lines.append(f"  {subkey}: {subvalue:.4f}")
                    else:
                        lines.append(f"  {subkey}: {subvalue}")
            elif isinstance(value, float):
                lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)

class BaseQueryBenchmarker(ABC):
    """
    Base abstract class for vector database query benchmarkers.
    
    Query benchmarkers provide tools to measure query performance in terms of
    latency, throughput, recall, and precision.
    """
    
    def __init__(
        self,
        db_connector: Any = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BaseQueryBenchmarker.
        
        Args:
            db_connector: Vector database connector instance
            params: Parameters for the benchmarker
        """
        self.db_connector = db_connector
        self.params = params or {}
        self.default_params = self._get_default_params()
    
    @abstractmethod
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for the benchmarker.
        
        Returns:
            Dict[str, Any]: Default parameters
        """
        pass
    
    @abstractmethod
    def benchmark(
        self,
        collection_name: str,
        query_vectors: Optional[List[List[float]]] = None,
        **kwargs
    ) -> BenchmarkResult:
        """
        Run a benchmark on a collection.
        
        Args:
            collection_name: Name of the collection to benchmark
            query_vectors: Query vectors for benchmarking (if None, generate random vectors)
            **kwargs: Additional parameters for the benchmark
            
        Returns:
            BenchmarkResult: Benchmark results
        """
        pass
    
    def generate_random_queries(
        self,
        dimension: int,
        num_queries: int = 100,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate random query vectors for benchmarking.
        
        Args:
            dimension: Dimensionality of vectors
            num_queries: Number of query vectors to generate
            normalize: Whether to normalize vectors
            
        Returns:
            List[List[float]]: List of random query vectors
        """
        try:
            import numpy as np
            
            # Generate random vectors
            vectors = np.random.rand(num_queries, dimension).astype(np.float32)
            
            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / norms
            
            return vectors.tolist()
        
        except ImportError:
            logger.error("NumPy not available. Install with: pip install numpy")
            # Fallback to simple random vectors
            import random
            vectors = []
            for _ in range(num_queries):
                vector = [random.random() for _ in range(dimension)]
                if normalize:
                    magnitude = sum(x**2 for x in vector) ** 0.5
                    vector = [x/magnitude for x in vector]
                vectors.append(vector)
            return vectors
    
    def _check_db_connector(self):
        """
        Check if the database connector is provided and connected.
        
        Raises:
            ValueError: If the database connector is not provided or not connected
        """
        if self.db_connector is None:
            raise ValueError("Database connector not provided.")
        
        if not hasattr(self.db_connector, 'is_connected') or not self.db_connector.is_connected():
            raise ValueError("Database connector is not connected.")
    
    def _merge_params(self, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Merge default parameters with provided parameters.
        
        Args:
            parameters: Parameters to merge with defaults
            
        Returns:
            Dict[str, Any]: Merged parameters
        """
        merged = self.default_params.copy()
        if parameters:
            merged.update(parameters)
        return merged

class LatencyBenchmarker(BaseQueryBenchmarker):
    """
    Benchmarker for measuring vector database query latency.
    """
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for latency benchmarking.
        
        Returns:
            Dict[str, Any]: Default parameters
        """
        return {
            "num_queries": 50,    # Number of queries to run
            "top_k": 10,           # Number of results to retrieve per query
            "warmup_runs": 5,      # Number of warmup runs before benchmarking
            "include_percentiles": True  # Whether to include percentile metrics
        }
    
    def benchmark(
        self,
        collection_name: str,
        query_vectors: Optional[List[List[float]]] = None,
        num_queries: Optional[int] = None,
        top_k: Optional[int] = None,
        warmup_runs: Optional[int] = None,
        include_percentiles: Optional[bool] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark query latency on a collection.
        
        Args:
            collection_name: Name of the collection to benchmark
            query_vectors: Query vectors for benchmarking (if None, generate random vectors)
            num_queries: Number of queries to run (overrides default)
            top_k: Number of results to retrieve per query (overrides default)
            warmup_runs: Number of warmup runs before benchmarking (overrides default)
            include_percentiles: Whether to include percentile metrics (overrides default)
            filter: Filter to apply to queries
            
        Returns:
            BenchmarkResult: Latency benchmark results
        """
        self._check_db_connector()
        
        # Merge parameters
        params = self._merge_params({
            "num_queries": num_queries,
            "top_k": top_k,
            "warmup_runs": warmup_runs,
            "include_percentiles": include_percentiles
        })
        
        num_queries = params["num_queries"]
        top_k = params["top_k"]
        warmup_runs = params["warmup_runs"]
        include_percentiles = params["include_percentiles"]
        
        try:
            # Get collection info for dimension
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            
            # Generate random query vectors if not provided
            if query_vectors is None:
                query_vectors = self.generate_random_queries(dimension, num_queries)
            
            # Limit number of queries to available vectors
            num_queries = min(len(query_vectors), num_queries)
            
            # Run warmup queries
            logger.info(f"Running {warmup_runs} warmup queries...")
            for i in range(warmup_runs):
                warmup_idx = i % num_queries  # Cycle through available queries
                self.db_connector.search(
                    collection_name=collection_name,
                    query_vector=query_vectors[warmup_idx],
                    top_k=top_k,
                    filter=filter
                )
            
            # Run benchmark queries
            logger.info(f"Running {num_queries} benchmark queries...")
            latencies = []
            
            for i in range(num_queries):
                start_time = time.time()
                self.db_connector.search(
                    collection_name=collection_name,
                    query_vector=query_vectors[i],
                    top_k=top_k,
                    filter=filter
                )
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            # Basic results
            results = {
                "collection_name": collection_name,
                "benchmark_params": {
                    "num_queries": num_queries,
                    "top_k": top_k,
                    "warmup_runs": warmup_runs,
                    "filter_applied": filter is not None
                },
                "latency_ms": {
                    "avg": avg_latency,
                    "median": median_latency,
                    "min": min_latency,
                    "max": max_latency
                },
                "throughput_qps": 1000 / avg_latency  # Queries per second
            }
            
            # Add percentiles if requested
            if include_percentiles and len(latencies) >= 10:
                latencies.sort()
                p90_idx = int(0.9 * len(latencies))
                p95_idx = int(0.95 * len(latencies))
                p99_idx = int(0.99 * len(latencies))
                
                results["latency_ms"]["p90"] = latencies[p90_idx]
                results["latency_ms"]["p95"] = latencies[p95_idx]
                results["latency_ms"]["p99"] = latencies[p99_idx]
            
            return BenchmarkResult("latency", results)
        
        except Exception as e:
            logger.error(f"Error benchmarking latency: {str(e)}")
            return BenchmarkResult("latency", {"error": str(e)})

class ThroughputBenchmarker(BaseQueryBenchmarker):
    """
    Benchmarker for measuring vector database query throughput.
    """
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for throughput benchmarking.
        
        Returns:
            Dict[str, Any]: Default parameters
        """
        return {
            "num_queries": 100,    # Number of queries to run
            "top_k": 10,           # Number of results to retrieve per query
            "warmup_runs": 5,      # Number of warmup runs before benchmarking
            "concurrent_queries": 4,  # Number of concurrent queries
            "time_based": False,   # Whether to run for a fixed time instead of fixed queries
            "duration_seconds": 30  # Duration in seconds if time_based is True
        }
    
    def benchmark(
        self,
        collection_name: str,
        query_vectors: Optional[List[List[float]]] = None,
        num_queries: Optional[int] = None,
        top_k: Optional[int] = None,
        warmup_runs: Optional[int] = None,
        concurrent_queries: Optional[int] = None,
        time_based: Optional[bool] = None,
        duration_seconds: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark query throughput on a collection.
        
        Args:
            collection_name: Name of the collection to benchmark
            query_vectors: Query vectors for benchmarking (if None, generate random vectors)
            num_queries: Number of queries to run (overrides default)
            top_k: Number of results to retrieve per query (overrides default)
            warmup_runs: Number of warmup runs before benchmarking (overrides default)
            concurrent_queries: Number of concurrent queries (overrides default)
            time_based: Whether to run for a fixed time instead of fixed queries (overrides default)
            duration_seconds: Duration in seconds if time_based is True (overrides default)
            filter: Filter to apply to queries
            
        Returns:
            BenchmarkResult: Throughput benchmark results
        """
        self._check_db_connector()
        
        # Merge parameters
        params = self._merge_params({
            "num_queries": num_queries,
            "top_k": top_k,
            "warmup_runs": warmup_runs,
            "concurrent_queries": concurrent_queries,
            "time_based": time_based,
            "duration_seconds": duration_seconds
        })
        
        num_queries = params["num_queries"]
        top_k = params["top_k"]
        warmup_runs = params["warmup_runs"]
        concurrent_queries = params["concurrent_queries"]
        time_based = params["time_based"]
        duration_seconds = params["duration_seconds"]
        
        try:
            # Get collection info for dimension
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            
            # Generate random query vectors if not provided
            if query_vectors is None:
                # For time-based, generate more vectors
                if time_based:
                    query_vectors = self.generate_random_queries(dimension, max(1000, num_queries))
                else:
                    query_vectors = self.generate_random_queries(dimension, num_queries)
            
            # Run warmup queries
            logger.info(f"Running {warmup_runs} warmup queries...")
            for i in range(warmup_runs):
                warmup_idx = i % len(query_vectors)  # Cycle through available queries
                self.db_connector.search(
                    collection_name=collection_name,
                    query_vector=query_vectors[warmup_idx],
                    top_k=top_k,
                    filter=filter
                )
            
            # Function to run a single query and return latency
            def run_query(query_vector):
                start_time = time.time()
                self.db_connector.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    top_k=top_k,
                    filter=filter
                )
                end_time = time.time()
                return (end_time - start_time) * 1000  # Convert to ms
            
            # Run benchmark based on mode (time-based or query-based)
            if time_based:
                logger.info(f"Running throughput benchmark for {duration_seconds} seconds...")
                start_time = time.time()
                end_time = start_time + duration_seconds
                
                query_count = 0
                latencies = []
                
                # Continue until duration is reached
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_queries) as executor:
                    while time.time() < end_time:
                        # Submit batch of queries
                        batch_size = min(concurrent_queries, len(query_vectors))
                        futures = []
                        
                        for i in range(batch_size):
                            query_idx = (query_count + i) % len(query_vectors)
                            futures.append(executor.submit(run_query, query_vectors[query_idx]))
                        
                        # Collect results
                        for future in concurrent.futures.as_completed(futures):
                            latency = future.result()
                            latencies.append(latency)
                            query_count += 1
                
                total_time = time.time() - start_time
                
            else:
                # Query-based benchmark
                logger.info(f"Running {num_queries} benchmark queries with concurrency {concurrent_queries}...")
                latencies = []
                start_time = time.time()
                
                # Limit to available vectors
                num_queries = min(len(query_vectors), num_queries)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_queries) as executor:
                    futures = [executor.submit(run_query, query_vectors[i % len(query_vectors)]) for i in range(num_queries)]
                    
                    for future in concurrent.futures.as_completed(futures):
                        latency = future.result()
                        latencies.append(latency)
                
                total_time = time.time() - start_time
                query_count = num_queries
            
            # Calculate statistics
            if latencies:
                avg_latency = statistics.mean(latencies)
                median_latency = statistics.median(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                
                # Calculate throughput
                throughput = query_count / total_time
                
                # Calculate percentiles
                latencies.sort()
                p90_idx = int(0.9 * len(latencies))
                p95_idx = int(0.95 * len(latencies))
                p99_idx = int(0.99 * len(latencies))
                
                results = {
                    "collection_name": collection_name,
                    "benchmark_params": {
                        "mode": "time_based" if time_based else "query_based",
                        "concurrent_queries": concurrent_queries,
                        "duration_seconds" if time_based else "num_queries": duration_seconds if time_based else num_queries,
                        "top_k": top_k,
                        "filter_applied": filter is not None
                    },
                    "throughput": {
                        "queries_completed": query_count,
                        "total_time_seconds": total_time,
                        "queries_per_second": throughput
                    },
                    "latency_ms": {
                        "avg": avg_latency,
                        "median": median_latency,
                        "min": min_latency,
                        "max": max_latency,
                        "p90": latencies[p90_idx],
                        "p95": latencies[p95_idx],
                        "p99": latencies[p99_idx]
                    }
                }
            else:
                results = {
                    "collection_name": collection_name,
                    "error": "No queries completed during benchmark"
                }
            
            return BenchmarkResult("throughput", results)
        
        except Exception as e:
            logger.error(f"Error benchmarking throughput: {str(e)}")
            return BenchmarkResult("throughput", {"error": str(e)})

class RecallBenchmarker(BaseQueryBenchmarker):
    """
    Benchmarker for measuring vector database recall at different k values.
    """
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for recall benchmarking.
        
        Returns:
            Dict[str, Any]: Default parameters
        """
        return {
            "num_queries": 50,     # Number of queries to run
            "k_values": [1, 5, 10, 20, 50, 100],  # k values to measure recall at
            "ground_truth_k": 100  # k value for ground truth (exact search)
        }
    
    def benchmark(
        self,
        collection_name: str,
        query_vectors: Optional[List[List[float]]] = None,
        num_queries: Optional[int] = None,
        k_values: Optional[List[int]] = None,
        ground_truth_k: Optional[int] = None,
        ground_truth: Optional[List[List[str]]] = None,
        ground_truth_fn: Optional[Callable] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark query recall on a collection.
        
        Args:
            collection_name: Name of the collection to benchmark
            query_vectors: Query vectors for benchmarking (if None, generate random vectors)
            num_queries: Number of queries to run (overrides default)
            k_values: k values to measure recall at (overrides default)
            ground_truth_k: k value for ground truth (overrides default)
            ground_truth: Pre-computed ground truth results
            ground_truth_fn: Function to compute ground truth (takes query vector, returns list of ids)
            filter: Filter to apply to queries
            
        Returns:
            BenchmarkResult: Recall benchmark results
        """
        self._check_db_connector()
        
        # Merge parameters
        params = self._merge_params({
            "num_queries": num_queries,
            "k_values": k_values,
            "ground_truth_k": ground_truth_k
        })
        
        num_queries = params["num_queries"]
        k_values = params["k_values"]
        ground_truth_k = params["ground_truth_k"]
        
        try:
            # Get collection info for dimension
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            
            # Generate random query vectors if not provided
            if query_vectors is None:
                query_vectors = self.generate_random_queries(dimension, num_queries)
            
            # Limit number of queries to available vectors
            num_queries = min(len(query_vectors), num_queries)
            
            # Ensure max_k is computed
            max_k = max(k_values)
            if ground_truth is None and ground_truth_fn is None:
                # We need to compute ground truth with exhaustive search
                if hasattr(self.db_connector, "exhaustive_search"):
                    logger.info(f"Computing ground truth with exhaustive search...")
                    ground_truth = []
                    for i in range(num_queries):
                        exact_results = self.db_connector.exhaustive_search(
                            collection_name=collection_name,
                            query_vector=query_vectors[i],
                            top_k=ground_truth_k,
                            filter=filter
                        )
                        ground_truth.append([r["id"] for r in exact_results])
                else:
                    # Use regular search with large k as approximate ground truth
                    logger.warning("Exhaustive search not supported by connector. Using regular search with large k as ground truth.")
                    ground_truth = []
                    for i in range(num_queries):
                        results = self.db_connector.search(
                            collection_name=collection_name,
                            query_vector=query_vectors[i],
                            top_k=ground_truth_k,
                            filter=filter
                        )
                        ground_truth.append([r["id"] for r in results])
            elif ground_truth_fn:
                # Use provided function to compute ground truth
                logger.info(f"Computing ground truth with provided function...")
                ground_truth = []
                for i in range(num_queries):
                    truth = ground_truth_fn(query_vectors[i])
                    ground_truth.append(truth)
            
            # Run benchmark queries at different k values
            logger.info(f"Running {num_queries} benchmark queries at {len(k_values)} k values...")
            
            recalls = {k: [] for k in k_values}
            
            for i in range(num_queries):
                # Get ground truth for this query
                truth_set = set(ground_truth[i])
                
                # Run query at each k value
                for k in k_values:
                    if k > len(truth_set):
                        # Skip if k is larger than ground truth
                        continue
                    
                    results = self.db_connector.search(
                        collection_name=collection_name,
                        query_vector=query_vectors[i],
                        top_k=k,
                        filter=filter
                    )
                    
                    # Extract result ids
                    result_ids = [r["id"] for r in results]
                    
                    # Calculate recall@k
                    intersection = len(set(result_ids).intersection(truth_set))
                    recall = intersection / min(k, len(truth_set))
                    recalls[k].append(recall)
            
            # Calculate average recall for each k
            avg_recalls = {}
            for k, values in recalls.items():
                if values:
                    avg_recalls[f"recall@{k}"] = statistics.mean(values)
            
            results = {
                "collection_name": collection_name,
                "benchmark_params": {
                    "num_queries": num_queries,
                    "k_values": k_values,
                    "ground_truth_k": ground_truth_k,
                    "filter_applied": filter is not None
                },
                "recalls": avg_recalls
            }
            
            return BenchmarkResult("recall", results)
        
        except Exception as e:
            logger.error(f"Error benchmarking recall: {str(e)}")
            return BenchmarkResult("recall", {"error": str(e)})

class PrecisionBenchmarker(BaseQueryBenchmarker):
    """
    Benchmarker for measuring vector database precision at different k values.
    """
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for precision benchmarking.
        
        Returns:
            Dict[str, Any]: Default parameters
        """
        return {
            "num_queries": 50,     # Number of queries to run
            "k_values": [1, 5, 10, 20, 50, 100],  # k values to measure precision at
            "relevance_threshold": 0.7  # Similarity threshold for relevance
        }
    
    def benchmark(
        self,
        collection_name: str,
        query_vectors: Optional[List[List[float]]] = None,
        num_queries: Optional[int] = None,
        k_values: Optional[List[int]] = None,
        relevance_threshold: Optional[float] = None,
        relevance_fn: Optional[Callable] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark query precision on a collection.
        
        Args:
            collection_name: Name of the collection to benchmark
            query_vectors: Query vectors for benchmarking (if None, generate random vectors)
            num_queries: Number of queries to run (overrides default)
            k_values: k values to measure precision at (overrides default)
            relevance_threshold: Similarity threshold for relevance (overrides default)
            relevance_fn: Function to determine relevance (takes query vector and result, returns bool)
            filter: Filter to apply to queries
            
        Returns:
            BenchmarkResult: Precision benchmark results
        """
        self._check_db_connector()
        
        # Merge parameters
        params = self._merge_params({
            "num_queries": num_queries,
            "k_values": k_values,
            "relevance_threshold": relevance_threshold
        })
        
        num_queries = params["num_queries"]
        k_values = params["k_values"]
        relevance_threshold = params["relevance_threshold"]
        
        try:
            # Get collection info for dimension
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            
            # Generate random query vectors if not provided
            if query_vectors is None:
                query_vectors = self.generate_random_queries(dimension, num_queries)
            
            # Limit number of queries to available vectors
            num_queries = min(len(query_vectors), num_queries)
            
            # Define default relevance function if not provided
            if relevance_fn is None:
                def default_relevance_fn(query_vec, result):
                    # Use score as measure of relevance
                    return result["score"] >= relevance_threshold
                
                relevance_fn = default_relevance_fn
            
            # Run benchmark queries at different k values
            logger.info(f"Running {num_queries} benchmark queries at {len(k_values)} k values...")
            
            precisions = {k: [] for k in k_values}
            
            for i in range(num_queries):
                query_vec = query_vectors[i]
                
                # Run query at max k
                max_k = max(k_values)
                results = self.db_connector.search(
                    collection_name=collection_name,
                    query_vector=query_vec,
                    top_k=max_k,
                    filter=filter
                )
                
                # Calculate precision at each k
                for k in k_values:
                    if k > len(results):
                        # Skip if k is larger than results
                        continue
                    
                    # Get top-k results
                    top_k_results = results[:k]
                    
                    # Count relevant results
                    relevant_count = sum(1 for r in top_k_results if relevance_fn(query_vec, r))
                    
                    # Calculate precision@k
                    precision = relevant_count / k
                    precisions[k].append(precision)
            
            # Calculate average precision for each k
            avg_precisions = {}
            for k, values in precisions.items():
                if values:
                    avg_precisions[f"precision@{k}"] = statistics.mean(values)
            
            results = {
                "collection_name": collection_name,
                "benchmark_params": {
                    "num_queries": num_queries,
                    "k_values": k_values,
                    "relevance_threshold": relevance_threshold,
                    "filter_applied": filter is not None
                },
                "precisions": avg_precisions
            }
            
            return BenchmarkResult("precision", results)
        
        except Exception as e:
            logger.error(f"Error benchmarking precision: {str(e)}")
            return BenchmarkResult("precision", {"error": str(e)})

def get_query_benchmarker(
    benchmarker_type: str,
    db_connector: Any = None,
    params: Optional[Dict[str, Any]] = None
) -> BaseQueryBenchmarker:
    """
    Get a query benchmarker based on the benchmarker type.
    
    Args:
        benchmarker_type: Type of benchmarker ('latency', 'throughput', 'recall', 'precision')
        db_connector: Vector database connector instance
        params: Parameters for the benchmarker
        
    Returns:
        BaseQueryBenchmarker: Query benchmarker
        
    Raises:
        ValueError: If the benchmarker type is not supported
    """
    # Normalize benchmarker type
    benchmarker_type = benchmarker_type.lower().strip()
    
    # Create benchmarker based on type
    if benchmarker_type in ["latency", "latency_benchmarker"]:
        return LatencyBenchmarker(db_connector, params)
    
    elif benchmarker_type in ["throughput", "throughput_benchmarker"]:
        return ThroughputBenchmarker(db_connector, params)
    
    elif benchmarker_type in ["recall", "recall_benchmarker"]:
        return RecallBenchmarker(db_connector, params)
    
    elif benchmarker_type in ["precision", "precision_benchmarker"]:
        return PrecisionBenchmarker(db_connector, params)
    
    else:
        supported_types = ["latency", "throughput", "recall", "precision"]
        raise ValueError(f"Unsupported benchmarker type: {benchmarker_type}. Supported types: {supported_types}")

def run_benchmark(
    db_connector: Any,
    collection_name: str,
    benchmarker_type: str = "latency",
    query_vectors: Optional[List[List[float]]] = None,
    params: Optional[Dict[str, Any]] = None
) -> BenchmarkResult:
    """
    Run a benchmark using the appropriate benchmarker.
    
    Args:
        db_connector: Vector database connector instance
        collection_name: Name of the collection to benchmark
        benchmarker_type: Type of benchmarker ('latency', 'throughput', 'recall', 'precision')
        query_vectors: Query vectors for benchmarking (if None, generate random vectors)
        params: Parameters for the benchmark
        
    Returns:
        BenchmarkResult: Benchmark results
    """
    try:
        # Get benchmarker
        benchmarker = get_query_benchmarker(benchmarker_type, db_connector, params)
        
        # Run benchmark
        return benchmarker.benchmark(
            collection_name=collection_name,
            query_vectors=query_vectors,
            **(params or {})
        )
        
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        return BenchmarkResult(benchmarker_type, {"error": str(e)}) 