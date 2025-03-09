"""
Vector Database Index Optimizers

This module provides tools for optimizing vector database indices to improve
search performance, memory usage, and query latency.
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

class BaseIndexOptimizer(ABC):
    """
    Base abstract class for vector database index optimizers.
    
    Index optimizers provide tools to analyze and optimize vector database indices
    for better performance, memory usage, and query latency.
    """
    
    def __init__(
        self,
        db_connector: Any = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BaseIndexOptimizer.
        
        Args:
            db_connector: Vector database connector instance
            params: Parameters for the optimizer
        """
        self.db_connector = db_connector
        self.params = params or {}
        self.default_params = self._get_default_params()
    
    @abstractmethod
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for the optimizer.
        
        Returns:
            Dict[str, Any]: Default parameters
        """
        pass
    
    @abstractmethod
    def analyze_index(
        self,
        collection_name: str,
        stats_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Analyze the current index configuration and performance.
        
        Args:
            collection_name: Name of the collection to analyze
            stats_format: Format for the statistics ('json', 'text', 'dict')
            
        Returns:
            Dict[str, Any]: Analysis results with statistics and recommendations
        """
        pass
    
    @abstractmethod
    def optimize_index(
        self,
        collection_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize the index based on parameters or auto-tuning.
        
        Args:
            collection_name: Name of the collection to optimize
            parameters: Parameters for the optimization (if None, use auto-tuning)
            dry_run: If True, only show what would be done without making changes
            
        Returns:
            Dict[str, Any]: Results of the optimization process
        """
        pass
    
    @abstractmethod
    def estimate_memory_usage(
        self,
        collection_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimate memory usage for the index with current or proposed parameters.
        
        Args:
            collection_name: Name of the collection
            parameters: Proposed parameters (if None, use current parameters)
            
        Returns:
            Dict[str, Any]: Memory usage estimates
        """
        pass
    
    @abstractmethod
    def benchmark_index(
        self,
        collection_name: str,
        query_vectors: Optional[List[List[float]]] = None,
        num_queries: int = 100,
        top_k: int = 10,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark index performance with current or proposed parameters.
        
        Args:
            collection_name: Name of the collection
            query_vectors: Query vectors for benchmarking (if None, generate random vectors)
            num_queries: Number of queries to run if query_vectors is None
            top_k: Number of results to retrieve for each query
            parameters: Proposed parameters (if None, use current parameters)
            
        Returns:
            Dict[str, Any]: Benchmark results
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
    
    def _format_stats(self, stats: Dict[str, Any], format_type: str = "json") -> Any:
        """
        Format statistics based on the requested format.
        
        Args:
            stats: Statistics to format
            format_type: Format type ('json', 'text', 'dict')
            
        Returns:
            Any: Formatted statistics
        """
        if format_type.lower() == "json":
            return json.dumps(stats, indent=2)
        elif format_type.lower() == "text":
            lines = []
            self._dict_to_text(stats, lines)
            return "\n".join(lines)
        else:  # dict
            return stats
    
    def _dict_to_text(self, data: Dict[str, Any], lines: List[str], indent: int = 0):
        """
        Convert a dictionary to text lines.
        
        Args:
            data: Dictionary to convert
            lines: List to append lines to
            indent: Current indentation level
        """
        for key, value in data.items():
            prefix = "  " * indent
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                self._dict_to_text(value, lines, indent + 1)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    lines.append(f"{prefix}  -")
                    self._dict_to_text(item, lines, indent + 2)
            else:
                lines.append(f"{prefix}{key}: {value}")


class HNSWOptimizer(BaseIndexOptimizer):
    """
    Optimizer for HNSW (Hierarchical Navigable Small World) indices.
    
    HNSW is an algorithm for approximate nearest neighbor search that builds
    a hierarchical graph structure for efficient navigation.
    """
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for HNSW optimizer.
        
        Returns:
            Dict[str, Any]: Default parameters
        """
        return {
            "ef_construction": 100,    # Size of the dynamic list for the nearest neighbors (during construction)
            "ef_search": 50,           # Size of the dynamic list for the nearest neighbors (during search)
            "m": 16,                   # Maximum number of connections per element
            "max_elements": 1000000,   # Maximum number of elements in the index
            "index_thread_qty": 4      # Number of threads to use for indexing
        }
    
    def analyze_index(
        self,
        collection_name: str,
        stats_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Analyze the HNSW index configuration and performance.
        
        Args:
            collection_name: Name of the collection to analyze
            stats_format: Format for the statistics ('json', 'text', 'dict')
            
        Returns:
            Dict[str, Any]: Analysis results with statistics and recommendations
        """
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            count = collection_info.get("count", 0)
            
            # Basic stats
            stats = {
                "collection_name": collection_name,
                "vector_count": count,
                "dimension": dimension,
                "current_params": {},
                "recommendations": {}
            }
            
            # Extract current parameters (implementation specific)
            if hasattr(self.db_connector, "get_index_params"):
                current_params = self.db_connector.get_index_params(collection_name)
                stats["current_params"] = current_params
            
            # Generate recommendations
            recommendations = {}
            
            # ef_construction recommendation
            if count < 10000:
                recommendations["ef_construction"] = max(50, self.default_params["ef_construction"] // 2)
            elif count < 100000:
                recommendations["ef_construction"] = self.default_params["ef_construction"]
            else:
                recommendations["ef_construction"] = min(200, self.default_params["ef_construction"] * 2)
            
            # m recommendation
            if dimension <= 20:
                recommendations["m"] = 12
            elif dimension <= 100:
                recommendations["m"] = 16
            else:
                recommendations["m"] = 24
            
            # ef_search recommendation
            recommendations["ef_search"] = max(50, int(recommendations["m"] * 4))
            
            # thread recommendation
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            recommendations["index_thread_qty"] = max(1, min(cpu_count - 1, 4))
            
            stats["recommendations"] = recommendations
            
            return self._format_stats(stats, stats_format)
        
        except Exception as e:
            logger.error(f"Error analyzing HNSW index: {str(e)}")
            return {"error": str(e)}
    
    def optimize_index(
        self,
        collection_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize the HNSW index based on parameters or auto-tuning.
        
        Args:
            collection_name: Name of the collection to optimize
            parameters: Parameters for the optimization (if None, use auto-tuning)
            dry_run: If True, only show what would be done without making changes
            
        Returns:
            Dict[str, Any]: Results of the optimization process
        """
        self._check_db_connector()
        
        try:
            # If no parameters provided, analyze and use recommendations
            if parameters is None:
                analysis = self.analyze_index(collection_name, stats_format="dict")
                if "error" in analysis:
                    return analysis
                parameters = analysis.get("recommendations", {})
            
            # Merge with defaults
            params = self._merge_params(parameters)
            
            result = {
                "collection_name": collection_name,
                "optimizer": "HNSW",
                "parameters": params,
                "dry_run": dry_run
            }
            
            if not dry_run:
                # Implementation depends on database connector
                if hasattr(self.db_connector, "optimize_index"):
                    optimization_result = self.db_connector.optimize_index(
                        collection_name=collection_name,
                        index_type="hnsw",
                        parameters=params
                    )
                    result["status"] = "success"
                    result["details"] = optimization_result
                else:
                    # Best-effort optimization using recreate
                    collection_info = self.db_connector.get_collection_info(collection_name)
                    dimension = collection_info.get("dimension")
                    distance_metric = collection_info.get("distance_metric", "cosine")
                    
                    # For databases that don't support direct optimization, we need to:
                    # 1. Get all vectors and metadata from the collection
                    # 2. Delete the collection
                    # 3. Recreate the collection with optimized parameters
                    # 4. Reinsert all vectors and metadata
                    
                    # This is a simplified approach and might not be suitable for large collections
                    logger.warning(f"Direct index optimization not supported for {collection_name}. "
                                 f"Using recreate approach (might be slow for large collections).")
                    
                    # Get all vectors - this assumes the connector has a method to get all vectors
                    # which might not be efficient for very large collections
                    all_vectors = []
                    all_ids = []
                    all_metadata = []
                    
                    # This is a placeholder - you'd need to implement collection backup and restore
                    # based on your database connector's capabilities
                    result["status"] = "not_implemented"
                    result["details"] = "Direct optimization not supported for this database"
            
            return result
        
        except Exception as e:
            logger.error(f"Error optimizing HNSW index: {str(e)}")
            return {"error": str(e)}
    
    def estimate_memory_usage(
        self,
        collection_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimate memory usage for the HNSW index with current or proposed parameters.
        
        Args:
            collection_name: Name of the collection
            parameters: Proposed parameters (if None, use current parameters)
            
        Returns:
            Dict[str, Any]: Memory usage estimates
        """
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            count = collection_info.get("count", 0)
            
            # Get parameters
            if parameters is None:
                if hasattr(self.db_connector, "get_index_params"):
                    current_params = self.db_connector.get_index_params(collection_name)
                    parameters = current_params
                else:
                    # Use defaults if can't get current
                    parameters = self.default_params
            
            # Merge with defaults
            params = self._merge_params(parameters)
            
            # Estimate HNSW memory usage
            # Formula: vertices * (dim * 4 + m * 8) bytes
            # - Each vertex stores a vector (dim * 4 bytes) and m links (8 bytes per link)
            vector_data_size = count * dimension * 4  # 4 bytes per float32
            m = params.get("m", 16)
            link_data_size = count * m * 8  # 8 bytes per link (assuming 64-bit pointers)
            
            index_size = vector_data_size + link_data_size
            
            # Add overhead for multi-layer structure (about 10%)
            total_size = index_size * 1.1
            
            # Format sizes in human-readable form
            result = {
                "collection_name": collection_name,
                "vector_count": count,
                "dimension": dimension,
                "parameters": params,
                "estimated_memory": {
                    "vector_data_bytes": vector_data_size,
                    "link_data_bytes": link_data_size,
                    "total_bytes": total_size,
                    "total_mb": total_size / (1024 * 1024),
                    "total_gb": total_size / (1024 * 1024 * 1024)
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error estimating memory usage: {str(e)}")
            return {"error": str(e)}
    
    def benchmark_index(
        self,
        collection_name: str,
        query_vectors: Optional[List[List[float]]] = None,
        num_queries: int = 100,
        top_k: int = 10,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark HNSW index performance with current or proposed parameters.
        
        Args:
            collection_name: Name of the collection
            query_vectors: Query vectors for benchmarking (if None, generate random vectors)
            num_queries: Number of queries to run if query_vectors is None
            top_k: Number of results to retrieve for each query
            parameters: Proposed parameters (if None, use current parameters)
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            
            # Generate random queries if not provided
            if query_vectors is None:
                query_vectors = self.generate_random_queries(dimension, num_queries)
            
            # Check if we can optimize the index with the given parameters
            if parameters is not None:
                # This would require creating a temporary index with the new parameters
                # For now, we'll just log a warning
                logger.warning("Benchmarking with proposed parameters not supported yet. "
                             "Using current index configuration.")
            
            # Perform benchmark
            import time
            import statistics
            
            latencies = []
            num_vectors = min(len(query_vectors), num_queries)
            
            for i in range(num_vectors):
                start_time = time.time()
                results = self.db_connector.search(
                    collection_name=collection_name,
                    query_vector=query_vectors[i],
                    top_k=top_k
                )
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            p95_latency = None
            p99_latency = None
            
            try:
                latencies.sort()
                p95_idx = int(0.95 * len(latencies))
                p99_idx = int(0.99 * len(latencies))
                p95_latency = latencies[p95_idx]
                p99_latency = latencies[p99_idx]
            except:
                pass
            
            result = {
                "collection_name": collection_name,
                "benchmark_params": {
                    "num_queries": num_vectors,
                    "top_k": top_k
                },
                "latency_ms": {
                    "avg": avg_latency,
                    "median": median_latency,
                    "min": min_latency,
                    "max": max_latency,
                    "p95": p95_latency,
                    "p99": p99_latency
                },
                "throughput_qps": 1000 / avg_latency  # Queries per second
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error benchmarking index: {str(e)}")
            return {"error": str(e)}


class IVFFlatOptimizer(BaseIndexOptimizer):
    """
    Optimizer for IVFFlat (Inverted File with Flat Compression) indices.
    
    IVFFlat is an algorithm that partitions the vector space into clusters
    and performs exact search within relevant clusters.
    """
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for IVFFlat optimizer.
        
        Returns:
            Dict[str, Any]: Default parameters
        """
        return {
            "nlist": 100,              # Number of clusters/centroids
            "nprobe": 10,              # Number of clusters to search during query
            "max_elements": 1000000,   # Maximum number of elements in the index
            "index_thread_qty": 4      # Number of threads to use for indexing
        }
    
    def analyze_index(
        self,
        collection_name: str,
        stats_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Analyze the IVFFlat index configuration and performance.
        
        Args:
            collection_name: Name of the collection to analyze
            stats_format: Format for the statistics ('json', 'text', 'dict')
            
        Returns:
            Dict[str, Any]: Analysis results with statistics and recommendations
        """
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            count = collection_info.get("count", 0)
            
            # Basic stats
            stats = {
                "collection_name": collection_name,
                "vector_count": count,
                "dimension": dimension,
                "current_params": {},
                "recommendations": {}
            }
            
            # Extract current parameters (implementation specific)
            if hasattr(self.db_connector, "get_index_params"):
                current_params = self.db_connector.get_index_params(collection_name)
                stats["current_params"] = current_params
            
            # Generate recommendations
            recommendations = {}
            
            # nlist recommendation - typically sqrt(n) for balanced clusters
            optimal_nlist = int(count ** 0.5)
            recommendations["nlist"] = max(16, min(optimal_nlist, 8192))
            
            # nprobe recommendation - typically 1-2% of nlist
            recommendations["nprobe"] = max(1, int(recommendations["nlist"] * 0.02))
            
            # thread recommendation
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            recommendations["index_thread_qty"] = max(1, min(cpu_count - 1, 4))
            
            stats["recommendations"] = recommendations
            
            return self._format_stats(stats, stats_format)
        
        except Exception as e:
            logger.error(f"Error analyzing IVFFlat index: {str(e)}")
            return {"error": str(e)}
    
    def optimize_index(
        self,
        collection_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize the IVFFlat index based on parameters or auto-tuning.
        
        Args:
            collection_name: Name of the collection to optimize
            parameters: Parameters for the optimization (if None, use auto-tuning)
            dry_run: If True, only show what would be done without making changes
            
        Returns:
            Dict[str, Any]: Results of the optimization process
        """
        self._check_db_connector()
        
        try:
            # If no parameters provided, analyze and use recommendations
            if parameters is None:
                analysis = self.analyze_index(collection_name, stats_format="dict")
                if "error" in analysis:
                    return analysis
                parameters = analysis.get("recommendations", {})
            
            # Merge with defaults
            params = self._merge_params(parameters)
            
            result = {
                "collection_name": collection_name,
                "optimizer": "IVFFlat",
                "parameters": params,
                "dry_run": dry_run
            }
            
            if not dry_run:
                # Implementation depends on database connector
                if hasattr(self.db_connector, "optimize_index"):
                    optimization_result = self.db_connector.optimize_index(
                        collection_name=collection_name,
                        index_type="ivfflat",
                        parameters=params
                    )
                    result["status"] = "success"
                    result["details"] = optimization_result
                else:
                    # Similar approach to HNSW optimizer for databases without direct optimization
                    result["status"] = "not_implemented"
                    result["details"] = "Direct optimization not supported for this database"
            
            return result
        
        except Exception as e:
            logger.error(f"Error optimizing IVFFlat index: {str(e)}")
            return {"error": str(e)}
    
    def estimate_memory_usage(
        self,
        collection_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimate memory usage for the IVFFlat index with current or proposed parameters.
        
        Args:
            collection_name: Name of the collection
            parameters: Proposed parameters (if None, use current parameters)
            
        Returns:
            Dict[str, Any]: Memory usage estimates
        """
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            count = collection_info.get("count", 0)
            
            # Get parameters
            if parameters is None:
                if hasattr(self.db_connector, "get_index_params"):
                    current_params = self.db_connector.get_index_params(collection_name)
                    parameters = current_params
                else:
                    # Use defaults if can't get current
                    parameters = self.default_params
            
            # Merge with defaults
            params = self._merge_params(parameters)
            
            # Estimate IVFFlat memory usage
            # Formula: (centroids * dim * 4) + (vectors * dim * 4) + (centroids * 100) bytes
            nlist = params.get("nlist", 100)
            
            centroid_data_size = nlist * dimension * 4  # 4 bytes per float32
            vector_data_size = count * dimension * 4  # 4 bytes per float32
            index_overhead = nlist * 100  # Rough estimate for cluster metadata
            
            index_size = centroid_data_size + vector_data_size + index_overhead
            
            # Add some overhead (about 5%)
            total_size = index_size * 1.05
            
            # Format sizes in human-readable form
            result = {
                "collection_name": collection_name,
                "vector_count": count,
                "dimension": dimension,
                "parameters": params,
                "estimated_memory": {
                    "centroid_data_bytes": centroid_data_size,
                    "vector_data_bytes": vector_data_size,
                    "index_overhead_bytes": index_overhead,
                    "total_bytes": total_size,
                    "total_mb": total_size / (1024 * 1024),
                    "total_gb": total_size / (1024 * 1024 * 1024)
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error estimating memory usage: {str(e)}")
            return {"error": str(e)}
    
    def benchmark_index(
        self,
        collection_name: str,
        query_vectors: Optional[List[List[float]]] = None,
        num_queries: int = 100,
        top_k: int = 10,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark IVFFlat index performance with current or proposed parameters.
        
        Args:
            collection_name: Name of the collection
            query_vectors: Query vectors for benchmarking (if None, generate random vectors)
            num_queries: Number of queries to run if query_vectors is None
            top_k: Number of results to retrieve for each query
            parameters: Proposed parameters (if None, use current parameters)
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        # Implementation similar to HNSW benchmark_index but with IVFFlat specific options
        # For now, we'll reuse the same benchmarking code as it's database-agnostic
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            if "dimension" not in collection_info:
                raise ValueError(f"Could not determine dimension for collection {collection_name}")
            
            dimension = collection_info.get("dimension")
            
            # Generate random queries if not provided
            if query_vectors is None:
                query_vectors = self.generate_random_queries(dimension, num_queries)
            
            # Check if we can optimize the index with the given parameters
            if parameters is not None:
                logger.warning("Benchmarking with proposed parameters not supported yet. "
                             "Using current index configuration.")
            
            # Perform benchmark
            import time
            import statistics
            
            latencies = []
            num_vectors = min(len(query_vectors), num_queries)
            
            # If the connector supports search_with_params, use it
            search_func = getattr(self.db_connector, "search_with_params", None)
            if search_func is None:
                search_func = self.db_connector.search
                search_kwargs = {
                    "collection_name": collection_name,
                    "top_k": top_k
                }
            else:
                search_kwargs = {
                    "collection_name": collection_name,
                    "top_k": top_k,
                    "search_params": parameters or {}
                }
            
            for i in range(num_vectors):
                start_time = time.time()
                results = search_func(
                    query_vector=query_vectors[i],
                    **search_kwargs
                )
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            p95_latency = None
            p99_latency = None
            
            try:
                latencies.sort()
                p95_idx = int(0.95 * len(latencies))
                p99_idx = int(0.99 * len(latencies))
                p95_latency = latencies[p95_idx]
                p99_latency = latencies[p99_idx]
            except:
                pass
            
            result = {
                "collection_name": collection_name,
                "benchmark_params": {
                    "num_queries": num_vectors,
                    "top_k": top_k
                },
                "latency_ms": {
                    "avg": avg_latency,
                    "median": median_latency,
                    "min": min_latency,
                    "max": max_latency,
                    "p95": p95_latency,
                    "p99": p99_latency
                },
                "throughput_qps": 1000 / avg_latency  # Queries per second
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error benchmarking index: {str(e)}")
            return {"error": str(e)}


class PqIndexOptimizer(BaseIndexOptimizer):
    """
    Optimizer for Product Quantization (PQ) indices.
    
    Product Quantization is a compression technique that divides vectors into subvectors,
    quantizes each subvector separately, and combines the quantized subvectors.
    This significantly reduces memory usage and speeds up similarity search.
    """
    
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for PQ optimization.
        
        Returns:
            Dict[str, Any]: Default parameters
        """
        return {
            "num_subvectors": 8,        # Number of subvectors to divide each vector into
            "bits_per_subvector": 8,    # Bits per subvector (usually 8 for 256 centroids)
            "training_sample_count": 10000,  # Number of vectors to use for training
            "max_iterations": 25,       # Maximum number of iterations for k-means
            "epsilon": 0.001,           # Convergence threshold for k-means
            "use_opq": False,           # Whether to use Optimized Product Quantization
            "use_residual": False,      # Whether to use Residual Product Quantization
            "search_nprobe": 10         # Number of clusters to search during query
        }
    
    def analyze_index(
        self,
        collection_name: str,
        stats_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Analyze the current index configuration and performance.
        
        Args:
            collection_name: Name of the collection to analyze
            stats_format: Format for the statistics ('json', 'text', 'dict')
            
        Returns:
            Dict[str, Any]: Analysis results with statistics and recommendations
        """
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            # Check if collection exists
            if "error" in collection_info:
                return {"error": f"Collection '{collection_name}' not found or error retrieving info"}
            
            # Get collection stats
            vector_count = self.db_connector.count_vectors(collection_name)
            
            # Extract dimensionality
            dimension = self._extract_dimension(collection_info)
            if dimension is None:
                return {"error": "Could not determine vector dimension from collection info"}
            
            # Get index information if available
            index_params = self._extract_index_params(collection_info)
            is_pq = self._is_pq_index(index_params)
            
            # Generate analysis
            analysis = {
                "collection_name": collection_name,
                "vector_count": vector_count,
                "dimension": dimension,
                "current_index": {
                    "type": "product_quantization" if is_pq else "unknown",
                    "parameters": index_params or {}
                },
                "recommendations": self._generate_recommendations(dimension, vector_count, index_params)
            }
            
            # Add current memory estimates if PQ is used
            if is_pq:
                analysis["current_memory_usage"] = self._estimate_memory(
                    dimension, vector_count, index_params
                )
            
            # Add optimized memory estimates
            optimized_params = self._generate_recommended_params(dimension, vector_count)
            analysis["optimized_memory_usage"] = self._estimate_memory(
                dimension, vector_count, optimized_params
            )
            
            # Return formatted results
            return self._format_stats(analysis, stats_format)
            
        except Exception as e:
            logger.error(f"Error analyzing PQ index: {str(e)}")
            return {"error": str(e)}
    
    def optimize_index(
        self,
        collection_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize the index based on parameters or auto-tuning.
        
        Args:
            collection_name: Name of the collection to optimize
            parameters: Parameters for the optimization (if None, use auto-tuning)
            dry_run: If True, only show what would be done without making changes
            
        Returns:
            Dict[str, Any]: Results of the optimization process
        """
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            # Check if collection exists
            if "error" in collection_info:
                return {"error": f"Collection '{collection_name}' not found or error retrieving info"}
            
            # Extract dimensionality
            dimension = self._extract_dimension(collection_info)
            if dimension is None:
                return {"error": "Could not determine vector dimension from collection info"}
            
            # Get vector count
            vector_count = self.db_connector.count_vectors(collection_name)
            
            # Get existing index parameters
            existing_params = self._extract_index_params(collection_info)
            
            # Merge with provided parameters or generate recommended params
            if parameters:
                optimized_params = self._merge_params(parameters)
            else:
                optimized_params = self._generate_recommended_params(dimension, vector_count)
            
            # Calculate improvements
            current_memory = self._estimate_memory(dimension, vector_count, existing_params)
            optimized_memory = self._estimate_memory(dimension, vector_count, optimized_params)
            
            memory_reduction = 0
            if current_memory.get("total_kb", 0) > 0:
                memory_reduction = 1 - (optimized_memory.get("total_kb", 0) / current_memory.get("total_kb", 0))
            
            # Prepare result
            result = {
                "collection_name": collection_name,
                "vector_count": vector_count,
                "dimension": dimension,
                "existing_params": existing_params,
                "optimized_params": optimized_params,
                "memory_before": current_memory,
                "memory_after": optimized_memory,
                "memory_reduction_percentage": round(memory_reduction * 100, 2),
                "dry_run": dry_run
            }
            
            # Apply optimization if not a dry run
            if not dry_run:
                # The actual implementation depends on database-specific methods
                # Here we'll use a generic approach that might need adapter code for specific databases
                
                # For databases that support direct index rebuilding:
                if hasattr(self.db_connector, "create_index"):
                    success = self._rebuild_index(collection_name, optimized_params)
                    result["success"] = success
                    if not success:
                        result["error"] = "Failed to rebuild index with optimized parameters"
                else:
                    result["success"] = False
                    result["error"] = "Database connector does not support index rebuilding"
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing PQ index: {str(e)}")
            return {"error": str(e)}
    
    def estimate_memory_usage(
        self,
        collection_name: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Estimate memory usage for the index with current or proposed parameters.
        
        Args:
            collection_name: Name of the collection
            parameters: Proposed parameters (if None, use current parameters)
            
        Returns:
            Dict[str, Any]: Memory usage estimates
        """
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            # Check if collection exists
            if "error" in collection_info:
                return {"error": f"Collection '{collection_name}' not found or error retrieving info"}
            
            # Extract dimensionality
            dimension = self._extract_dimension(collection_info)
            if dimension is None:
                return {"error": "Could not determine vector dimension from collection info"}
            
            # Get vector count
            vector_count = self.db_connector.count_vectors(collection_name)
            
            # Get parameters
            if parameters:
                params = self._merge_params(parameters)
            else:
                params = self._extract_index_params(collection_info)
                if not params or not self._is_pq_index(params):
                    params = self._generate_recommended_params(dimension, vector_count)
            
            # Estimate memory usage
            memory_usage = self._estimate_memory(dimension, vector_count, params)
            
            # Calculate uncompressed size for comparison
            uncompressed_bytes = vector_count * dimension * 4  # 4 bytes per float
            compression_ratio = 0
            if uncompressed_bytes > 0:
                compression_ratio = memory_usage.get("vectors_kb", 0) * 1024 / uncompressed_bytes
            
            # Add results
            result = {
                "collection_name": collection_name,
                "vector_count": vector_count,
                "dimension": dimension,
                "parameters": params,
                "memory_usage": memory_usage,
                "uncompressed_size_kb": round(uncompressed_bytes / 1024, 2),
                "compression_ratio": round(compression_ratio, 4),
                "space_saving_percentage": round((1 - compression_ratio) * 100, 2) if compression_ratio > 0 else 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error estimating PQ memory usage: {str(e)}")
            return {"error": str(e)}
    
    def benchmark_index(
        self,
        collection_name: str,
        query_vectors: Optional[List[List[float]]] = None,
        num_queries: int = 100,
        top_k: int = 10,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark index performance with current or proposed parameters.
        
        Args:
            collection_name: Name of the collection
            query_vectors: Query vectors for benchmarking (if None, generate random vectors)
            num_queries: Number of queries to run if query_vectors is None
            top_k: Number of results to retrieve for each query
            parameters: Proposed parameters (if None, use current parameters)
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        self._check_db_connector()
        
        try:
            # Get collection info
            collection_info = self.db_connector.get_collection_info(collection_name)
            
            # Check if collection exists
            if "error" in collection_info:
                return {"error": f"Collection '{collection_name}' not found or error retrieving info"}
            
            # Extract dimensionality
            dimension = self._extract_dimension(collection_info)
            if dimension is None:
                return {"error": "Could not determine vector dimension from collection info"}
            
            # Generate query vectors if not provided
            if query_vectors is None:
                query_vectors = self.generate_random_queries(dimension, num_queries)
            else:
                num_queries = len(query_vectors)
            
            # Get existing index parameters
            existing_params = self._extract_index_params(collection_info)
            
            # For comparing current vs. optimized, we need to run benchmarks twice
            # First with existing configuration, then with optimized one
            
            # Benchmark current configuration
            current_results = self._run_benchmark(
                collection_name, query_vectors, top_k, existing_params
            )
            
            # Prepare results
            result = {
                "collection_name": collection_name,
                "dimension": dimension,
                "num_queries": num_queries,
                "top_k": top_k,
                "existing_params": existing_params,
                "current_performance": current_results
            }
            
            # If parameters are provided, benchmark with optimized configuration
            if parameters:
                optimized_params = self._merge_params(parameters)
                
                # Only run this if we can reconfigure the index
                if hasattr(self.db_connector, "create_index"):
                    # Save current state to restore later
                    current_config_saved = True
                    
                    try:
                        # Apply optimized parameters temporarily
                        success = self._rebuild_index(collection_name, optimized_params)
                        
                        if success:
                            # Run benchmark with optimized parameters
                            optimized_results = self._run_benchmark(
                                collection_name, query_vectors, top_k, optimized_params
                            )
                            
                            result["optimized_params"] = optimized_params
                            result["optimized_performance"] = optimized_results
                            
                            # Calculate improvements
                            if current_results.get("avg_query_time_ms", 0) > 0:
                                speed_improvement = 1 - (
                                    optimized_results.get("avg_query_time_ms", 0) / 
                                    current_results.get("avg_query_time_ms", 0)
                                )
                                result["speed_improvement_percentage"] = round(speed_improvement * 100, 2)
                        else:
                            result["warning"] = "Could not apply optimized parameters for benchmarking"
                    
                    finally:
                        # Restore original configuration if we changed it
                        if current_config_saved:
                            self._rebuild_index(collection_name, existing_params)
                else:
                    result["warning"] = "Database connector does not support index reconfiguration for comparative benchmarking"
            
            return result
            
        except Exception as e:
            logger.error(f"Error benchmarking PQ index: {str(e)}")
            return {"error": str(e)}
    
    def _extract_dimension(self, collection_info: Dict[str, Any]) -> Optional[int]:
        """
        Extract vector dimension from collection info.
        
        Args:
            collection_info: Collection information
            
        Returns:
            Optional[int]: Vector dimension if found, None otherwise
        """
        # Try common paths where dimension might be stored
        if "dimension" in collection_info:
            return collection_info["dimension"]
        
        if "vector_size" in collection_info:
            return collection_info["vector_size"]
        
        if "schema" in collection_info:
            schema = collection_info["schema"]
            
            # Look in schema for vector field with dimension
            if isinstance(schema, dict):
                for field_name, field_info in schema.items():
                    if isinstance(field_info, dict) and "dimension" in field_info:
                        return field_info["dimension"]
                    if isinstance(field_info, dict) and "dim" in field_info:
                        return field_info["dim"]
        
        # For more specific database schemas
        if "config" in collection_info:
            config = collection_info["config"]
            if isinstance(config, dict):
                if "dimensions" in config:
                    return config["dimensions"]
                if "vector_size" in config:
                    return config["vector_size"]
        
        return None
    
    def _extract_index_params(self, collection_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract index parameters from collection info.
        
        Args:
            collection_info: Collection information
            
        Returns:
            Dict[str, Any]: Index parameters
        """
        # Initialize with empty params
        params = {}
        
        # Try common paths where index parameters might be stored
        if "index_params" in collection_info:
            params = collection_info["index_params"]
        
        elif "index" in collection_info:
            index_info = collection_info["index"]
            if isinstance(index_info, dict):
                params = index_info.get("params", {})
                
                # If index type is stored separately, add it
                if "type" in index_info:
                    params["index_type"] = index_info["type"]
        
        elif "indexes" in collection_info:
            indexes = collection_info["indexes"]
            if isinstance(indexes, list) and indexes:
                # Use the first index that looks like PQ
                for idx in indexes:
                    if isinstance(idx, dict):
                        if idx.get("index_type", "").lower() in ["pq", "product_quantization"]:
                            params = idx.get("params", {})
                            params["index_type"] = idx.get("index_type")
                            break
        
        return params
    
    def _is_pq_index(self, params: Dict[str, Any]) -> bool:
        """
        Check if the index parameters indicate a PQ index.
        
        Args:
            params: Index parameters
            
        Returns:
            bool: True if the index is a PQ index, False otherwise
        """
        if not params:
            return False
        
        # Check for PQ-specific parameters
        if "index_type" in params and params["index_type"].lower() in ["pq", "product_quantization"]:
            return True
        
        # Check for typical PQ parameters
        pq_params = ["num_subvectors", "bits_per_subvector", "subquantizers", "m", "code_size"]
        return any(param in params for param in pq_params)
    
    def _generate_recommendations(
        self, 
        dimension: int, 
        vector_count: int,
        current_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate recommendations for PQ index parameters.
        
        Args:
            dimension: Vector dimension
            vector_count: Number of vectors
            current_params: Current index parameters
            
        Returns:
            Dict[str, Any]: Recommendations
        """
        recommendations = {}
        
        # Generate recommended parameters
        recommended_params = self._generate_recommended_params(dimension, vector_count)
        
        # If we have current parameters, compare and make specific recommendations
        if current_params and self._is_pq_index(current_params):
            # Check number of subvectors
            current_subvectors = current_params.get("num_subvectors", current_params.get("m", 0))
            recommended_subvectors = recommended_params.get("num_subvectors")
            
            if current_subvectors != recommended_subvectors:
                recommendations["subvectors"] = (
                    f"Change number of subvectors from {current_subvectors} to {recommended_subvectors} "
                    f"for better balance between memory usage and recall"
                )
            
            # Check bits per subvector
            current_bits = current_params.get("bits_per_subvector", current_params.get("code_size", 0))
            recommended_bits = recommended_params.get("bits_per_subvector")
            
            if current_bits != recommended_bits:
                recommendations["bits"] = (
                    f"Change bits per subvector from {current_bits} to {recommended_bits} "
                    f"for better quantization precision"
                )
            
            # Check if OPQ would be beneficial
            if not current_params.get("use_opq", False) and dimension > 100:
                recommendations["opq"] = (
                    "Consider using Optimized Product Quantization (OPQ) for better "
                    "accuracy with high-dimensional vectors"
                )
        else:
            # General recommendations for new PQ index
            recommendations["general"] = "Implement Product Quantization index with the following parameters:"
            recommendations["parameters"] = recommended_params
            
            # Additional recommendations
            if dimension > 100:
                recommendations["opq"] = "Consider using Optimized Product Quantization (OPQ) for high-dimensional vectors"
            
            if vector_count > 1000000:
                recommendations["hybrid"] = (
                    "For very large collections, consider a hybrid approach with IVF+PQ "
                    "for improved search performance"
                )
        
        return recommendations
    
    def _generate_recommended_params(self, dimension: int, vector_count: int) -> Dict[str, Any]:
        """
        Generate recommended PQ parameters based on vector dimension and count.
        
        Args:
            dimension: Vector dimension
            vector_count: Number of vectors
            
        Returns:
            Dict[str, Any]: Recommended parameters
        """
        # Start with default parameters
        params = self._get_default_params()
        
        # Adjust number of subvectors based on dimension
        # General rule: num_subvectors should divide dimension evenly
        # and be in reasonable range (4-16 for most use cases)
        optimal_subvectors = max(4, min(16, dimension // 4))
        
        # Find a divisor of dimension close to optimal_subvectors
        divisors = [d for d in range(4, min(dimension, 32)) if dimension % d == 0]
        if divisors:
            # Find closest divisor to optimal_subvectors
            num_subvectors = min(divisors, key=lambda d: abs(d - optimal_subvectors))
        else:
            # If no exact divisor, use the closest reasonable value
            num_subvectors = max(4, min(16, round(dimension / round(dimension / optimal_subvectors))))
        
        params["num_subvectors"] = num_subvectors
        
        # Adjust bits per subvector based on desired accuracy
        # Standard is 8 bits (256 centroids), but can use 6 (64 centroids) or 10 (1024 centroids)
        # More bits = better accuracy but more memory
        if vector_count < 100000:
            # For small collections, we can afford more precision
            params["bits_per_subvector"] = 8
        elif vector_count > 10000000:
            # For very large collections, might need to reduce precision
            params["bits_per_subvector"] = 8
        else:
            # Default for most collections
            params["bits_per_subvector"] = 8
        
        # Adjust training sample count based on vector count
        params["training_sample_count"] = min(vector_count, max(10000, round(vector_count * 0.1)))
        
        # Adjust search parameters based on vector count
        if vector_count > 1000000:
            params["search_nprobe"] = 16
        elif vector_count < 100000:
            params["search_nprobe"] = 8
        
        # Consider OPQ for high-dimensional vectors
        params["use_opq"] = dimension > 100
        
        return params
    
    def _estimate_memory(
        self,
        dimension: int,
        vector_count: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate memory usage for PQ index.
        
        Args:
            dimension: Vector dimension
            vector_count: Number of vectors
            params: Index parameters
            
        Returns:
            Dict[str, Any]: Memory usage estimates
        """
        # Extract parameters
        num_subvectors = params.get("num_subvectors", params.get("m", 8))
        bits_per_subvector = params.get("bits_per_subvector", params.get("code_size", 8))
        
        # Calculate memory for PQ codes
        # Each vector is encoded using num_subvectors subvectors, each using bits_per_subvector bits
        bits_per_vector = num_subvectors * bits_per_subvector
        bytes_per_vector = (bits_per_vector + 7) // 8  # Round up to nearest byte
        
        # Total memory for encoded vectors
        vector_memory_bytes = vector_count * bytes_per_vector
        
        # Calculate memory for codebooks
        # Each subvector has 2^bits_per_subvector centroids, each of dimension/num_subvectors floats
        centroids_per_subvector = 2 ** bits_per_subvector
        floats_per_centroid = dimension // num_subvectors
        codebook_memory_bytes = num_subvectors * centroids_per_subvector * floats_per_centroid * 4  # 4 bytes per float
        
        # Calculate memory for other metadata (IDs, etc.)
        # Assuming 8 bytes per vector for ID and minimal metadata
        metadata_memory_bytes = vector_count * 8
        
        # Calculate total memory
        total_memory_bytes = vector_memory_bytes + codebook_memory_bytes + metadata_memory_bytes
        
        # Return results in KB
        return {
            "vectors_kb": round(vector_memory_bytes / 1024, 2),
            "codebooks_kb": round(codebook_memory_bytes / 1024, 2),
            "metadata_kb": round(metadata_memory_bytes / 1024, 2),
            "total_kb": round(total_memory_bytes / 1024, 2),
            "total_mb": round(total_memory_bytes / (1024 * 1024), 2)
        }
    
    def _rebuild_index(self, collection_name: str, params: Dict[str, Any]) -> bool:
        """
        Rebuild index with new parameters.
        
        Args:
            collection_name: Name of the collection
            params: New index parameters
            
        Returns:
            bool: True if index was rebuilt successfully, False otherwise
        """
        try:
            # Implementation depends on database-specific API
            # Here's a generic approach that may need to be adapted
            
            # Check if database connector supports creating an index with parameters
            if hasattr(self.db_connector, "create_index"):
                index_params = {
                    "index_type": "pq",
                    **params
                }
                
                # Some connectors may have specific method signature
                if hasattr(self.db_connector, "create_index_with_params"):
                    success = self.db_connector.create_index_with_params(
                        collection_name, index_params
                    )
                else:
                    # Try the more generic approach
                    success = self.db_connector.create_index(
                        collection_name, "pq", params
                    )
                
                return success
            
            # For databases that don't support direct index creation/modification
            logger.warning(
                f"Database connector for collection '{collection_name}' does not support "
                "direct index rebuilding with PQ parameters"
            )
            return False
            
        except Exception as e:
            logger.error(f"Error rebuilding PQ index: {str(e)}")
            return False
    
    def _run_benchmark(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run benchmark queries and measure performance.
        
        Args:
            collection_name: Name of the collection
            query_vectors: Query vectors
            top_k: Number of results to retrieve
            params: Index parameters
            
        Returns:
            Dict[str, Any]: Benchmark results
        """
        try:
            import time
            import numpy as np
            
            # Prepare search parameters if needed
            search_params = {}
            if params:
                # Extract search-relevant parameters
                search_params = {
                    "nprobe": params.get("search_nprobe", 10)
                }
            
            # Run queries and measure times
            query_times = []
            result_counts = []
            
            for query_vector in query_vectors:
                start_time = time.time()
                
                # Pass search parameters if the connector supports it
                if hasattr(self.db_connector, "search_with_params"):
                    results = self.db_connector.search_with_params(
                        collection_name, query_vector, top_k, search_params
                    )
                else:
                    # Fall back to standard search
                    results = self.db_connector.search(
                        collection_name, query_vector, top_k
                    )
                
                end_time = time.time()
                query_time_ms = (end_time - start_time) * 1000
                
                query_times.append(query_time_ms)
                result_counts.append(len(results) if results else 0)
            
            # Calculate statistics
            avg_query_time = np.mean(query_times)
            min_query_time = np.min(query_times)
            max_query_time = np.max(query_times)
            p95_query_time = np.percentile(query_times, 95)
            p99_query_time = np.percentile(query_times, 99)
            
            avg_result_count = np.mean(result_counts)
            
            # Return results
            return {
                "num_queries": len(query_vectors),
                "avg_query_time_ms": round(avg_query_time, 2),
                "min_query_time_ms": round(min_query_time, 2),
                "max_query_time_ms": round(max_query_time, 2),
                "p95_query_time_ms": round(p95_query_time, 2),
                "p99_query_time_ms": round(p99_query_time, 2),
                "avg_result_count": round(avg_result_count, 2),
                "qps": round(1000 / avg_query_time, 2)  # Queries per second
            }
            
        except Exception as e:
            logger.error(f"Error running benchmark: {str(e)}")
            return {"error": str(e)}


def get_index_optimizer(
    optimizer_type: str,
    db_connector: Any = None,
    params: Optional[Dict[str, Any]] = None
) -> BaseIndexOptimizer:
    """
    Get an index optimizer based on the optimizer type.
    
    Args:
        optimizer_type: Type of optimizer ("hnsw", "ivfflat", "pq", "ann")
        db_connector: Vector database connector instance
        params: Parameters for the optimizer
        
    Returns:
        BaseIndexOptimizer: Index optimizer
        
    Raises:
        ValueError: If the optimizer type is not supported
    """
    # Normalize optimizer type
    optimizer_type = optimizer_type.lower().strip()
    
    # Create optimizer based on type
    if optimizer_type in ["hnsw", "hierarchical_navigable_small_world"]:
        return HNSWOptimizer(db_connector, params)
    
    elif optimizer_type in ["ivfflat", "ivf", "inverted_file_flat"]:
        return IVFFlatOptimizer(db_connector, params)
    
    elif optimizer_type in ["pq", "product_quantization"]:
        return PqIndexOptimizer(db_connector, params)
    
    else:
        supported_types = ["hnsw", "ivfflat", "pq"]
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Supported types: {supported_types}")


def optimize_index(
    db_connector: Any,
    collection_name: str,
    optimizer_type: str = "auto",
    parameters: Optional[Dict[str, Any]] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Optimize an index using the appropriate optimizer.
    
    Args:
        db_connector: Vector database connector instance
        collection_name: Name of the collection to optimize
        optimizer_type: Type of optimizer ("auto", "hnsw", "ivfflat", "pq", "ann")
        parameters: Parameters for the optimization
        dry_run: If True, only show what would be done without making changes
        
    Returns:
        Dict[str, Any]: Results of the optimization process
    """
    try:
        # Auto-detect optimizer type if set to "auto"
        if optimizer_type.lower() == "auto":
            # Try to determine from database type and collection info
            if hasattr(db_connector, "get_collection_info"):
                collection_info = db_connector.get_collection_info(collection_name)
                
                # Look for index type in collection info
                if "index_type" in collection_info:
                    optimizer_type = collection_info["index_type"].lower()
                
                # If still auto, choose based on database type
                if optimizer_type.lower() == "auto":
                    db_type = getattr(db_connector, "db_type", "").lower()
                    if db_type in ["qdrant"]:
                        optimizer_type = "hnsw"  # Qdrant uses HNSW by default
                    elif db_type in ["postgres", "postgresql", "pgvector"]:
                        optimizer_type = "ivfflat"  # PostgreSQL with pgvector often uses IVFFlat
                    elif db_type in ["milvus"]:
                        optimizer_type = "pq"  # Milvus often benefits from PQ for large collections
                    else:
                        optimizer_type = "hnsw"  # Default to HNSW as it's widely supported
        
        # Get optimizer
        optimizer = get_index_optimizer(optimizer_type, db_connector, {})
        
        # Optimize index
        return optimizer.optimize_index(collection_name, parameters, dry_run)
        
    except Exception as e:
        logger.error(f"Error optimizing index: {str(e)}")
        return {"error": str(e)} 