#!/usr/bin/env python3
"""
Query latency monitor for RAG systems.

This module provides a latency monitor for measuring, tracking, and analyzing
query latency in RAG systems, focusing on the retrieval component.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import datetime
import time
import uuid
from .base import BaseLatencyMonitor


class QueryLatencyMonitor(BaseLatencyMonitor):
    """
    Latency monitor for query operations in RAG systems.
    
    This monitor focuses on measuring and analyzing latency for query operations
    in the retrieval component of RAG systems, helping identify performance
    bottlenecks and optimize query response times.
    
    Attributes:
        name (str): Name of the monitor.
        config (Dict[str, Any]): Configuration options for the monitor.
        precision (int): Number of decimal places for latency measurements.
        unit (str): Unit for latency measurements, e.g., 'ms', 's'.
        measurements (List[Dict[str, Any]]): List of latency measurements.
        _active_measurements (Dict[str, Dict[str, Any]]): Dictionary of active measurements.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        precision: int = 3,
        unit: str = "ms"
    ):
        """
        Initialize the query latency monitor.
        
        Args:
            name (str): Name of the monitor.
            config (Optional[Dict[str, Any]]): Configuration options for the monitor.
                Defaults to an empty dictionary.
            precision (int): Number of decimal places for latency measurements.
                Defaults to 3.
            unit (str): Unit for latency measurements. Defaults to 'ms'.
                Valid values: 'ns', 'us', 'ms', 's'.
        """
        super().__init__(name=name, config=config, precision=precision, unit=unit)
        self._active_measurements = {}
    
    def start_measurement(self, operation_id: Optional[str] = None, **kwargs) -> str:
        """
        Start measuring query latency for an operation.
        
        Args:
            operation_id (Optional[str]): ID for the operation.
                If not provided, a unique ID will be generated.
            **kwargs: Additional metadata for the measurement.
                Common metadata for query operations:
                - query (str): Query text or representation.
                - collection (str): Vector database collection or index.
                - vector_db (str): Vector database type (e.g., 'chroma', 'pinecone').
                - k (int): Number of results to retrieve.
                - filter (Dict[str, Any]): Query filter.
        
        Returns:
            str: Operation ID for the measurement.
        """
        # Generate operation ID if not provided
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        
        # Record start time
        start_time = time.time()
        
        # Prepare measurement
        measurement = {
            "operation_id": operation_id,
            "start_time": start_time,
            "start_timestamp": datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        # Store active measurement
        with self._lock:
            self._active_measurements[operation_id] = measurement
        
        return operation_id
    
    def end_measurement(self, operation_id: str, **kwargs) -> Dict[str, Any]:
        """
        End measuring query latency for an operation and calculate results.
        
        Args:
            operation_id (str): ID for the operation.
            **kwargs: Additional metadata for the measurement.
                Common metadata for query operations:
                - num_results (int): Number of results retrieved.
                - vector_db_time (float): Time spent in the vector database.
                - reranking_time (float): Time spent in reranking.
                - preprocessing_time (float): Time spent in preprocessing.
        
        Returns:
            Dict[str, Any]: Measurement results, including latency.
            
        Raises:
            ValueError: If the operation ID is not found in active measurements.
        """
        # Get end time
        end_time = time.time()
        
        # Get active measurement
        with self._lock:
            if operation_id not in self._active_measurements:
                raise ValueError(f"Operation ID not found: {operation_id}")
            
            measurement = self._active_measurements.pop(operation_id)
        
        # Calculate latency
        start_time = measurement.get("start_time", end_time)
        latency_seconds = end_time - start_time
        latency = self._convert_time_to_unit(latency_seconds)
        
        # Prepare result
        result = {
            **measurement,
            **kwargs,
            "end_time": end_time,
            "end_timestamp": datetime.datetime.now().isoformat(),
            "latency": round(latency, self.precision),
            "latency_unit": self.unit,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Remove internal fields
        result.pop("start_time", None)
        
        # Record measurement
        with self._lock:
            self.measurements.append(result)
        
        return result
    
    def measure(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure query latency for a function call.
        
        Args:
            func (Callable): Function to measure.
            *args: Arguments for the function.
            **kwargs: Keyword arguments for the function.
                Special keys:
                - _metadata (Dict[str, Any]): Additional metadata for the measurement.
        
        Returns:
            Dict[str, Any]: Measurement results, including latency and function result.
        """
        # Extract metadata
        metadata = kwargs.pop("_metadata", {})
        
        # Start measurement
        operation_id = self.start_measurement(**metadata)
        
        try:
            # Call the function
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Calculate function execution time
            execution_time = end_time - start_time
            
            # Extract results metadata if available
            results_metadata = {}
            if isinstance(result, dict) and "metadata" in result:
                results_metadata = result.get("metadata", {})
            
            # Prepare additional metadata
            additional_metadata = {
                "execution_time": round(self._convert_time_to_unit(execution_time), self.precision),
                "execution_time_unit": self.unit,
                "status": "success"
            }
            
            # Add results metadata if available
            if results_metadata:
                additional_metadata["results_metadata"] = results_metadata
            
            # End measurement
            self.end_measurement(operation_id, **additional_metadata)
            
            return {
                "result": result,
                "latency": additional_metadata["execution_time"],
                "latency_unit": self.unit,
                "operation_id": operation_id
            }
        except Exception as e:
            # End measurement with error
            self.end_measurement(operation_id, status="error", error=str(e))
            
            # Re-raise the exception
            raise
    
    def analyze_by_query_type(self, 
                              start_time: Optional[datetime.datetime] = None,
                              end_time: Optional[datetime.datetime] = None
                             ) -> Dict[str, Any]:
        """
        Analyze query latency by query type.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by query type.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by query type
        query_types = {}
        for m in measurements:
            query_type = m.get("query_type", "unknown")
            if query_type not in query_types:
                query_types[query_type] = []
            query_types[query_type].append(m)
        
        # Calculate statistics for each query type
        results = {}
        for query_type, type_measurements in query_types.items():
            latencies = [m.get("latency", 0) for m in type_measurements]
            
            if not latencies:
                continue
            
            results[query_type] = {
                "count": len(latencies),
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results
    
    def analyze_by_collection(self, 
                             start_time: Optional[datetime.datetime] = None,
                             end_time: Optional[datetime.datetime] = None
                            ) -> Dict[str, Any]:
        """
        Analyze query latency by collection.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by collection.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by collection
        collections = {}
        for m in measurements:
            collection = m.get("collection", "unknown")
            if collection not in collections:
                collections[collection] = []
            collections[collection].append(m)
        
        # Calculate statistics for each collection
        results = {}
        for collection, coll_measurements in collections.items():
            latencies = [m.get("latency", 0) for m in coll_measurements]
            
            if not latencies:
                continue
            
            results[collection] = {
                "count": len(latencies),
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results
    
    def analyze_query_complexity(self, 
                               start_time: Optional[datetime.datetime] = None,
                               end_time: Optional[datetime.datetime] = None
                              ) -> Dict[str, Any]:
        """
        Analyze query latency by query complexity.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by query complexity.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by query complexity
        # (assuming query complexity is indicated by query length or token count)
        complexity_groups = {
            "simple": [],
            "medium": [],
            "complex": []
        }
        
        for m in measurements:
            # Determine complexity based on query length or token count
            query = m.get("query", "")
            query_tokens = m.get("query_tokens", 0)
            
            if query_tokens > 0:
                # Use token count if available
                if query_tokens < 10:
                    complexity = "simple"
                elif query_tokens < 50:
                    complexity = "medium"
                else:
                    complexity = "complex"
            elif isinstance(query, str):
                # Use query length as a proxy for complexity
                if len(query) < 50:
                    complexity = "simple"
                elif len(query) < 200:
                    complexity = "medium"
                else:
                    complexity = "complex"
            else:
                # Default to medium if complexity can't be determined
                complexity = "medium"
            
            complexity_groups[complexity].append(m)
        
        # Calculate statistics for each complexity group
        results = {}
        for complexity, group_measurements in complexity_groups.items():
            latencies = [m.get("latency", 0) for m in group_measurements]
            
            if not latencies:
                continue
            
            results[complexity] = {
                "count": len(latencies),
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results 