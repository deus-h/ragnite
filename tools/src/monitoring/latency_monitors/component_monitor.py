#!/usr/bin/env python3
"""
Component latency monitor for RAG systems.

This module provides a latency monitor for measuring, tracking, and analyzing
latency of specific components in RAG systems, such as embedding generation,
vector database queries, or reranking.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import datetime
import time
import uuid
from .base import BaseLatencyMonitor


class ComponentLatencyMonitor(BaseLatencyMonitor):
    """
    Latency monitor for specific components in RAG systems.
    
    This monitor focuses on measuring and analyzing latency for specific components
    in RAG systems, such as embedding generation, vector database queries, or reranking,
    helping identify performance bottlenecks within individual components.
    
    Attributes:
        name (str): Name of the monitor.
        component (str): Name of the component being monitored.
        config (Dict[str, Any]): Configuration options for the monitor.
        precision (int): Number of decimal places for latency measurements.
        unit (str): Unit for latency measurements, e.g., 'ms', 's'.
        measurements (List[Dict[str, Any]]): List of latency measurements.
        _active_measurements (Dict[str, Dict[str, Any]]): Dictionary of active measurements.
    """
    
    def __init__(
        self,
        name: str,
        component: str,
        config: Optional[Dict[str, Any]] = None,
        precision: int = 3,
        unit: str = "ms"
    ):
        """
        Initialize the component latency monitor.
        
        Args:
            name (str): Name of the monitor.
            component (str): Name of the component being monitored.
            config (Optional[Dict[str, Any]]): Configuration options for the monitor.
                Defaults to an empty dictionary.
            precision (int): Number of decimal places for latency measurements.
                Defaults to 3.
            unit (str): Unit for latency measurements. Defaults to 'ms'.
                Valid values: 'ns', 'us', 'ms', 's'.
        """
        super().__init__(name=name, config=config, precision=precision, unit=unit)
        self.component = component
        self._active_measurements = {}
    
    def start_measurement(self, operation_id: Optional[str] = None, **kwargs) -> str:
        """
        Start measuring component latency for an operation.
        
        Args:
            operation_id (Optional[str]): ID for the operation.
                If not provided, a unique ID will be generated.
            **kwargs: Additional metadata for the measurement.
                Common metadata for component operations:
                - subcomponent (str): Name of the subcomponent.
                - parent_operation_id (str): ID of the parent operation.
                - input_size (int): Size of the input data.
                - configuration (Dict[str, Any]): Configuration for the component.
        
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
            "component": self.component,
            "start_time": start_time,
            "start_timestamp": datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        # Store active measurement
        with self._lock:
            self._active_measurements[operation_id] = measurement
        
        return operation_id
    
    def record_subcomponent_time(
        self,
        operation_id: str,
        subcomponent: str,
        elapsed_time: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Record subcomponent time for an ongoing operation.
        
        Args:
            operation_id (str): ID for the operation.
            subcomponent (str): Name of the subcomponent.
            elapsed_time (float): Elapsed time for the subcomponent in seconds.
            **kwargs: Additional metadata for the subcomponent measurement.
        
        Returns:
            Dict[str, Any]: Updated measurement.
            
        Raises:
            ValueError: If the operation ID is not found in active measurements.
        """
        # Convert elapsed time to the monitor's unit
        latency = self._convert_time_to_unit(elapsed_time)
        
        # Get active measurement
        with self._lock:
            if operation_id not in self._active_measurements:
                raise ValueError(f"Operation ID not found: {operation_id}")
            
            measurement = self._active_measurements[operation_id]
            
            # Initialize subcomponents dictionary if not present
            if "subcomponents" not in measurement:
                measurement["subcomponents"] = {}
            
            # Add subcomponent time
            measurement["subcomponents"][subcomponent] = {
                "latency": round(latency, self.precision),
                "latency_unit": self.unit,
                **kwargs
            }
        
        return measurement
    
    def end_measurement(self, operation_id: str, **kwargs) -> Dict[str, Any]:
        """
        End measuring component latency for an operation and calculate results.
        
        Args:
            operation_id (str): ID for the operation.
            **kwargs: Additional metadata for the measurement.
                Common metadata for component operations:
                - output_size (int): Size of the output data.
                - status (str): Status of the operation (e.g., 'success', 'error').
                - error (str): Error message if operation failed.
                - cache_hit (bool): Whether the operation resulted in a cache hit.
                - throughput (float): Throughput of the operation (e.g., tokens/second).
        
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
        
        # Calculate throughput if input and output sizes are available
        input_size = measurement.get("input_size")
        output_size = kwargs.get("output_size")
        
        if latency_seconds > 0:
            if input_size is not None:
                kwargs["input_throughput"] = round(input_size / latency_seconds, self.precision)
            
            if output_size is not None:
                kwargs["output_throughput"] = round(output_size / latency_seconds, self.precision)
        
        # Calculate subcomponent percentages if subcomponents exist
        if "subcomponents" in measurement:
            subcomponents = measurement["subcomponents"]
            subcomponent_total = 0
            
            # Calculate total subcomponent time
            for subcomponent, data in subcomponents.items():
                subcomponent_total += data.get("latency", 0)
            
            # Calculate percentages
            for subcomponent, data in subcomponents.items():
                if subcomponent_total > 0:
                    data["percentage"] = round((data.get("latency", 0) / subcomponent_total) * 100, 1)
                else:
                    data["percentage"] = 0
            
            # Calculate percentage of total time
            if latency > 0:
                kwargs["subcomponents_percentage"] = round((subcomponent_total / latency) * 100, 1)
        
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
        Measure component latency for a function call.
        
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
            
            # Extract result metadata if available
            result_metadata = {}
            if isinstance(result, dict):
                # Extract common metadata fields
                for key in ["output_size", "status", "cache_hit", "throughput"]:
                    if key in result:
                        result_metadata[key] = result[key]
                
                # Extract subcomponent times if available
                if "subcomponent_times" in result and isinstance(result["subcomponent_times"], dict):
                    for subcomponent, elapsed_time in result["subcomponent_times"].items():
                        if isinstance(elapsed_time, (int, float)):
                            self.record_subcomponent_time(
                                operation_id=operation_id,
                                subcomponent=subcomponent,
                                elapsed_time=elapsed_time
                            )
            
            # Prepare additional metadata
            additional_metadata = {
                "execution_time": round(self._convert_time_to_unit(execution_time), self.precision),
                "execution_time_unit": self.unit,
                "status": "success",
                **result_metadata
            }
            
            # End measurement
            self.end_measurement(operation_id, **additional_metadata)
            
            return {
                "result": result,
                "latency": additional_metadata["execution_time"],
                "latency_unit": self.unit,
                "operation_id": operation_id,
                **result_metadata
            }
        except Exception as e:
            # End measurement with error
            self.end_measurement(operation_id, status="error", error=str(e))
            
            # Re-raise the exception
            raise
    
    def analyze_by_subcomponent(self, 
                              start_time: Optional[datetime.datetime] = None,
                              end_time: Optional[datetime.datetime] = None
                             ) -> Dict[str, Any]:
        """
        Analyze component latency by subcomponent.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by subcomponent.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Initialize subcomponent data
        subcomponents = {}
        total_operations = 0
        total_latency = 0
        
        # Process measurements
        for m in measurements:
            total_operations += 1
            total_latency += m.get("latency", 0)
            
            # Skip measurements without subcomponents
            if "subcomponents" not in m or not isinstance(m["subcomponents"], dict):
                continue
            
            # Process each subcomponent
            for subcomponent_name, subcomponent_data in m["subcomponents"].items():
                if subcomponent_name not in subcomponents:
                    subcomponents[subcomponent_name] = {
                        "count": 0,
                        "total_latency": 0,
                        "latencies": []
                    }
                
                # Add subcomponent data
                subcomponents[subcomponent_name]["count"] += 1
                subcomponent_latency = subcomponent_data.get("latency", 0)
                subcomponents[subcomponent_name]["total_latency"] += subcomponent_latency
                subcomponents[subcomponent_name]["latencies"].append(subcomponent_latency)
        
        # Calculate statistics for each subcomponent
        results = {
            "component": self.component,
            "total_operations": total_operations,
            "total_latency": round(total_latency, self.precision),
            "subcomponents": {}
        }
        
        for subcomponent_name, subcomponent_data in subcomponents.items():
            latencies = subcomponent_data["latencies"]
            
            if not latencies:
                continue
            
            results["subcomponents"][subcomponent_name] = {
                "count": subcomponent_data["count"],
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "percentage_of_operations": round((subcomponent_data["count"] / total_operations) * 100 if total_operations > 0 else 0, 1),
                "percentage_of_total_latency": round((subcomponent_data["total_latency"] / total_latency) * 100 if total_latency > 0 else 0, 1),
                "unit": self.unit
            }
        
        return results
    
    def analyze_by_configuration(self, 
                               configuration_key: str,
                               start_time: Optional[datetime.datetime] = None,
                               end_time: Optional[datetime.datetime] = None
                              ) -> Dict[str, Any]:
        """
        Analyze component latency by configuration setting.
        
        Args:
            configuration_key (str): Configuration key to analyze by.
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by configuration value.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by configuration value
        config_values = {}
        
        for m in measurements:
            # Skip measurements without configuration
            if "configuration" not in m or not isinstance(m["configuration"], dict):
                continue
            
            # Get configuration value
            configuration = m["configuration"]
            config_value = configuration.get(configuration_key, "unknown")
            
            # Convert complex values to string
            if not isinstance(config_value, (str, int, float, bool, type(None))):
                config_value = str(config_value)
            
            if config_value not in config_values:
                config_values[config_value] = []
            
            config_values[config_value].append(m)
        
        # Calculate statistics for each configuration value
        results = {}
        for value, value_measurements in config_values.items():
            latencies = [m.get("latency", 0) for m in value_measurements]
            
            if not latencies:
                continue
            
            results[str(value)] = {
                "count": len(latencies),
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results
    
    def analyze_by_input_size(self, 
                            size_ranges: Optional[List[int]] = None,
                            start_time: Optional[datetime.datetime] = None,
                            end_time: Optional[datetime.datetime] = None
                           ) -> Dict[str, Any]:
        """
        Analyze component latency by input size.
        
        Args:
            size_ranges (Optional[List[int]]): List of size range boundaries.
                Defaults to [10, 100, 1000, 10000].
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by input size range.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Set default size ranges if not provided
        if size_ranges is None:
            size_ranges = [10, 100, 1000, 10000]
        
        # Ensure size ranges are sorted
        size_ranges = sorted(size_ranges)
        
        # Initialize size range groups
        size_groups = {}
        for i in range(len(size_ranges) + 1):
            if i == 0:
                group_name = f"<{size_ranges[0]}"
            elif i == len(size_ranges):
                group_name = f">={size_ranges[-1]}"
            else:
                group_name = f"{size_ranges[i-1]}-{size_ranges[i]-1}"
            
            size_groups[group_name] = []
        
        # Group measurements by input size
        for m in measurements:
            # Skip measurements without input size
            if "input_size" not in m:
                continue
            
            input_size = m.get("input_size", 0)
            
            # Determine group
            group_name = f">={size_ranges[-1]}"
            for i, size in enumerate(size_ranges):
                if input_size < size:
                    if i == 0:
                        group_name = f"<{size}"
                    else:
                        group_name = f"{size_ranges[i-1]}-{size-1}"
                    break
            
            size_groups[group_name].append(m)
        
        # Calculate statistics for each size group
        results = {}
        for group_name, group_measurements in size_groups.items():
            latencies = [m.get("latency", 0) for m in group_measurements]
            
            if not latencies:
                continue
            
            results[group_name] = {
                "count": len(latencies),
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results
    
    def analyze_cache_efficiency(self, 
                               start_time: Optional[datetime.datetime] = None,
                               end_time: Optional[datetime.datetime] = None
                              ) -> Dict[str, Any]:
        """
        Analyze component cache efficiency.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Cache efficiency statistics.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by cache hit status
        cache_hits = []
        cache_misses = []
        
        for m in measurements:
            cache_hit = m.get("cache_hit", False)
            
            if cache_hit:
                cache_hits.append(m)
            else:
                cache_misses.append(m)
        
        # Calculate statistics
        total_operations = len(cache_hits) + len(cache_misses)
        hit_rate = round((len(cache_hits) / total_operations) * 100, 1) if total_operations > 0 else 0
        
        hit_latencies = [m.get("latency", 0) for m in cache_hits]
        miss_latencies = [m.get("latency", 0) for m in cache_misses]
        
        result = {
            "total_operations": total_operations,
            "cache_hits": len(cache_hits),
            "cache_misses": len(cache_misses),
            "hit_rate": hit_rate,
            "miss_rate": round(100 - hit_rate, 1),
            "latency_reduction": {}
        }
        
        # Calculate latency reduction if both hits and misses exist
        if hit_latencies and miss_latencies:
            avg_hit_latency = sum(hit_latencies) / len(hit_latencies)
            avg_miss_latency = sum(miss_latencies) / len(miss_latencies)
            
            result["latency_reduction"] = {
                "avg_hit_latency": round(avg_hit_latency, self.precision),
                "avg_miss_latency": round(avg_miss_latency, self.precision),
                "absolute_reduction": round(avg_miss_latency - avg_hit_latency, self.precision),
                "percentage_reduction": round(((avg_miss_latency - avg_hit_latency) / avg_miss_latency) * 100, 1) if avg_miss_latency > 0 else 0,
                "unit": self.unit
            }
        
        return result 