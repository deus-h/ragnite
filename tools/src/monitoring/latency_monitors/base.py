#!/usr/bin/env python3
"""
Base class for latency monitors in RAG systems.

This module provides the abstract base class for all latency monitors,
which are responsible for measuring, tracking, and analyzing latency
across different components and operations in RAG systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
import datetime
import statistics
import time
import json
import uuid
import threading
import functools


class BaseLatencyMonitor(ABC):
    """
    Abstract base class for latency monitors.
    
    Latency monitors measure, track, and analyze latency across different
    components and operations in RAG systems, helping identify performance
    bottlenecks and optimize response times.
    
    Attributes:
        name (str): Name of the monitor.
        config (Dict[str, Any]): Configuration options for the monitor.
        precision (int): Number of decimal places for latency measurements.
        unit (str): Unit for latency measurements, e.g., 'ms', 's'.
        measurements (List[Dict[str, Any]]): List of latency measurements.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        precision: int = 3,
        unit: str = "ms"
    ):
        """
        Initialize the latency monitor.
        
        Args:
            name (str): Name of the monitor.
            config (Optional[Dict[str, Any]]): Configuration options for the monitor.
                Defaults to an empty dictionary.
            precision (int): Number of decimal places for latency measurements.
                Defaults to 3.
            unit (str): Unit for latency measurements. Defaults to 'ms'.
                Valid values: 'ns', 'us', 'ms', 's'.
        """
        self.name = name
        self.config = config or {}
        self.precision = precision
        self.unit = unit
        self.measurements = []
        self._lock = threading.Lock()
    
    @abstractmethod
    def start_measurement(self, operation_id: Optional[str] = None, **kwargs) -> str:
        """
        Start measuring latency for an operation.
        
        Args:
            operation_id (Optional[str]): ID for the operation.
                If not provided, a unique ID will be generated.
            **kwargs: Additional metadata for the measurement.
        
        Returns:
            str: Operation ID for the measurement.
        """
        pass
    
    @abstractmethod
    def end_measurement(self, operation_id: str, **kwargs) -> Dict[str, Any]:
        """
        End measuring latency for an operation and calculate results.
        
        Args:
            operation_id (str): ID for the operation.
            **kwargs: Additional metadata for the measurement.
        
        Returns:
            Dict[str, Any]: Measurement results, including latency.
        """
        pass
    
    @abstractmethod
    def measure(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure latency for a function call.
        
        Args:
            func (Callable): Function to measure.
            *args: Arguments for the function.
            **kwargs: Keyword arguments for the function.
        
        Returns:
            Dict[str, Any]: Measurement results, including latency and function result.
        """
        pass
    
    def get_measurements(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get latency measurements within the specified time period.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
            filter_func (Optional[Callable[[Dict[str, Any]], bool]]): Function for filtering measurements.
                If provided, only measurements for which the function returns True will be included.
            limit (int): Maximum number of measurements to return. Defaults to 100.
        
        Returns:
            List[Dict[str, Any]]: List of measurement results.
        """
        with self._lock:
            measurements = self.measurements.copy()
        
        # Filter by start time if provided
        if start_time:
            start_time_iso = start_time.isoformat()
            measurements = [m for m in measurements if m.get("timestamp", "") >= start_time_iso]
            
        # Filter by end time if provided
        if end_time:
            end_time_iso = end_time.isoformat()
            measurements = [m for m in measurements if m.get("timestamp", "") <= end_time_iso]
            
        # Apply custom filter if provided
        if filter_func:
            measurements = [m for m in measurements if filter_func(m)]
            
        # Apply limit
        measurements = measurements[-limit:]
        
        return measurements
    
    def calculate_statistics(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate statistics for latency measurements.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
            filter_func (Optional[Callable[[Dict[str, Any]], bool]]): Function for filtering measurements.
                If provided, only measurements for which the function returns True will be included.
        
        Returns:
            Dict[str, Any]: Statistics for latency measurements.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            filter_func=filter_func,
            limit=10000  # Large limit to include all measurements
        )
        
        if not measurements:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "std_dev": None,
                "unit": self.unit
            }
        
        latencies = [m.get("latency", 0) for m in measurements]
        
        p90_index = int(len(latencies) * 0.9)
        p95_index = int(len(latencies) * 0.95)
        p99_index = int(len(latencies) * 0.99)
        
        sorted_latencies = sorted(latencies)
        
        return {
            "count": len(measurements),
            "min": round(min(latencies), self.precision),
            "max": round(max(latencies), self.precision),
            "mean": round(statistics.mean(latencies), self.precision),
            "median": round(statistics.median(latencies), self.precision),
            "p90": round(sorted_latencies[p90_index], self.precision),
            "p95": round(sorted_latencies[p95_index], self.precision),
            "p99": round(sorted_latencies[p99_index], self.precision),
            "std_dev": round(statistics.stdev(latencies) if len(latencies) > 1 else 0, self.precision),
            "unit": self.unit
        }
    
    def export_measurements(
        self,
        format: str = "json",
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None,
        limit: int = 1000
    ) -> str:
        """
        Export latency measurements in the specified format.
        
        Args:
            format (str): Format for the export. Defaults to 'json'.
                Valid values: 'json', 'csv'.
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
            filter_func (Optional[Callable[[Dict[str, Any]], bool]]): Function for filtering measurements.
                If provided, only measurements for which the function returns True will be included.
            limit (int): Maximum number of measurements to export. Defaults to 1000.
        
        Returns:
            str: Exported measurements in the specified format.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            filter_func=filter_func,
            limit=limit
        )
        
        if format.lower() == "json":
            return json.dumps(measurements, indent=2)
        elif format.lower() == "csv":
            if not measurements:
                return ""
            
            # Get all unique keys across all measurements
            keys = set()
            for m in measurements:
                keys.update(m.keys())
            keys = sorted(keys)
            
            # Create CSV header
            csv_lines = [",".join(keys)]
            
            # Create CSV rows
            for m in measurements:
                values = [str(m.get(k, "")) for k in keys]
                csv_lines.append(",".join(values))
            
            return "\n".join(csv_lines)
        else:
            raise ValueError(f"Unsupported format: {format}. Valid formats: 'json', 'csv'.")
    
    def reset_measurements(self) -> None:
        """
        Reset all latency measurements.
        """
        with self._lock:
            self.measurements = []
    
    def _convert_time_to_unit(self, time_seconds: float) -> float:
        """
        Convert time from seconds to the monitor's unit.
        
        Args:
            time_seconds (float): Time in seconds.
        
        Returns:
            float: Time in the monitor's unit.
        """
        if self.unit == "ns":
            return time_seconds * 1e9
        elif self.unit == "us":
            return time_seconds * 1e6
        elif self.unit == "ms":
            return time_seconds * 1e3
        elif self.unit == "s":
            return time_seconds
        else:
            # Default to milliseconds
            return time_seconds * 1e3
    
    def as_decorator(self, operation_name: Optional[str] = None, include_args: bool = False):
        """
        Create a decorator for measuring function latency.
        
        Args:
            operation_name (Optional[str]): Name for the operation.
                If not provided, the function name will be used.
            include_args (bool): Whether to include function arguments in the measurement metadata.
                Defaults to False.
        
        Returns:
            Callable: Decorator function.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Use function name if operation_name is not provided
                op_name = operation_name or func.__name__
                
                # Prepare metadata
                metadata = {
                    "operation_name": op_name,
                    "function": func.__name__
                }
                
                # Include arguments if requested
                if include_args:
                    metadata["args"] = str(args)
                    metadata["kwargs"] = str(kwargs)
                
                # Start measurement
                operation_id = self.start_measurement(**metadata)
                
                try:
                    # Call the function
                    result = func(*args, **kwargs)
                    
                    # Add success status to metadata
                    metadata["status"] = "success"
                    
                    # End measurement
                    self.end_measurement(operation_id, **metadata)
                    
                    return result
                except Exception as e:
                    # Add error status to metadata
                    metadata["status"] = "error"
                    metadata["error"] = str(e)
                    
                    # End measurement
                    self.end_measurement(operation_id, **metadata)
                    
                    # Re-raise the exception
                    raise
            
            return wrapper
        
        return decorator 