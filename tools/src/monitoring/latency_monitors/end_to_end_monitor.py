#!/usr/bin/env python3
"""
End-to-end latency monitor for RAG systems.

This module provides a latency monitor for measuring, tracking, and analyzing
end-to-end latency in RAG systems, from initial request to final response.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import datetime
import time
import uuid
from .base import BaseLatencyMonitor


class EndToEndLatencyMonitor(BaseLatencyMonitor):
    """
    Latency monitor for end-to-end operations in RAG systems.
    
    This monitor focuses on measuring and analyzing latency for complete
    end-to-end operations in RAG systems, from initial request to final response,
    helping identify performance bottlenecks and optimize overall response times.
    
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
        Initialize the end-to-end latency monitor.
        
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
        Start measuring end-to-end latency for an operation.
        
        Args:
            operation_id (Optional[str]): ID for the operation.
                If not provided, a unique ID will be generated.
            **kwargs: Additional metadata for the measurement.
                Common metadata for end-to-end operations:
                - query (str): User query or request.
                - user_id (str): ID of the user making the request.
                - session_id (str): ID of the user session.
                - request_source (str): Source of the request (e.g., 'web', 'api', 'mobile').
                - with_retrieval (bool): Whether the operation includes retrieval.
                - with_generation (bool): Whether the operation includes generation.
                - models (List[str]): List of models used in the operation.
        
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
    
    def record_component_time(
        self,
        operation_id: str,
        component: str,
        start_time: float,
        end_time: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Record component-specific time for an ongoing operation.
        
        Args:
            operation_id (str): ID for the operation.
            component (str): Name of the component (e.g., 'retrieval', 'generation').
            start_time (float): Start time for the component operation.
            end_time (float): End time for the component operation.
            **kwargs: Additional metadata for the component measurement.
        
        Returns:
            Dict[str, Any]: Updated measurement.
            
        Raises:
            ValueError: If the operation ID is not found in active measurements.
        """
        # Calculate component latency
        latency_seconds = end_time - start_time
        latency = self._convert_time_to_unit(latency_seconds)
        
        # Get active measurement
        with self._lock:
            if operation_id not in self._active_measurements:
                raise ValueError(f"Operation ID not found: {operation_id}")
            
            measurement = self._active_measurements[operation_id]
            
            # Initialize components dictionary if not present
            if "components" not in measurement:
                measurement["components"] = {}
            
            # Add component time
            measurement["components"][component] = {
                "latency": round(latency, self.precision),
                "latency_unit": self.unit,
                "start_time": start_time,
                "end_time": end_time,
                **kwargs
            }
        
        return measurement
    
    def end_measurement(self, operation_id: str, **kwargs) -> Dict[str, Any]:
        """
        End measuring end-to-end latency for an operation and calculate results.
        
        Args:
            operation_id (str): ID for the operation.
            **kwargs: Additional metadata for the measurement.
                Common metadata for end-to-end operations:
                - response (str): Response text or data.
                - status (str): Status of the operation (e.g., 'success', 'error').
                - error (str): Error message if operation failed.
                - total_tokens (int): Total number of tokens used.
                - total_cost (float): Total cost of the operation.
        
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
        
        # Calculate component percentages if components exist
        if "components" in measurement:
            components = measurement["components"]
            component_total = 0
            
            # Calculate total component time
            for component, data in components.items():
                component_total += data.get("latency", 0)
            
            # Calculate percentages
            for component, data in components.items():
                if component_total > 0:
                    data["percentage"] = round((data.get("latency", 0) / component_total) * 100, 1)
                else:
                    data["percentage"] = 0
        
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
        if "components" in result:
            for component in result["components"].values():
                component.pop("start_time", None)
                component.pop("end_time", None)
        
        # Record measurement
        with self._lock:
            self.measurements.append(result)
        
        return result
    
    def measure(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Measure end-to-end latency for a function call.
        
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
                for key in ["status", "error", "total_tokens", "total_cost"]:
                    if key in result:
                        result_metadata[key] = result[key]
                
                # Extract component times if available
                if "component_times" in result and isinstance(result["component_times"], dict):
                    for component, component_time in result["component_times"].items():
                        if isinstance(component_time, dict) and "start_time" in component_time and "end_time" in component_time:
                            self.record_component_time(
                                operation_id=operation_id,
                                component=component,
                                start_time=component_time["start_time"],
                                end_time=component_time["end_time"],
                                **{k: v for k, v in component_time.items() if k not in ["start_time", "end_time"]}
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
    
    def analyze_by_request_source(self, 
                                start_time: Optional[datetime.datetime] = None,
                                end_time: Optional[datetime.datetime] = None
                               ) -> Dict[str, Any]:
        """
        Analyze end-to-end latency by request source.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by request source.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by request source
        sources = {}
        for m in measurements:
            source = m.get("request_source", "unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append(m)
        
        # Calculate statistics for each source
        results = {}
        for source, source_measurements in sources.items():
            latencies = [m.get("latency", 0) for m in source_measurements]
            
            if not latencies:
                continue
            
            results[source] = {
                "count": len(latencies),
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results
    
    def analyze_component_breakdown(self, 
                                  start_time: Optional[datetime.datetime] = None,
                                  end_time: Optional[datetime.datetime] = None
                                 ) -> Dict[str, Any]:
        """
        Analyze component latency breakdown.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by component.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Initialize component data
        components = {}
        total_operations = 0
        total_latency = 0
        
        # Process measurements
        for m in measurements:
            total_operations += 1
            total_latency += m.get("latency", 0)
            
            # Skip measurements without components
            if "components" not in m or not isinstance(m["components"], dict):
                continue
            
            # Process each component
            for component_name, component_data in m["components"].items():
                if component_name not in components:
                    components[component_name] = {
                        "count": 0,
                        "total_latency": 0,
                        "latencies": []
                    }
                
                # Add component data
                components[component_name]["count"] += 1
                component_latency = component_data.get("latency", 0)
                components[component_name]["total_latency"] += component_latency
                components[component_name]["latencies"].append(component_latency)
        
        # Calculate statistics for each component
        results = {
            "total_operations": total_operations,
            "total_latency": round(total_latency, self.precision),
            "components": {}
        }
        
        for component_name, component_data in components.items():
            latencies = component_data["latencies"]
            
            if not latencies:
                continue
            
            results["components"][component_name] = {
                "count": component_data["count"],
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "percentage_of_operations": round((component_data["count"] / total_operations) * 100 if total_operations > 0 else 0, 1),
                "percentage_of_total_latency": round((component_data["total_latency"] / total_latency) * 100 if total_latency > 0 else 0, 1),
                "unit": self.unit
            }
        
        return results
    
    def analyze_by_operation_type(self, 
                                start_time: Optional[datetime.datetime] = None,
                                end_time: Optional[datetime.datetime] = None
                               ) -> Dict[str, Any]:
        """
        Analyze end-to-end latency by operation type.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by operation type.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Define operation types based on included components
        operation_types = {
            "retrieval_only": [],
            "generation_only": [],
            "retrieval_and_generation": [],
            "other": []
        }
        
        for m in measurements:
            with_retrieval = m.get("with_retrieval", False)
            with_generation = m.get("with_generation", False)
            
            if with_retrieval and with_generation:
                operation_types["retrieval_and_generation"].append(m)
            elif with_retrieval:
                operation_types["retrieval_only"].append(m)
            elif with_generation:
                operation_types["generation_only"].append(m)
            else:
                operation_types["other"].append(m)
        
        # Calculate statistics for each operation type
        results = {}
        for operation_type, type_measurements in operation_types.items():
            latencies = [m.get("latency", 0) for m in type_measurements]
            
            if not latencies:
                continue
            
            results[operation_type] = {
                "count": len(latencies),
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results
    
    def analyze_performance_trends(self, 
                                 interval: str = "day",
                                 start_time: Optional[datetime.datetime] = None,
                                 end_time: Optional[datetime.datetime] = None
                                ) -> Dict[str, Any]:
        """
        Analyze end-to-end latency performance trends over time.
        
        Args:
            interval (str): Time interval for grouping. Defaults to 'day'.
                Valid values: 'hour', 'day', 'week', 'month'.
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by time interval.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by time interval
        intervals = {}
        
        for m in measurements:
            timestamp_str = m.get("timestamp", "")
            
            try:
                timestamp = datetime.datetime.fromisoformat(timestamp_str)
                
                # Format timestamp based on interval
                if interval == "hour":
                    interval_key = timestamp.strftime("%Y-%m-%d %H:00")
                elif interval == "day":
                    interval_key = timestamp.strftime("%Y-%m-%d")
                elif interval == "week":
                    # ISO week with year
                    interval_key = f"{timestamp.strftime('%Y')}-W{timestamp.strftime('%V')}"
                elif interval == "month":
                    interval_key = timestamp.strftime("%Y-%m")
                else:
                    # Default to day
                    interval_key = timestamp.strftime("%Y-%m-%d")
                
                if interval_key not in intervals:
                    intervals[interval_key] = []
                
                intervals[interval_key].append(m)
            except (ValueError, TypeError):
                # Skip measurements with invalid timestamps
                continue
        
        # Calculate statistics for each interval
        results = {}
        for interval_key, interval_measurements in intervals.items():
            latencies = [m.get("latency", 0) for m in interval_measurements]
            
            if not latencies:
                continue
            
            results[interval_key] = {
                "count": len(latencies),
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results 