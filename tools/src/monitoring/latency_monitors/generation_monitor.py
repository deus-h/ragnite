#!/usr/bin/env python3
"""
Generation latency monitor for RAG systems.

This module provides a latency monitor for measuring, tracking, and analyzing
generation latency in RAG systems, focusing on the LLM response generation.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import datetime
import time
import uuid
from .base import BaseLatencyMonitor


class GenerationLatencyMonitor(BaseLatencyMonitor):
    """
    Latency monitor for generation operations in RAG systems.
    
    This monitor focuses on measuring and analyzing latency for generation operations
    in the LLM component of RAG systems, helping identify performance
    bottlenecks and optimize response times.
    
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
        Initialize the generation latency monitor.
        
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
        Start measuring generation latency for an operation.
        
        Args:
            operation_id (Optional[str]): ID for the operation.
                If not provided, a unique ID will be generated.
            **kwargs: Additional metadata for the measurement.
                Common metadata for generation operations:
                - model (str): LLM model name or version.
                - prompt (str): Prompt text.
                - prompt_tokens (int): Number of tokens in the prompt.
                - max_tokens (int): Maximum number of tokens to generate.
                - temperature (float): Temperature for generation.
                - provider (str): LLM provider (e.g., 'openai', 'anthropic').
        
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
        End measuring generation latency for an operation and calculate results.
        
        Args:
            operation_id (str): ID for the operation.
            **kwargs: Additional metadata for the measurement.
                Common metadata for generation operations:
                - completion (str): Generated text.
                - completion_tokens (int): Number of tokens in the generated text.
                - total_tokens (int): Total number of tokens (prompt + completion).
                - finish_reason (str): Reason for finishing generation.
                - time_to_first_token (float): Time to generate the first token.
                - tokens_per_second (float): Generation speed in tokens per second.
        
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
        
        # Calculate tokens per second if token counts are available
        completion_tokens = kwargs.get("completion_tokens")
        if completion_tokens is not None and completion_tokens > 0 and latency_seconds > 0:
            tokens_per_second = completion_tokens / latency_seconds
            kwargs["tokens_per_second"] = round(tokens_per_second, self.precision)
        
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
        Measure generation latency for a function call.
        
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
            
            # Extract completion metadata if available
            completion_metadata = {}
            if isinstance(result, dict):
                # Extract completion tokens and other metadata
                if "completion_tokens" in result:
                    completion_metadata["completion_tokens"] = result.get("completion_tokens")
                elif "usage" in result and isinstance(result["usage"], dict):
                    completion_metadata["completion_tokens"] = result["usage"].get("completion_tokens")
                
                # Extract other metadata
                for key in ["finish_reason", "total_tokens", "model"]:
                    if key in result:
                        completion_metadata[key] = result[key]
            
            # Calculate tokens per second if token counts are available
            if "completion_tokens" in completion_metadata and completion_metadata["completion_tokens"] > 0:
                tokens_per_second = completion_metadata["completion_tokens"] / execution_time
                completion_metadata["tokens_per_second"] = round(tokens_per_second, self.precision)
            
            # Prepare additional metadata
            additional_metadata = {
                "execution_time": round(self._convert_time_to_unit(execution_time), self.precision),
                "execution_time_unit": self.unit,
                "status": "success",
                **completion_metadata
            }
            
            # End measurement
            self.end_measurement(operation_id, **additional_metadata)
            
            return {
                "result": result,
                "latency": additional_metadata["execution_time"],
                "latency_unit": self.unit,
                "operation_id": operation_id,
                **completion_metadata
            }
        except Exception as e:
            # End measurement with error
            self.end_measurement(operation_id, status="error", error=str(e))
            
            # Re-raise the exception
            raise
    
    def analyze_by_model(self, 
                       start_time: Optional[datetime.datetime] = None,
                       end_time: Optional[datetime.datetime] = None
                      ) -> Dict[str, Any]:
        """
        Analyze generation latency by model.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by model.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by model
        models = {}
        for m in measurements:
            model = m.get("model", "unknown")
            if model not in models:
                models[model] = []
            models[model].append(m)
        
        # Calculate statistics for each model
        results = {}
        for model, model_measurements in models.items():
            latencies = [m.get("latency", 0) for m in model_measurements]
            tokens_per_second = [m.get("tokens_per_second", 0) for m in model_measurements if m.get("tokens_per_second", 0) > 0]
            
            if not latencies:
                continue
            
            results[model] = {
                "count": len(latencies),
                "latency": {
                    "min": round(min(latencies), self.precision),
                    "max": round(max(latencies), self.precision),
                    "mean": round(sum(latencies) / len(latencies), self.precision),
                    "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                    "unit": self.unit
                }
            }
            
            # Add tokens per second statistics if available
            if tokens_per_second:
                results[model]["tokens_per_second"] = {
                    "min": round(min(tokens_per_second), self.precision),
                    "max": round(max(tokens_per_second), self.precision),
                    "mean": round(sum(tokens_per_second) / len(tokens_per_second), self.precision)
                }
        
        return results
    
    def analyze_by_prompt_length(self, 
                               start_time: Optional[datetime.datetime] = None,
                               end_time: Optional[datetime.datetime] = None
                              ) -> Dict[str, Any]:
        """
        Analyze generation latency by prompt length.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Latency statistics by prompt length category.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by prompt length
        length_groups = {
            "short": [],
            "medium": [],
            "long": []
        }
        
        for m in measurements:
            # Determine length category based on prompt tokens
            prompt_tokens = m.get("prompt_tokens", 0)
            
            if prompt_tokens > 0:
                if prompt_tokens < 100:
                    length_category = "short"
                elif prompt_tokens < 1000:
                    length_category = "medium"
                else:
                    length_category = "long"
            else:
                # Default to medium if length can't be determined
                length_category = "medium"
            
            length_groups[length_category].append(m)
        
        # Calculate statistics for each length category
        results = {}
        for category, group_measurements in length_groups.items():
            latencies = [m.get("latency", 0) for m in group_measurements]
            
            if not latencies:
                continue
            
            results[category] = {
                "count": len(latencies),
                "min": round(min(latencies), self.precision),
                "max": round(max(latencies), self.precision),
                "mean": round(sum(latencies) / len(latencies), self.precision),
                "p90": round(sorted(latencies)[int(len(latencies) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results
    
    def analyze_time_to_first_token(self, 
                                  start_time: Optional[datetime.datetime] = None,
                                  end_time: Optional[datetime.datetime] = None
                                 ) -> Dict[str, Any]:
        """
        Analyze time to first token by model.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Time to first token statistics by model.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by model
        models = {}
        for m in measurements:
            # Skip measurements without time_to_first_token
            if "time_to_first_token" not in m:
                continue
                
            model = m.get("model", "unknown")
            if model not in models:
                models[model] = []
            models[model].append(m)
        
        # Calculate statistics for each model
        results = {}
        for model, model_measurements in models.items():
            times = [m.get("time_to_first_token", 0) for m in model_measurements]
            
            if not times:
                continue
            
            results[model] = {
                "count": len(times),
                "min": round(min(times), self.precision),
                "max": round(max(times), self.precision),
                "mean": round(sum(times) / len(times), self.precision),
                "p90": round(sorted(times)[int(len(times) * 0.9)], self.precision),
                "unit": self.unit
            }
        
        return results
    
    def analyze_tokens_per_second(self, 
                                start_time: Optional[datetime.datetime] = None,
                                end_time: Optional[datetime.datetime] = None
                               ) -> Dict[str, Any]:
        """
        Analyze tokens per second by model.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering measurements.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering measurements.
                If not provided, no upper bound will be applied.
        
        Returns:
            Dict[str, Any]: Tokens per second statistics by model.
        """
        measurements = self.get_measurements(
            start_time=start_time,
            end_time=end_time,
            limit=10000  # Large limit to include all measurements
        )
        
        # Group measurements by model
        models = {}
        for m in measurements:
            # Skip measurements without tokens_per_second
            if "tokens_per_second" not in m:
                continue
                
            model = m.get("model", "unknown")
            if model not in models:
                models[model] = []
            models[model].append(m)
        
        # Calculate statistics for each model
        results = {}
        for model, model_measurements in models.items():
            tps_values = [m.get("tokens_per_second", 0) for m in model_measurements]
            
            if not tps_values:
                continue
            
            results[model] = {
                "count": len(tps_values),
                "min": round(min(tps_values), self.precision),
                "max": round(max(tps_values), self.precision),
                "mean": round(sum(tps_values) / len(tps_values), self.precision),
                "p90": round(sorted(tps_values)[int(len(tps_values) * 0.9)], self.precision)
            }
        
        return results 