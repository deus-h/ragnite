"""
Base Performance Tracker

This module defines the BasePerformanceTracker abstract class that all performance trackers
must implement. It provides a common interface for tracking various performance metrics.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import time
import json
import os


class BasePerformanceTracker(ABC):
    """
    Abstract base class for performance trackers.
    
    Performance trackers monitor various aspects of RAG system performance,
    such as latency, throughput, memory usage, and CPU usage.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the tracker.
        metrics (Dict[str, List[Dict[str, Any]]]): Dictionary of tracked metrics.
        start_time (datetime): Time when the tracker was initialized.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance tracker.
        
        Args:
            config: Optional configuration dictionary for the tracker.
        """
        self.config = config or {}
        self.metrics = {}
        self.start_time = datetime.now()
    
    @abstractmethod
    def start_tracking(self, track_id: Optional[str] = None) -> str:
        """
        Start tracking a performance metric.
        
        Args:
            track_id: Optional identifier for the tracking session.
                If not provided, a unique ID will be generated.
            
        Returns:
            A tracking ID that can be used to stop tracking and retrieve metrics.
        """
        pass
    
    @abstractmethod
    def stop_tracking(self, track_id: str) -> Dict[str, Any]:
        """
        Stop tracking a performance metric and return the results.
        
        Args:
            track_id: The tracking ID returned by start_tracking.
            
        Returns:
            A dictionary containing the tracked metrics.
        """
        pass
    
    @abstractmethod
    def get_current_value(self, track_id: Optional[str] = None) -> Any:
        """
        Get the current value of a tracked metric.
        
        Args:
            track_id: Optional tracking ID. If not provided, returns the
                current value without association to a tracking session.
            
        Returns:
            The current value of the tracked metric.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Reset all tracking data.
        """
        pass
    
    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all tracked metrics or metrics for a specific name.
        
        Args:
            metric_name: Optional name of the metric to retrieve.
                If not provided, returns all metrics.
            
        Returns:
            A dictionary containing the tracked metrics.
        """
        if metric_name is None:
            return self.metrics
        
        return {metric_name: self.metrics.get(metric_name, [])}
    
    def save_metrics(self, file_path: str, format: str = "json") -> None:
        """
        Save tracked metrics to a file.
        
        Args:
            file_path: Path to save the metrics file.
            format: Format to save the metrics in (currently only "json" is supported).
        """
        if format.lower() != "json":
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        with open(file_path, "w") as f:
            # Add metadata to the saved metrics
            data = {
                "tracker_type": self.__class__.__name__,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "config": self.config,
                "metrics": self.metrics
            }
            json.dump(data, f, indent=2)
    
    def load_metrics(self, file_path: str) -> Dict[str, Any]:
        """
        Load tracked metrics from a file.
        
        Args:
            file_path: Path to load the metrics from.
            
        Returns:
            A dictionary containing the loaded metrics.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metrics file not found: {file_path}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # Extract and set the metrics
        self.metrics = data.get("metrics", {})
        
        # Return the full data including metadata
        return data
    
    def calculate_statistics(self, metric_name: str) -> Dict[str, Any]:
        """
        Calculate statistics for a specific metric.
        
        Args:
            metric_name: Name of the metric to calculate statistics for.
            
        Returns:
            A dictionary containing statistics like min, max, mean, etc.
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = [m.get("value", 0) for m in self.metrics[metric_name] if "value" in m]
        
        if not values:
            return {}
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "count": len(values),
            "sum": sum(values),
            "last": values[-1]
        }
    
    def __repr__(self) -> str:
        """
        Return a string representation of the tracker.
        
        Returns:
            A string describing the tracker and its metrics.
        """
        metrics_summary = {k: len(v) for k, v in self.metrics.items()}
        return f"{self.__class__.__name__}(metrics={metrics_summary}, start_time={self.start_time.isoformat()})"
    
    def __enter__(self):
        """
        Context manager entry point. Starts tracking with a new ID.
        
        Returns:
            The tracking ID.
        """
        self._context_id = self.start_tracking()
        return self._context_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point. Stops tracking with the stored ID.
        
        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        if hasattr(self, "_context_id"):
            self.stop_tracking(self._context_id)
            delattr(self, "_context_id") 