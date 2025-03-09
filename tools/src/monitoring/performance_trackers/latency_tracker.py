"""
Latency Tracker

This module provides the LatencyTracker class for tracking the latency of operations
in a RAG system, such as query processing, retrieval, and generation.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime

from .base_performance_tracker import BasePerformanceTracker


class LatencyTracker(BasePerformanceTracker):
    """
    Tracker for measuring the latency of operations.
    
    This tracker measures the time it takes to complete operations in a RAG system,
    such as query processing, retrieval, and generation. It can track multiple
    operations concurrently and provides statistics on latency.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the tracker.
        metrics (Dict[str, List[Dict[str, Any]]]): Dictionary of tracked latency metrics.
        start_time (datetime): Time when the tracker was initialized.
        active_tracks (Dict[str, Dict[str, Any]]): Dictionary of currently active tracking sessions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the latency tracker.
        
        Args:
            config: Optional configuration dictionary for the tracker.
                - metric_name: Name to use for the latency metric (default: "latency").
                - time_unit: Unit to use for time measurements (default: "ms").
                - include_timestamps: Whether to include start and end timestamps in the metrics (default: True).
        """
        super().__init__(config)
        
        # Set default configuration values
        if 'metric_name' not in self.config:
            self.config['metric_name'] = "latency"
        if 'time_unit' not in self.config:
            self.config['time_unit'] = "ms"  # milliseconds
        if 'include_timestamps' not in self.config:
            self.config['include_timestamps'] = True
            
        # Initialize the metrics dictionary with the configured metric name
        self.metrics[self.config['metric_name']] = []
        
        # Dictionary to store active tracking sessions
        self.active_tracks = {}
    
    def start_tracking(self, track_id: Optional[str] = None, label: Optional[str] = None) -> str:
        """
        Start tracking the latency of an operation.
        
        Args:
            track_id: Optional identifier for the tracking session.
                If not provided, a unique ID will be generated.
            label: Optional label to identify what is being tracked (e.g., "query_processing").
            
        Returns:
            A tracking ID that can be used to stop tracking and retrieve metrics.
        """
        # Generate a tracking ID if none was provided
        if track_id is None:
            track_id = str(uuid.uuid4())
            
        # Record the start time
        start_time = time.time()
        
        # Store the tracking session
        self.active_tracks[track_id] = {
            "start_time": start_time,
            "label": label
        }
        
        return track_id
    
    def stop_tracking(self, track_id: str) -> Dict[str, Any]:
        """
        Stop tracking latency and return the results.
        
        Args:
            track_id: The tracking ID returned by start_tracking.
            
        Returns:
            A dictionary containing the latency metrics.
            
        Raises:
            ValueError: If the track_id is not found in active tracks.
        """
        if track_id not in self.active_tracks:
            raise ValueError(f"Tracking session with ID {track_id} not found.")
            
        # Record the end time
        end_time = time.time()
        
        # Get the tracking session
        track = self.active_tracks[track_id]
        start_time = track["start_time"]
        label = track["label"]
        
        # Calculate the latency in the specified time unit
        latency_seconds = end_time - start_time
        
        if self.config['time_unit'] == "ms":
            latency = latency_seconds * 1000  # Convert to milliseconds
        elif self.config['time_unit'] == "us":
            latency = latency_seconds * 1000000  # Convert to microseconds
        elif self.config['time_unit'] == "ns":
            latency = latency_seconds * 1000000000  # Convert to nanoseconds
        else:  # Default to seconds
            latency = latency_seconds
            
        # Prepare the metric data
        metric_data = {
            "track_id": track_id,
            "value": latency,
            "timestamp": datetime.now().isoformat(),
            "unit": self.config['time_unit']
        }
        
        if label:
            metric_data["label"] = label
            
        if self.config['include_timestamps']:
            metric_data["start_time"] = datetime.fromtimestamp(start_time).isoformat()
            metric_data["end_time"] = datetime.fromtimestamp(end_time).isoformat()
            
        # Add the metric to the metrics list
        metric_name = self.config['metric_name']
        self.metrics[metric_name].append(metric_data)
        
        # Remove the tracking session from active tracks
        del self.active_tracks[track_id]
        
        return metric_data
    
    def get_current_value(self, track_id: Optional[str] = None) -> float:
        """
        Get the current latency for an active tracking session.
        
        Args:
            track_id: Optional tracking ID. If provided, returns the current
                latency for that tracking session. If not provided, returns the
                current latency for the last tracking session.
            
        Returns:
            The current latency in the configured time unit.
            
        Raises:
            ValueError: If the track_id is not found in active tracks or if there are no active tracks.
        """
        if not self.active_tracks:
            raise ValueError("No active tracking sessions.")
            
        # If no track_id was provided, use the most recently started tracking session
        if track_id is None:
            # Sort active tracks by start time and get the most recent one
            track_id = sorted(self.active_tracks.items(), key=lambda x: x[1]["start_time"])[-1][0]
            
        if track_id not in self.active_tracks:
            raise ValueError(f"Tracking session with ID {track_id} not found.")
            
        # Get the tracking session
        track = self.active_tracks[track_id]
        start_time = track["start_time"]
        
        # Calculate the current latency
        current_time = time.time()
        latency_seconds = current_time - start_time
        
        # Convert to the specified time unit
        if self.config['time_unit'] == "ms":
            return latency_seconds * 1000
        elif self.config['time_unit'] == "us":
            return latency_seconds * 1000000
        elif self.config['time_unit'] == "ns":
            return latency_seconds * 1000000000
        else:
            return latency_seconds
    
    def reset(self) -> None:
        """
        Reset all tracking data.
        
        This method clears all tracked metrics and active tracking sessions.
        """
        # Clear the metrics
        self.metrics = {self.config['metric_name']: []}
        
        # Clear active tracking sessions
        self.active_tracks = {}
        
        # Reset the start time
        self.start_time = datetime.now()
    
    def get_latency_statistics(self, label: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for the tracked latencies, optionally filtered by label.
        
        Args:
            label: Optional label to filter the metrics by.
            
        Returns:
            A dictionary containing statistics like min, max, mean, etc.
        """
        metric_name = self.config['metric_name']
        
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
            
        # Filter metrics by label if provided
        if label:
            values = [m.get("value", 0) for m in self.metrics[metric_name] 
                     if "value" in m and m.get("label") == label]
        else:
            values = [m.get("value", 0) for m in self.metrics[metric_name] if "value" in m]
            
        if not values:
            return {}
            
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
            "count": len(values),
            "sum": sum(values),
            "last": values[-1],
            "unit": self.config['time_unit']
        }
    
    def get_latency_by_label(self) -> Dict[str, Dict[str, Any]]:
        """
        Get latency statistics grouped by label.
        
        Returns:
            A dictionary mapping labels to their latency statistics.
        """
        metric_name = self.config['metric_name']
        
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
            
        # Group metrics by label
        labels = set(m.get("label") for m in self.metrics[metric_name] if "label" in m)
        
        return {label: self.get_latency_statistics(label) for label in labels}
    
    def track_function(self, func: Callable, *args, label: Optional[str] = None, **kwargs) -> Any:
        """
        Track the latency of a function call.
        
        Args:
            func: The function to track.
            *args: Arguments to pass to the function.
            label: Optional label to identify what is being tracked.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The result of the function call.
        """
        track_id = self.start_tracking(label=label)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            self.stop_tracking(track_id)
    
    def __call__(self, func: Callable) -> Callable:
        """
        Use the tracker as a decorator to track the latency of a function.
        
        Args:
            func: The function to track.
            
        Returns:
            A wrapped function that tracks latency.
        """
        def wrapper(*args, **kwargs):
            return self.track_function(func, *args, label=func.__name__, **kwargs)
        
        return wrapper 