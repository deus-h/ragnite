"""
Throughput Tracker

This module provides the ThroughputTracker class for tracking the throughput of operations
in a RAG system, such as queries per second, documents processed per second, etc.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime
from collections import deque

from .base_performance_tracker import BasePerformanceTracker


class ThroughputTracker(BasePerformanceTracker):
    """
    Tracker for measuring the throughput of operations.
    
    This tracker measures how many operations can be completed in a given time period,
    such as queries per second, documents processed per second, or tokens generated per second.
    It can track multiple operation types concurrently and provides statistics on throughput.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the tracker.
        metrics (Dict[str, List[Dict[str, Any]]]): Dictionary of tracked throughput metrics.
        start_time (datetime): Time when the tracker was initialized.
        active_tracks (Dict[str, Dict[str, Any]]): Dictionary of currently active tracking sessions.
        event_queues (Dict[str, deque]): Dictionary of event queues for calculating throughput.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the throughput tracker.
        
        Args:
            config: Optional configuration dictionary for the tracker.
                - metric_name: Name to use for the throughput metric (default: "throughput").
                - time_unit: Unit to use for time measurements (default: "s").
                - window_size: Size of the sliding window for throughput calculation (default: 60).
                - include_timestamps: Whether to include timestamps in the metrics (default: True).
        """
        super().__init__(config)
        
        # Set default configuration values
        if 'metric_name' not in self.config:
            self.config['metric_name'] = "throughput"
        if 'time_unit' not in self.config:
            self.config['time_unit'] = "s"  # seconds
        if 'window_size' not in self.config:
            self.config['window_size'] = 60  # 60 seconds sliding window
        if 'include_timestamps' not in self.config:
            self.config['include_timestamps'] = True
            
        # Initialize the metrics dictionary with the configured metric name
        self.metrics[self.config['metric_name']] = []
        
        # Dictionary to store active tracking sessions
        self.active_tracks = {}
        
        # Dictionary to store event queues for each label
        self.event_queues = {}
    
    def start_tracking(self, track_id: Optional[str] = None, label: Optional[str] = None) -> str:
        """
        Start tracking the throughput of an operation.
        
        Args:
            track_id: Optional identifier for the tracking session.
                If not provided, a unique ID will be generated.
            label: Optional label to identify what is being tracked (e.g., "queries").
            
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
            "label": label or "default",
            "count": 0  # Initialize count to 0
        }
        
        # Initialize event queue for this label if it doesn't exist
        if label not in self.event_queues:
            self.event_queues[label or "default"] = deque()
        
        return track_id
    
    def record_event(self, track_id: str, count: int = 1) -> Dict[str, Any]:
        """
        Record an event for throughput calculation.
        
        Args:
            track_id: The tracking ID returned by start_tracking.
            count: Number of operations to record (default: 1).
            
        Returns:
            A dictionary containing the updated tracking session.
            
        Raises:
            ValueError: If the track_id is not found in active tracks.
        """
        if track_id not in self.active_tracks:
            raise ValueError(f"Tracking session with ID {track_id} not found.")
            
        # Get the tracking session
        track = self.active_tracks[track_id]
        label = track["label"]
        
        # Record the event timestamp and count
        event_time = time.time()
        self.event_queues[label].append((event_time, count))
        
        # Update the count in the tracking session
        track["count"] += count
        
        # Remove events outside the sliding window
        window_start = event_time - self.config['window_size']
        while self.event_queues[label] and self.event_queues[label][0][0] < window_start:
            self.event_queues[label].popleft()
        
        return track
    
    def stop_tracking(self, track_id: str) -> Dict[str, Any]:
        """
        Stop tracking throughput and return the results.
        
        Args:
            track_id: The tracking ID returned by start_tracking.
            
        Returns:
            A dictionary containing the throughput metrics.
            
        Raises:
            ValueError: If the track_id is not found in active tracks.
        """
        if track_id not in self.active_tracks:
            raise ValueError(f"Tracking session with ID {track_id} not found.")
            
        # Get the tracking session
        track = self.active_tracks[track_id]
        start_time = track["start_time"]
        label = track["label"]
        count = track["count"]
        
        # Record the end time
        end_time = time.time()
        
        # Calculate the duration in the specified time unit
        duration_seconds = end_time - start_time
        
        # Calculate the throughput (events per time unit)
        if duration_seconds > 0:
            throughput = count / duration_seconds
        else:
            throughput = 0
            
        # Convert to the specified time unit
        if self.config['time_unit'] == "m":
            throughput *= 60  # Convert to per minute
        elif self.config['time_unit'] == "h":
            throughput *= 3600  # Convert to per hour
        
        # Prepare the metric data
        metric_data = {
            "track_id": track_id,
            "value": throughput,
            "count": count,
            "duration": duration_seconds,
            "timestamp": datetime.now().isoformat(),
            "unit": f"events/{self.config['time_unit']}"
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
    
    def get_current_value(self, track_id: Optional[str] = None, label: Optional[str] = None) -> float:
        """
        Get the current throughput for an active tracking session or a label.
        
        This calculates the throughput based on the events in the sliding window.
        
        Args:
            track_id: Optional tracking ID. If provided, returns the current
                throughput for that tracking session.
            label: Optional label. If provided instead of track_id, returns the
                current throughput for all tracking sessions with that label.
            
        Returns:
            The current throughput in the configured time unit.
            
        Raises:
            ValueError: If neither track_id nor label is provided and there are no active tracks,
                or if the provided track_id is not found.
        """
        if track_id is not None:
            # Get throughput for a specific tracking session
            if track_id not in self.active_tracks:
                raise ValueError(f"Tracking session with ID {track_id} not found.")
                
            track = self.active_tracks[track_id]
            label = track["label"]
            
        elif label is not None:
            # Get throughput for a specific label
            if label not in self.event_queues:
                raise ValueError(f"No events found for label {label}.")
        else:
            # If neither track_id nor label is provided
            if not self.active_tracks:
                raise ValueError("No active tracking sessions.")
                
            # Use the most recently started tracking session
            track_id = sorted(self.active_tracks.items(), key=lambda x: x[1]["start_time"])[-1][0]
            track = self.active_tracks[track_id]
            label = track["label"]
        
        # Calculate the throughput from the event queue
        if label not in self.event_queues or not self.event_queues[label]:
            return 0.0
            
        # Get current time and window start time
        current_time = time.time()
        window_start = current_time - self.config['window_size']
        
        # Remove events outside the sliding window
        while self.event_queues[label] and self.event_queues[label][0][0] < window_start:
            self.event_queues[label].popleft()
            
        # Calculate the total count within the window
        total_count = sum(count for _, count in self.event_queues[label])
        
        # Calculate the window duration
        if self.event_queues[label]:
            window_duration = current_time - max(window_start, self.event_queues[label][0][0])
        else:
            window_duration = 0
            
        # Calculate the throughput (events per time unit)
        if window_duration > 0:
            throughput = total_count / window_duration
        else:
            throughput = 0
            
        # Convert to the specified time unit
        if self.config['time_unit'] == "m":
            throughput *= 60  # Convert to per minute
        elif self.config['time_unit'] == "h":
            throughput *= 3600  # Convert to per hour
            
        return throughput
    
    def reset(self) -> None:
        """
        Reset all tracking data.
        
        This method clears all tracked metrics, active tracking sessions, and event queues.
        """
        # Clear the metrics
        self.metrics = {self.config['metric_name']: []}
        
        # Clear active tracking sessions
        self.active_tracks = {}
        
        # Clear event queues
        self.event_queues = {}
        
        # Reset the start time
        self.start_time = datetime.now()
    
    def get_throughput_statistics(self, label: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for the tracked throughput, optionally filtered by label.
        
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
            "unit": f"events/{self.config['time_unit']}"
        }
    
    def get_throughput_by_label(self) -> Dict[str, Dict[str, Any]]:
        """
        Get throughput statistics grouped by label.
        
        Returns:
            A dictionary mapping labels to their throughput statistics.
        """
        metric_name = self.config['metric_name']
        
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
            
        # Group metrics by label
        labels = set(m.get("label") for m in self.metrics[metric_name] if "label" in m)
        
        return {label: self.get_throughput_statistics(label) for label in labels}
    
    def start_batch(self, label: Optional[str] = None) -> str:
        """
        Start tracking throughput for a batch of operations.
        
        This is a convenience method that calls start_tracking.
        
        Args:
            label: Optional label to identify what is being tracked.
            
        Returns:
            A tracking ID that can be used with record_batch and stop_batch.
        """
        return self.start_tracking(label=label)
    
    def record_batch(self, track_id: str, count: int) -> Dict[str, Any]:
        """
        Record a batch of operations for throughput calculation.
        
        This is a convenience method that calls record_event.
        
        Args:
            track_id: The tracking ID returned by start_batch.
            count: Number of operations in the batch.
            
        Returns:
            A dictionary containing the updated tracking session.
        """
        return self.record_event(track_id, count)
    
    def stop_batch(self, track_id: str) -> Dict[str, Any]:
        """
        Stop tracking throughput for a batch of operations and return the results.
        
        This is a convenience method that calls stop_tracking.
        
        Args:
            track_id: The tracking ID returned by start_batch.
            
        Returns:
            A dictionary containing the throughput metrics.
        """
        return self.stop_tracking(track_id)
    
    def track_function_throughput(self, func: Callable, *args, 
                                 label: Optional[str] = None,
                                 count_func: Optional[Callable[[Any], int]] = None,
                                 **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Track the throughput of a function call.
        
        Args:
            func: The function to track.
            *args: Arguments to pass to the function.
            label: Optional label to identify what is being tracked.
            count_func: Optional function to calculate the count from the result.
                If not provided, a count of 1 is used.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            A tuple containing the result of the function call and the throughput metrics.
        """
        track_id = self.start_tracking(label=label)
        try:
            result = func(*args, **kwargs)
            
            # Calculate the count
            if count_func is not None:
                count = count_func(result)
            else:
                count = 1
                
            self.record_event(track_id, count)
            
            return result, self.stop_tracking(track_id)
        except:
            self.stop_tracking(track_id)
            raise 