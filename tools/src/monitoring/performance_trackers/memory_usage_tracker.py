"""
Memory Usage Tracker

This module provides the MemoryUsageTracker class for tracking memory usage
in a RAG system, including RAM usage, GPU memory, and memory peaks.
"""

import time
import uuid
import os
import platform
import psutil
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_performance_tracker import BasePerformanceTracker


class MemoryUsageTracker(BasePerformanceTracker):
    """
    Tracker for measuring memory usage.
    
    This tracker measures the memory usage of a RAG system, including RAM usage,
    GPU memory (if available), and memory peaks. It can track memory usage over time
    and provides statistics on memory usage.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the tracker.
        metrics (Dict[str, List[Dict[str, Any]]]): Dictionary of tracked memory metrics.
        start_time (datetime): Time when the tracker was initialized.
        active_tracks (Dict[str, Dict[str, Any]]): Dictionary of currently active tracking sessions.
        process (psutil.Process): Current process being monitored.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory usage tracker.
        
        Args:
            config: Optional configuration dictionary for the tracker.
                - metric_name: Name to use for the memory metric (default: "memory_usage").
                - track_gpu: Whether to track GPU memory usage (default: True if torch is available).
                - gpu_device: GPU device to track (default: 0).
                - unit: Unit to use for memory measurements (default: "MB").
                - track_peak: Whether to track peak memory usage (default: True).
                - sampling_interval: Interval in seconds for memory usage sampling (default: 1.0).
                - include_timestamps: Whether to include timestamps in the metrics (default: True).
        """
        super().__init__(config)
        
        # Set default configuration values
        if 'metric_name' not in self.config:
            self.config['metric_name'] = "memory_usage"
        if 'track_gpu' not in self.config:
            self.config['track_gpu'] = TORCH_AVAILABLE
        if 'gpu_device' not in self.config:
            self.config['gpu_device'] = 0
        if 'unit' not in self.config:
            self.config['unit'] = "MB"  # megabytes
        if 'track_peak' not in self.config:
            self.config['track_peak'] = True
        if 'sampling_interval' not in self.config:
            self.config['sampling_interval'] = 1.0  # 1 second
        if 'include_timestamps' not in self.config:
            self.config['include_timestamps'] = True
            
        # Initialize the metrics dictionary with the configured metric name
        self.metrics[self.config['metric_name']] = []
        
        # Dictionary to store active tracking sessions
        self.active_tracks = {}
        
        # Get the current process
        self.process = psutil.Process(os.getpid())
    
    def start_tracking(self, track_id: Optional[str] = None, label: Optional[str] = None) -> str:
        """
        Start tracking memory usage.
        
        Args:
            track_id: Optional identifier for the tracking session.
                If not provided, a unique ID will be generated.
            label: Optional label to identify what is being tracked (e.g., "model_loading").
            
        Returns:
            A tracking ID that can be used to stop tracking and retrieve metrics.
        """
        # Generate a tracking ID if none was provided
        if track_id is None:
            track_id = str(uuid.uuid4())
            
        # Record the start time
        start_time = time.time()
        
        # Get the initial memory usage
        memory_usage = self._get_memory_usage()
        
        # Store the tracking session
        self.active_tracks[track_id] = {
            "start_time": start_time,
            "label": label,
            "initial_memory": memory_usage,
            "peak_memory": memory_usage,
            "samples": []
        }
        
        # Start the sampling thread if sampling is enabled
        if self.config['sampling_interval'] > 0:
            # Add the first sample
            self.active_tracks[track_id]["samples"].append({
                "timestamp": start_time,
                "memory": memory_usage
            })
        
        return track_id
    
    def record_sample(self, track_id: str) -> Dict[str, Any]:
        """
        Record a memory usage sample for a tracking session.
        
        Args:
            track_id: The tracking ID returned by start_tracking.
            
        Returns:
            A dictionary containing the memory usage sample.
            
        Raises:
            ValueError: If the track_id is not found in active tracks.
        """
        if track_id not in self.active_tracks:
            raise ValueError(f"Tracking session with ID {track_id} not found.")
            
        # Get the tracking session
        track = self.active_tracks[track_id]
        
        # Get the current memory usage
        memory_usage = self._get_memory_usage()
        
        # Record the sample
        sample = {
            "timestamp": time.time(),
            "memory": memory_usage
        }
        
        track["samples"].append(sample)
        
        # Update peak memory if tracking peaks
        if self.config['track_peak']:
            for key in memory_usage:
                if memory_usage[key] > track["peak_memory"].get(key, 0):
                    track["peak_memory"][key] = memory_usage[key]
        
        return sample
    
    def stop_tracking(self, track_id: str) -> Dict[str, Any]:
        """
        Stop tracking memory usage and return the results.
        
        Args:
            track_id: The tracking ID returned by start_tracking.
            
        Returns:
            A dictionary containing the memory usage metrics.
            
        Raises:
            ValueError: If the track_id is not found in active tracks.
        """
        if track_id not in self.active_tracks:
            raise ValueError(f"Tracking session with ID {track_id} not found.")
            
        # Get the tracking session
        track = self.active_tracks[track_id]
        start_time = track["start_time"]
        label = track["label"]
        initial_memory = track["initial_memory"]
        
        # Record the end time
        end_time = time.time()
        
        # Get the final memory usage
        final_memory = self._get_memory_usage()
        
        # Calculate the memory usage change
        memory_change = {}
        for key in final_memory:
            if key in initial_memory:
                memory_change[key] = final_memory[key] - initial_memory[key]
        
        # Get the peak memory usage
        peak_memory = track["peak_memory"] if self.config['track_peak'] else final_memory
        
        # Prepare the metric data
        metric_data = {
            "track_id": track_id,
            "value": final_memory,
            "initial": initial_memory,
            "change": memory_change,
            "peak": peak_memory,
            "timestamp": datetime.now().isoformat(),
            "unit": self.config['unit']
        }
        
        if label:
            metric_data["label"] = label
            
        if self.config['include_timestamps']:
            metric_data["start_time"] = datetime.fromtimestamp(start_time).isoformat()
            metric_data["end_time"] = datetime.fromtimestamp(end_time).isoformat()
            
        # Add samples if available
        if track["samples"]:
            metric_data["samples"] = [
                {
                    "timestamp": datetime.fromtimestamp(s["timestamp"]).isoformat(),
                    "memory": s["memory"]
                }
                for s in track["samples"]
            ]
            
        # Add the metric to the metrics list
        metric_name = self.config['metric_name']
        self.metrics[metric_name].append(metric_data)
        
        # Remove the tracking session from active tracks
        del self.active_tracks[track_id]
        
        return metric_data
    
    def get_current_value(self, track_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get the current memory usage.
        
        Args:
            track_id: Optional tracking ID. If provided, returns the memory change
                since the start of that tracking session. If not provided, returns
                the current absolute memory usage.
            
        Returns:
            A dictionary containing memory usage values for RAM and GPU (if tracked).
        """
        # Get the current memory usage
        current_usage = self._get_memory_usage()
        
        if track_id is not None:
            # Calculate the change since the start of tracking
            if track_id not in self.active_tracks:
                raise ValueError(f"Tracking session with ID {track_id} not found.")
                
            # Get the initial memory usage
            initial_usage = self.active_tracks[track_id]["initial_memory"]
            
            # Calculate the change
            memory_change = {}
            for key in current_usage:
                if key in initial_usage:
                    memory_change[key] = current_usage[key] - initial_usage[key]
                    
            return memory_change
        else:
            # Return the absolute memory usage
            return current_usage
    
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
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """
        Get the current memory usage of the process and system.
        
        Returns:
            A dictionary containing memory usage values in the configured unit.
        """
        result = {}
        
        # Get RAM usage
        process_mem_info = self.process.memory_info()
        system_mem_info = psutil.virtual_memory()
        
        # Convert to the configured unit
        if self.config['unit'] == "KB":
            # Memory values in kilobytes
            conversion = 1024
        elif self.config['unit'] == "MB":
            # Memory values in megabytes
            conversion = 1024 * 1024
        elif self.config['unit'] == "GB":
            # Memory values in gigabytes
            conversion = 1024 * 1024 * 1024
        else:
            # Memory values in bytes (default)
            conversion = 1
            
        # Process memory
        result["process_rss"] = process_mem_info.rss / conversion  # Resident Set Size
        result["process_vms"] = process_mem_info.vms / conversion  # Virtual Memory Size
        
        # System memory
        result["system_total"] = system_mem_info.total / conversion
        result["system_available"] = system_mem_info.available / conversion
        result["system_used"] = system_mem_info.used / conversion
        result["system_free"] = system_mem_info.free / conversion
        result["system_percent"] = system_mem_info.percent  # Percentage (not converted)
        
        # GPU memory if tracked and available
        if self.config['track_gpu'] and TORCH_AVAILABLE:
            gpu_device = self.config['gpu_device']
            
            try:
                if torch.cuda.is_available() and gpu_device < torch.cuda.device_count():
                    # Get GPU memory info
                    gpu_mem_info = torch.cuda.mem_get_info(gpu_device)
                    result["gpu_free"] = gpu_mem_info[0] / conversion
                    result["gpu_total"] = gpu_mem_info[1] / conversion
                    result["gpu_used"] = (gpu_mem_info[1] - gpu_mem_info[0]) / conversion
                    result["gpu_percent"] = (result["gpu_used"] / result["gpu_total"]) * 100
                    
                    # Get allocated GPU memory for this process
                    result["gpu_allocated"] = torch.cuda.memory_allocated(gpu_device) / conversion
                    result["gpu_reserved"] = torch.cuda.memory_reserved(gpu_device) / conversion
                    
                    # Get max allocated memory
                    result["gpu_max_allocated"] = torch.cuda.max_memory_allocated(gpu_device) / conversion
                    result["gpu_max_reserved"] = torch.cuda.max_memory_reserved(gpu_device) / conversion
            except (RuntimeError, AttributeError):
                # Skip GPU memory tracking if there's an error
                pass
                
        return result
    
    def get_memory_statistics(self, label: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for the tracked memory usage, optionally filtered by label.
        
        Args:
            label: Optional label to filter the metrics by.
            
        Returns:
            A dictionary containing statistics for each memory metric.
        """
        metric_name = self.config['metric_name']
        
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
            
        # Filter metrics by label if provided
        if label:
            metrics = [m for m in self.metrics[metric_name] if m.get("label") == label]
        else:
            metrics = self.metrics[metric_name]
            
        if not metrics:
            return {}
            
        # Get all memory keys in the metrics
        all_keys = set()
        for metric in metrics:
            if "value" in metric and isinstance(metric["value"], dict):
                all_keys.update(metric["value"].keys())
                
        # Calculate statistics for each memory key
        result = {}
        for key in all_keys:
            values = [m["value"].get(key, 0) for m in metrics if "value" in m and isinstance(m["value"], dict)]
            if values:
                result[key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "last": values[-1],
                    "unit": self.config['unit']
                }
                
                # Add peak values if available
                peak_values = [m["peak"].get(key, 0) for m in metrics if "peak" in m and isinstance(m["peak"], dict)]
                if peak_values:
                    result[key]["peak_max"] = max(peak_values)
                    result[key]["peak_mean"] = sum(peak_values) / len(peak_values)
                    
                # Add change values if available
                change_values = [m["change"].get(key, 0) for m in metrics if "change" in m and isinstance(m["change"], dict)]
                if change_values:
                    result[key]["change_min"] = min(change_values)
                    result[key]["change_max"] = max(change_values)
                    result[key]["change_mean"] = sum(change_values) / len(change_values)
                    
        return result
    
    def track_function_memory(self, func: Callable, *args, 
                             label: Optional[str] = None,
                             sampling_interval: Optional[float] = None,
                             **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Track the memory usage of a function call.
        
        Args:
            func: The function to track.
            *args: Arguments to pass to the function.
            label: Optional label to identify what is being tracked.
            sampling_interval: Optional sampling interval in seconds.
                If provided, overrides the configured sampling_interval.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            A tuple containing the result of the function call and the memory usage metrics.
        """
        # Save the original sampling interval
        original_interval = self.config['sampling_interval']
        
        # Set the sampling interval if provided
        if sampling_interval is not None:
            self.config['sampling_interval'] = sampling_interval
            
        track_id = self.start_tracking(label=label)
        
        try:
            # Take a sample before the function call
            self.record_sample(track_id)
            
            # Call the function
            result = func(*args, **kwargs)
            
            # Take a sample after the function call
            self.record_sample(track_id)
            
            # Get the memory usage metrics
            metrics = self.stop_tracking(track_id)
            
            return result, metrics
        finally:
            # Restore the original sampling interval
            self.config['sampling_interval'] = original_interval 