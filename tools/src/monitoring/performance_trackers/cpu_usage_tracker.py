"""
CPU Usage Tracker

This module provides the CPUUsageTracker class for tracking CPU usage
in a RAG system, including process CPU usage, system CPU usage, and per-core usage.
"""

import time
import uuid
import os
import psutil
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime
from threading import Thread, Event

from .base_performance_tracker import BasePerformanceTracker


class CPUUsageTracker(BasePerformanceTracker):
    """
    Tracker for measuring CPU usage.
    
    This tracker measures the CPU usage of a RAG system, including process CPU usage,
    system CPU usage, and per-core usage. It can track CPU usage over time with sampling
    and provides statistics on CPU usage.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the tracker.
        metrics (Dict[str, List[Dict[str, Any]]]): Dictionary of tracked CPU metrics.
        start_time (datetime): Time when the tracker was initialized.
        active_tracks (Dict[str, Dict[str, Any]]): Dictionary of currently active tracking sessions.
        process (psutil.Process): Current process being monitored.
        sampling_threads (Dict[str, Thread]): Dictionary of sampling threads per tracking session.
        stop_events (Dict[str, Event]): Dictionary of stop events for sampling threads.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CPU usage tracker.
        
        Args:
            config: Optional configuration dictionary for the tracker.
                - metric_name: Name to use for the CPU metric (default: "cpu_usage").
                - track_per_core: Whether to track per-core CPU usage (default: True).
                - track_process: Whether to track process CPU usage (default: True).
                - track_system: Whether to track system CPU usage (default: True).
                - sampling_interval: Interval in seconds for CPU usage sampling (default: 0.5).
                - include_timestamps: Whether to include timestamps in the metrics (default: True).
        """
        super().__init__(config)
        
        # Set default configuration values
        if 'metric_name' not in self.config:
            self.config['metric_name'] = "cpu_usage"
        if 'track_per_core' not in self.config:
            self.config['track_per_core'] = True
        if 'track_process' not in self.config:
            self.config['track_process'] = True
        if 'track_system' not in self.config:
            self.config['track_system'] = True
        if 'sampling_interval' not in self.config:
            self.config['sampling_interval'] = 0.5  # 0.5 second
        if 'include_timestamps' not in self.config:
            self.config['include_timestamps'] = True
            
        # Initialize the metrics dictionary with the configured metric name
        self.metrics[self.config['metric_name']] = []
        
        # Dictionary to store active tracking sessions
        self.active_tracks = {}
        
        # Get the current process
        self.process = psutil.Process(os.getpid())
        
        # Initialize sampling threads and stop events
        self.sampling_threads = {}
        self.stop_events = {}
    
    def start_tracking(self, track_id: Optional[str] = None, label: Optional[str] = None) -> str:
        """
        Start tracking CPU usage.
        
        Args:
            track_id: Optional identifier for the tracking session.
                If not provided, a unique ID will be generated.
            label: Optional label to identify what is being tracked (e.g., "model_inference").
            
        Returns:
            A tracking ID that can be used to stop tracking and retrieve metrics.
        """
        # Generate a tracking ID if none was provided
        if track_id is None:
            track_id = str(uuid.uuid4())
            
        # Record the start time
        start_time = time.time()
        
        # Get the initial CPU usage
        cpu_usage = self._get_cpu_usage()
        
        # Store the tracking session
        self.active_tracks[track_id] = {
            "start_time": start_time,
            "label": label,
            "samples": [{
                "timestamp": start_time,
                "cpu": cpu_usage
            }],
            "peak": {
                "process": cpu_usage.get("process", 0),
                "system": cpu_usage.get("system", 0),
                "cores": cpu_usage.get("cores", {})
            }
        }
        
        # Start the sampling thread if sampling is enabled
        if self.config['sampling_interval'] > 0:
            self.stop_events[track_id] = Event()
            self.sampling_threads[track_id] = Thread(
                target=self._sampling_thread,
                args=(track_id, self.stop_events[track_id], self.config['sampling_interval']),
                daemon=True
            )
            self.sampling_threads[track_id].start()
        
        return track_id
    
    def _sampling_thread(self, track_id: str, stop_event: Event, interval: float) -> None:
        """
        Thread function for sampling CPU usage at regular intervals.
        
        Args:
            track_id: The tracking ID for the session being sampled.
            stop_event: Event to signal when to stop sampling.
            interval: Sampling interval in seconds.
        """
        while not stop_event.is_set() and track_id in self.active_tracks:
            try:
                self.record_sample(track_id)
            except ValueError:
                # Tracking session may have been removed
                break
                
            # Wait for the next sample or until the stop event is set
            stop_event.wait(interval)
    
    def record_sample(self, track_id: str) -> Dict[str, Any]:
        """
        Record a CPU usage sample for a tracking session.
        
        Args:
            track_id: The tracking ID returned by start_tracking.
            
        Returns:
            A dictionary containing the CPU usage sample.
            
        Raises:
            ValueError: If the track_id is not found in active tracks.
        """
        if track_id not in self.active_tracks:
            raise ValueError(f"Tracking session with ID {track_id} not found.")
            
        # Get the tracking session
        track = self.active_tracks[track_id]
        
        # Get the current CPU usage
        cpu_usage = self._get_cpu_usage()
        
        # Record the sample
        sample = {
            "timestamp": time.time(),
            "cpu": cpu_usage
        }
        
        track["samples"].append(sample)
        
        # Update peak values
        if cpu_usage.get("process", 0) > track["peak"]["process"]:
            track["peak"]["process"] = cpu_usage.get("process", 0)
            
        if cpu_usage.get("system", 0) > track["peak"]["system"]:
            track["peak"]["system"] = cpu_usage.get("system", 0)
            
        if "cores" in cpu_usage:
            for core, usage in cpu_usage["cores"].items():
                if core not in track["peak"]["cores"] or usage > track["peak"]["cores"][core]:
                    track["peak"]["cores"][core] = usage
        
        return sample
    
    def stop_tracking(self, track_id: str) -> Dict[str, Any]:
        """
        Stop tracking CPU usage and return the results.
        
        Args:
            track_id: The tracking ID returned by start_tracking.
            
        Returns:
            A dictionary containing the CPU usage metrics.
            
        Raises:
            ValueError: If the track_id is not found in active tracks.
        """
        if track_id not in self.active_tracks:
            raise ValueError(f"Tracking session with ID {track_id} not found.")
            
        # Stop the sampling thread if it exists
        if track_id in self.stop_events:
            self.stop_events[track_id].set()
            if track_id in self.sampling_threads and self.sampling_threads[track_id].is_alive():
                self.sampling_threads[track_id].join(timeout=1.0)
                
            # Clean up
            del self.stop_events[track_id]
            del self.sampling_threads[track_id]
            
        # Get the tracking session
        track = self.active_tracks[track_id]
        start_time = track["start_time"]
        label = track["label"]
        samples = track["samples"]
        peak = track["peak"]
        
        # Record the end time
        end_time = time.time()
        
        # Add a final sample if needed
        if samples[-1]["timestamp"] < end_time - self.config['sampling_interval']:
            self.record_sample(track_id)
            samples = track["samples"]  # Update samples after adding a new one
            
        # Calculate average CPU usage
        if len(samples) > 1:
            # Process CPU
            if self.config['track_process']:
                process_values = [s["cpu"].get("process", 0) for s in samples if "cpu" in s]
                avg_process = sum(process_values) / len(process_values) if process_values else 0
            else:
                avg_process = 0
                
            # System CPU
            if self.config['track_system']:
                system_values = [s["cpu"].get("system", 0) for s in samples if "cpu" in s]
                avg_system = sum(system_values) / len(system_values) if system_values else 0
            else:
                avg_system = 0
                
            # Per-core CPU
            if self.config['track_per_core']:
                avg_cores = {}
                # Get all cores that have been tracked
                all_cores = set()
                for sample in samples:
                    if "cpu" in sample and "cores" in sample["cpu"]:
                        all_cores.update(sample["cpu"]["cores"].keys())
                        
                # Calculate average for each core
                for core in all_cores:
                    core_values = [s["cpu"]["cores"].get(core, 0) for s in samples 
                                  if "cpu" in s and "cores" in s["cpu"]]
                    avg_cores[core] = sum(core_values) / len(core_values) if core_values else 0
            else:
                avg_cores = {}
        else:
            # Only one sample, use its values
            cpu = samples[0]["cpu"] if samples else {}
            avg_process = cpu.get("process", 0)
            avg_system = cpu.get("system", 0)
            avg_cores = cpu.get("cores", {})
            
        # Prepare the average CPU usage
        avg_cpu = {
            "process": avg_process,
            "system": avg_system,
            "cores": avg_cores
        }
        
        # Prepare the metric data
        metric_data = {
            "track_id": track_id,
            "value": avg_cpu,
            "peak": peak,
            "timestamp": datetime.now().isoformat(),
            "duration": end_time - start_time
        }
        
        if label:
            metric_data["label"] = label
            
        if self.config['include_timestamps']:
            metric_data["start_time"] = datetime.fromtimestamp(start_time).isoformat()
            metric_data["end_time"] = datetime.fromtimestamp(end_time).isoformat()
            
        # Add samples if there are more than one
        if len(samples) > 1:
            metric_data["samples"] = [
                {
                    "timestamp": datetime.fromtimestamp(s["timestamp"]).isoformat(),
                    "cpu": s["cpu"]
                }
                for s in samples
            ]
            
        # Add the metric to the metrics list
        metric_name = self.config['metric_name']
        self.metrics[metric_name].append(metric_data)
        
        # Remove the tracking session from active tracks
        del self.active_tracks[track_id]
        
        return metric_data
    
    def get_current_value(self, track_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current CPU usage.
        
        Args:
            track_id: Optional tracking ID. If provided, returns the average CPU usage
                for that tracking session so far. If not provided, returns the
                current absolute CPU usage.
            
        Returns:
            A dictionary containing CPU usage values for process, system, and cores.
        """
        if track_id is not None:
            # Calculate the average CPU usage for the tracking session
            if track_id not in self.active_tracks:
                raise ValueError(f"Tracking session with ID {track_id} not found.")
                
            # Get the samples
            samples = self.active_tracks[track_id]["samples"]
            
            if not samples:
                return {}
                
            # Calculate average CPU usage
            if len(samples) > 1:
                # Process CPU
                if self.config['track_process']:
                    process_values = [s["cpu"].get("process", 0) for s in samples if "cpu" in s]
                    avg_process = sum(process_values) / len(process_values) if process_values else 0
                else:
                    avg_process = 0
                    
                # System CPU
                if self.config['track_system']:
                    system_values = [s["cpu"].get("system", 0) for s in samples if "cpu" in s]
                    avg_system = sum(system_values) / len(system_values) if system_values else 0
                else:
                    avg_system = 0
                    
                # Per-core CPU
                if self.config['track_per_core']:
                    avg_cores = {}
                    # Get all cores that have been tracked
                    all_cores = set()
                    for sample in samples:
                        if "cpu" in sample and "cores" in sample["cpu"]:
                            all_cores.update(sample["cpu"]["cores"].keys())
                            
                    # Calculate average for each core
                    for core in all_cores:
                        core_values = [s["cpu"]["cores"].get(core, 0) for s in samples 
                                      if "cpu" in s and "cores" in s["cpu"]]
                        avg_cores[core] = sum(core_values) / len(core_values) if core_values else 0
                else:
                    avg_cores = {}
                    
                return {
                    "process": avg_process,
                    "system": avg_system,
                    "cores": avg_cores
                }
            else:
                # Only one sample, use its values
                return samples[0]["cpu"] if samples else {}
        else:
            # Return the current CPU usage
            return self._get_cpu_usage()
    
    def reset(self) -> None:
        """
        Reset all tracking data.
        
        This method clears all tracked metrics and active tracking sessions.
        """
        # Stop all sampling threads
        for track_id in list(self.stop_events.keys()):
            self.stop_events[track_id].set()
            if track_id in self.sampling_threads and self.sampling_threads[track_id].is_alive():
                self.sampling_threads[track_id].join(timeout=1.0)
                
        # Clear sampling threads and stop events
        self.sampling_threads = {}
        self.stop_events = {}
        
        # Clear the metrics
        self.metrics = {self.config['metric_name']: []}
        
        # Clear active tracking sessions
        self.active_tracks = {}
        
        # Reset the start time
        self.start_time = datetime.now()
    
    def _get_cpu_usage(self) -> Dict[str, Any]:
        """
        Get the current CPU usage of the process and system.
        
        Returns:
            A dictionary containing CPU usage values as percentages.
        """
        result = {}
        
        # Get process CPU usage if enabled
        if self.config['track_process']:
            try:
                result["process"] = self.process.cpu_percent(interval=0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                result["process"] = 0
        
        # Get system CPU usage if enabled
        if self.config['track_system']:
            result["system"] = psutil.cpu_percent(interval=0)
        
        # Get per-core CPU usage if enabled
        if self.config['track_per_core']:
            per_cpu = psutil.cpu_percent(interval=0, percpu=True)
            result["cores"] = {f"core_{i}": per_cpu[i] for i in range(len(per_cpu))}
            
        return result
    
    def get_cpu_statistics(self, label: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for the tracked CPU usage, optionally filtered by label.
        
        Args:
            label: Optional label to filter the metrics by.
            
        Returns:
            A dictionary containing statistics for process, system, and per-core CPU usage.
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
            
        result = {}
        
        # Process CPU statistics
        process_values = [m["value"].get("process", 0) for m in metrics if "value" in m]
        if process_values:
            result["process"] = {
                "min": min(process_values),
                "max": max(process_values),
                "mean": sum(process_values) / len(process_values),
                "last": process_values[-1],
                "peak": max(m["peak"].get("process", 0) for m in metrics if "peak" in m)
            }
            
        # System CPU statistics
        system_values = [m["value"].get("system", 0) for m in metrics if "value" in m]
        if system_values:
            result["system"] = {
                "min": min(system_values),
                "max": max(system_values),
                "mean": sum(system_values) / len(system_values),
                "last": system_values[-1],
                "peak": max(m["peak"].get("system", 0) for m in metrics if "peak" in m)
            }
            
        # Per-core CPU statistics
        if self.config['track_per_core']:
            # Get all cores that have been tracked
            all_cores = set()
            for metric in metrics:
                if "value" in metric and "cores" in metric["value"]:
                    all_cores.update(metric["value"]["cores"].keys())
                    
            # Calculate statistics for each core
            cores_stats = {}
            for core in all_cores:
                core_values = [m["value"]["cores"].get(core, 0) for m in metrics 
                              if "value" in m and "cores" in m["value"]]
                if core_values:
                    peak_values = [m["peak"]["cores"].get(core, 0) for m in metrics 
                                  if "peak" in m and "cores" in m["peak"]]
                    peak = max(peak_values) if peak_values else 0
                    
                    cores_stats[core] = {
                        "min": min(core_values),
                        "max": max(core_values),
                        "mean": sum(core_values) / len(core_values),
                        "last": core_values[-1],
                        "peak": peak
                    }
                    
            if cores_stats:
                result["cores"] = cores_stats
                
        return result
    
    def track_function_cpu(self, func: Callable, *args, 
                         label: Optional[str] = None,
                         sampling_interval: Optional[float] = None,
                         **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Track the CPU usage of a function call.
        
        Args:
            func: The function to track.
            *args: Arguments to pass to the function.
            label: Optional label to identify what is being tracked.
            sampling_interval: Optional sampling interval in seconds.
                If provided, overrides the configured sampling_interval.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            A tuple containing the result of the function call and the CPU usage metrics.
        """
        # Save the original sampling interval
        original_interval = self.config['sampling_interval']
        
        # Set the sampling interval if provided
        if sampling_interval is not None:
            self.config['sampling_interval'] = sampling_interval
            
        track_id = self.start_tracking(label=label)
        
        try:
            # Call the function
            result = func(*args, **kwargs)
            
            # Get the CPU usage metrics
            metrics = self.stop_tracking(track_id)
            
            return result, metrics
        except Exception as e:
            # Stop tracking in case of an exception
            self.stop_tracking(track_id)
            raise e
        finally:
            # Restore the original sampling interval
            self.config['sampling_interval'] = original_interval 