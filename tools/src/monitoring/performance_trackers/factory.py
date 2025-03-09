"""
Performance Tracker Factory

This module provides a factory function for creating performance tracker instances
based on type and configuration.
"""

from typing import Any, Dict, Optional, Union, Type
from .base_performance_tracker import BasePerformanceTracker
from .latency_tracker import LatencyTracker
from .throughput_tracker import ThroughputTracker
from .memory_usage_tracker import MemoryUsageTracker
from .cpu_usage_tracker import CPUUsageTracker

def get_performance_tracker(
    tracker_type: str,
    config: Optional[Dict[str, Any]] = None
) -> BasePerformanceTracker:
    """
    Factory function to create a performance tracker of the specified type.
    
    Args:
        tracker_type: Type of performance tracker to create.
            Valid values are 'latency', 'throughput', 'memory', and 'cpu'.
        config: Optional configuration dictionary for the tracker.
            
    Returns:
        A performance tracker instance of the requested type.
        
    Raises:
        ValueError: If the specified tracker type is not supported.
    """
    tracker_classes: Dict[str, Type[BasePerformanceTracker]] = {
        'latency': LatencyTracker,
        'throughput': ThroughputTracker,
        'memory': MemoryUsageTracker,
        'cpu': CPUUsageTracker
    }
    
    # Normalize type for case-insensitive matching
    normalized_type = tracker_type.lower()
    
    if normalized_type not in tracker_classes:
        valid_types = ', '.join(tracker_classes.keys())
        raise ValueError(
            f"Unsupported performance tracker type: '{tracker_type}'. "
            f"Valid types are: {valid_types}"
        )
    
    # Create and return the tracker instance
    return tracker_classes[normalized_type](config) 