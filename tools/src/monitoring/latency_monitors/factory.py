#!/usr/bin/env python3
"""
Factory function for latency monitors in RAG systems.

This module provides a factory function for creating latency monitor instances
based on the specified type and configuration.
"""

from typing import Dict, Any, Optional
from .base import BaseLatencyMonitor
from .query_monitor import QueryLatencyMonitor
from .generation_monitor import GenerationLatencyMonitor
from .end_to_end_monitor import EndToEndLatencyMonitor
from .component_monitor import ComponentLatencyMonitor


def get_latency_monitor(
    monitor_type: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseLatencyMonitor:
    """
    Create a latency monitor instance based on the specified type and configuration.
    
    Args:
        monitor_type (str): Type of latency monitor to create.
            Valid values: 'query', 'generation', 'end_to_end', 'component'.
        name (Optional[str]): Name of the monitor.
            If not provided, a default name will be used based on the monitor type.
        config (Optional[Dict[str, Any]]): Configuration options for the monitor.
            Defaults to an empty dictionary.
        **kwargs: Additional keyword arguments specific to each monitor type:
            - For 'component': component (str) - Name of the component being monitored.
            
    Returns:
        BaseLatencyMonitor: Latency monitor instance.
        
    Raises:
        ValueError: If an unsupported monitor type is specified or if required arguments are missing.
    """
    # Normalize monitor type
    monitor_type = monitor_type.lower().strip()
    
    # Set default name if not provided
    if name is None:
        name = f"{monitor_type}_latency_monitor"
    
    # Set default config if not provided
    if config is None:
        config = {}
    
    # Create monitor based on type
    if monitor_type == "query":
        return QueryLatencyMonitor(name=name, config=config)
    
    elif monitor_type == "generation":
        return GenerationLatencyMonitor(name=name, config=config)
    
    elif monitor_type == "end_to_end":
        return EndToEndLatencyMonitor(name=name, config=config)
    
    elif monitor_type == "component":
        # Get component name from kwargs
        component = kwargs.get("component")
        if not component:
            raise ValueError("Component name is required for component latency monitor.")
        
        return ComponentLatencyMonitor(name=name, component=component, config=config)
    
    else:
        valid_types = ["query", "generation", "end_to_end", "component"]
        raise ValueError(
            f"Unsupported monitor type: '{monitor_type}'. "
            f"Valid types are: {', '.join(valid_types)}."
        )


def create_monitor_from_config(config: Dict[str, Any]) -> BaseLatencyMonitor:
    """
    Create a latency monitor instance from a configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary with the following structure:
            {
                "type": str,  # Type of monitor (required)
                "name": str,  # Name of monitor (optional)
                "config": Dict[str, Any],  # Monitor configuration (optional)
                "component": str,  # Component name for component monitor (optional)
                "precision": int,  # Precision for latency measurements (optional)
                "unit": str  # Unit for latency measurements (optional)
            }
    
    Returns:
        BaseLatencyMonitor: Latency monitor instance.
    
    Raises:
        ValueError: If the configuration is invalid or incomplete.
    """
    # Verify required fields
    if "type" not in config:
        raise ValueError("Monitor type is required in configuration.")
    
    monitor_type = config["type"]
    name = config.get("name")
    monitor_config = config.get("config", {})
    
    # Extract optional parameters
    kwargs = {}
    
    if "component" in config:
        kwargs["component"] = config["component"]
    
    if "precision" in config:
        kwargs["precision"] = config["precision"]
    
    if "unit" in config:
        kwargs["unit"] = config["unit"]
    
    # Create and return the monitor
    return get_latency_monitor(
        monitor_type=monitor_type,
        name=name,
        config=monitor_config,
        **kwargs
    )


def create_default_monitors(prefix: str = "default") -> Dict[str, BaseLatencyMonitor]:
    """
    Create a set of default latency monitors for common use cases.
    
    Args:
        prefix (str): Prefix for monitor names. Defaults to "default".
    
    Returns:
        Dict[str, BaseLatencyMonitor]: Dictionary of latency monitors.
    """
    # Create default monitors
    monitors = {
        "query": QueryLatencyMonitor(name=f"{prefix}_query_latency_monitor"),
        "generation": GenerationLatencyMonitor(name=f"{prefix}_generation_latency_monitor"),
        "end_to_end": EndToEndLatencyMonitor(name=f"{prefix}_end_to_end_latency_monitor"),
        "embedding": ComponentLatencyMonitor(
            name=f"{prefix}_embedding_latency_monitor",
            component="embedding"
        ),
        "vector_db": ComponentLatencyMonitor(
            name=f"{prefix}_vector_db_latency_monitor",
            component="vector_db"
        ),
        "reranking": ComponentLatencyMonitor(
            name=f"{prefix}_reranking_latency_monitor",
            component="reranking"
        )
    }
    
    return monitors 