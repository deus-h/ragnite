# Performance Trackers

Performance Trackers monitor various aspects of a RAG system's performance, including latency, throughput, memory usage, and CPU usage. They help identify bottlenecks, optimize performance, and ensure the system meets performance requirements.

## Overview

Performance monitoring is crucial for RAG systems, which can be computationally intensive and may need to handle high volumes of queries with low latency. Performance trackers provide both real-time monitoring and historical data collection, allowing you to:

- Measure and optimize the latency of critical operations
- Track throughput to ensure the system can handle the required load
- Monitor memory usage to detect leaks and optimize resource usage
- Measure CPU utilization to identify bottlenecks and balance workloads

## Available Performance Trackers

### LatencyTracker

Tracks the time it takes to complete operations in a RAG system.

**Features:**
- Measures operation latency with high precision
- Supports multiple time units (seconds, milliseconds, microseconds, nanoseconds)
- Provides statistical analysis of latency data
- Can be used as a decorator for function-level latency tracking
- Supports context manager interface for clean tracking in code blocks

**Use Cases:**
- Measure query processing time
- Track retrieval latency
- Monitor generation time for LLM responses
- Identify slow operations in the RAG pipeline

### ThroughputTracker

Measures how many operations can be completed in a given time period.

**Features:**
- Tracks events per second, minute, or hour
- Supports batch processing with cumulative counts
- Uses a sliding window for real-time throughput calculation
- Provides statistical analysis of throughput
- Tracks both instantaneous and average throughput

**Use Cases:**
- Measure queries per second
- Track document processing rate
- Monitor token generation speed
- Assess system capacity under load

### MemoryUsageTracker

Monitors memory consumption across various components of the system.

**Features:**
- Tracks process and system memory usage
- Optional GPU memory tracking for systems with CUDA
- Samples memory at configurable intervals
- Records peak memory usage
- Provides detailed memory metrics (RSS, VMS, etc.)

**Use Cases:**
- Detect memory leaks
- Monitor model loading memory footprint
- Track batch processing memory requirements
- Optimize memory usage for large-scale RAG systems

### CPUUsageTracker

Monitors CPU utilization for both the process and the system.

**Features:**
- Tracks process-specific and system-wide CPU usage
- Optional per-core CPU utilization monitoring
- Configurable sampling at regular intervals
- Detects CPU bottlenecks and peak usage
- Supports multi-threaded monitoring

**Use Cases:**
- Monitor CPU usage during batch processing
- Track CPU utilization for concurrent operations
- Identify CPU-bound components in the RAG pipeline
- Optimize thread and process allocation

## Usage

### Basic Usage

You can use performance trackers by creating an instance and calling the `start_tracking` and `stop_tracking` methods:

```python
from tools.src.monitoring.performance_trackers import LatencyTracker

# Create a tracker
latency_tracker = LatencyTracker({'time_unit': 'ms'})

# Start tracking
track_id = latency_tracker.start_tracking(label="retrieval")

# Perform the operation you want to track
results = retrieve_documents(query)

# Stop tracking and get the results
metrics = latency_tracker.stop_tracking(track_id)

print(f"Retrieval latency: {metrics['value']} ms")
```

### Context Manager

All trackers support the context manager interface for cleaner code:

```python
from tools.src.monitoring.performance_trackers import LatencyTracker

latency_tracker = LatencyTracker({'time_unit': 'ms'})

# Use the context manager
with latency_tracker as track_id:
    results = retrieve_documents(query)

# The tracking is automatically stopped when exiting the context
```

### Decorator Pattern

The `LatencyTracker` can be used as a decorator to track function execution time:

```python
from tools.src.monitoring.performance_trackers import LatencyTracker

latency_tracker = LatencyTracker({'time_unit': 'ms'})

@latency_tracker
def retrieve_documents(query):
    # Retrieval logic here
    return results

# The function call is automatically tracked
results = retrieve_documents(query)
```

### Batch Tracking

The `ThroughputTracker` provides specialized methods for batch processing:

```python
from tools.src.monitoring.performance_trackers import ThroughputTracker

throughput_tracker = ThroughputTracker({'time_unit': 's'})

# Start tracking a batch
batch_id = throughput_tracker.start_batch(label="document_processing")

# Process multiple batches
for batch in document_batches:
    process_batch(batch)
    throughput_tracker.record_batch(batch_id, len(batch))

# Get the final throughput
metrics = throughput_tracker.stop_batch(batch_id)

print(f"Processing throughput: {metrics['value']} items/s")
```

### Function Tracking

Each tracker provides specialized functions for tracking function execution:

```python
from tools.src.monitoring.performance_trackers import MemoryUsageTracker

memory_tracker = MemoryUsageTracker({'unit': 'MB'})

# Track memory usage of a function call
result, metrics = memory_tracker.track_function_memory(
    load_model,
    model_name="gpt-3.5-turbo",
    sampling_interval=0.1  # Sample every 0.1 seconds
)

print(f"Peak memory usage: {metrics['peak']['process_rss']} MB")
```

### Factory Function

A factory function is provided for creating trackers dynamically:

```python
from tools.src.monitoring.performance_trackers import get_performance_tracker

# Create trackers using the factory
latency_tracker = get_performance_tracker("latency", {"time_unit": "ms"})
throughput_tracker = get_performance_tracker("throughput", {"time_unit": "s"})
memory_tracker = get_performance_tracker("memory", {"unit": "MB"})
cpu_tracker = get_performance_tracker("cpu", {"sampling_interval": 0.5})
```

## Configuration Options

### LatencyTracker Options

```python
{
    'metric_name': 'latency',  # Name for the metric
    'time_unit': 'ms',         # Time unit: 's', 'ms', 'us', 'ns'
    'include_timestamps': True  # Whether to include timestamps in metrics
}
```

### ThroughputTracker Options

```python
{
    'metric_name': 'throughput',  # Name for the metric
    'time_unit': 's',           # Time unit: 's', 'm', 'h'
    'window_size': 60,          # Size of sliding window in seconds
    'include_timestamps': True   # Whether to include timestamps in metrics
}
```

### MemoryUsageTracker Options

```python
{
    'metric_name': 'memory_usage',  # Name for the metric
    'track_gpu': True,           # Whether to track GPU memory
    'gpu_device': 0,             # GPU device to track
    'unit': 'MB',                # Memory unit: 'B', 'KB', 'MB', 'GB'
    'track_peak': True,          # Whether to track peak memory
    'sampling_interval': 1.0,    # Interval for sampling in seconds
    'include_timestamps': True    # Whether to include timestamps in metrics
}
```

### CPUUsageTracker Options

```python
{
    'metric_name': 'cpu_usage',    # Name for the metric
    'track_per_core': True,      # Whether to track per-core usage
    'track_process': True,       # Whether to track process CPU
    'track_system': True,        # Whether to track system CPU
    'sampling_interval': 0.5,    # Interval for sampling in seconds
    'include_timestamps': True    # Whether to include timestamps in metrics
}
```

## Saving and Loading Metrics

All trackers support saving metrics to a file and loading them later:

```python
# Save metrics to a file
tracker.save_metrics("metrics.json")

# Load metrics from a file
data = tracker.load_metrics("metrics.json")
```

## Statistical Analysis

All trackers provide methods to calculate statistics for the tracked metrics:

```python
# Get statistics for a specific metric
stats = latency_tracker.get_latency_statistics(label="retrieval")
print(f"Min: {stats['min']} ms, Max: {stats['max']} ms, Mean: {stats['mean']} ms")

# Get statistics grouped by label
grouped_stats = latency_tracker.get_latency_by_label()
for label, stats in grouped_stats.items():
    print(f"{label}: Min: {stats['min']} ms, Max: {stats['max']} ms")
```

## Examples

For complete examples of how to use each performance tracker, see the [performance_trackers_example.py](../../../../../examples/monitoring/performance_trackers/performance_trackers_example.py) script. 