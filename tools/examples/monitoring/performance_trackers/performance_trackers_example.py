#!/usr/bin/env python3
"""
Performance Trackers Example

This script demonstrates how to use the performance trackers to monitor
various aspects of a RAG system's performance, including latency, throughput,
memory usage, and CPU usage.
"""

import sys
import time
import os
import random
import json
import numpy as np
from typing import Dict, List, Any, Optional
from threading import Thread

sys.path.append("../../../..")  # Add the repository root to the path

from tools.src.monitoring.performance_trackers import (
    BasePerformanceTracker,
    LatencyTracker,
    ThroughputTracker,
    MemoryUsageTracker,
    CPUUsageTracker,
    get_performance_tracker
)

# Sample workloads for the different trackers
def dummy_retrieval(delay: float = 0.1, size: int = 5) -> List[Dict[str, Any]]:
    """
    Simulate a retrieval operation with a specified delay.
    
    Args:
        delay: Time delay in seconds.
        size: Number of results to retrieve.
        
    Returns:
        A list of dummy retrieval results.
    """
    time.sleep(delay)  # Simulate latency
    return [
        {
            "document_id": f"doc_{i}",
            "score": random.random(),
            "content": f"This is document {i} content..." * 10
        }
        for i in range(size)
    ]

def dummy_embedding_generation(text_list: List[str], dimension: int = 768) -> List[List[float]]:
    """
    Simulate embedding generation with a varying delay based on input size.
    
    Args:
        text_list: List of texts to embed.
        dimension: Embedding dimension.
        
    Returns:
        A list of embeddings.
    """
    # Simulate embedding computation time that scales with text length
    delay = sum(len(text) for text in text_list) * 0.0001
    time.sleep(delay)
    
    # Generate random embeddings
    embeddings = []
    for _ in text_list:
        # Ensure the embeddings are normalized
        embedding = np.random.randn(dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())
        
    return embeddings

def dummy_llm_generation(prompt: str, max_tokens: int = 100) -> str:
    """
    Simulate LLM generation with a varying delay based on output size.
    
    Args:
        prompt: Input prompt.
        max_tokens: Maximum number of tokens to generate.
        
    Returns:
        Generated text.
    """
    # Simulate generation time that scales with output size
    delay = max_tokens * 0.01
    time.sleep(delay)
    
    # Generate dummy text
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", 
             "adipiscing", "elit", "sed", "do", "eiusmod", "tempor", 
             "incididunt", "ut", "labore", "et", "dolore", "magna", "aliqua"]
    
    output_words = []
    for _ in range(random.randint(max_tokens // 5, max_tokens // 3)):
        output_words.append(random.choice(words))
        
    return " ".join(output_words)

def cpu_intensive_task(iterations: int = 1000000) -> float:
    """
    Perform a CPU-intensive task.
    
    Args:
        iterations: Number of iterations for the computation.
        
    Returns:
        Result of the computation.
    """
    result = 0
    for i in range(iterations):
        result += (i * i) % 1000
        if i % 10000 == 0:
            result = result % 1000000  # Prevent overflow
            
    return result

def memory_intensive_task(size_mb: int = 100, hold_time: float = 2.0) -> None:
    """
    Perform a memory-intensive task.
    
    Args:
        size_mb: Size in MB to allocate.
        hold_time: Time to hold the memory allocation in seconds.
    """
    # Allocate memory (1MB = 1024*1024 bytes)
    bytes_per_mb = 1024 * 1024
    data = bytearray(size_mb * bytes_per_mb)
    
    # Fill with random data to ensure it's actually allocated
    for i in range(0, len(data), 1024):
        data[i] = random.randint(0, 255)
        
    # Hold the memory for the specified time
    time.sleep(hold_time)
    
    # Let the memory be freed when the function returns

def run_concurrent_tasks(num_tasks: int, task_func, *args, **kwargs) -> None:
    """
    Run multiple tasks concurrently.
    
    Args:
        num_tasks: Number of concurrent tasks to run.
        task_func: Function to run for each task.
        *args: Arguments to pass to the task function.
        **kwargs: Keyword arguments to pass to the task function.
    """
    threads = []
    for _ in range(num_tasks):
        thread = Thread(target=task_func, args=args, kwargs=kwargs)
        thread.start()
        threads.append(thread)
        
    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Examples for each tracker
def run_latency_tracker_example() -> None:
    """
    Demonstrate the use of the LatencyTracker.
    """
    print("\n=== Latency Tracker Example ===")
    
    # Create a latency tracker
    latency_tracker = LatencyTracker({
        'time_unit': 'ms'  # Track in milliseconds
    })
    
    print("Tracking latency for retrieval operations...")
    
    # Track a single operation
    track_id = latency_tracker.start_tracking(label="retrieval")
    dummy_retrieval(delay=0.2, size=10)
    result = latency_tracker.stop_tracking(track_id)
    
    print(f"Single retrieval latency: {result['value']:.2f} {result['unit']}")
    
    # Track latency using the context manager
    print("\nTracking latency using context manager...")
    with latency_tracker as track_id:
        dummy_retrieval(delay=0.15, size=5)
    
    # Track several operations with different parameters
    delays = [0.05, 0.1, 0.15, 0.2, 0.25]
    for i, delay in enumerate(delays):
        track_id = latency_tracker.start_tracking(label=f"retrieval_{i}")
        dummy_retrieval(delay=delay, size=5)
        latency_tracker.stop_tracking(track_id)
    
    # Get statistics for a specific label
    stats = latency_tracker.get_latency_statistics(label="retrieval")
    print("\nLatency statistics for 'retrieval' label:")
    print(f"Min: {stats.get('min', 0):.2f} ms")
    print(f"Max: {stats.get('max', 0):.2f} ms")
    print(f"Mean: {stats.get('mean', 0):.2f} ms")
    print(f"Median: {stats.get('median', 0):.2f} ms")
    
    # Track latency of a function using a decorator
    @latency_tracker
    def retrieval_function(delay, size):
        return dummy_retrieval(delay, size)
    
    print("\nTracking latency using decorator...")
    result = retrieval_function(0.1, 5)
    print(f"Retrieved {len(result)} documents.")
    
    # Save metrics to a file
    temp_file = "latency_metrics.json"
    latency_tracker.save_metrics(temp_file)
    print(f"\nSaved metrics to {temp_file}")
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)

def run_throughput_tracker_example() -> None:
    """
    Demonstrate the use of the ThroughputTracker.
    """
    print("\n=== Throughput Tracker Example ===")
    
    # Create a throughput tracker
    throughput_tracker = ThroughputTracker({
        'time_unit': 's',  # Events per second
        'window_size': 10  # 10-second sliding window
    })
    
    print("Tracking throughput for embedding generation...")
    
    # Start tracking
    track_id = throughput_tracker.start_tracking(label="embedding")
    
    # Process several batches
    batch_sizes = [10, 20, 30, 40, 50]
    for batch_size in batch_sizes:
        # Generate random texts
        texts = [f"This is text {i}" for i in range(batch_size)]
        
        # Generate embeddings
        embeddings = dummy_embedding_generation(texts)
        
        # Record the event with the batch size
        throughput_tracker.record_event(track_id, batch_size)
        
        # Short pause between batches
        time.sleep(0.5)
    
    # Stop tracking
    result = throughput_tracker.stop_tracking(track_id)
    
    print(f"Average throughput: {result['value']:.2f} {result['unit']}")
    print(f"Total processed: {result['count']} items")
    print(f"Duration: {result['duration']:.2f} seconds")
    
    # Tracking throughput with batches
    print("\nTracking throughput with explicit batches...")
    
    # Start a batch
    batch_id = throughput_tracker.start_batch(label="batch_processing")
    
    # Process several batches with varying sizes
    for _ in range(5):
        batch_size = random.randint(10, 100)
        
        # Simulate batch processing
        time.sleep(random.uniform(0.1, 0.5))
        
        # Record the batch
        throughput_tracker.record_batch(batch_id, batch_size)
    
    # Stop the batch
    batch_result = throughput_tracker.stop_batch(batch_id)
    
    print(f"Batch processing throughput: {batch_result['value']:.2f} {batch_result['unit']}")
    print(f"Total processed in batch: {batch_result['count']} items")
    
    # Track throughput of a function
    print("\nTracking function throughput...")
    
    def count_embeddings(result):
        """Count the number of embeddings in the result."""
        return len(result)
    
    # Track with a function that calculates the count
    result, metrics = throughput_tracker.track_function_throughput(
        dummy_embedding_generation,
        [f"Text {i}" for i in range(25)],
        label="function_throughput",
        count_func=count_embeddings
    )
    
    print(f"Function throughput: {metrics['value']:.2f} {metrics['unit']}")
    print(f"Generated {len(result)} embeddings.")

def run_memory_usage_tracker_example() -> None:
    """
    Demonstrate the use of the MemoryUsageTracker.
    """
    print("\n=== Memory Usage Tracker Example ===")
    
    # Create a memory usage tracker
    memory_tracker = MemoryUsageTracker({
        'unit': 'MB',  # Track in megabytes
        'track_peak': True
    })
    
    print("Tracking memory usage for allocation tasks...")
    
    # Get initial memory usage
    initial_memory = memory_tracker.get_current_value()
    print(f"Initial process memory: {initial_memory.get('process_rss', 0):.2f} MB")
    
    # Track a memory-intensive task
    track_id = memory_tracker.start_tracking(label="memory_allocation")
    
    # Record a sample before allocation
    memory_tracker.record_sample(track_id)
    
    # Allocate memory
    memory_intensive_task(size_mb=50, hold_time=1.0)
    
    # Record a sample during allocation
    memory_tracker.record_sample(track_id)
    
    # Free the memory and let it be garbage collected
    time.sleep(0.5)
    
    # Record a sample after freeing
    memory_tracker.record_sample(track_id)
    
    # Stop tracking
    result = memory_tracker.stop_tracking(track_id)
    
    print(f"Final process memory: {result['value'].get('process_rss', 0):.2f} MB")
    print(f"Peak process memory: {result['peak'].get('process_rss', 0):.2f} MB")
    print(f"Memory change: {result['change'].get('process_rss', 0):.2f} MB")
    
    # Track multiple allocations
    print("\nTracking multiple memory allocations...")
    sizes = [20, 40, 60, 80, 100]
    
    for size in sizes:
        track_id = memory_tracker.start_tracking(label=f"allocation_{size}MB")
        memory_intensive_task(size_mb=size, hold_time=0.5)
        memory_tracker.stop_tracking(track_id)
    
    # Get memory statistics
    stats = memory_tracker.get_memory_statistics()
    if stats and 'process_rss' in stats:
        print("\nMemory statistics for process RSS:")
        print(f"Min: {stats['process_rss'].get('min', 0):.2f} MB")
        print(f"Max: {stats['process_rss'].get('max', 0):.2f} MB")
        print(f"Mean: {stats['process_rss'].get('mean', 0):.2f} MB")
        if 'peak_max' in stats['process_rss']:
            print(f"Peak max: {stats['process_rss']['peak_max']:.2f} MB")
    
    # Track memory usage of a function
    print("\nTracking function memory usage...")
    
    result, metrics = memory_tracker.track_function_memory(
        memory_intensive_task,
        size_mb=75,
        hold_time=1.0,
        label="function_memory",
        sampling_interval=0.1  # Sample every 0.1 seconds
    )
    
    print(f"Function memory peak: {metrics['peak'].get('process_rss', 0):.2f} MB")
    print(f"Samples collected: {len(metrics.get('samples', []))}")

def run_cpu_usage_tracker_example() -> None:
    """
    Demonstrate the use of the CPUUsageTracker.
    """
    print("\n=== CPU Usage Tracker Example ===")
    
    # Create a CPU usage tracker
    cpu_tracker = CPUUsageTracker({
        'sampling_interval': 0.2,  # Sample every 0.2 seconds
        'track_per_core': True,
        'track_process': True,
        'track_system': True
    })
    
    print("Tracking CPU usage for compute-intensive tasks...")
    
    # Get initial CPU usage
    initial_cpu = cpu_tracker.get_current_value()
    print(f"Initial process CPU: {initial_cpu.get('process', 0):.2f}%")
    print(f"Initial system CPU: {initial_cpu.get('system', 0):.2f}%")
    
    # Track a CPU-intensive task
    track_id = cpu_tracker.start_tracking(label="cpu_intensive")
    
    # Run a CPU-intensive computation
    cpu_intensive_task(iterations=5000000)
    
    # Stop tracking
    result = cpu_tracker.stop_tracking(track_id)
    
    print(f"Average process CPU: {result['value'].get('process', 0):.2f}%")
    print(f"Average system CPU: {result['value'].get('system', 0):.2f}%")
    print(f"Peak process CPU: {result['peak'].get('process', 0):.2f}%")
    
    # Track multiple concurrent tasks
    print("\nTracking multiple concurrent CPU tasks...")
    
    track_id = cpu_tracker.start_tracking(label="concurrent_tasks")
    
    # Run several CPU-intensive tasks concurrently
    run_concurrent_tasks(4, cpu_intensive_task, iterations=2000000)
    
    # Stop tracking
    concurrent_result = cpu_tracker.stop_tracking(track_id)
    
    print(f"Concurrent tasks avg process CPU: {concurrent_result['value'].get('process', 0):.2f}%")
    print(f"Concurrent tasks avg system CPU: {concurrent_result['value'].get('system', 0):.2f}%")
    
    # Track CPU usage by core
    if 'cores' in concurrent_result['value']:
        print("\nPer-core CPU usage:")
        cores = concurrent_result['value']['cores']
        for core, usage in cores.items():
            print(f"{core}: {usage:.2f}%")
    
    # Track CPU usage of a function
    print("\nTracking function CPU usage...")
    
    result, metrics = cpu_tracker.track_function_cpu(
        cpu_intensive_task,
        iterations=3000000,
        label="function_cpu",
        sampling_interval=0.1  # Sample every 0.1 seconds
    )
    
    print(f"Function CPU usage: {metrics['value'].get('process', 0):.2f}%")
    print(f"Samples collected: {len(metrics.get('samples', []))}")

def run_factory_function_example() -> None:
    """
    Demonstrate the use of the factory function to create trackers.
    """
    print("\n=== Factory Function Example ===")
    
    # Create different types of trackers using the factory function
    trackers = {
        "latency": get_performance_tracker("latency", {"time_unit": "ms"}),
        "throughput": get_performance_tracker("throughput", {"time_unit": "s"}),
        "memory": get_performance_tracker("memory", {"unit": "MB"}),
        "cpu": get_performance_tracker("cpu", {"sampling_interval": 0.5})
    }
    
    print("Created trackers using factory function:")
    for name, tracker in trackers.items():
        print(f"- {name}: {tracker.__class__.__name__}")
    
    # Use each tracker for a simple task
    print("\nRunning tasks with each tracker:")
    
    # Latency tracker
    with trackers["latency"] as track_id:
        dummy_retrieval(delay=0.1)
    latency_result = trackers["latency"].metrics["latency"][-1]
    print(f"Latency: {latency_result['value']:.2f} ms")
    
    # Throughput tracker
    throughput_id = trackers["throughput"].start_tracking()
    for _ in range(3):
        trackers["throughput"].record_event(throughput_id, 10)
        time.sleep(0.2)
    throughput_result = trackers["throughput"].stop_tracking(throughput_id)
    print(f"Throughput: {throughput_result['value']:.2f} events/s")
    
    # Memory tracker
    memory_id = trackers["memory"].start_tracking()
    memory_intensive_task(size_mb=25, hold_time=0.5)
    memory_result = trackers["memory"].stop_tracking(memory_id)
    print(f"Memory change: {memory_result['change'].get('process_rss', 0):.2f} MB")
    
    # CPU tracker
    cpu_id = trackers["cpu"].start_tracking()
    cpu_intensive_task(iterations=1000000)
    cpu_result = trackers["cpu"].stop_tracking(cpu_id)
    print(f"CPU usage: {cpu_result['value'].get('process', 0):.2f}%")
    
    # Error handling with invalid type
    print("\nError handling with invalid type:")
    try:
        invalid_tracker = get_performance_tracker("invalid_type")
        print("This should not be printed")
    except ValueError as e:
        print(f"Error: {e}")

def main() -> None:
    """
    Run all performance tracker examples.
    """
    print("Performance Trackers Example")
    print("This script demonstrates how to use the performance trackers to monitor system performance.")
    
    try:
        # Run examples for each tracker
        run_latency_tracker_example()
        run_throughput_tracker_example()
        run_memory_usage_tracker_example()
        run_cpu_usage_tracker_example()
        run_factory_function_example()
        
        print("\nExamples completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 