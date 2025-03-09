"""
End-to-End Metrics

This module provides metrics for evaluating the end-to-end performance of RAG systems.
"""

import time
import psutil
import numpy as np
from typing import List, Dict, Any, Union, Optional, Set, Tuple, Callable


class EndToEndMetrics:
    """
    A class for evaluating the end-to-end performance of RAG systems.
    """
    
    def __init__(self):
        """
        Initialize the end-to-end metrics calculator.
        """
        pass
    
    def task_completion(
        self,
        outputs: List[str],
        expected_outputs: List[str],
        task_evaluator: Optional[Callable[[str, str], float]] = None,
    ) -> float:
        """
        Evaluate task completion rate.
        
        Args:
            outputs: List of system outputs
            expected_outputs: List of expected outputs
            task_evaluator: Optional custom function to evaluate task completion
            
        Returns:
            Task completion score (0 to 1)
        """
        if not outputs or not expected_outputs:
            return 0.0
        
        # If lengths don't match, use the shorter length
        eval_length = min(len(outputs), len(expected_outputs))
        
        # If a custom evaluator is provided, use it
        if task_evaluator:
            scores = [
                task_evaluator(outputs[i], expected_outputs[i])
                for i in range(eval_length)
            ]
            return sum(scores) / len(scores)
        
        # Simple exact match as a default metric
        matches = sum(1 for i in range(eval_length) if outputs[i] == expected_outputs[i])
        return matches / eval_length
    
    def user_satisfaction(
        self,
        user_ratings: List[float],
    ) -> Dict[str, float]:
        """
        Calculate user satisfaction metrics.
        
        Args:
            user_ratings: List of user ratings (0 to 5)
            
        Returns:
            Dictionary of satisfaction metrics
        """
        if not user_ratings:
            return {
                "average": 0.0,
                "median": 0.0,
                "normalized_average": 0.0,
            }
        
        # Calculate basic statistics
        average = sum(user_ratings) / len(user_ratings)
        median = sorted(user_ratings)[len(user_ratings) // 2]
        
        # Normalize to 0-1 scale (assuming ratings are 0-5)
        normalized_average = average / 5.0
        
        # Calculate distribution
        distribution = {}
        for rating in range(6):  # 0 to 5
            distribution[rating] = sum(1 for r in user_ratings if r == rating) / len(user_ratings)
        
        return {
            "average": average,
            "median": median,
            "normalized_average": normalized_average,
            "distribution": distribution,
        }
    
    def measure_efficiency(
        self,
        query_func: Callable,
        queries: List[str],
        measure_memory: bool = True,
    ) -> Dict[str, Any]:
        """
        Measure efficiency metrics for a RAG system.
        
        Args:
            query_func: Function that executes a query and returns results
            queries: List of queries to evaluate
            measure_memory: Whether to measure memory usage
            
        Returns:
            Dictionary of efficiency metrics
        """
        if not queries:
            return {
                "average_latency": 0.0,
                "throughput": 0.0,
                "memory_usage": None,
            }
        
        # Measure latency and throughput
        latencies = []
        memory_usages = []
        
        # Get initial memory usage
        if measure_memory:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        
        for query in queries:
            # Measure individual query latency
            query_start = time.time()
            _ = query_func(query)
            query_end = time.time()
            
            latency = query_end - query_start
            latencies.append(latency)
            
            # Measure memory usage after each query
            if measure_memory:
                current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                memory_usages.append(current_memory - initial_memory)
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        average_latency = sum(latencies) / len(latencies)
        throughput = len(queries) / total_time if total_time > 0 else 0
        
        results = {
            "average_latency": average_latency,
            "throughput": throughput,
            "total_time": total_time,
            "latency_p50": np.percentile(latencies, 50) if latencies else 0,
            "latency_p90": np.percentile(latencies, 90) if latencies else 0,
            "latency_p99": np.percentile(latencies, 99) if latencies else 0,
        }
        
        # Add memory metrics if measured
        if measure_memory and memory_usages:
            results["memory_usage"] = {
                "average": sum(memory_usages) / len(memory_usages),
                "peak": max(memory_usages),
                "final": memory_usages[-1],
            }
        
        return results
    
    def robustness(
        self,
        query_func: Callable,
        standard_queries: List[str],
        adversarial_queries: List[str],
        evaluation_func: Callable,
    ) -> Dict[str, Any]:
        """
        Evaluate robustness by comparing performance on standard vs. adversarial queries.
        
        Args:
            query_func: Function that executes a query and returns results
            standard_queries: List of standard queries
            adversarial_queries: List of adversarial or challenging queries
            evaluation_func: Function to evaluate quality of responses
            
        Returns:
            Dictionary of robustness metrics
        """
        if not standard_queries or not adversarial_queries:
            return {
                "standard_score": 0.0,
                "adversarial_score": 0.0,
                "robustness_ratio": 0.0,
            }
        
        # Evaluate standard queries
        standard_results = [query_func(query) for query in standard_queries]
        standard_scores = [evaluation_func(result) for result in standard_results]
        standard_avg = sum(standard_scores) / len(standard_scores) if standard_scores else 0
        
        # Evaluate adversarial queries
        adversarial_results = [query_func(query) for query in adversarial_queries]
        adversarial_scores = [evaluation_func(result) for result in adversarial_results]
        adversarial_avg = sum(adversarial_scores) / len(adversarial_scores) if adversarial_scores else 0
        
        # Calculate robustness ratio
        robustness_ratio = adversarial_avg / standard_avg if standard_avg > 0 else 0
        
        return {
            "standard_score": standard_avg,
            "adversarial_score": adversarial_avg,
            "robustness_ratio": robustness_ratio,
            "standard_distribution": standard_scores,
            "adversarial_distribution": adversarial_scores,
        }
    
    def evaluate_all(
        self,
        outputs: List[str],
        expected_outputs: List[str],
        user_ratings: Optional[List[float]] = None,
        efficiency_data: Optional[Dict[str, Any]] = None,
        robustness_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end performance using multiple metrics.
        
        Args:
            outputs: List of system outputs
            expected_outputs: List of expected outputs
            user_ratings: Optional list of user ratings
            efficiency_data: Optional efficiency metrics
            robustness_data: Optional robustness metrics
            
        Returns:
            Dictionary of evaluation results
        """
        results = {}
        
        # Task completion
        results["task_completion"] = self.task_completion(outputs, expected_outputs)
        
        # User satisfaction if available
        if user_ratings:
            results["user_satisfaction"] = self.user_satisfaction(user_ratings)
        
        # Efficiency if available
        if efficiency_data:
            results["efficiency"] = efficiency_data
        
        # Robustness if available
        if robustness_data:
            results["robustness"] = robustness_data
        
        # Calculate overall end-to-end score
        overall_score = 0.0
        num_metrics = 0
        
        if "task_completion" in results:
            overall_score += results["task_completion"] * 0.4  # 40% weight
            num_metrics += 0.4
        
        if "user_satisfaction" in results and "normalized_average" in results["user_satisfaction"]:
            overall_score += results["user_satisfaction"]["normalized_average"] * 0.4  # 40% weight
            num_metrics += 0.4
        
        if "robustness" in results and "robustness_ratio" in results["robustness"]:
            overall_score += results["robustness"]["robustness_ratio"] * 0.2  # 20% weight
            num_metrics += 0.2
        
        if num_metrics > 0:
            results["overall_score"] = overall_score / num_metrics
        else:
            results["overall_score"] = 0.0
        
        return results 