#!/usr/bin/env python3
"""
Feature usage analyzer for RAG systems.

This module provides the FeatureUsageAnalyzer class, which is responsible for
tracking and analyzing feature usage in RAG systems.
"""

from typing import Dict, List, Any, Optional, Set, Counter as CounterType
from collections import Counter, defaultdict
import datetime
import statistics
from .base import BaseUsageAnalyzer


class FeatureUsageAnalyzer(BaseUsageAnalyzer):
    """
    Analyzer for feature usage in RAG systems.
    
    This analyzer tracks and analyzes feature usage, including:
    - Feature popularity
    - Feature usage patterns
    - Feature combinations
    - Feature usage by user segment
    - Feature usage over time
    
    Attributes:
        name (str): Name of the analyzer.
        data_dir (str): Directory to store analysis data.
        config (Dict[str, Any]): Configuration options for the analyzer.
    """
    
    def __init__(
        self,
        name: str = "feature_usage_analyzer",
        data_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the feature usage analyzer.
        
        Args:
            name (str): Name of the analyzer. Defaults to "feature_usage_analyzer".
            data_dir (Optional[str]): Directory to store analysis data.
                Defaults to './usage_data'.
            config (Optional[Dict[str, Any]]): Configuration options for the analyzer.
                Defaults to an empty dictionary.
        """
        super().__init__(name, data_dir, config)
        
        # Set default config values
        self.config.setdefault("time_window", "day")  # Options: "hour", "day", "week", "month"
        self.config.setdefault("top_combinations", 10)
        self.config.setdefault("user_segments", {})  # User segment definitions
    
    def track(self, event: Dict[str, Any]) -> None:
        """
        Track a feature usage event.
        
        Args:
            event (Dict[str, Any]): The feature usage event to track.
                Must contain 'feature_id' key.
                May optionally contain 'user_id', 'session_id', 'timestamp',
                'parameters', and 'result'.
        """
        if "feature_id" not in event:
            raise ValueError("Event must contain a 'feature_id' key")
        
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add to data
        self.data.append(event)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the tracked feature usage data and return insights.
        
        Returns:
            Dict[str, Any]: Analysis results, including:
                - feature_popularity: Metrics on feature popularity
                - usage_patterns: Analysis of feature usage patterns
                - feature_combinations: Analysis of feature combinations
                - segment_analysis: Analysis by user segment
                - time_analysis: Analysis of feature usage over time
        """
        if not self.data:
            return {"error": "No data to analyze"}
        
        return {
            "feature_popularity": self._analyze_feature_popularity(),
            "usage_patterns": self._analyze_usage_patterns(),
            "feature_combinations": self._analyze_feature_combinations(),
            "segment_analysis": self._analyze_by_segment(),
            "time_analysis": self._analyze_time_distribution()
        }
    
    def _analyze_feature_popularity(self) -> Dict[str, Any]:
        """
        Analyze feature popularity.
        
        Returns:
            Dict[str, Any]: Feature popularity metrics.
        """
        # Count feature usage
        feature_counts = Counter()
        user_counts = defaultdict(set)
        
        for event in self.data:
            feature_id = event["feature_id"]
            feature_counts[feature_id] += 1
            
            # Count unique users per feature
            if "user_id" in event:
                user_counts[feature_id].add(event["user_id"])
        
        # Calculate unique users per feature
        unique_users = {
            feature_id: len(users) for feature_id, users in user_counts.items()
        }
        
        # Calculate average usage per user
        avg_usage_per_user = {}
        for feature_id, count in feature_counts.items():
            if feature_id in unique_users and unique_users[feature_id] > 0:
                avg_usage_per_user[feature_id] = count / unique_users[feature_id]
            else:
                avg_usage_per_user[feature_id] = 0
        
        return {
            "total_usage": dict(feature_counts),
            "unique_users": unique_users,
            "avg_usage_per_user": avg_usage_per_user
        }
    
    def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """
        Analyze feature usage patterns.
        
        Returns:
            Dict[str, Any]: Feature usage pattern analysis.
        """
        # Analyze parameter usage
        parameter_usage = defaultdict(lambda: defaultdict(Counter))
        result_distribution = defaultdict(Counter)
        
        for event in self.data:
            feature_id = event["feature_id"]
            
            # Analyze parameters
            if "parameters" in event and isinstance(event["parameters"], dict):
                for param_name, param_value in event["parameters"].items():
                    # Convert param_value to string for counting
                    if isinstance(param_value, (list, dict)):
                        param_str = str(type(param_value).__name__)
                    else:
                        param_str = str(param_value)
                    
                    parameter_usage[feature_id][param_name][param_str] += 1
            
            # Analyze results
            if "result" in event:
                result = event["result"]
                if isinstance(result, dict) and "status" in result:
                    result_distribution[feature_id][result["status"]] += 1
                else:
                    result_distribution[feature_id]["unknown"] += 1
        
        # Convert to regular dictionaries
        param_usage_dict = {}
        for feature_id, params in parameter_usage.items():
            param_usage_dict[feature_id] = {}
            for param_name, values in params.items():
                param_usage_dict[feature_id][param_name] = dict(values)
        
        result_dist_dict = {}
        for feature_id, results in result_distribution.items():
            result_dist_dict[feature_id] = dict(results)
        
        return {
            "parameter_usage": param_usage_dict,
            "result_distribution": result_dist_dict
        }
    
    def _analyze_feature_combinations(self) -> Dict[str, Any]:
        """
        Analyze feature combinations.
        
        Returns:
            Dict[str, Any]: Feature combination analysis.
        """
        # Group events by session
        session_features = defaultdict(set)
        
        for event in self.data:
            if "session_id" in event:
                session_id = event["session_id"]
                feature_id = event["feature_id"]
                session_features[session_id].add(feature_id)
        
        # Count feature combinations
        combinations = Counter()
        
        for session_id, features in session_features.items():
            if len(features) >= 2:
                # Sort features to ensure consistent combination representation
                sorted_features = sorted(features)
                combination = ",".join(sorted_features)
                combinations[combination] += 1
        
        # Get top combinations
        top_combinations = {}
        for combo, count in combinations.most_common(self.config["top_combinations"]):
            top_combinations[combo] = count
        
        # Calculate co-occurrence matrix
        all_features = set()
        for features in session_features.values():
            all_features.update(features)
        
        co_occurrence = defaultdict(lambda: defaultdict(int))
        
        for features in session_features.values():
            feature_list = list(features)
            for i in range(len(feature_list)):
                for j in range(i + 1, len(feature_list)):
                    feature1 = feature_list[i]
                    feature2 = feature_list[j]
                    co_occurrence[feature1][feature2] += 1
                    co_occurrence[feature2][feature1] += 1
        
        # Convert to regular dictionary
        co_occurrence_dict = {}
        for feature1, co_features in co_occurrence.items():
            co_occurrence_dict[feature1] = dict(co_features)
        
        return {
            "top_combinations": top_combinations,
            "co_occurrence": co_occurrence_dict
        }
    
    def _analyze_by_segment(self) -> Dict[str, Any]:
        """
        Analyze feature usage by user segment.
        
        Returns:
            Dict[str, Any]: Feature usage by segment.
        """
        # Get user segments
        user_segments = self.config["user_segments"]
        
        if not user_segments:
            # Create default segments based on usage frequency
            user_usage = defaultdict(int)
            for event in self.data:
                if "user_id" in event:
                    user_usage[event["user_id"]] += 1
            
            # Calculate usage statistics
            if user_usage:
                usage_values = list(user_usage.values())
                avg_usage = statistics.mean(usage_values)
                
                # Define segments
                user_segments = {
                    "high_usage": lambda count: count > 2 * avg_usage,
                    "medium_usage": lambda count: avg_usage <= count <= 2 * avg_usage,
                    "low_usage": lambda count: count < avg_usage
                }
                
                # Assign users to segments
                segmented_users = defaultdict(list)
                for user_id, count in user_usage.items():
                    for segment_name, condition in user_segments.items():
                        if condition(count):
                            segmented_users[segment_name].append(user_id)
                            break
                
                # Convert to sets for faster lookup
                segmented_users = {
                    segment: set(users) for segment, users in segmented_users.items()
                }
            else:
                return {"error": "No user data available for segmentation"}
        else:
            # Use provided segment definitions
            # Assume user_segments is a dict mapping segment names to user ID lists
            segmented_users = {
                segment: set(users) for segment, users in user_segments.items()
            }
        
        # Analyze feature usage by segment
        segment_usage = defaultdict(lambda: defaultdict(int))
        
        for event in self.data:
            if "user_id" in event:
                user_id = event["user_id"]
                feature_id = event["feature_id"]
                
                # Find user's segment
                for segment_name, users in segmented_users.items():
                    if user_id in users:
                        segment_usage[segment_name][feature_id] += 1
                        break
        
        # Convert to regular dictionary
        segment_usage_dict = {}
        for segment_name, features in segment_usage.items():
            segment_usage_dict[segment_name] = dict(features)
        
        return segment_usage_dict
    
    def _analyze_time_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Analyze feature usage over time.
        
        Returns:
            Dict[str, Dict[str, int]]: Feature usage by time period.
        """
        time_window = self.config["time_window"]
        feature_time_periods = defaultdict(lambda: defaultdict(int))
        
        for event in self.data:
            feature_id = event["feature_id"]
            timestamp = datetime.datetime.fromisoformat(event["timestamp"])
            
            if time_window == "hour":
                period = timestamp.strftime("%Y-%m-%d %H:00")
            elif time_window == "day":
                period = timestamp.strftime("%Y-%m-%d")
            elif time_window == "week":
                # Get the start of the week (Monday)
                start_of_week = timestamp - datetime.timedelta(days=timestamp.weekday())
                period = start_of_week.strftime("%Y-%m-%d")
            elif time_window == "month":
                period = timestamp.strftime("%Y-%m")
            else:
                period = timestamp.strftime("%Y-%m-%d")
            
            feature_time_periods[feature_id][period] += 1
        
        # Convert to regular dictionary
        time_distribution = {}
        for feature_id, periods in feature_time_periods.items():
            time_distribution[feature_id] = dict(sorted(periods.items()))
        
        return time_distribution 