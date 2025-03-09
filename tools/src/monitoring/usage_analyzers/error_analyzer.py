#!/usr/bin/env python3
"""
Error analyzer for RAG systems.

This module provides the ErrorAnalyzer class, which is responsible for
tracking and analyzing errors in RAG systems.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
import datetime
import re
import statistics
from .base import BaseUsageAnalyzer


class ErrorAnalyzer(BaseUsageAnalyzer):
    """
    Analyzer for errors in RAG systems.
    
    This analyzer tracks and analyzes errors, including:
    - Error frequency
    - Error types
    - Error patterns
    - Error impact
    - Error resolution
    
    Attributes:
        name (str): Name of the analyzer.
        data_dir (str): Directory to store analysis data.
        config (Dict[str, Any]): Configuration options for the analyzer.
    """
    
    def __init__(
        self,
        name: str = "error_analyzer",
        data_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the error analyzer.
        
        Args:
            name (str): Name of the analyzer. Defaults to "error_analyzer".
            data_dir (Optional[str]): Directory to store analysis data.
                Defaults to './usage_data'.
            config (Optional[Dict[str, Any]]): Configuration options for the analyzer.
                Defaults to an empty dictionary.
        """
        super().__init__(name, data_dir, config)
        
        # Set default config values
        self.config.setdefault("time_window", "day")  # Options: "hour", "day", "week", "month"
        self.config.setdefault("severity_levels", ["critical", "high", "medium", "low"])
        self.config.setdefault("error_categories", {
            "retrieval": r"retriev|fetch|find|search",
            "generation": r"generat|creat|produc",
            "parsing": r"pars|extract|format",
            "authentication": r"auth|login|credential",
            "rate_limit": r"rate|limit|throttl",
            "timeout": r"timeout|timed out|too long",
            "connection": r"connect|network|server",
            "validation": r"valid|schema|format"
        })
    
    def track(self, event: Dict[str, Any]) -> None:
        """
        Track an error event.
        
        Args:
            event (Dict[str, Any]): The error event to track.
                Must contain 'error_message' key.
                May optionally contain 'error_code', 'error_type', 'severity',
                'component', 'user_id', 'session_id', 'timestamp', and 'resolution'.
        """
        if "error_message" not in event:
            raise ValueError("Event must contain an 'error_message' key")
        
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add error category if not present
        if "error_category" not in event:
            event["error_category"] = self._categorize_error(event["error_message"])
        
        # Add severity if not present
        if "severity" not in event:
            event["severity"] = self._determine_severity(event)
        
        # Add to data
        self.data.append(event)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the tracked error data and return insights.
        
        Returns:
            Dict[str, Any]: Analysis results, including:
                - error_frequency: Metrics on error frequency
                - error_types: Analysis of error types
                - error_patterns: Analysis of error patterns
                - error_impact: Analysis of error impact
                - error_resolution: Analysis of error resolution
        """
        if not self.data:
            return {"error": "No data to analyze"}
        
        return {
            "error_frequency": self._analyze_error_frequency(),
            "error_types": self._analyze_error_types(),
            "error_patterns": self._analyze_error_patterns(),
            "error_impact": self._analyze_error_impact(),
            "error_resolution": self._analyze_error_resolution()
        }
    
    def _analyze_error_frequency(self) -> Dict[str, Any]:
        """
        Analyze error frequency.
        
        Returns:
            Dict[str, Any]: Error frequency metrics.
        """
        # Count errors by time period
        time_window = self.config["time_window"]
        time_periods = {}
        
        for event in self.data:
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
            
            time_periods[period] = time_periods.get(period, 0) + 1
        
        # Calculate error rate if we have total requests
        error_rate = None
        if "total_requests" in self.config:
            total_requests = self.config["total_requests"]
            if isinstance(total_requests, int) and total_requests > 0:
                error_rate = len(self.data) / total_requests
        
        return {
            "total_errors": len(self.data),
            "error_rate": error_rate,
            "time_distribution": dict(sorted(time_periods.items()))
        }
    
    def _analyze_error_types(self) -> Dict[str, Any]:
        """
        Analyze error types.
        
        Returns:
            Dict[str, Any]: Error type analysis.
        """
        # Count errors by type, code, and category
        error_types = Counter()
        error_codes = Counter()
        error_categories = Counter()
        severity_counts = Counter()
        component_counts = Counter()
        
        for event in self.data:
            # Count error types
            if "error_type" in event:
                error_types[event["error_type"]] += 1
            
            # Count error codes
            if "error_code" in event:
                error_codes[str(event["error_code"])] += 1
            
            # Count error categories
            error_categories[event["error_category"]] += 1
            
            # Count severity levels
            severity_counts[event["severity"]] += 1
            
            # Count components
            if "component" in event:
                component_counts[event["component"]] += 1
        
        return {
            "error_types": dict(error_types),
            "error_codes": dict(error_codes),
            "error_categories": dict(error_categories),
            "severity_distribution": dict(severity_counts),
            "component_distribution": dict(component_counts)
        }
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """
        Analyze error patterns.
        
        Returns:
            Dict[str, Any]: Error pattern analysis.
        """
        # Extract common patterns from error messages
        error_messages = [event["error_message"] for event in self.data]
        
        # Find common words
        common_words = self._find_common_words(error_messages)
        
        # Find common phrases
        common_phrases = self._find_common_phrases(error_messages)
        
        # Analyze co-occurrence of error types and categories
        type_category_co_occurrence = defaultdict(Counter)
        
        for event in self.data:
            if "error_type" in event:
                error_type = event["error_type"]
                error_category = event["error_category"]
                type_category_co_occurrence[error_type][error_category] += 1
        
        # Convert to regular dictionary
        co_occurrence_dict = {}
        for error_type, categories in type_category_co_occurrence.items():
            co_occurrence_dict[error_type] = dict(categories)
        
        return {
            "common_words": common_words,
            "common_phrases": common_phrases,
            "type_category_co_occurrence": co_occurrence_dict
        }
    
    def _analyze_error_impact(self) -> Dict[str, Any]:
        """
        Analyze error impact.
        
        Returns:
            Dict[str, Any]: Error impact analysis.
        """
        # Count affected users and sessions
        affected_users = set()
        affected_sessions = set()
        
        for event in self.data:
            if "user_id" in event:
                affected_users.add(event["user_id"])
            
            if "session_id" in event:
                affected_sessions.add(event["session_id"])
        
        # Count errors by severity
        severity_counts = Counter()
        for event in self.data:
            severity_counts[event["severity"]] += 1
        
        # Calculate impact score
        severity_weights = {
            "critical": 100,
            "high": 50,
            "medium": 20,
            "low": 5
        }
        
        impact_score = sum(
            count * severity_weights.get(severity, 1)
            for severity, count in severity_counts.items()
        )
        
        return {
            "affected_users": len(affected_users),
            "affected_sessions": len(affected_sessions),
            "severity_distribution": dict(severity_counts),
            "impact_score": impact_score
        }
    
    def _analyze_error_resolution(self) -> Dict[str, Any]:
        """
        Analyze error resolution.
        
        Returns:
            Dict[str, Any]: Error resolution analysis.
        """
        # Count resolved errors
        resolved_count = 0
        resolution_times = []
        resolution_methods = Counter()
        
        for event in self.data:
            if "resolution" in event and event["resolution"].get("resolved", False):
                resolved_count += 1
                
                # Count resolution methods
                if "method" in event["resolution"]:
                    resolution_methods[event["resolution"]["method"]] += 1
                
                # Calculate resolution time if available
                if "resolved_at" in event["resolution"]:
                    error_time = datetime.datetime.fromisoformat(event["timestamp"])
                    resolution_time = datetime.datetime.fromisoformat(
                        event["resolution"]["resolved_at"]
                    )
                    resolution_minutes = (resolution_time - error_time).total_seconds() / 60
                    resolution_times.append(resolution_minutes)
        
        # Calculate resolution rate
        resolution_rate = resolved_count / len(self.data) if self.data else 0
        
        # Calculate resolution time statistics
        resolution_time_stats = {}
        if resolution_times:
            resolution_time_stats = {
                "min": min(resolution_times),
                "max": max(resolution_times),
                "avg": statistics.mean(resolution_times),
                "median": statistics.median(resolution_times)
            }
        
        return {
            "resolved_count": resolved_count,
            "unresolved_count": len(self.data) - resolved_count,
            "resolution_rate": resolution_rate,
            "resolution_time_minutes": resolution_time_stats,
            "resolution_methods": dict(resolution_methods)
        }
    
    def _categorize_error(self, error_message: str) -> str:
        """
        Categorize an error based on its message.
        
        Args:
            error_message (str): The error message.
            
        Returns:
            str: Error category.
        """
        error_message = error_message.lower()
        
        for category, pattern in self.config["error_categories"].items():
            if re.search(pattern, error_message):
                return category
        
        return "other"
    
    def _determine_severity(self, event: Dict[str, Any]) -> str:
        """
        Determine the severity of an error.
        
        Args:
            event (Dict[str, Any]): The error event.
            
        Returns:
            str: Error severity.
        """
        # Use default severity levels
        severity_levels = self.config["severity_levels"]
        
        # Check for explicit severity indicators in the error message
        error_message = event["error_message"].lower()
        
        if "critical" in error_message or "fatal" in error_message:
            return "critical"
        elif "high" in error_message or "severe" in error_message:
            return "high"
        elif "medium" in error_message or "moderate" in error_message:
            return "medium"
        elif "low" in error_message or "minor" in error_message:
            return "low"
        
        # Check for error code severity if available
        if "error_code" in event:
            error_code = event["error_code"]
            if isinstance(error_code, int):
                if error_code >= 500:
                    return "high"
                elif error_code >= 400:
                    return "medium"
                else:
                    return "low"
        
        # Default to medium severity
        return "medium"
    
    def _find_common_words(self, messages: List[str], top_n: int = 20) -> Dict[str, int]:
        """
        Find common words in error messages.
        
        Args:
            messages (List[str]): List of error messages.
            top_n (int): Number of top words to return.
            
        Returns:
            Dict[str, int]: Dictionary of common words and their frequencies.
        """
        # Combine all messages
        all_text = " ".join(messages).lower()
        
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', all_text)
        
        # Filter out common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "in", "on", "at", "to", "for", "with",
            "by", "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "from", "up", "down", "of",
            "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "can", "will",
            "just", "should", "now"
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counter = Counter(filtered_words)
        
        # Return the most common words
        return dict(word_counter.most_common(top_n))
    
    def _find_common_phrases(self, messages: List[str], top_n: int = 10, min_length: int = 3) -> Dict[str, int]:
        """
        Find common phrases in error messages.
        
        Args:
            messages (List[str]): List of error messages.
            top_n (int): Number of top phrases to return.
            min_length (int): Minimum number of words in a phrase.
            
        Returns:
            Dict[str, int]: Dictionary of common phrases and their frequencies.
        """
        # Extract n-grams from messages
        phrases = []
        
        for message in messages:
            # Normalize message
            message = message.lower()
            
            # Remove punctuation and split into words
            words = re.findall(r'\b\w+\b', message)
            
            # Extract n-grams
            for n in range(min_length, min(len(words) + 1, 6)):  # Phrases of length 3 to 5
                for i in range(len(words) - n + 1):
                    phrase = " ".join(words[i:i+n])
                    phrases.append(phrase)
        
        # Count phrase frequencies
        phrase_counter = Counter(phrases)
        
        # Return the most common phrases
        return dict(phrase_counter.most_common(top_n)) 