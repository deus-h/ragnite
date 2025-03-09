#!/usr/bin/env python3
"""
Query analyzer for RAG systems.

This module provides the QueryAnalyzer class, which is responsible for
tracking and analyzing query patterns in RAG systems.
"""

from typing import Dict, List, Any, Optional, Counter as CounterType
from collections import Counter
import datetime
import re
from .base import BaseUsageAnalyzer


class QueryAnalyzer(BaseUsageAnalyzer):
    """
    Analyzer for query patterns in RAG systems.
    
    This analyzer tracks and analyzes query patterns, including:
    - Most common query terms
    - Query length distribution
    - Query frequency over time
    - Query complexity
    - Query categories
    
    Attributes:
        name (str): Name of the analyzer.
        data_dir (str): Directory to store analysis data.
        config (Dict[str, Any]): Configuration options for the analyzer.
    """
    
    def __init__(
        self,
        name: str = "query_analyzer",
        data_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the query analyzer.
        
        Args:
            name (str): Name of the analyzer. Defaults to "query_analyzer".
            data_dir (Optional[str]): Directory to store analysis data.
                Defaults to './usage_data'.
            config (Optional[Dict[str, Any]]): Configuration options for the analyzer.
                Defaults to an empty dictionary.
        """
        super().__init__(name, data_dir, config)
        
        # Set default config values
        self.config.setdefault("min_term_length", 3)
        self.config.setdefault("max_common_terms", 20)
        self.config.setdefault("time_window", "day")  # Options: "hour", "day", "week", "month"
        self.config.setdefault("complexity_factors", {
            "length": 1.0,
            "unique_terms": 1.5,
            "special_chars": 0.5
        })
        
        # Initialize category patterns
        self.category_patterns = self.config.get("category_patterns", {
            "factual": r"\b(what|who|when|where|why|how)\b.*\?",
            "instructional": r"\b(how to|steps|guide|tutorial|explain)\b",
            "comparative": r"\b(compare|versus|vs|difference|better|best)\b",
            "opinion": r"\b(opinion|think|feel|believe|recommend)\b",
            "clarification": r"\b(clarify|explain|elaborate|mean)\b"
        })
    
    def track(self, event: Dict[str, Any]) -> None:
        """
        Track a query event.
        
        Args:
            event (Dict[str, Any]): The query event to track.
                Must contain a 'query' key with the query text.
                May optionally contain 'user_id', 'session_id', and 'timestamp'.
        """
        if "query" not in event:
            raise ValueError("Event must contain a 'query' key")
        
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add query metadata
        query_text = event["query"]
        event["query_metadata"] = {
            "length": len(query_text),
            "word_count": len(query_text.split()),
            "complexity": self._calculate_complexity(query_text),
            "category": self._categorize_query(query_text)
        }
        
        # Add to data
        self.data.append(event)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the tracked query data and return insights.
        
        Returns:
            Dict[str, Any]: Analysis results, including:
                - common_terms: Most common terms in queries
                - length_distribution: Distribution of query lengths
                - time_analysis: Query frequency over time
                - complexity_stats: Statistics on query complexity
                - category_distribution: Distribution of query categories
        """
        if not self.data:
            return {"error": "No data to analyze"}
        
        return {
            "common_terms": self._analyze_common_terms(),
            "length_distribution": self._analyze_length_distribution(),
            "time_analysis": self._analyze_time_distribution(),
            "complexity_stats": self._analyze_complexity(),
            "category_distribution": self._analyze_categories()
        }
    
    def _analyze_common_terms(self) -> Dict[str, int]:
        """
        Analyze the most common terms in queries.
        
        Returns:
            Dict[str, int]: Dictionary of common terms and their frequencies.
        """
        all_terms = []
        min_length = self.config["min_term_length"]
        
        for event in self.data:
            query = event["query"].lower()
            # Remove punctuation and split into terms
            terms = re.findall(r'\b\w+\b', query)
            # Filter out short terms
            terms = [term for term in terms if len(term) >= min_length]
            all_terms.extend(terms)
        
        # Count term frequencies
        term_counter = Counter(all_terms)
        
        # Return the most common terms
        return dict(term_counter.most_common(self.config["max_common_terms"]))
    
    def _analyze_length_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of query lengths.
        
        Returns:
            Dict[str, Any]: Statistics on query lengths.
        """
        lengths = [event["query_metadata"]["length"] for event in self.data]
        word_counts = [event["query_metadata"]["word_count"] for event in self.data]
        
        return {
            "character_length": {
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths) / len(lengths),
                "distribution": self._get_distribution(lengths)
            },
            "word_count": {
                "min": min(word_counts),
                "max": max(word_counts),
                "avg": sum(word_counts) / len(word_counts),
                "distribution": self._get_distribution(word_counts)
            }
        }
    
    def _analyze_time_distribution(self) -> Dict[str, int]:
        """
        Analyze the distribution of queries over time.
        
        Returns:
            Dict[str, int]: Query frequency by time period.
        """
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
        
        return dict(sorted(time_periods.items()))
    
    def _analyze_complexity(self) -> Dict[str, Any]:
        """
        Analyze query complexity.
        
        Returns:
            Dict[str, Any]: Statistics on query complexity.
        """
        complexities = [event["query_metadata"]["complexity"] for event in self.data]
        
        return {
            "min": min(complexities),
            "max": max(complexities),
            "avg": sum(complexities) / len(complexities),
            "distribution": self._get_distribution(complexities)
        }
    
    def _analyze_categories(self) -> Dict[str, int]:
        """
        Analyze the distribution of query categories.
        
        Returns:
            Dict[str, int]: Frequency of each query category.
        """
        categories = [event["query_metadata"]["category"] for event in self.data]
        category_counter = Counter(categories)
        
        return dict(category_counter)
    
    def _calculate_complexity(self, query: str) -> float:
        """
        Calculate the complexity of a query.
        
        Args:
            query (str): The query text.
            
        Returns:
            float: Complexity score.
        """
        factors = self.config["complexity_factors"]
        
        # Length factor
        length_score = len(query) * factors["length"] / 100  # Normalize by 100
        
        # Unique terms factor
        terms = set(re.findall(r'\b\w+\b', query.lower()))
        unique_terms_score = len(terms) * factors["unique_terms"] / 10  # Normalize by 10
        
        # Special characters factor
        special_chars = re.findall(r'[^\w\s]', query)
        special_chars_score = len(special_chars) * factors["special_chars"]
        
        return length_score + unique_terms_score + special_chars_score
    
    def _categorize_query(self, query: str) -> str:
        """
        Categorize a query based on patterns.
        
        Args:
            query (str): The query text.
            
        Returns:
            str: Query category.
        """
        query = query.lower()
        
        for category, pattern in self.category_patterns.items():
            if re.search(pattern, query):
                return category
        
        return "other"
    
    def _get_distribution(self, values: List[int]) -> Dict[str, int]:
        """
        Get the distribution of values.
        
        Args:
            values (List[int]): List of values.
            
        Returns:
            Dict[str, int]: Distribution of values.
        """
        counter = Counter(values)
        return dict(sorted(counter.items())) 