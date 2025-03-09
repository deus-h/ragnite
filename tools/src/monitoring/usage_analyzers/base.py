#!/usr/bin/env python3
"""
Base class for usage analyzers in RAG systems.

This module provides the abstract base class for all usage analyzers,
which are responsible for tracking and analyzing how users interact with
the RAG system, identifying patterns, and providing insights.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import json
import os
import datetime
from pathlib import Path


class BaseUsageAnalyzer(ABC):
    """
    Abstract base class for usage analyzers.
    
    Usage analyzers track and analyze how users interact with the RAG system,
    identifying patterns in queries, tracking user sessions, monitoring feature
    usage, and analyzing errors.
    
    Attributes:
        name (str): Name of the analyzer.
        data_dir (str): Directory to store analysis data.
        config (Dict[str, Any]): Configuration options for the analyzer.
    """
    
    def __init__(
        self,
        name: str,
        data_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the usage analyzer.
        
        Args:
            name (str): Name of the analyzer.
            data_dir (Optional[str]): Directory to store analysis data.
                Defaults to './usage_data'.
            config (Optional[Dict[str, Any]]): Configuration options for the analyzer.
                Defaults to an empty dictionary.
        """
        self.name = name
        self.data_dir = data_dir or './usage_data'
        self.config = config or {}
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data storage
        self.data: List[Dict[str, Any]] = []
    
    @abstractmethod
    def track(self, event: Dict[str, Any]) -> None:
        """
        Track a usage event.
        
        Args:
            event (Dict[str, Any]): The event to track.
        """
        pass
    
    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the tracked data and return insights.
        
        Returns:
            Dict[str, Any]: Analysis results.
        """
        pass
    
    def save_data(self, filename: Optional[str] = None) -> str:
        """
        Save the tracked data to a file.
        
        Args:
            filename (Optional[str]): Name of the file to save data to.
                If not provided, a default name will be generated.
                
        Returns:
            str: Path to the saved file.
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_data_{timestamp}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        return filepath
    
    def load_data(self, filepath: str) -> None:
        """
        Load tracked data from a file.
        
        Args:
            filepath (str): Path to the file to load data from.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            self.data = json.load(f)
    
    def clear_data(self) -> None:
        """Clear all tracked data."""
        self.data = []
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the tracked data.
        
        Returns:
            Dict[str, Any]: Summary of the tracked data.
        """
        return {
            "analyzer_name": self.name,
            "event_count": len(self.data),
            "time_range": self._get_time_range(),
        }
    
    def _get_time_range(self) -> Dict[str, str]:
        """
        Get the time range of the tracked data.
        
        Returns:
            Dict[str, str]: Start and end times of the tracked data.
        """
        if not self.data:
            return {"start": None, "end": None}
        
        timestamps = []
        for event in self.data:
            if "timestamp" in event:
                timestamps.append(event["timestamp"])
        
        if not timestamps:
            return {"start": None, "end": None}
        
        return {
            "start": min(timestamps),
            "end": max(timestamps)
        } 