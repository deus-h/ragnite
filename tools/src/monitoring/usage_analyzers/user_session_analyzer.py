#!/usr/bin/env python3
"""
User session analyzer for RAG systems.

This module provides the UserSessionAnalyzer class, which is responsible for
tracking and analyzing user sessions in RAG systems.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
import datetime
import statistics
from .base import BaseUsageAnalyzer


class UserSessionAnalyzer(BaseUsageAnalyzer):
    """
    Analyzer for user sessions in RAG systems.
    
    This analyzer tracks and analyzes user sessions, including:
    - Session duration
    - Session activity
    - User engagement
    - Session flow
    - User retention
    
    Attributes:
        name (str): Name of the analyzer.
        data_dir (str): Directory to store analysis data.
        config (Dict[str, Any]): Configuration options for the analyzer.
    """
    
    def __init__(
        self,
        name: str = "user_session_analyzer",
        data_dir: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the user session analyzer.
        
        Args:
            name (str): Name of the analyzer. Defaults to "user_session_analyzer".
            data_dir (Optional[str]): Directory to store analysis data.
                Defaults to './usage_data'.
            config (Optional[Dict[str, Any]]): Configuration options for the analyzer.
                Defaults to an empty dictionary.
        """
        super().__init__(name, data_dir, config)
        
        # Set default config values
        self.config.setdefault("session_timeout", 30)  # minutes
        self.config.setdefault("min_session_events", 2)
        self.config.setdefault("retention_periods", ["day", "week", "month"])
    
    def track(self, event: Dict[str, Any]) -> None:
        """
        Track a user session event.
        
        Args:
            event (Dict[str, Any]): The session event to track.
                Must contain 'user_id' and 'event_type' keys.
                May optionally contain 'session_id' and 'timestamp'.
        """
        if "user_id" not in event:
            raise ValueError("Event must contain a 'user_id' key")
        
        if "event_type" not in event:
            raise ValueError("Event must contain an 'event_type' key")
        
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add session_id if not present
        if "session_id" not in event:
            # Generate session_id from user_id and timestamp
            timestamp = datetime.datetime.fromisoformat(event["timestamp"])
            date_str = timestamp.strftime("%Y%m%d")
            event["session_id"] = f"{event['user_id']}_{date_str}"
        
        # Add to data
        self.data.append(event)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the tracked session data and return insights.
        
        Returns:
            Dict[str, Any]: Analysis results, including:
                - session_stats: Statistics on session duration and activity
                - user_engagement: Metrics on user engagement
                - session_flow: Analysis of session flow
                - user_retention: Metrics on user retention
        """
        if not self.data:
            return {"error": "No data to analyze"}
        
        # Group events by session
        sessions = self._group_events_by_session()
        
        # Filter out sessions with too few events
        min_events = self.config["min_session_events"]
        valid_sessions = {
            session_id: events for session_id, events in sessions.items()
            if len(events) >= min_events
        }
        
        if not valid_sessions:
            return {"error": "No valid sessions to analyze"}
        
        return {
            "session_stats": self._analyze_session_stats(valid_sessions),
            "user_engagement": self._analyze_user_engagement(valid_sessions),
            "session_flow": self._analyze_session_flow(valid_sessions),
            "user_retention": self._analyze_user_retention()
        }
    
    def _group_events_by_session(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group events by session ID.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary of session events.
        """
        sessions = defaultdict(list)
        
        for event in sorted(self.data, key=lambda e: e["timestamp"]):
            sessions[event["session_id"]].append(event)
        
        return dict(sessions)
    
    def _analyze_session_stats(self, sessions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze session statistics.
        
        Args:
            sessions (Dict[str, List[Dict[str, Any]]]): Dictionary of session events.
            
        Returns:
            Dict[str, Any]: Session statistics.
        """
        durations = []
        event_counts = []
        
        for session_id, events in sessions.items():
            # Calculate session duration
            start_time = datetime.datetime.fromisoformat(events[0]["timestamp"])
            end_time = datetime.datetime.fromisoformat(events[-1]["timestamp"])
            duration_minutes = (end_time - start_time).total_seconds() / 60
            
            durations.append(duration_minutes)
            event_counts.append(len(events))
        
        return {
            "session_count": len(sessions),
            "duration_minutes": {
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "avg": statistics.mean(durations) if durations else 0,
                "median": statistics.median(durations) if durations else 0
            },
            "events_per_session": {
                "min": min(event_counts),
                "max": max(event_counts),
                "avg": statistics.mean(event_counts),
                "median": statistics.median(event_counts)
            }
        }
    
    def _analyze_user_engagement(self, sessions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze user engagement.
        
        Args:
            sessions (Dict[str, List[Dict[str, Any]]]): Dictionary of session events.
            
        Returns:
            Dict[str, Any]: User engagement metrics.
        """
        # Get unique users
        users = set()
        user_sessions = defaultdict(list)
        event_types = Counter()
        
        for session_id, events in sessions.items():
            user_id = events[0]["user_id"]
            users.add(user_id)
            user_sessions[user_id].append(session_id)
            
            # Count event types
            for event in events:
                event_types[event["event_type"]] += 1
        
        # Calculate sessions per user
        sessions_per_user = [len(sessions) for user, sessions in user_sessions.items()]
        
        return {
            "unique_users": len(users),
            "sessions_per_user": {
                "min": min(sessions_per_user),
                "max": max(sessions_per_user),
                "avg": statistics.mean(sessions_per_user),
                "median": statistics.median(sessions_per_user)
            },
            "event_type_distribution": dict(event_types)
        }
    
    def _analyze_session_flow(self, sessions: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze session flow.
        
        Args:
            sessions (Dict[str, List[Dict[str, Any]]]): Dictionary of session events.
            
        Returns:
            Dict[str, Any]: Session flow analysis.
        """
        # Analyze event sequences
        event_sequences = []
        first_events = Counter()
        last_events = Counter()
        event_transitions = defaultdict(Counter)
        
        for session_id, events in sessions.items():
            # Extract event types
            event_types = [event["event_type"] for event in events]
            event_sequences.append(event_types)
            
            # Record first and last events
            first_events[event_types[0]] += 1
            last_events[event_types[-1]] += 1
            
            # Record event transitions
            for i in range(len(event_types) - 1):
                current_event = event_types[i]
                next_event = event_types[i + 1]
                event_transitions[current_event][next_event] += 1
        
        # Convert transitions to dictionary
        transitions_dict = {}
        for current_event, next_events in event_transitions.items():
            transitions_dict[current_event] = dict(next_events)
        
        return {
            "first_events": dict(first_events),
            "last_events": dict(last_events),
            "event_transitions": transitions_dict
        }
    
    def _analyze_user_retention(self) -> Dict[str, Any]:
        """
        Analyze user retention.
        
        Returns:
            Dict[str, Any]: User retention metrics.
        """
        # Get all users and their activity dates
        user_activity = defaultdict(set)
        
        for event in self.data:
            user_id = event["user_id"]
            timestamp = datetime.datetime.fromisoformat(event["timestamp"])
            
            # Add activity date
            user_activity[user_id].add(timestamp.date())
        
        # Calculate retention for different periods
        retention = {}
        
        for period in self.config["retention_periods"]:
            retention[period] = self._calculate_retention(user_activity, period)
        
        return retention
    
    def _calculate_retention(
        self,
        user_activity: Dict[str, Set[datetime.date]],
        period: str
    ) -> Dict[str, float]:
        """
        Calculate user retention for a specific period.
        
        Args:
            user_activity (Dict[str, Set[datetime.date]]): Dictionary of user activity dates.
            period (str): Retention period ("day", "week", or "month").
            
        Returns:
            Dict[str, float]: Retention rates.
        """
        # Get all activity dates
        all_dates = set()
        for dates in user_activity.values():
            all_dates.update(dates)
        
        if not all_dates:
            return {}
        
        # Sort dates
        sorted_dates = sorted(all_dates)
        min_date = sorted_dates[0]
        max_date = sorted_dates[-1]
        
        # Define period delta
        if period == "day":
            delta = datetime.timedelta(days=1)
        elif period == "week":
            delta = datetime.timedelta(weeks=1)
        elif period == "month":
            delta = datetime.timedelta(days=30)  # Approximate
        else:
            delta = datetime.timedelta(days=1)
        
        # Calculate retention
        retention_rates = {}
        current_date = min_date
        
        while current_date <= max_date:
            next_date = current_date + delta
            
            # Get users active on current date
            current_users = set()
            for user_id, dates in user_activity.items():
                if current_date in dates:
                    current_users.add(user_id)
            
            # Get users active on next date
            next_users = set()
            for user_id, dates in user_activity.items():
                if any(d >= next_date and d < next_date + delta for d in dates):
                    next_users.add(user_id)
            
            # Calculate retention rate
            if current_users:
                retained_users = current_users.intersection(next_users)
                retention_rate = len(retained_users) / len(current_users)
                
                # Format date as string
                date_str = current_date.strftime("%Y-%m-%d")
                retention_rates[date_str] = retention_rate
            
            current_date = next_date
        
        return retention_rates 