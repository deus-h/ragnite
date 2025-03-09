#!/usr/bin/env python3
"""
Base class for cost estimators in RAG systems.

This module provides the abstract base class for all cost estimators,
which are responsible for calculating and tracking the costs associated
with using different services and infrastructure in RAG systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import datetime
import json
import uuid


class BaseCostEstimator(ABC):
    """
    Abstract base class for cost estimators.
    
    Cost estimators calculate and track the costs associated with using
    different services and infrastructure in RAG systems, helping users
    understand and optimize their expenses.
    
    Attributes:
        name (str): Name of the estimator.
        config (Dict[str, Any]): Configuration options for the estimator.
        price_data (Dict[str, Any]): Pricing data for the service or infrastructure.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        price_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the cost estimator.
        
        Args:
            name (str): Name of the estimator.
            config (Optional[Dict[str, Any]]): Configuration options for the estimator.
                Defaults to an empty dictionary.
            price_data (Optional[Dict[str, Any]]): Pricing data for the service or infrastructure.
                If not provided, default pricing data will be used.
        """
        self.name = name
        self.config = config or {}
        self.price_data = price_data or self._get_default_price_data()
        self.usage_records = []
        self.cost_id_counter = 0
    
    @abstractmethod
    def _get_default_price_data(self) -> Dict[str, Any]:
        """
        Get default pricing data for the service or infrastructure.
        
        Returns:
            Dict[str, Any]: Default pricing data.
        """
        pass
    
    @abstractmethod
    def estimate_cost(
        self, 
        usage: Dict[str, Any],
        time_period: Optional[Dict[str, datetime.datetime]] = None
    ) -> Dict[str, Any]:
        """
        Estimate the cost based on usage and time period.
        
        Args:
            usage (Dict[str, Any]): Usage data for the service or infrastructure.
            time_period (Optional[Dict[str, datetime.datetime]]): Time period for the cost estimation.
                Should contain 'start' and 'end' keys with datetime values.
                If not provided, the entire usage history will be considered.
        
        Returns:
            Dict[str, Any]: Cost estimation result with detailed breakdown.
        """
        pass
    
    def record_usage(self, usage_data: Dict[str, Any]) -> str:
        """
        Record usage data for later cost estimation.
        
        Args:
            usage_data (Dict[str, Any]): Usage data to record.
        
        Returns:
            str: Unique ID for the usage record.
        """
        # Generate a unique ID for the usage record
        record_id = str(uuid.uuid4())
        
        # Add timestamp and ID to the usage data
        timestamp = datetime.datetime.now().isoformat()
        record = {
            "id": record_id,
            "timestamp": timestamp,
            **usage_data
        }
        
        # Add to usage records
        self.usage_records.append(record)
        
        return record_id
    
    def get_usage_records(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get usage records within the specified time period.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering records.
                If not provided, no lower bound will be applied.
            end_time (Optional[datetime.datetime]): End time for filtering records.
                If not provided, no upper bound will be applied.
            limit (int): Maximum number of records to return. Defaults to 100.
        
        Returns:
            List[Dict[str, Any]]: List of usage records.
        """
        records = self.usage_records
        
        # Filter by start time if provided
        if start_time:
            start_time_iso = start_time.isoformat()
            records = [r for r in records if r["timestamp"] >= start_time_iso]
            
        # Filter by end time if provided
        if end_time:
            end_time_iso = end_time.isoformat()
            records = [r for r in records if r["timestamp"] <= end_time_iso]
            
        # Apply limit
        records = records[-limit:]
        
        return records
    
    def export_cost_report(
        self,
        time_period: Optional[Dict[str, datetime.datetime]] = None,
        format: str = "json"
    ) -> str:
        """
        Export a cost report for the specified time period.
        
        Args:
            time_period (Optional[Dict[str, datetime.datetime]]): Time period for the cost report.
                Should contain 'start' and 'end' keys with datetime values.
                If not provided, the entire usage history will be considered.
            format (str): Format for the cost report. Defaults to 'json'.
                Supported formats: 'json', 'csv'.
        
        Returns:
            str: Cost report in the specified format.
        """
        # Get usage records for the specified time period
        if time_period and time_period.get("start") and time_period.get("end"):
            records = self.get_usage_records(
                start_time=time_period["start"],
                end_time=time_period["end"],
                limit=10000  # High limit to get all records
            )
        else:
            records = self.usage_records
        
        # Estimate cost based on the records
        usage = self._aggregate_usage(records)
        cost_estimation = self.estimate_cost(usage, time_period)
        
        # Export in the specified format
        if format.lower() == "json":
            return json.dumps(cost_estimation, indent=2)
        elif format.lower() == "csv":
            # Convert to CSV
            lines = ["timestamp,service,resource,quantity,unit_price,cost"]
            for item in cost_estimation.get("items", []):
                line = f"{item.get('timestamp', '')},{item.get('service', '')},{item.get('resource', '')}," \
                       f"{item.get('quantity', 0)},{item.get('unit_price', 0)},{item.get('cost', 0)}"
                lines.append(line)
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported formats: 'json', 'csv'.")
    
    def _aggregate_usage(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate usage data from multiple records.
        
        Args:
            records (List[Dict[str, Any]]): List of usage records.
        
        Returns:
            Dict[str, Any]: Aggregated usage data.
        """
        # Default implementation - should be overridden by subclasses
        # for more specific aggregation logic
        result = {}
        for record in records:
            for key, value in record.items():
                if key not in ["id", "timestamp"]:
                    if key not in result:
                        result[key] = value
                    elif isinstance(value, (int, float)):
                        result[key] = result.get(key, 0) + value
        
        return result
    
    def update_price_data(self, new_price_data: Dict[str, Any]) -> None:
        """
        Update the pricing data used by the estimator.
        
        Args:
            new_price_data (Dict[str, Any]): New pricing data to use.
        """
        self.price_data = new_price_data
    
    def reset_usage_records(self) -> None:
        """
        Reset all usage records.
        """
        self.usage_records = [] 