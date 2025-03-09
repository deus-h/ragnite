#!/usr/bin/env python3
"""
Composite cost estimator for combining multiple cost estimators in RAG systems.

This module provides a composite cost estimator that can combine multiple cost
estimators to provide a total cost estimation for RAG systems.
"""

import datetime
from typing import Dict, List, Any, Optional, Union
import json
from .base import BaseCostEstimator


class CompositeCostEstimator(BaseCostEstimator):
    """
    Composite cost estimator for combining multiple estimators.
    
    This estimator aggregates the results of multiple cost estimators to provide
    a comprehensive cost estimation for RAG systems that use multiple services
    and infrastructure components.
    
    Attributes:
        name (str): Name of the estimator.
        config (Dict[str, Any]): Configuration options for the estimator.
        estimators (List[BaseCostEstimator]): List of cost estimators to combine.
    """
    
    def __init__(
        self,
        name: str,
        estimators: List[BaseCostEstimator],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the composite cost estimator.
        
        Args:
            name (str): Name of the estimator.
            estimators (List[BaseCostEstimator]): List of cost estimators to combine.
            config (Optional[Dict[str, Any]]): Configuration options for the estimator.
                Defaults to an empty dictionary.
        """
        self.estimators = estimators
        super().__init__(name=name, config=config, price_data={})
    
    def _get_default_price_data(self) -> Dict[str, Any]:
        """
        Get default pricing data.
        
        For the composite estimator, this is an empty dictionary as pricing data
        is handled by the individual estimators.
        
        Returns:
            Dict[str, Any]: Empty dictionary.
        """
        return {}
    
    def estimate_cost(
        self, 
        usage: Dict[str, Any],
        time_period: Optional[Dict[str, datetime.datetime]] = None
    ) -> Dict[str, Any]:
        """
        Estimate the total cost by combining the results of multiple estimators.
        
        Args:
            usage (Dict[str, Any]): Usage data for multiple services and infrastructure.
                The keys should correspond to the estimator types.
                Expected format:
                {
                    "openai": {...},  # OpenAI usage data
                    "anthropic": {...},  # Anthropic usage data
                    "cloud": {...}  # Cloud infrastructure usage data
                }
            time_period (Optional[Dict[str, datetime.datetime]]): Time period for the cost estimation.
                Should contain 'start' and 'end' keys with datetime values.
                If not provided, the entire usage history will be considered.
        
        Returns:
            Dict[str, Any]: Total cost estimation result with detailed breakdown.
        """
        total_cost = 0.0
        component_costs = []
        
        # Process each estimator
        for estimator in self.estimators:
            # Get the usage data for this estimator type
            estimator_type = self._get_estimator_type(estimator)
            estimator_usage = usage.get(estimator_type, {})
            
            # Skip if no usage data for this estimator
            if not estimator_usage:
                continue
            
            # Estimate cost using this estimator
            cost_result = estimator.estimate_cost(estimator_usage, time_period)
            
            # Add to total cost
            component_cost = cost_result.get("total_cost", 0.0)
            total_cost += component_cost
            
            # Add to component costs
            component_costs.append({
                "type": estimator_type,
                "name": estimator.name,
                "total_cost": component_cost,
                "details": cost_result
            })
        
        # Prepare result
        time_info = {}
        if time_period:
            time_info = {
                "start_time": time_period.get("start", "").isoformat() if time_period.get("start") else None,
                "end_time": time_period.get("end", "").isoformat() if time_period.get("end") else None
            }
        
        result = {
            "name": self.name,
            "time_period": time_info,
            "total_cost": round(total_cost, 4),
            "currency": "USD",
            "components": component_costs
        }
        
        return result
    
    def _get_estimator_type(self, estimator: BaseCostEstimator) -> str:
        """
        Get the type of an estimator based on its class name.
        
        Args:
            estimator (BaseCostEstimator): The estimator instance.
        
        Returns:
            str: The estimator type as a lowercase string.
        """
        class_name = estimator.__class__.__name__
        
        # Remove "CostEstimator" suffix and convert to lowercase
        if class_name.endswith("CostEstimator"):
            estimator_type = class_name[:-12].lower()
        else:
            estimator_type = class_name.lower()
        
        return estimator_type
    
    def record_usage(self, usage_data: Dict[str, Any]) -> str:
        """
        Record usage data for later cost estimation and forward to individual estimators.
        
        Args:
            usage_data (Dict[str, Any]): Usage data to record.
                The keys should correspond to the estimator types.
        
        Returns:
            str: Unique ID for the usage record.
        """
        # Record usage data in the composite estimator
        record_id = super().record_usage(usage_data)
        
        # Forward usage data to individual estimators
        for estimator in self.estimators:
            estimator_type = self._get_estimator_type(estimator)
            if estimator_type in usage_data:
                estimator.record_usage(usage_data[estimator_type])
        
        return record_id
    
    def _aggregate_usage(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate usage data from multiple records.
        
        Args:
            records (List[Dict[str, Any]]): List of usage records.
        
        Returns:
            Dict[str, Any]: Aggregated usage data.
        """
        # Initialize result with empty dictionaries for each estimator type
        result = {}
        
        # Group records by estimator type
        grouped_records = {}
        
        for record in records:
            for key, value in record.items():
                if key not in ["id", "timestamp"]:
                    if key not in grouped_records:
                        grouped_records[key] = []
                    grouped_records[key].append(value)
        
        # Aggregate usage data for each estimator type
        for estimator in self.estimators:
            estimator_type = self._get_estimator_type(estimator)
            if estimator_type in grouped_records:
                result[estimator_type] = estimator._aggregate_usage(grouped_records[estimator_type])
        
        return result
    
    def add_estimator(self, estimator: BaseCostEstimator) -> None:
        """
        Add an estimator to the composite estimator.
        
        Args:
            estimator (BaseCostEstimator): The estimator to add.
        """
        self.estimators.append(estimator)
    
    def remove_estimator(self, estimator_name: str) -> bool:
        """
        Remove an estimator from the composite estimator.
        
        Args:
            estimator_name (str): The name of the estimator to remove.
        
        Returns:
            bool: True if the estimator was removed, False otherwise.
        """
        for i, estimator in enumerate(self.estimators):
            if estimator.name == estimator_name:
                self.estimators.pop(i)
                return True
        return False
    
    def get_estimator(self, estimator_name: str) -> Optional[BaseCostEstimator]:
        """
        Get an estimator by name.
        
        Args:
            estimator_name (str): The name of the estimator to get.
        
        Returns:
            Optional[BaseCostEstimator]: The estimator instance if found, None otherwise.
        """
        for estimator in self.estimators:
            if estimator.name == estimator_name:
                return estimator
        return None
    
    def reset_usage_records(self) -> None:
        """
        Reset all usage records in the composite estimator and its component estimators.
        """
        super().reset_usage_records()
        
        # Reset usage records in individual estimators
        for estimator in self.estimators:
            estimator.reset_usage_records() 