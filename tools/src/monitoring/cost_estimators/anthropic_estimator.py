#!/usr/bin/env python3
"""
Cost estimator for Anthropic API usage in RAG systems.

This module provides a cost estimator for calculating and tracking the costs
associated with using Anthropic's API services in RAG systems.
"""

import datetime
from typing import Dict, List, Any, Optional, Union
import json
from .base import BaseCostEstimator


class AnthropicCostEstimator(BaseCostEstimator):
    """
    Cost estimator for Anthropic API usage.
    
    This estimator calculates and tracks the costs associated with using
    Anthropic's API services in RAG systems, including different Claude models.
    
    Attributes:
        name (str): Name of the estimator.
        config (Dict[str, Any]): Configuration options for the estimator.
        price_data (Dict[str, Any]): Pricing data for Anthropic API services.
    """
    
    def _get_default_price_data(self) -> Dict[str, Any]:
        """
        Get default pricing data for Anthropic API services.
        
        Returns:
            Dict[str, Any]: Default pricing data for Anthropic API services.
        """
        # Default pricing data as of March 2025
        # Prices are in USD per 1M tokens
        return {
            "claude-3.7-sonnet": {
                "input": 3.00,
                "output": 15.00
            },
            "claude-3.5-sonnet": {
                "input": 3.00,
                "output": 15.00
            },
            "claude-3.5-haiku": {
                "input": 0.80,
                "output": 4.00
            },
            "claude-3-opus": {
                "input": 15.00,
                "output": 75.00
            },
            "claude-3-sonnet": {
                "input": 3.00,
                "output": 15.00
            },
            "claude-3-haiku": {
                "input": 0.25,
                "output": 1.25
            },
            "claude-2.1": {
                "input": 8.00,
                "output": 24.00
            },
            "claude-2": {
                "input": 8.00,
                "output": 24.00
            },
            "claude-instant-1.2": {
                "input": 1.63,
                "output": 5.51
            },
            "claude-instant-1": {
                "input": 1.63,
                "output": 5.51
            }
        }
    
    def estimate_cost(
        self, 
        usage: Dict[str, Any],
        time_period: Optional[Dict[str, datetime.datetime]] = None
    ) -> Dict[str, Any]:
        """
        Estimate the cost based on Anthropic API usage.
        
        Args:
            usage (Dict[str, Any]): Usage data for Anthropic API services.
                Expected format:
                {
                    "model": str,
                    "input_tokens": int,
                    "output_tokens": int,
                    "requests": int  # Optional for request-based tracking
                }
            time_period (Optional[Dict[str, datetime.datetime]]): Time period for the cost estimation.
                Should contain 'start' and 'end' keys with datetime values.
                If not provided, the entire usage history will be considered.
        
        Returns:
            Dict[str, Any]: Cost estimation result with detailed breakdown.
        """
        total_cost = 0.0
        items = []
        
        model = usage.get("model", "claude-3-sonnet")
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        requests = usage.get("requests", 0)
        
        # Get model pricing
        model_pricing = self.price_data.get(model)
        if not model_pricing:
            # Use default pricing if model not found
            model_pricing = self.price_data["claude-3-sonnet"]
        
        # Calculate costs
        input_cost = (input_tokens / 1000000) * model_pricing["input"]
        output_cost = (output_tokens / 1000000) * model_pricing["output"]
        total_model_cost = input_cost + output_cost
        total_cost = total_model_cost
        
        # Add to items
        items.append({
            "service": "Anthropic",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "requests": requests,
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total_cost": round(total_model_cost, 4)
        })
        
        # Prepare result
        time_info = {}
        if time_period:
            time_info = {
                "start_time": time_period.get("start", "").isoformat() if time_period.get("start") else None,
                "end_time": time_period.get("end", "").isoformat() if time_period.get("end") else None
            }
        
        result = {
            "service": "Anthropic",
            "time_period": time_info,
            "total_cost": round(total_cost, 4),
            "currency": "USD",
            "items": items
        }
        
        return result
    
    def _aggregate_usage(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate Anthropic API usage data from multiple records.
        
        Args:
            records (List[Dict[str, Any]]): List of usage records.
        
        Returns:
            Dict[str, Any]: Aggregated usage data.
        """
        result = {
            "model": "claude-3-sonnet",  # Will be overridden if present in records
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0
        }
        
        # Track models with highest usage
        model_usage = {}
        
        for record in records:
            # Skip non-Anthropic records
            if "model" not in record:
                continue
                
            model = record.get("model", "claude-3-sonnet")
            input_tokens = record.get("input_tokens", 0)
            output_tokens = record.get("output_tokens", 0)
            requests = record.get("requests", 0)
            
            result["input_tokens"] += input_tokens
            result["output_tokens"] += output_tokens
            result["requests"] += requests
            
            # Track model usage
            if model not in model_usage:
                model_usage[model] = 0
            model_usage[model] += input_tokens + output_tokens
        
        # Set the most used model
        if model_usage:
            most_used_model = max(model_usage.items(), key=lambda x: x[1])[0]
            result["model"] = most_used_model
        
        return result 