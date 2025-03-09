#!/usr/bin/env python3
"""
Cost estimator for xAI's Grok API usage in RAG systems.

This module provides a cost estimator for calculating and tracking the costs
associated with using xAI's Grok API services in RAG systems.
"""

import datetime
from typing import Dict, List, Any, Optional, Union
import json
from .base import BaseCostEstimator


class GrokCostEstimator(BaseCostEstimator):
    """
    Cost estimator for xAI's Grok API usage.
    
    This estimator calculates and tracks the costs associated with using
    xAI's Grok API services in RAG systems, including different Grok models.
    
    Attributes:
        name (str): Name of the estimator.
        config (Dict[str, Any]): Configuration options for the estimator.
        price_data (Dict[str, Any]): Pricing data for Grok API services.
    """
    
    def _get_default_price_data(self) -> Dict[str, Any]:
        """
        Get default pricing data for xAI's Grok API services.
        
        Returns:
            Dict[str, Any]: Default pricing data for Grok API services.
        """
        # Default pricing data as of March 2025
        # Prices are in USD per 1M tokens
        return {
            # Grok models
            "grok-3": {
                "input": 2.00,
                "output": 10.00
            },
            "grok-3-thinking": {
                "input": 5.00,
                "output": 25.00
            },
            "grok-3-reasoning": {
                "input": 10.00,
                "output": 30.00
            },
            "grok-2": {
                "input": 2.00,
                "output": 10.00
            },
            "grok-1.5": {
                "input": 0.70,
                "output": 3.50
            },
            "grok-1": {
                "input": 0.50,
                "output": 1.50
            },
            # Embedding models
            "grok-embedding": 0.10,
            "grok-embedding-large": 0.30,
            # Vision models
            "grok-vision": {
                "input_text": 2.00,
                "output": 10.00,
                "input_image": {
                    "standard": 0.02,  # Per image
                    "high_res": 0.05   # Per image
                }
            }
        }
    
    def estimate_cost(
        self, 
        usage: Dict[str, Any],
        time_period: Optional[Dict[str, datetime.datetime]] = None
    ) -> Dict[str, Any]:
        """
        Estimate the cost based on Grok API usage.
        
        Args:
            usage (Dict[str, Any]): Usage data for Grok API services.
                Expected format:
                {
                    "completion": {
                        "model": str,
                        "input_tokens": int,
                        "output_tokens": int
                    },
                    "embedding": {
                        "model": str,
                        "tokens": int
                    },
                    "vision": {
                        "model": str,
                        "input_text_tokens": int,
                        "output_tokens": int,
                        "input_images": {
                            "standard": int,
                            "high_res": int
                        }
                    }
                }
            time_period (Optional[Dict[str, datetime.datetime]]): Time period for the cost estimation.
                Should contain 'start' and 'end' keys with datetime values.
                If not provided, the entire usage history will be considered.
        
        Returns:
            Dict[str, Any]: Cost estimation result with detailed breakdown.
        """
        total_cost = 0.0
        items = []
        
        # Process completion usage
        if "completion" in usage:
            completion_usage = usage["completion"]
            model = completion_usage.get("model", "grok-3")
            input_tokens = completion_usage.get("input_tokens", 0)
            output_tokens = completion_usage.get("output_tokens", 0)
            
            # Get pricing for the model
            model_pricing = self.price_data.get(model)
            if not model_pricing:
                # Use grok-3 as default if model not found
                model_pricing = self.price_data.get("grok-3", {"input": 2.00, "output": 10.00})
            
            # Calculate costs
            input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
            output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
            total_model_cost = input_cost + output_cost
            
            # Add to total cost
            total_cost += total_model_cost
            
            # Add to items
            items.append({
                "service": "xAI Grok",
                "model": model,
                "type": "completion",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_model_cost, 6)
            })
        
        # Process embedding usage
        if "embedding" in usage:
            embedding_usage = usage["embedding"]
            model = embedding_usage.get("model", "grok-embedding")
            tokens = embedding_usage.get("tokens", 0)
            
            # Get pricing for the model
            model_price = self.price_data.get(model)
            if not model_price:
                # Use default embedding model price if model not found
                model_price = self.price_data.get("grok-embedding", 0.10)
            
            # Calculate cost
            embedding_cost = (tokens / 1_000_000) * model_price
            
            # Add to total cost
            total_cost += embedding_cost
            
            # Add to items
            items.append({
                "service": "xAI Grok",
                "model": model,
                "type": "embedding",
                "tokens": tokens,
                "cost": round(embedding_cost, 6)
            })
        
        # Process vision usage
        if "vision" in usage:
            vision_usage = usage["vision"]
            model = vision_usage.get("model", "grok-vision")
            input_text_tokens = vision_usage.get("input_text_tokens", 0)
            output_tokens = vision_usage.get("output_tokens", 0)
            input_images = vision_usage.get("input_images", {"standard": 0, "high_res": 0})
            
            # Get pricing for the model
            vision_pricing = self.price_data.get(model)
            if not vision_pricing:
                # Use default vision model pricing if model not found
                vision_pricing = self.price_data.get("grok-vision", {
                    "input_text": 2.00,
                    "output": 10.00,
                    "input_image": {
                        "standard": 0.02,
                        "high_res": 0.05
                    }
                })
            
            # Calculate costs
            input_text_cost = (input_text_tokens / 1_000_000) * vision_pricing["input_text"]
            output_cost = (output_tokens / 1_000_000) * vision_pricing["output"]
            
            # Calculate image costs
            standard_image_cost = input_images.get("standard", 0) * vision_pricing["input_image"]["standard"]
            high_res_image_cost = input_images.get("high_res", 0) * vision_pricing["input_image"]["high_res"]
            image_cost = standard_image_cost + high_res_image_cost
            
            # Calculate total vision cost
            total_vision_cost = input_text_cost + output_cost + image_cost
            
            # Add to total cost
            total_cost += total_vision_cost
            
            # Add to items
            items.append({
                "service": "xAI Grok",
                "model": model,
                "type": "vision",
                "input_text_tokens": input_text_tokens,
                "output_tokens": output_tokens,
                "standard_images": input_images.get("standard", 0),
                "high_res_images": input_images.get("high_res", 0),
                "input_text_cost": round(input_text_cost, 6),
                "output_cost": round(output_cost, 6),
                "image_cost": round(image_cost, 6),
                "total_cost": round(total_vision_cost, 6)
            })
        
        # Prepare result
        result = {
            "total_cost": round(total_cost, 6),
            "currency": "USD",
            "provider": "xAI",
            "items": items
        }
        
        # Add time period if provided
        if time_period and "start" in time_period and "end" in time_period:
            result["time_period"] = {
                "start": time_period["start"].isoformat(),
                "end": time_period["end"].isoformat()
            }
        
        return result
    
    def _aggregate_usage(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate usage data from multiple records.
        
        Args:
            records (List[Dict[str, Any]]): List of usage records.
        
        Returns:
            Dict[str, Any]: Aggregated usage data.
        """
        # Initialize aggregated usage
        aggregated = {
            "completion": {},
            "embedding": {},
            "vision": {}
        }
        
        # Process each record
        for record in records:
            # Skip id and timestamp fields
            if "id" in record:
                del record["id"]
            if "timestamp" in record:
                del record["timestamp"]
            
            # Aggregate completion usage
            if "completion" in record:
                completion = record["completion"]
                model = completion.get("model")
                
                if model not in aggregated["completion"]:
                    aggregated["completion"][model] = {
                        "model": model,
                        "input_tokens": 0,
                        "output_tokens": 0
                    }
                
                aggregated["completion"][model]["input_tokens"] += completion.get("input_tokens", 0)
                aggregated["completion"][model]["output_tokens"] += completion.get("output_tokens", 0)
            
            # Aggregate embedding usage
            if "embedding" in record:
                embedding = record["embedding"]
                model = embedding.get("model")
                
                if model not in aggregated["embedding"]:
                    aggregated["embedding"][model] = {
                        "model": model,
                        "tokens": 0
                    }
                
                aggregated["embedding"][model]["tokens"] += embedding.get("tokens", 0)
            
            # Aggregate vision usage
            if "vision" in record:
                vision = record["vision"]
                model = vision.get("model")
                
                if model not in aggregated["vision"]:
                    aggregated["vision"][model] = {
                        "model": model,
                        "input_text_tokens": 0,
                        "output_tokens": 0,
                        "input_images": {
                            "standard": 0,
                            "high_res": 0
                        }
                    }
                
                aggregated["vision"][model]["input_text_tokens"] += vision.get("input_text_tokens", 0)
                aggregated["vision"][model]["output_tokens"] += vision.get("output_tokens", 0)
                
                if "input_images" in vision:
                    aggregated["vision"][model]["input_images"]["standard"] += vision["input_images"].get("standard", 0)
                    aggregated["vision"][model]["input_images"]["high_res"] += vision["input_images"].get("high_res", 0)
        
        # Flatten the aggregated data
        result = {}
        
        # Add completion usage
        completion_models = list(aggregated["completion"].values())
        if completion_models:
            if len(completion_models) == 1:
                result["completion"] = completion_models[0]
            else:
                # Multiple models, create separate entries
                for i, model_data in enumerate(completion_models):
                    result[f"completion_{i+1}"] = model_data
        
        # Add embedding usage
        embedding_models = list(aggregated["embedding"].values())
        if embedding_models:
            if len(embedding_models) == 1:
                result["embedding"] = embedding_models[0]
            else:
                # Multiple models, create separate entries
                for i, model_data in enumerate(embedding_models):
                    result[f"embedding_{i+1}"] = model_data
        
        # Add vision usage
        vision_models = list(aggregated["vision"].values())
        if vision_models:
            if len(vision_models) == 1:
                result["vision"] = vision_models[0]
            else:
                # Multiple models, create separate entries
                for i, model_data in enumerate(vision_models):
                    result[f"vision_{i+1}"] = model_data
        
        return result 