#!/usr/bin/env python3
"""
Cost estimator for OpenAI API usage in RAG systems.

This module provides a cost estimator for calculating and tracking the costs
associated with using OpenAI's API services in RAG systems.
"""

import datetime
from typing import Dict, List, Any, Optional, Union
import json
from .base import BaseCostEstimator


class OpenAICostEstimator(BaseCostEstimator):
    """
    Cost estimator for OpenAI API usage.
    
    This estimator calculates and tracks the costs associated with using
    OpenAI's API services in RAG systems, including different models for
    embeddings and completions.
    
    Attributes:
        name (str): Name of the estimator.
        config (Dict[str, Any]): Configuration options for the estimator.
        price_data (Dict[str, Any]): Pricing data for OpenAI API services.
    """
    
    def _get_default_price_data(self) -> Dict[str, Any]:
        """
        Get default pricing data for OpenAI API services.
        
        Returns:
            Dict[str, Any]: Default pricing data for OpenAI API services.
        """
        # Default pricing data as of March 2025
        # Prices are in USD per 1K tokens
        return {
            "completion_models": {
                "gpt-4o": {
                    "input": 2.5,
                    "output": 10.0
                },
                "gpt-4o-mini": {
                    "input": 0.15,
                    "output": 0.6
                },
                "gpt-4o-realtime-preview": {
                    "input": 5.0,
                    "output": 20.0
                },
                "gpt-4o-audio-preview": {
                    "input": 2.5,
                    "output": 10.0
                },
                "gpt-4o-mini-audio-preview": {
                    "input": 0.15,
                    "output": 0.6
                },
                "gpt-4o-mini-realtime-preview": {
                    "input": 0.6,
                    "output": 2.4
                },
                "gpt-4.5-preview": {
                    "input": 75.0,
                    "output": 150.0
                },
                "o3-mini": {
                    "input": 1.1,
                    "output": 4.4
                },
                "o1": {
                    "input": 15.0,
                    "output": 60.0
                },
                "o1-preview": {
                    "input": 15.0,
                    "output": 60.0
                },
                "o1-mini": {
                    "input": 1.1,
                    "output": 4.4
                },
                "gpt-4-turbo-preview": {
                    "input": 10.0,
                    "output": 30.0
                },
                "gpt-3.5-turbo": {
                    "input": 0.5,
                    "output": 1.5
                },
                "gpt-4": {
                    "input": 30.0,
                    "output": 60.0
                }
            },
            "embedding_models": {
                "text-embedding-3-small": 0.02,
                "text-embedding-3-large": 0.13,
                "text-embedding-ada-002": 0.1
            },
            "vision_models": {
                "gpt-4o-audio-preview": {
                    "input_text": 2.5,
                    "output": 10.0,
                    "input_audio": 40.0
                },
                "gpt-4o-realtime-preview": {
                    "input_text": 5.0,
                    "output": 20.0,
                    "input_audio": 40.0,
                    "input_image": {
                        "low_res": 0.01,  # Per image
                        "high_res": 0.03  # Per image
                    }
                },
                "gpt-4o-mini-audio-preview": {
                    "input_text": 0.15,
                    "output": 0.6,
                    "input_audio": 10.0
                },
                "gpt-4o-mini-realtime-preview": {
                    "input_text": 0.6,
                    "output": 2.4,
                    "input_audio": 10.0,
                    "input_image": {
                        "low_res": 0.01,  # Per image
                        "high_res": 0.03  # Per image
                    }
                }
            },
            "fine_tuning": {
                "gpt-3.5-turbo": {
                    "training": 3.0,
                    "input": 0.5,
                    "output": 1.5
                },
                "gpt-4o": {
                    "training": 3.75,
                    "input": 3.75,
                    "output": 15.0
                },
                "gpt-4o-mini": {
                    "training": 0.3,
                    "input": 0.3,
                    "output": 1.2
                }
            }
        }
    
    def estimate_cost(
        self, 
        usage: Dict[str, Any],
        time_period: Optional[Dict[str, datetime.datetime]] = None
    ) -> Dict[str, Any]:
        """
        Estimate the cost based on OpenAI API usage.
        
        Args:
            usage (Dict[str, Any]): Usage data for OpenAI API services.
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
                            "low_res": int,
                            "high_res": int
                        }
                    },
                    "fine_tuning": {
                        "model": str,
                        "training_tokens": int,
                        "input_tokens": int,
                        "output_tokens": int
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
            model = completion_usage.get("model", "gpt-3.5-turbo")
            input_tokens = completion_usage.get("input_tokens", 0)
            output_tokens = completion_usage.get("output_tokens", 0)
            
            # Get model pricing
            model_pricing = self.price_data["completion_models"].get(model)
            if not model_pricing:
                # Use default pricing if model not found
                model_pricing = self.price_data["completion_models"]["gpt-3.5-turbo"]
            
            # Calculate costs
            input_cost = (input_tokens / 1000) * model_pricing["input"]
            output_cost = (output_tokens / 1000) * model_pricing["output"]
            completion_cost = input_cost + output_cost
            total_cost += completion_cost
            
            # Add to items
            items.append({
                "service": "OpenAI",
                "category": "Completion",
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(completion_cost, 4)
            })
        
        # Process embedding usage
        if "embedding" in usage:
            embedding_usage = usage["embedding"]
            model = embedding_usage.get("model", "text-embedding-ada-002")
            tokens = embedding_usage.get("tokens", 0)
            
            # Get model pricing
            model_price = self.price_data["embedding_models"].get(model, 0.10)  # Default to ada-002 price
            
            # Calculate cost
            embedding_cost = (tokens / 1000) * model_price
            total_cost += embedding_cost
            
            # Add to items
            items.append({
                "service": "OpenAI",
                "category": "Embedding",
                "model": model,
                "tokens": tokens,
                "total_cost": round(embedding_cost, 4)
            })
        
        # Process vision usage
        if "vision" in usage:
            vision_usage = usage["vision"]
            model = vision_usage.get("model", "gpt-4-vision")
            input_text_tokens = vision_usage.get("input_text_tokens", 0)
            output_tokens = vision_usage.get("output_tokens", 0)
            input_images = vision_usage.get("input_images", {"low_res": 0, "high_res": 0})
            
            # Get model pricing
            model_pricing = self.price_data["vision_models"].get(model)
            if not model_pricing:
                # Use default pricing if model not found
                model_pricing = self.price_data["vision_models"]["gpt-4-vision"]
            
            # Calculate costs
            input_text_cost = (input_text_tokens / 1000) * model_pricing["input_text"]
            output_cost = (output_tokens / 1000) * model_pricing["output"]
            image_cost = (
                input_images.get("low_res", 0) * model_pricing["input_image"]["low_res"] +
                input_images.get("high_res", 0) * model_pricing["input_image"]["high_res"]
            )
            vision_cost = input_text_cost + output_cost + image_cost
            total_cost += vision_cost
            
            # Add to items
            items.append({
                "service": "OpenAI",
                "category": "Vision",
                "model": model,
                "input_text_tokens": input_text_tokens,
                "output_tokens": output_tokens,
                "input_images": input_images,
                "input_text_cost": round(input_text_cost, 4),
                "output_cost": round(output_cost, 4),
                "image_cost": round(image_cost, 4),
                "total_cost": round(vision_cost, 4)
            })
        
        # Process fine-tuning usage
        if "fine_tuning" in usage:
            fine_tuning_usage = usage["fine_tuning"]
            model = fine_tuning_usage.get("model", "gpt-3.5-turbo")
            training_tokens = fine_tuning_usage.get("training_tokens", 0)
            input_tokens = fine_tuning_usage.get("input_tokens", 0)
            output_tokens = fine_tuning_usage.get("output_tokens", 0)
            
            # Get model pricing
            model_pricing = self.price_data["fine_tuning"].get(model)
            if not model_pricing:
                # Use default pricing if model not found
                model_pricing = self.price_data["fine_tuning"]["gpt-3.5-turbo"]
            
            # Calculate costs
            training_cost = (training_tokens / 1000) * model_pricing["training"]
            input_cost = (input_tokens / 1000) * model_pricing["input"]
            output_cost = (output_tokens / 1000) * model_pricing["output"]
            fine_tuning_cost = training_cost + input_cost + output_cost
            total_cost += fine_tuning_cost
            
            # Add to items
            items.append({
                "service": "OpenAI",
                "category": "Fine-tuning",
                "model": model,
                "training_tokens": training_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "training_cost": round(training_cost, 4),
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(fine_tuning_cost, 4)
            })
        
        # Prepare result
        time_info = {}
        if time_period:
            time_info = {
                "start_time": time_period.get("start", "").isoformat() if time_period.get("start") else None,
                "end_time": time_period.get("end", "").isoformat() if time_period.get("end") else None
            }
        
        result = {
            "service": "OpenAI",
            "time_period": time_info,
            "total_cost": round(total_cost, 4),
            "currency": "USD",
            "items": items
        }
        
        return result
    
    def _aggregate_usage(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate OpenAI API usage data from multiple records.
        
        Args:
            records (List[Dict[str, Any]]): List of usage records.
        
        Returns:
            Dict[str, Any]: Aggregated usage data.
        """
        result = {
            "completion": {
                "model": "gpt-3.5-turbo",  # Will be overridden if present in records
                "input_tokens": 0,
                "output_tokens": 0
            },
            "embedding": {
                "model": "text-embedding-ada-002",  # Will be overridden if present in records
                "tokens": 0
            },
            "vision": {
                "model": "gpt-4-vision",  # Will be overridden if present in records
                "input_text_tokens": 0,
                "output_tokens": 0,
                "input_images": {
                    "low_res": 0,
                    "high_res": 0
                }
            },
            "fine_tuning": {
                "model": "gpt-3.5-turbo",  # Will be overridden if present in records
                "training_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }
        }
        
        # Track models with highest usage for each category
        model_usage = {
            "completion": {},
            "embedding": {},
            "vision": {},
            "fine_tuning": {}
        }
        
        for record in records:
            # Process completion usage
            if "completion" in record:
                completion = record["completion"]
                model = completion.get("model", "gpt-3.5-turbo")
                input_tokens = completion.get("input_tokens", 0)
                output_tokens = completion.get("output_tokens", 0)
                
                result["completion"]["input_tokens"] += input_tokens
                result["completion"]["output_tokens"] += output_tokens
                
                # Track model usage
                if model not in model_usage["completion"]:
                    model_usage["completion"][model] = 0
                model_usage["completion"][model] += input_tokens + output_tokens
            
            # Process embedding usage
            if "embedding" in record:
                embedding = record["embedding"]
                model = embedding.get("model", "text-embedding-ada-002")
                tokens = embedding.get("tokens", 0)
                
                result["embedding"]["tokens"] += tokens
                
                # Track model usage
                if model not in model_usage["embedding"]:
                    model_usage["embedding"][model] = 0
                model_usage["embedding"][model] += tokens
            
            # Process vision usage
            if "vision" in record:
                vision = record["vision"]
                model = vision.get("model", "gpt-4-vision")
                input_text_tokens = vision.get("input_text_tokens", 0)
                output_tokens = vision.get("output_tokens", 0)
                input_images = vision.get("input_images", {"low_res": 0, "high_res": 0})
                
                result["vision"]["input_text_tokens"] += input_text_tokens
                result["vision"]["output_tokens"] += output_tokens
                result["vision"]["input_images"]["low_res"] += input_images.get("low_res", 0)
                result["vision"]["input_images"]["high_res"] += input_images.get("high_res", 0)
                
                # Track model usage
                if model not in model_usage["vision"]:
                    model_usage["vision"][model] = 0
                model_usage["vision"][model] += input_text_tokens + output_tokens
            
            # Process fine-tuning usage
            if "fine_tuning" in record:
                fine_tuning = record["fine_tuning"]
                model = fine_tuning.get("model", "gpt-3.5-turbo")
                training_tokens = fine_tuning.get("training_tokens", 0)
                input_tokens = fine_tuning.get("input_tokens", 0)
                output_tokens = fine_tuning.get("output_tokens", 0)
                
                result["fine_tuning"]["training_tokens"] += training_tokens
                result["fine_tuning"]["input_tokens"] += input_tokens
                result["fine_tuning"]["output_tokens"] += output_tokens
                
                # Track model usage
                if model not in model_usage["fine_tuning"]:
                    model_usage["fine_tuning"][model] = 0
                model_usage["fine_tuning"][model] += training_tokens + input_tokens + output_tokens
        
        # Set the most used model for each category
        for category, models in model_usage.items():
            if models:
                most_used_model = max(models.items(), key=lambda x: x[1])[0]
                result[category]["model"] = most_used_model
        
        return result 