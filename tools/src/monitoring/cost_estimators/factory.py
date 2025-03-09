#!/usr/bin/env python3
"""
Factory function for cost estimators in RAG systems.

This module provides a factory function for creating cost estimator instances
based on the specified type and configuration.
"""

from typing import Dict, List, Any, Optional, Union
import datetime
from .base import BaseCostEstimator
from .openai_estimator import OpenAICostEstimator
from .anthropic_estimator import AnthropicCostEstimator
from .grok_estimator import GrokCostEstimator
from .cloud_estimator import CloudCostEstimator
from .composite_estimator import CompositeCostEstimator


def get_cost_estimator(
    estimator_type: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseCostEstimator:
    """
    Create a cost estimator instance based on the specified type and configuration.
    
    Args:
        estimator_type (str): Type of cost estimator to create.
            Valid values: 'openai', 'anthropic', 'grok', 'cloud', 'composite'.
        name (Optional[str]): Name of the estimator.
            If not provided, a default name will be used based on the estimator type.
        config (Optional[Dict[str, Any]]): Configuration options for the estimator.
            Defaults to an empty dictionary.
        **kwargs: Additional keyword arguments specific to each estimator type:
            - For 'cloud': cloud_provider (str) - Cloud provider. Valid values: 'aws', 'gcp', 'azure'.
            - For 'composite': estimators (List[BaseCostEstimator]) - List of cost estimators to combine.
            
    Returns:
        BaseCostEstimator: Cost estimator instance.
        
    Raises:
        ValueError: If an unsupported estimator type is specified or if required arguments are missing.
    """
    # Normalize estimator type
    estimator_type = estimator_type.lower().strip()
    
    # Set default name if not provided
    if name is None:
        name = f"{estimator_type}_cost_estimator"
    
    # Set default config if not provided
    if config is None:
        config = {}
    
    # Create estimator based on type
    if estimator_type == "openai":
        return OpenAICostEstimator(name=name, config=config)
    
    elif estimator_type == "anthropic":
        return AnthropicCostEstimator(name=name, config=config)
    
    elif estimator_type in ["grok", "xai"]:
        return GrokCostEstimator(name=name, config=config)
    
    elif estimator_type == "cloud":
        # Get cloud provider from kwargs
        cloud_provider = kwargs.get("cloud_provider", "aws")
        return CloudCostEstimator(name=name, cloud_provider=cloud_provider, config=config)
    
    elif estimator_type == "composite":
        # Get estimators from kwargs
        estimators = kwargs.get("estimators")
        if not estimators:
            raise ValueError("Estimators list is required for composite estimator.")
        return CompositeCostEstimator(name=name, estimators=estimators, config=config)
    
    else:
        valid_types = ["openai", "anthropic", "grok", "xai", "cloud", "composite"]
        raise ValueError(
            f"Unsupported estimator type: '{estimator_type}'. "
            f"Valid types are: {', '.join(valid_types)}."
        )


def create_estimator_from_config(config: Dict[str, Any]) -> BaseCostEstimator:
    """
    Create a cost estimator instance from a configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary with the following structure:
            {
                "type": str,  # Type of estimator (required)
                "name": str,  # Name of estimator (optional)
                "config": Dict[str, Any],  # Estimator configuration (optional)
                "cloud_provider": str,  # Cloud provider for cloud estimator (optional)
                "estimators": List[Dict[str, Any]]  # Estimator configs for composite estimator (optional)
            }
    
    Returns:
        BaseCostEstimator: Cost estimator instance.
    
    Raises:
        ValueError: If the configuration is invalid or incomplete.
    """
    # Verify required fields
    if "type" not in config:
        raise ValueError("Estimator type is required in configuration.")
    
    estimator_type = config["type"]
    name = config.get("name")
    estimator_config = config.get("config", {})
    
    # Handle specific arguments for different estimator types
    kwargs = {}
    
    if estimator_type == "cloud":
        kwargs["cloud_provider"] = config.get("cloud_provider", "aws")
    
    elif estimator_type == "composite":
        if "estimators" not in config:
            raise ValueError("Estimators list is required for composite estimator in configuration.")
        
        # Recursively create child estimators
        child_estimators = []
        for child_config in config["estimators"]:
            child_estimator = create_estimator_from_config(child_config)
            child_estimators.append(child_estimator)
        
        kwargs["estimators"] = child_estimators
    
    # Create and return the estimator
    return get_cost_estimator(
        estimator_type=estimator_type,
        name=name,
        config=estimator_config,
        **kwargs
    )


def create_default_composite_estimator(name: str = "default_cost_estimator") -> CompositeCostEstimator:
    """
    Create a default composite estimator with standard estimators for RAG systems.
    
    This function creates a composite estimator that combines OpenAI, Anthropic,
    Grok, and AWS cloud estimators, which are commonly used in RAG systems.
    
    Args:
        name (str): Name of the composite estimator. Defaults to "default_cost_estimator".
    
    Returns:
        CompositeCostEstimator: Default composite estimator for RAG systems.
    """
    # Create individual estimators
    openai_estimator = OpenAICostEstimator(name="openai_cost_estimator")
    anthropic_estimator = AnthropicCostEstimator(name="anthropic_cost_estimator")
    grok_estimator = GrokCostEstimator(name="grok_cost_estimator")
    cloud_estimator = CloudCostEstimator(name="aws_cost_estimator", cloud_provider="aws")
    
    # Create and return composite estimator
    return CompositeCostEstimator(
        name=name,
        estimators=[openai_estimator, anthropic_estimator, grok_estimator, cloud_estimator]
    ) 