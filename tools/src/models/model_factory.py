"""
Model Factory

This module provides factory functions for creating model provider instances.
"""

import logging
import os
from typing import Dict, Any, Optional, Union

from .base_model import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .local_model_provider import LocalModelProvider
from .xai_provider import XAIProvider
from .google_provider import GoogleAIProvider

# Configure logging
logger = logging.getLogger(__name__)


def get_model_provider(
    provider_type: str,
    **kwargs
) -> LLMProvider:
    """
    Factory function to create a model provider.
    
    Args:
        provider_type: Type of provider to create:
            - "openai": OpenAI provider (GPT-4, GPT-3.5)
            - "anthropic": Anthropic provider (Claude)
            - "local": Local model provider (Llama, Mistral)
            - "xai": xAI provider (Grok)
            - "google": Google AI provider (Gemini)
            - Or a path to a local model
        **kwargs: Provider-specific parameters
            
    Common parameters:
        api_key: API key for cloud providers
        model: Model name (default varies by provider)
            
    OpenAI provider:
        embedding_model: Model to use for embeddings (default: "text-embedding-3-small")
        organization: OpenAI organization ID
        
    Anthropic provider:
        No specific parameters beyond api_key and model
    
    xAI provider:
        No specific parameters beyond api_key and model
        
    Google AI provider:
        project_id: Google Cloud project ID (optional)
        
    Local model provider:
        model_name_or_path: Path to local model or HuggingFace model name (required if not in provider_type)
        embedding_model: Model to use for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")
        device: Device to run the model on (default: CUDA if available, else CPU)
        load_in_8bit: Whether to load the model in 8-bit precision (default: False)
        load_in_4bit: Whether to load the model in 4-bit precision (default: False)
            
    Returns:
        An instance of the specified provider
        
    Raises:
        ValueError: If provider_type is not supported or for invalid configurations
        ImportError: If required dependencies are not installed
    """
    provider_type = provider_type.lower()
    
    # Check if provider_type is a path to a local model
    if os.path.exists(provider_type):
        logger.info(f"Provider type '{provider_type}' is a path, using LocalModelProvider")
        return LocalModelProvider(model_name_or_path=provider_type, **kwargs)
    
    # Create the appropriate provider based on type
    if provider_type == "openai":
        try:
            return OpenAIProvider(**kwargs)
        except ImportError as e:
            logger.error(f"Could not create OpenAI provider: {e}")
            raise
    
    elif provider_type == "anthropic":
        try:
            return AnthropicProvider(**kwargs)
        except ImportError as e:
            logger.error(f"Could not create Anthropic provider: {e}")
            raise
    
    elif provider_type == "xai":
        try:
            return XAIProvider(**kwargs)
        except ImportError as e:
            logger.error(f"Could not create xAI provider: {e}")
            raise
    
    elif provider_type == "google":
        try:
            return GoogleAIProvider(**kwargs)
        except ImportError as e:
            logger.error(f"Could not create Google AI provider: {e}")
            raise
    
    elif provider_type == "local":
        try:
            return LocalModelProvider(**kwargs)
        except ImportError as e:
            logger.error(f"Could not create Local model provider: {e}")
            raise
    
    else:
        # Try to infer provider type from model name
        if provider_type.startswith(("gpt-", "text-embedding-", "dall-e")):
            logger.info(f"Model name '{provider_type}' looks like an OpenAI model")
            try:
                return OpenAIProvider(model=provider_type, **kwargs)
            except ImportError as e:
                logger.error(f"Could not create OpenAI provider: {e}")
                raise
                
        elif provider_type.startswith("claude"):
            logger.info(f"Model name '{provider_type}' looks like an Anthropic model")
            try:
                return AnthropicProvider(model=provider_type, **kwargs)
            except ImportError as e:
                logger.error(f"Could not create Anthropic provider: {e}")
                raise
        
        elif provider_type.startswith("grok"):
            logger.info(f"Model name '{provider_type}' looks like an xAI model")
            try:
                return XAIProvider(model=provider_type, **kwargs)
            except ImportError as e:
                logger.error(f"Could not create xAI provider: {e}")
                raise
                
        elif provider_type.startswith("gemini"):
            logger.info(f"Model name '{provider_type}' looks like a Google AI model")
            try:
                return GoogleAIProvider(model=provider_type, **kwargs)
            except ImportError as e:
                logger.error(f"Could not create Google AI provider: {e}")
                raise
                
        elif any(name in provider_type for name in ["llama", "mistral", "mixtral"]):
            logger.info(f"Model name '{provider_type}' looks like a local/HuggingFace model")
            try:
                return LocalModelProvider(model_name_or_path=provider_type, **kwargs)
            except ImportError as e:
                logger.error(f"Could not create Local model provider: {e}")
                raise
                
        else:
            supported_providers = ["openai", "anthropic", "xai", "google", "local"]
            raise ValueError(
                f"Unsupported provider type: {provider_type}. "
                f"Supported types: {supported_providers} or a path to a local model"
            )


def get_provider_for_model(model_name: str, **kwargs) -> LLMProvider:
    """
    Factory function to create a model provider appropriate for the given model.
    
    Args:
        model_name: Name of the model to use
        **kwargs: Additional parameters to pass to the provider
            
    Returns:
        An instance of the appropriate provider for the model
    """
    model_name = model_name.lower()
    
    # Determine the provider based on the model name
    if model_name.startswith(("gpt-", "text-embedding-", "dall-e")):
        provider_type = "openai"
    elif model_name.startswith("claude"):
        provider_type = "anthropic"
    else:
        # Default to local for other models
        provider_type = "local"
    
    return get_model_provider(provider_type, model=model_name, **kwargs)


def create_smart_router(
    providers: Dict[str, LLMProvider],
    default_provider: str,
    routing_rules: Optional[Dict[str, Any]] = None
) -> 'SmartModelRouter':
    """
    Create a smart router that can route requests to different model providers based on rules.
    
    Args:
        providers: Dictionary mapping provider names to LLMProvider instances
        default_provider: Name of the default provider to use
        routing_rules: Optional dictionary of routing rules
            
    Returns:
        A SmartModelRouter instance
        
    Note: 
        This is just a stub for now. The SmartModelRouter class will be implemented in a separate module.
    """
    raise NotImplementedError("SmartModelRouter is not yet implemented") 