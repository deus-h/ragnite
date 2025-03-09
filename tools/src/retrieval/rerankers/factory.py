"""
Reranker Factory

This module provides the factory function for creating rerankers.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable

from .base_reranker import BaseReranker
from .cross_encoder_reranker import CrossEncoderReranker
from .mono_t5_reranker import MonoT5Reranker
from .llm_reranker import LLMReranker
from .ensemble_reranker import EnsembleReranker

# Configure logging
logger = logging.getLogger(__name__)


def get_reranker(
    reranker_type: str,
    model_name: Optional[str] = None,
    rerankers: Optional[List[BaseReranker]] = None,
    llm_provider: Optional[Callable] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseReranker:
    """
    Factory function for creating rerankers.
    
    Args:
        reranker_type: Type of reranker to create. Options:
            - 'cross_encoder': Reranker based on cross-encoder models.
            - 'mono_t5': Reranker based on MonoT5 models.
            - 'llm': Reranker based on large language models.
            - 'ensemble': Reranker that combines multiple rerankers.
        model_name: Name or path of the model to use for cross_encoder or mono_t5 rerankers.
        rerankers: List of reranker objects for ensemble reranker.
        llm_provider: Function that provides access to an LLM for the LLM reranker.
        config: Additional configuration options for the reranker.
        **kwargs: Additional keyword arguments.
    
    Returns:
        Configured reranker instance.
    
    Raises:
        ValueError: If the reranker type is not supported or if required parameters are missing.
    """
    config = config or {}
    
    if reranker_type == "cross_encoder":
        if model_name is None:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info(f"Using default cross-encoder model: {model_name}")
        
        return CrossEncoderReranker(
            model_name=model_name,
            device=kwargs.get("device"),
            batch_size=kwargs.get("batch_size", 16),
            max_length=kwargs.get("max_length", 512),
            config=config
        )
    
    elif reranker_type == "mono_t5":
        if model_name is None:
            model_name = "castorini/monot5-base-msmarco"
            logger.info(f"Using default MonoT5 model: {model_name}")
        
        return MonoT5Reranker(
            model_name=model_name,
            device=kwargs.get("device"),
            batch_size=kwargs.get("batch_size", 8),
            max_length=kwargs.get("max_length", 512),
            config=config
        )
    
    elif reranker_type == "llm":
        if llm_provider is None:
            # Check if we can use OpenAI or Anthropic
            if "provider" in kwargs:
                provider = kwargs.pop("provider")
                
                if provider == "openai":
                    logger.info("Using OpenAI provider for LLM reranker")
                    return LLMReranker.from_openai(
                        api_key=kwargs.pop("api_key", None),
                        model=kwargs.pop("model", "gpt-3.5-turbo"),
                        temperature=kwargs.pop("temperature", 0.0),
                        scoring_method=kwargs.pop("scoring_method", "direct"),
                        prompt_template=kwargs.pop("prompt_template", None),
                        batch_size=kwargs.pop("batch_size", 4),
                        config=config,
                        **kwargs
                    )
                elif provider == "anthropic":
                    logger.info("Using Anthropic provider for LLM reranker")
                    return LLMReranker.from_anthropic(
                        api_key=kwargs.pop("api_key", None),
                        model=kwargs.pop("model", "claude-3-haiku-20240307"),
                        temperature=kwargs.pop("temperature", 0.0),
                        scoring_method=kwargs.pop("scoring_method", "direct"),
                        prompt_template=kwargs.pop("prompt_template", None),
                        batch_size=kwargs.pop("batch_size", 4),
                        config=config,
                        **kwargs
                    )
                else:
                    raise ValueError(f"Unknown provider: {provider}. "
                                     f"Supported providers are 'openai' and 'anthropic'.")
            else:
                raise ValueError("LLM reranker requires an llm_provider function or a provider parameter.")
        
        return LLMReranker(
            llm_provider=llm_provider,
            scoring_method=kwargs.get("scoring_method", "direct"),
            prompt_template=kwargs.get("prompt_template"),
            batch_size=kwargs.get("batch_size", 4),
            config=config
        )
    
    elif reranker_type == "ensemble":
        if rerankers is None or len(rerankers) == 0:
            raise ValueError("Ensemble reranker requires a list of rerankers.")
        
        return EnsembleReranker(
            rerankers=rerankers,
            weights=kwargs.get("weights"),
            combination_method=kwargs.get("combination_method", "weighted_average"),
            config=config
        )
    
    else:
        raise ValueError(f"Unsupported reranker type: {reranker_type}. "
                        f"Supported types are: 'cross_encoder', 'mono_t5', 'llm', 'ensemble'.") 