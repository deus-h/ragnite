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
from .cohere_reranker import CohereReranker
from .cascade_reranker import CascadeReranker

# Configure logging
logger = logging.getLogger(__name__)


def get_reranker(
    reranker_type: str,
    model_name: Optional[str] = None,
    rerankers: Optional[List[BaseReranker]] = None,
    llm_provider: Optional[Callable] = None,
    api_key: Optional[str] = None,
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
            - 'cohere': Reranker using Cohere's Rerank API.
            - 'cascade': Multi-stage cascade reranking pipeline.
        model_name: Name or path of the model to use for cross_encoder, mono_t5, or cohere rerankers.
        rerankers: List of reranker objects for ensemble or cascade rerankers.
        llm_provider: Function that provides access to an LLM for the LLM reranker.
        api_key: API key for services like Cohere.
        config: Additional configuration options for the reranker.
        **kwargs: Additional keyword arguments.
    
    Returns:
        Configured reranker instance.
    
    Raises:
        ValueError: If the reranker type is not supported or if required parameters are missing.
    """
    config = config or {}
    reranker_type = reranker_type.lower()
    
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
            raise ValueError("LLM provider is required for LLM reranker")
        
        return LLMReranker(
            llm_provider=llm_provider,
            config=config
        )
    
    elif reranker_type == "ensemble":
        if not rerankers:
            raise ValueError("List of rerankers is required for ensemble reranker")
        
        return EnsembleReranker(
            rerankers=rerankers,
            config=config
        )
    
    elif reranker_type == "cohere":
        if api_key is None:
            raise ValueError("API key is required for Cohere reranker")
        
        # Use the provided model name or default to the English rerank model
        cohere_model = model_name or "rerank-english-v2.0"
        
        return CohereReranker(
            api_key=api_key,
            model=cohere_model,
            config=config
        )
    
    elif reranker_type == "cascade":
        if not rerankers:
            raise ValueError("List of rerankers is required for cascade reranker")
        
        return CascadeReranker(
            rerankers=rerankers,
            config=config
        )
    
    else:
        valid_types = ["cross_encoder", "mono_t5", "llm", "ensemble", "cohere", "cascade"]
        raise ValueError(f"Unsupported reranker type: {reranker_type}. Valid types: {valid_types}") 