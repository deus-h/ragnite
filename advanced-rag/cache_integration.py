"""
Cache Integration Module

This module provides integration between RAG pipelines and the caching infrastructure.
It includes wrapper classes and utilities to easily add caching capabilities to
existing RAG components with minimal code changes.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Type
import os
import numpy as np
from pathlib import Path

try:
    # Try relative imports for when used inside the package
    from .caching.cache_manager import CacheManager
    from .caching.embedding_cache import EmbeddingCache
    from .caching.semantic_cache import SemanticQueryCache
    from .caching.result_cache import ResultCache
    from .caching.prompt_cache import PromptCache
    from .caching.cache_dashboard import CacheDashboard
except ImportError:
    # Fallback to direct imports for when used externally
    from caching.cache_manager import CacheManager
    from caching.embedding_cache import EmbeddingCache
    from caching.semantic_cache import SemanticQueryCache
    from caching.result_cache import ResultCache
    from caching.prompt_cache import PromptCache
    from caching.cache_dashboard import CacheDashboard

# Configure logging
logger = logging.getLogger(__name__)

class CachedEmbeddings:
    """
    A wrapper around embedding models that transparently uses the embedding cache.
    Compatible with LangChain's embedding interface.
    """
    
    def __init__(
        self,
        embedding_model,
        cache_dir: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        namespace: str = "default",
        ttl: int = 86400 * 30,  # 30 days default
    ):
        """
        Initialize the cached embeddings wrapper.
        
        Args:
            embedding_model: The base embedding model to wrap (e.g., OpenAIEmbeddings)
            cache_dir: Directory to store cache files
            cache_manager: Optional existing cache manager to use
            namespace: Cache namespace to use
            ttl: Time-to-live for cached embeddings in seconds
        """
        self.embedding_model = embedding_model
        
        # Set up cache manager if not provided
        if cache_manager is None:
            if cache_dir is None:
                cache_dir = os.path.join(os.path.expanduser("~"), ".ragnite", "cache")
            
            self.cache_manager = CacheManager(cache_dir=cache_dir)
        else:
            self.cache_manager = cache_manager
        
        # Create embedding cache
        self.embedding_cache = EmbeddingCache(
            cache_manager=self.cache_manager,
            namespace=namespace,
            ttl=ttl
        )
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents"""
        # This could be more sophisticated with batch retrieval, but for now we'll just call the sync version
        return self.embed_documents(texts)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents with caching.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embeddings as float lists
        """
        results = []
        cache_hits = 0
        
        for text in texts:
            cached_embedding = self.embedding_cache.get(text)
            
            if cached_embedding is not None:
                results.append(cached_embedding)
                cache_hits += 1
            else:
                # Get a single embedding from the base model
                single_result = self.embedding_model.embed_documents([text])[0]
                self.embedding_cache.set(text, single_result)
                results.append(single_result)
        
        hit_rate = cache_hits / len(texts) if texts else 0
        logger.debug(f"Embedding cache hit rate: {hit_rate:.2f} ({cache_hits}/{len(texts)})")
        
        return results
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query"""
        return self.embed_query(text)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query with caching.
        
        Args:
            text: Query text to embed
            
        Returns:
            Query embedding as float list
        """
        cached_embedding = self.embedding_cache.get(text, query=True)
        
        if cached_embedding is not None:
            logger.debug("Query embedding cache hit")
            return cached_embedding
        
        # Get the embedding from the base model
        result = self.embedding_model.embed_query(text)
        self.embedding_cache.set(text, result, query=True)
        
        return result


class CachedLLM:
    """
    A wrapper around LLM models that uses result cache and prompt cache.
    Compatible with LangChain's LLM interface.
    """
    
    def __init__(
        self,
        llm_model,
        cache_dir: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        result_namespace: str = "llm_results",
        prompt_namespace: str = "llm_prompts",
        ttl: int = 86400 * 7,  # 7 days default
        semantic_similarity_threshold: float = 0.95
    ):
        """
        Initialize the cached LLM wrapper.
        
        Args:
            llm_model: The base LLM to wrap
            cache_dir: Directory to store cache files
            cache_manager: Optional existing cache manager to use
            result_namespace: Cache namespace for results
            prompt_namespace: Cache namespace for prompts
            ttl: Time-to-live for cached results in seconds
            semantic_similarity_threshold: Threshold for semantic similarity in the cache
        """
        self.llm = llm_model
        
        # Set up cache manager if not provided
        if cache_manager is None:
            if cache_dir is None:
                cache_dir = os.path.join(os.path.expanduser("~"), ".ragnite", "cache")
            
            self.cache_manager = CacheManager(cache_dir=cache_dir)
        else:
            self.cache_manager = cache_manager
        
        # Create caches
        self.result_cache = ResultCache(
            cache_manager=self.cache_manager,
            namespace=result_namespace,
            ttl=ttl
        )
        
        self.prompt_cache = PromptCache(
            cache_manager=self.cache_manager,
            namespace=prompt_namespace,
            ttl=ttl
        )
        
        self.semantic_cache = SemanticQueryCache(
            cache_manager=self.cache_manager,
            namespace=result_namespace,
            similarity_threshold=semantic_similarity_threshold,
            ttl=ttl
        )
    
    def invoke(self, input_data, **kwargs):
        """
        Invoke the model with caching.
        
        Args:
            input_data: Input to pass to the model
            **kwargs: Additional arguments for the model
            
        Returns:
            Model output
        """
        # For semantic caching, we use the raw input string if available
        cache_key = str(input_data)
        if isinstance(input_data, dict) and "content" in input_data:
            semantic_key = input_data["content"]
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], dict) and "content" in input_data[0]:
            semantic_key = input_data[0]["content"]
        else:
            semantic_key = cache_key
        
        # Try exact cache first
        cached_result = self.result_cache.get(cache_key)
        if cached_result is not None:
            logger.debug("LLM exact cache hit")
            return cached_result
        
        # Try semantic cache next
        semantic_result = self.semantic_cache.get(semantic_key)
        if semantic_result is not None:
            logger.debug("LLM semantic cache hit")
            return semantic_result
        
        # No cache hit, call the model
        start_time = time.time()
        result = self.llm.invoke(input_data, **kwargs)
        elapsed = time.time() - start_time
        
        # Cache the result
        self.result_cache.set(cache_key, result)
        self.semantic_cache.set(semantic_key, result)
        
        logger.debug(f"LLM call completed in {elapsed:.2f}s")
        return result
    
    # Pass through other attributes to the wrapped LLM
    def __getattr__(self, name):
        return getattr(self.llm, name)


class RAGCacheManager:
    """
    Manager for integrating caching with RAG pipelines.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        enable_embedding_cache: bool = True,
        enable_result_cache: bool = True,
        enable_semantic_cache: bool = True,
        enable_prompt_cache: bool = True,
        dashboard_port: Optional[int] = None
    ):
        """
        Initialize the RAG cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            enable_embedding_cache: Whether to enable embedding cache
            enable_result_cache: Whether to enable result cache
            enable_semantic_cache: Whether to enable semantic cache
            enable_prompt_cache: Whether to enable prompt cache
            dashboard_port: Port to run the dashboard on (None for no dashboard)
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".ragnite", "cache")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache_dir = cache_dir
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        
        # Initialize enabled caches
        self.embedding_cache = None
        self.result_cache = None
        self.semantic_cache = None
        self.prompt_cache = None
        
        if enable_embedding_cache:
            self.embedding_cache = EmbeddingCache(cache_manager=self.cache_manager)
        
        if enable_result_cache:
            self.result_cache = ResultCache(cache_manager=self.cache_manager)
        
        if enable_semantic_cache:
            self.semantic_cache = SemanticQueryCache(cache_manager=self.cache_manager)
        
        if enable_prompt_cache:
            self.prompt_cache = PromptCache(cache_manager=self.cache_manager)
        
        # Start dashboard if requested
        self.dashboard = None
        if dashboard_port is not None:
            self.dashboard = CacheDashboard(
                cache_manager=self.cache_manager,
                port=dashboard_port
            )
            self.dashboard.start()
    
    def wrap_embedding_model(self, embedding_model):
        """
        Wrap an embedding model with caching.
        
        Args:
            embedding_model: Embedding model to wrap
            
        Returns:
            Wrapped embedding model with caching
        """
        return CachedEmbeddings(
            embedding_model=embedding_model,
            cache_manager=self.cache_manager
        )
    
    def wrap_llm(self, llm_model):
        """
        Wrap an LLM with caching.
        
        Args:
            llm_model: LLM to wrap
            
        Returns:
            Wrapped LLM with caching
        """
        return CachedLLM(
            llm_model=llm_model,
            cache_manager=self.cache_manager
        )
    
    def enhance_rag_pipeline(self, rag_pipeline):
        """
        Enhance a RAG pipeline with caching capabilities.
        
        Args:
            rag_pipeline: RAG pipeline to enhance
            
        Returns:
            Enhanced RAG pipeline
        """
        # Wrap the embedding model if it exists
        if hasattr(rag_pipeline, 'embedding_model'):
            rag_pipeline.embedding_model = self.wrap_embedding_model(rag_pipeline.embedding_model)
        
        # Wrap the LLM if it exists
        if hasattr(rag_pipeline, 'llm'):
            rag_pipeline.llm = self.wrap_llm(rag_pipeline.llm)
        
        return rag_pipeline
    
    def clear_caches(self):
        """Clear all caches"""
        self.cache_manager.clear_all()
    
    def get_cache_stats(self):
        """Get statistics about the caches"""
        return self.cache_manager.get_stats()
    
    def shutdown(self):
        """Shutdown the cache manager and dashboard"""
        if self.dashboard:
            self.dashboard.stop()


def add_caching_to_pipeline(pipeline, cache_dir=None, dashboard_port=None):
    """
    Utility function to easily add caching to an existing RAG pipeline.
    
    Args:
        pipeline: RAG pipeline to enhance with caching
        cache_dir: Optional directory for cache files
        dashboard_port: Optional port for the cache dashboard
        
    Returns:
        Enhanced pipeline with caching
    """
    cache_manager = RAGCacheManager(
        cache_dir=cache_dir,
        dashboard_port=dashboard_port
    )
    
    return cache_manager.enhance_rag_pipeline(pipeline)


# Example usage
if __name__ == "__main__":
    # Example with a basic RAG pipeline
    from basic_rag.src.rag_pipeline import RAGPipeline
    
    # Create a RAG pipeline
    rag = RAGPipeline(
        embedding_model="text-embedding-ada-002",
        model_name="gpt-3.5-turbo"
    )
    
    # Add caching
    rag_with_cache = add_caching_to_pipeline(
        rag,
        cache_dir="~/.ragnite/cache",
        dashboard_port=8080
    )
    
    # Use the RAG pipeline as normal - caching happens automatically
    result = rag_with_cache.query("What is retrieval-augmented generation?")
    print(result) 