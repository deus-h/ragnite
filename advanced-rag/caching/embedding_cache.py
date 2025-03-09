"""
Embedding Cache Module

This module provides specialized caching for text and image embeddings,
reducing redundant API calls and computation for embedding generation.
"""

import logging
import hashlib
import time
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from pathlib import Path
import os

try:
    from .cache_manager import CacheManager, LRUCache
except ImportError:
    from cache_manager import CacheManager, LRUCache

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    Specialized cache for text and image embeddings that reduces
    redundant computation and API calls.
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        text_cache_size: int = 50000,
        image_cache_size: int = 10000,
        namespace: str = "default",
        persistent_cache_dir: Optional[str] = None,
        auto_persist: bool = False,
        persist_interval: int = 600,  # 10 minutes
        text_hash_method: str = "md5"  # "md5", "sha256", or "xxhash"
    ):
        """
        Initialize the embedding cache.
        
        Args:
            cache_manager: Optional central cache manager to use
            text_cache_size: Maximum number of text embeddings to cache
            image_cache_size: Maximum number of image embeddings to cache
            namespace: Namespace to use for cache keys
            persistent_cache_dir: Directory to store persistent cache files
            auto_persist: Whether to automatically persist cache to disk
            persist_interval: Interval for automatic cache persistence in seconds
            text_hash_method: Method to use for hashing text
        """
        # Use provided cache manager or create a new one
        self.cache_manager = cache_manager
        self.using_external_manager = cache_manager is not None
        
        # If not using external manager, create local caches
        if not self.using_external_manager:
            self.text_cache = LRUCache(text_cache_size)
            self.image_cache = LRUCache(image_cache_size)
            
            # Setup persistence if enabled
            self.persistent_cache_dir = persistent_cache_dir
            if self.persistent_cache_dir:
                os.makedirs(self.persistent_cache_dir, exist_ok=True)
        
        # Configuration
        self.namespace = namespace
        self.auto_persist = auto_persist
        self.persist_interval = persist_interval
        self.text_hash_method = text_hash_method
        
        # Tracking
        self.last_persist_time = time.time()
        self.metrics = {
            "text_hits": 0,
            "text_misses": 0,
            "image_hits": 0,
            "image_misses": 0,
            "saved_api_calls": 0,
            "estimated_cost_saved": 0.0
        }
        
        self.cost_per_token = {
            "text_embedding": 0.0001,  # $0.0001 per 1000 tokens
            "image_embedding": 0.002   # $0.002 per image
        }
        
        # Load persistent cache if available
        if not self.using_external_manager and self.persistent_cache_dir:
            self._load_persistent_cache()
    
    def _format_text_key(self, text: str, model: Optional[str] = None) -> str:
        """
        Create a cache key for text.
        
        Args:
            text: Text to create a key for
            model: Optional model identifier
            
        Returns:
            Cache key
        """
        # Hash the text to create a stable key
        if self.text_hash_method == "md5":
            text_hash = hashlib.md5(text.encode()).hexdigest()
        elif self.text_hash_method == "sha256":
            text_hash = hashlib.sha256(text.encode()).hexdigest()
        else:  # Default to md5
            text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Include model in key if provided
        if model:
            return f"{self.namespace}:text:{model}:{text_hash}"
        else:
            return f"{self.namespace}:text:{text_hash}"
    
    def _format_image_key(self, image_id: str, model: Optional[str] = None) -> str:
        """
        Create a cache key for an image.
        
        Args:
            image_id: Image identifier
            model: Optional model identifier
            
        Returns:
            Cache key
        """
        # Include model in key if provided
        if model:
            return f"{self.namespace}:image:{model}:{image_id}"
        else:
            return f"{self.namespace}:image:{image_id}"
    
    def get_text_embedding(self, text: str, model: Optional[str] = None) -> Tuple[bool, Optional[List[float]]]:
        """
        Get a text embedding from the cache.
        
        Args:
            text: Text to get embedding for
            model: Optional model identifier
            
        Returns:
            Tuple of (hit, embedding) where:
                hit: Whether the embedding was found in the cache
                embedding: The cached embedding if hit is True, otherwise None
        """
        key = self._format_text_key(text, model)
        
        if self.using_external_manager:
            hit, embedding = self.cache_manager.get_embedding(key)
        else:
            embedding = self.text_cache.get(key)
            hit = embedding is not None
            
            # Update metrics
            if hit:
                self.metrics["text_hits"] += 1
                self.metrics["saved_api_calls"] += 1
                # Estimate tokens saved (rough approximation)
                tokens = len(text.split()) * 1.3  # Rough token estimate
                self.metrics["estimated_cost_saved"] += (tokens / 1000) * self.cost_per_token["text_embedding"]
            else:
                self.metrics["text_misses"] += 1
        
        return hit, embedding
    
    def put_text_embedding(self, text: str, embedding: List[float], model: Optional[str] = None) -> None:
        """
        Add a text embedding to the cache.
        
        Args:
            text: Text the embedding was generated for
            embedding: Embedding to cache
            model: Optional model identifier
        """
        key = self._format_text_key(text, model)
        
        if self.using_external_manager:
            self.cache_manager.put_embedding(key, embedding)
        else:
            self.text_cache.put(key, embedding)
            
            # Auto-persist if enabled
            if self.auto_persist and self.persistent_cache_dir:
                current_time = time.time()
                if current_time - self.last_persist_time > self.persist_interval:
                    self._persist_cache()
                    self.last_persist_time = current_time
    
    def get_image_embedding(self, image_id: str, model: Optional[str] = None) -> Tuple[bool, Optional[List[float]]]:
        """
        Get an image embedding from the cache.
        
        Args:
            image_id: Image identifier
            model: Optional model identifier
            
        Returns:
            Tuple of (hit, embedding) where:
                hit: Whether the embedding was found in the cache
                embedding: The cached embedding if hit is True, otherwise None
        """
        key = self._format_image_key(image_id, model)
        
        if self.using_external_manager:
            hit, embedding = self.cache_manager.get_embedding(key)
        else:
            embedding = self.image_cache.get(key)
            hit = embedding is not None
            
            # Update metrics
            if hit:
                self.metrics["image_hits"] += 1
                self.metrics["saved_api_calls"] += 1
                self.metrics["estimated_cost_saved"] += self.cost_per_token["image_embedding"]
            else:
                self.metrics["image_misses"] += 1
        
        return hit, embedding
    
    def put_image_embedding(self, image_id: str, embedding: List[float], model: Optional[str] = None) -> None:
        """
        Add an image embedding to the cache.
        
        Args:
            image_id: Image identifier
            embedding: Embedding to cache
            model: Optional model identifier
        """
        key = self._format_image_key(image_id, model)
        
        if self.using_external_manager:
            self.cache_manager.put_embedding(key, embedding)
        else:
            self.image_cache.put(key, embedding)
            
            # Auto-persist if enabled
            if self.auto_persist and self.persistent_cache_dir:
                current_time = time.time()
                if current_time - self.last_persist_time > self.persist_interval:
                    self._persist_cache()
                    self.last_persist_time = current_time
    
    def create_cached_embedder(self, 
                              embed_function: Callable, 
                              is_image_embedder: bool = False,
                              model_name: Optional[str] = None) -> Callable:
        """
        Create a cached wrapper around an embedding function.
        
        Args:
            embed_function: Function that generates embeddings
            is_image_embedder: Whether the function generates image embeddings
            model_name: Optional model name
            
        Returns:
            Cached embedding function
        """
        if is_image_embedder:
            def cached_image_embedder(image_data):
                # If image_data is a dict with an ID, use that as the key
                image_id = image_data.get("id") if isinstance(image_data, dict) else str(hash(str(image_data)))
                
                # Check cache
                hit, embedding = self.get_image_embedding(image_id, model_name)
                if hit:
                    return embedding
                
                # Cache miss, generate embedding
                embedding = embed_function(image_data)
                
                # Cache the result
                self.put_image_embedding(image_id, embedding, model_name)
                
                return embedding
            
            return cached_image_embedder
        
        else:
            def cached_text_embedder(text):
                # Check cache
                hit, embedding = self.get_text_embedding(text, model_name)
                if hit:
                    return embedding
                
                # Cache miss, generate embedding
                embedding = embed_function(text)
                
                # Cache the result
                self.put_text_embedding(text, embedding, model_name)
                
                return embedding
            
            return cached_text_embedder
    
    def clear(self, text_only: bool = False, image_only: bool = False) -> None:
        """
        Clear the embedding cache.
        
        Args:
            text_only: Whether to only clear text embeddings
            image_only: Whether to only clear image embeddings
        """
        if self.using_external_manager:
            # Cannot selectively clear with external manager
            logger.warning("Cannot selectively clear with external cache manager")
            return
        
        if not text_only:
            self.image_cache.clear()
        
        if not image_only:
            self.text_cache.clear()
        
        # Reset relevant metrics
        if not text_only:
            self.metrics["image_hits"] = 0
            self.metrics["image_misses"] = 0
        
        if not image_only:
            self.metrics["text_hits"] = 0
            self.metrics["text_misses"] = 0
        
        self.metrics["saved_api_calls"] = 0
        self.metrics["estimated_cost_saved"] = 0.0
        
        logger.info(f"Embedding cache cleared (text_only={text_only}, image_only={image_only})")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get embedding cache performance metrics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        metrics = self.metrics.copy()
        
        # Add cache sizes
        if not self.using_external_manager:
            metrics["text_cache_size"] = self.text_cache.size()
            metrics["image_cache_size"] = self.image_cache.size()
        
        # Calculate hit rates
        text_total = metrics["text_hits"] + metrics["text_misses"]
        image_total = metrics["image_hits"] + metrics["image_misses"]
        
        metrics["text_hit_rate"] = metrics["text_hits"] / text_total if text_total > 0 else 0.0
        metrics["image_hit_rate"] = metrics["image_hits"] / image_total if image_total > 0 else 0.0
        
        # Overall metrics
        total_hits = metrics["text_hits"] + metrics["image_hits"]
        total_misses = metrics["text_misses"] + metrics["image_misses"]
        total = total_hits + total_misses
        
        metrics["overall_hit_rate"] = total_hits / total if total > 0 else 0.0
        
        return metrics
    
    def _persist_cache(self) -> None:
        """Persist cache to disk."""
        if not self.persistent_cache_dir or self.using_external_manager:
            return
        
        import pickle
        
        # Save text cache
        text_cache_path = Path(self.persistent_cache_dir) / f"{self.namespace}_text_cache.pkl"
        try:
            with open(text_cache_path, "wb") as f:
                pickle.dump(self.text_cache.cache, f)
        except Exception as e:
            logger.error(f"Error persisting text cache: {str(e)}")
        
        # Save image cache
        image_cache_path = Path(self.persistent_cache_dir) / f"{self.namespace}_image_cache.pkl"
        try:
            with open(image_cache_path, "wb") as f:
                pickle.dump(self.image_cache.cache, f)
        except Exception as e:
            logger.error(f"Error persisting image cache: {str(e)}")
        
        logger.info(f"Embedding cache persisted to {self.persistent_cache_dir}")
    
    def _load_persistent_cache(self) -> None:
        """Load persistent cache from disk."""
        if not self.persistent_cache_dir or self.using_external_manager:
            return
        
        import pickle
        
        # Load text cache
        text_cache_path = Path(self.persistent_cache_dir) / f"{self.namespace}_text_cache.pkl"
        if text_cache_path.exists():
            try:
                with open(text_cache_path, "rb") as f:
                    self.text_cache.cache = pickle.load(f)
                logger.info(f"Loaded text cache with {self.text_cache.size()} entries")
            except Exception as e:
                logger.error(f"Error loading text cache: {str(e)}")
        
        # Load image cache
        image_cache_path = Path(self.persistent_cache_dir) / f"{self.namespace}_image_cache.pkl"
        if image_cache_path.exists():
            try:
                with open(image_cache_path, "rb") as f:
                    self.image_cache.cache = pickle.load(f)
                logger.info(f"Loaded image cache with {self.image_cache.size()} entries")
            except Exception as e:
                logger.error(f"Error loading image cache: {str(e)}")
    
    def persist(self) -> None:
        """
        Manually persist cache to disk.
        """
        if self.using_external_manager:
            if hasattr(self.cache_manager, "persist") and callable(self.cache_manager.persist):
                self.cache_manager.persist()
                logger.info("Persisted cache via cache manager")
            else:
                logger.warning("Cache manager does not support persist operation")
        else:
            self._persist_cache()
            self.last_persist_time = time.time() 