"""
Cache Manager Module

This module provides a comprehensive caching infrastructure for the RAGNITE
RAG system to improve performance and reduce redundant computation and API calls.
"""

import logging
import time
import json
import os
import hashlib
import pickle
from typing import Dict, Any, List, Tuple, Optional, Callable, Union, Set
from datetime import datetime, timedelta
import threading
from collections import OrderedDict, defaultdict
import numpy as np
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class LRUCache:
    """
    Simple LRU (Least Recently Used) cache implementation.
    """
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize the LRU cache.
        
        Args:
            capacity: Maximum number of items to store
        """
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> Any:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key not in self.cache:
            return None
        
        # Move item to the end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """
        Add an item to the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # If key exists, update and move to end
        if key in self.cache:
            self.cache[key] = value
            self.cache.move_to_end(key)
            return
        
        # Add new item
        self.cache[key] = value
        
        # Remove oldest item if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def remove(self, key: str) -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        return len(self.cache)
    
    def keys(self) -> List[str]:
        """
        Get all keys in the cache.
        
        Returns:
            List of cache keys
        """
        return list(self.cache.keys())


class TimeBasedCache:
    """
    Cache implementation with time-based invalidation.
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize the time-based cache.
        
        Args:
            ttl_seconds: Time to live in seconds (default: 1 hour)
        """
        self.cache = {}  # key -> (value, expiry_timestamp)
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Any:
        """
        Get an item from the cache if it exists and is not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            return None
        
        value, expiry = self.cache[key]
        
        # Check if expired
        if time.time() > expiry:
            del self.cache[key]
            return None
        
        return value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Add an item to the cache with expiry time.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Custom TTL for this specific item (optional)
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)
    
    def remove(self, key: str) -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
    
    def cleanup(self) -> int:
        """
        Remove all expired items from the cache.
        
        Returns:
            Number of items removed
        """
        now = time.time()
        expired_keys = [k for k, (_, exp) in self.cache.items() if now > exp]
        
        for k in expired_keys:
            del self.cache[k]
        
        return len(expired_keys)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        return len(self.cache)
    
    def keys(self) -> List[str]:
        """
        Get all keys in the cache.
        
        Returns:
            List of cache keys
        """
        return list(self.cache.keys())


class SemanticCache:
    """
    Semantic cache for storing and retrieving embeddings and results
    based on semantic similarity.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        similarity_function: Optional[Callable] = None,
        max_size: int = 10000
    ):
        """
        Initialize the semantic cache.
        
        Args:
            similarity_threshold: Threshold for considering items semantically similar
            similarity_function: Function to compute similarity between embeddings
            max_size: Maximum number of items to store
        """
        self.cache = {}  # key -> (embedding, value)
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        
        # Default to cosine similarity if no function provided
        if similarity_function is None:
            self.similarity_function = self._cosine_similarity
        else:
            self.similarity_function = similarity_function
        
        # LRU tracking
        self.lru_list = []
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        if not a or not b:
            return 0.0
            
        try:
            a_array = np.array(a)
            b_array = np.array(b)
            
            norm_a = np.linalg.norm(a_array)
            norm_b = np.linalg.norm(b_array)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return np.dot(a_array, b_array) / (norm_a * norm_b)
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {str(e)}")
            return 0.0
    
    def get(self, key: str, embedding: List[float]) -> Tuple[bool, Any]:
        """
        Get an item from the cache based on key OR semantic similarity.
        
        Args:
            key: Cache key
            embedding: Embedding vector for semantic matching
            
        Returns:
            Tuple of (hit, value) where:
                hit: Whether a match was found
                value: The cached value if hit is True, otherwise None
        """
        # First, try exact key match
        if key in self.cache:
            _, value = self.cache[key]
            self._update_lru(key)
            return True, value
        
        # If embedding provided, try semantic matching
        if embedding:
            best_match = None
            best_score = 0
            
            for cache_key, (cached_embedding, cached_value) in self.cache.items():
                if not cached_embedding:
                    continue
                    
                score = self.similarity_function(embedding, cached_embedding)
                
                if score > self.similarity_threshold and score > best_score:
                    best_match = cache_key
                    best_score = score
            
            if best_match:
                _, value = self.cache[best_match]
                self._update_lru(best_match)
                return True, value
        
        return False, None
    
    def put(self, key: str, embedding: List[float], value: Any) -> None:
        """
        Add an item to the semantic cache.
        
        Args:
            key: Cache key
            embedding: Embedding vector for semantic matching
            value: Value to cache
        """
        # If over capacity, remove least recently used item
        if len(self.cache) >= self.max_size and self.lru_list:
            lru_key = self.lru_list.pop(0)
            if lru_key in self.cache:
                del self.cache[lru_key]
        
        # Add new item
        self.cache[key] = (embedding, value)
        self._update_lru(key)
    
    def _update_lru(self, key: str) -> None:
        """
        Update the LRU tracking for a key.
        
        Args:
            key: Cache key to mark as recently used
        """
        # Remove if exists
        if key in self.lru_list:
            self.lru_list.remove(key)
        
        # Add to end (most recently used)
        self.lru_list.append(key)
    
    def remove(self, key: str) -> None:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key
        """
        if key in self.cache:
            del self.cache[key]
        
        if key in self.lru_list:
            self.lru_list.remove(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.lru_list.clear()
    
    def size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        return len(self.cache)
    
    def keys(self) -> List[str]:
        """
        Get all keys in the cache.
        
        Returns:
            List of cache keys
        """
        return list(self.cache.keys())


class DiskCache:
    """
    Disk-based cache for persistent storage of cached items.
    """
    
    def __init__(self, cache_dir: str = ".cache", ttl_seconds: int = 86400):
        """
        Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time to live in seconds (default: 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Index file for quick lookups
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        
        # Thread lock for index operations
        self.lock = threading.Lock()
    
    def _load_index(self) -> Dict[str, Any]:
        """
        Load the cache index from disk.
        
        Returns:
            Cache index dictionary
        """
        if not self.index_file.exists():
            return {"keys": {}, "last_cleanup": time.time()}
        
        try:
            with open(self.index_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache index: {str(e)}")
            return {"keys": {}, "last_cleanup": time.time()}
    
    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f)
        except Exception as e:
            logger.error(f"Error saving cache index: {str(e)}")
    
    def _get_file_path(self, key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to the cache file
        """
        # Use hash of key as filename to avoid filesystem issues
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"
    
    def get(self, key: str) -> Any:
        """
        Get an item from the disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            # Check if key exists in index
            if key not in self.index["keys"]:
                return None
            
            # Check if expired
            entry = self.index["keys"][key]
            if time.time() > entry["expiry"]:
                self.remove(key)
                return None
            
            # Get file path
            file_path = self._get_file_path(key)
            
            # Load cached item
            try:
                with open(file_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cached item: {str(e)}")
                # Remove invalid cache entry
                self.remove(key)
                return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Add an item to the disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Custom TTL for this specific item (optional)
        """
        with self.lock:
            # Get file path
            file_path = self._get_file_path(key)
            
            # Save cached item
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.error(f"Error saving cached item: {str(e)}")
                return
            
            # Update index
            ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds
            expiry = time.time() + ttl
            
            self.index["keys"][key] = {
                "file": file_path.name,
                "expiry": expiry,
                "created": time.time()
            }
            
            self._save_index()
    
    def remove(self, key: str) -> None:
        """
        Remove an item from the disk cache.
        
        Args:
            key: Cache key
        """
        with self.lock:
            if key not in self.index["keys"]:
                return
            
            # Get file path
            file_path = self._get_file_path(key)
            
            # Remove file
            try:
                if file_path.exists():
                    os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing cached file: {str(e)}")
            
            # Update index
            del self.index["keys"][key]
            self._save_index()
    
    def cleanup(self) -> int:
        """
        Remove all expired items from the disk cache.
        
        Returns:
            Number of items removed
        """
        with self.lock:
            now = time.time()
            expired_keys = [k for k, v in self.index["keys"].items() if now > v["expiry"]]
            
            for k in expired_keys:
                self.remove(k)
            
            # Update last cleanup time
            self.index["last_cleanup"] = now
            self._save_index()
            
            return len(expired_keys)
    
    def clear(self) -> None:
        """Clear the disk cache."""
        with self.lock:
            # Remove all cache files
            for key in list(self.index["keys"].keys()):
                self.remove(key)
            
            # Reset index
            self.index = {"keys": {}, "last_cleanup": time.time()}
            self._save_index()
    
    def size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        with self.lock:
            return len(self.index["keys"])
    
    def keys(self) -> List[str]:
        """
        Get all keys in the cache.
        
        Returns:
            List of cache keys
        """
        with self.lock:
            return list(self.index["keys"].keys())


class PromptTemplateCache:
    """
    Specialized cache for prompt templates.
    """
    
    def __init__(self, capacity: int = 100):
        """
        Initialize the prompt template cache.
        
        Args:
            capacity: Maximum number of templates to cache
        """
        self.templates = LRUCache(capacity)
        self.rendered = LRUCache(capacity * 10)  # Store more rendered templates
    
    def get_template(self, template_id: str) -> Any:
        """
        Get a template from the cache.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Cached template or None if not found
        """
        return self.templates.get(template_id)
    
    def put_template(self, template_id: str, template: Any) -> None:
        """
        Add a template to the cache.
        
        Args:
            template_id: Template identifier
            template: Template to cache
        """
        self.templates.put(template_id, template)
    
    def get_rendered(self, template_id: str, params_key: str) -> Any:
        """
        Get a rendered template from the cache.
        
        Args:
            template_id: Template identifier
            params_key: Key for the parameters used in rendering
            
        Returns:
            Cached rendered template or None if not found
        """
        key = f"{template_id}:{params_key}"
        return self.rendered.get(key)
    
    def put_rendered(self, template_id: str, params_key: str, rendered: Any) -> None:
        """
        Add a rendered template to the cache.
        
        Args:
            template_id: Template identifier
            params_key: Key for the parameters used in rendering
            rendered: Rendered template to cache
        """
        key = f"{template_id}:{params_key}"
        self.rendered.put(key, rendered)
    
    def clear(self) -> None:
        """Clear the prompt template cache."""
        self.templates.clear()
        self.rendered.clear()
    
    def size(self) -> Dict[str, int]:
        """
        Get the current size of the template cache.
        
        Returns:
            Dictionary with template and rendered counts
        """
        return {
            "templates": self.templates.size(),
            "rendered": self.rendered.size()
        }


class CacheManager:
    """
    Central manager for all cache types in the RAG system.
    """
    
    def __init__(
        self,
        enable_embedding_cache: bool = True,
        enable_semantic_cache: bool = True,
        enable_result_cache: bool = True,
        enable_prompt_cache: bool = True,
        enable_disk_cache: bool = False,
        cache_dir: str = ".cache",
        embedding_cache_size: int = 10000,
        semantic_cache_size: int = 5000,
        semantic_threshold: float = 0.92,
        result_cache_ttl: int = 3600,  # 1 hour
        disk_cache_ttl: int = 86400,  # 24 hours
        prompt_cache_size: int = 100,
        auto_persist: bool = False,
        cleanup_interval: int = 300  # 5 minutes
    ):
        """
        Initialize the cache manager.
        
        Args:
            enable_embedding_cache: Whether to enable the embedding cache
            enable_semantic_cache: Whether to enable the semantic cache
            enable_result_cache: Whether to enable the result cache
            enable_prompt_cache: Whether to enable the prompt cache
            enable_disk_cache: Whether to enable the disk cache
            cache_dir: Directory to store persistent cache files
            embedding_cache_size: Maximum size of the embedding cache
            semantic_cache_size: Maximum size of the semantic cache
            semantic_threshold: Threshold for semantic similarity
            result_cache_ttl: Time to live for result cache in seconds
            disk_cache_ttl: Time to live for disk cache in seconds
            prompt_cache_size: Maximum size of the prompt cache
            auto_persist: Whether to automatically persist cache to disk
            cleanup_interval: Interval for automatic cache cleanup in seconds
        """
        # Initialize caches based on enabled flags
        self.embedding_cache = LRUCache(embedding_cache_size) if enable_embedding_cache else None
        self.semantic_cache = SemanticCache(semantic_threshold, max_size=semantic_cache_size) if enable_semantic_cache else None
        self.result_cache = TimeBasedCache(result_cache_ttl) if enable_result_cache else None
        self.prompt_cache = PromptTemplateCache(prompt_cache_size) if enable_prompt_cache else None
        self.disk_cache = DiskCache(cache_dir, disk_cache_ttl) if enable_disk_cache else None
        
        # Configuration
        self.enable_embedding_cache = enable_embedding_cache
        self.enable_semantic_cache = enable_semantic_cache
        self.enable_result_cache = enable_result_cache
        self.enable_prompt_cache = enable_prompt_cache
        self.enable_disk_cache = enable_disk_cache
        self.auto_persist = auto_persist
        
        # Metrics
        self.metrics = {
            "embedding_cache_hits": 0,
            "embedding_cache_misses": 0,
            "semantic_cache_hits": 0,
            "semantic_cache_misses": 0,
            "result_cache_hits": 0,
            "result_cache_misses": 0,
            "prompt_cache_hits": 0,
            "prompt_cache_misses": 0,
            "disk_cache_hits": 0,
            "disk_cache_misses": 0
        }
        
        # Start background cleanup thread if any TTL-based caches are enabled
        self.cleanup_interval = cleanup_interval
        if (enable_result_cache or enable_disk_cache) and cleanup_interval > 0:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self) -> None:
        """Start a background thread for periodic cache cleanup."""
        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval)
                self.cleanup()
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
    
    def cleanup(self) -> Dict[str, int]:
        """
        Clean up expired items from all caches.
        
        Returns:
            Dictionary with cleanup counts for each cache
        """
        cleanup_counts = {}
        
        # Clean up result cache
        if self.enable_result_cache and self.result_cache:
            count = self.result_cache.cleanup()
            cleanup_counts["result_cache"] = count
        
        # Clean up disk cache
        if self.enable_disk_cache and self.disk_cache:
            count = self.disk_cache.cleanup()
            cleanup_counts["disk_cache"] = count
        
        logger.debug(f"Cache cleanup complete: {cleanup_counts}")
        return cleanup_counts
    
    # Embedding Cache Methods
    
    def get_embedding(self, key: str) -> Tuple[bool, Any]:
        """
        Get an embedding from the cache.
        
        Args:
            key: Cache key (e.g., text or image identifier)
            
        Returns:
            Tuple of (hit, embedding) where:
                hit: Whether the embedding was found in the cache
                embedding: The cached embedding if hit is True, otherwise None
        """
        if not self.enable_embedding_cache or not self.embedding_cache:
            return False, None
        
        embedding = self.embedding_cache.get(key)
        hit = embedding is not None
        
        # Update metrics
        if hit:
            self.metrics["embedding_cache_hits"] += 1
        else:
            self.metrics["embedding_cache_misses"] += 1
        
        return hit, embedding
    
    def put_embedding(self, key: str, embedding: Any) -> None:
        """
        Add an embedding to the cache.
        
        Args:
            key: Cache key (e.g., text or image identifier)
            embedding: Embedding to cache
        """
        if not self.enable_embedding_cache or not self.embedding_cache:
            return
        
        self.embedding_cache.put(key, embedding)
        
        # Auto-persist if enabled
        if self.auto_persist and self.enable_disk_cache and self.disk_cache:
            self.disk_cache.put(f"embedding:{key}", embedding)
    
    # Semantic Cache Methods
    
    def get_semantic(self, query: str, embedding: List[float]) -> Tuple[bool, Any]:
        """
        Get a result from the semantic cache based on query similarity.
        
        Args:
            query: Query string
            embedding: Query embedding for semantic matching
            
        Returns:
            Tuple of (hit, result) where:
                hit: Whether a semantically similar result was found
                result: The cached result if hit is True, otherwise None
        """
        if not self.enable_semantic_cache or not self.semantic_cache:
            return False, None
        
        hit, result = self.semantic_cache.get(query, embedding)
        
        # Update metrics
        if hit:
            self.metrics["semantic_cache_hits"] += 1
        else:
            self.metrics["semantic_cache_misses"] += 1
        
        return hit, result
    
    def put_semantic(self, query: str, embedding: List[float], result: Any) -> None:
        """
        Add a result to the semantic cache.
        
        Args:
            query: Query string
            embedding: Query embedding for semantic matching
            result: Result to cache
        """
        if not self.enable_semantic_cache or not self.semantic_cache:
            return
        
        self.semantic_cache.put(query, embedding, result)
    
    # Result Cache Methods
    
    def get_result(self, key: str) -> Tuple[bool, Any]:
        """
        Get a result from the cache.
        
        Args:
            key: Cache key (e.g., query or request identifier)
            
        Returns:
            Tuple of (hit, result) where:
                hit: Whether the result was found in the cache
                result: The cached result if hit is True, otherwise None
        """
        # Try memory cache first
        if self.enable_result_cache and self.result_cache:
            result = self.result_cache.get(key)
            if result is not None:
                self.metrics["result_cache_hits"] += 1
                return True, result
        
        # Try disk cache next
        if self.enable_disk_cache and self.disk_cache:
            result = self.disk_cache.get(f"result:{key}")
            if result is not None:
                # Add to memory cache for faster access next time
                if self.enable_result_cache and self.result_cache:
                    self.result_cache.put(key, result)
                
                self.metrics["disk_cache_hits"] += 1
                return True, result
        
        # Cache miss
        if self.enable_result_cache:
            self.metrics["result_cache_misses"] += 1
        elif self.enable_disk_cache:
            self.metrics["disk_cache_misses"] += 1
        
        return False, None
    
    def put_result(self, key: str, result: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Add a result to the cache.
        
        Args:
            key: Cache key (e.g., query or request identifier)
            result: Result to cache
            ttl_seconds: Custom TTL for this specific item (optional)
        """
        # Add to memory cache
        if self.enable_result_cache and self.result_cache:
            self.result_cache.put(key, result, ttl_seconds)
        
        # Add to disk cache if auto-persist is enabled
        if self.auto_persist and self.enable_disk_cache and self.disk_cache:
            self.disk_cache.put(f"result:{key}", result, ttl_seconds)
    
    # Prompt Cache Methods
    
    def get_template(self, template_id: str) -> Tuple[bool, Any]:
        """
        Get a prompt template from the cache.
        
        Args:
            template_id: Template identifier
            
        Returns:
            Tuple of (hit, template) where:
                hit: Whether the template was found in the cache
                template: The cached template if hit is True, otherwise None
        """
        if not self.enable_prompt_cache or not self.prompt_cache:
            return False, None
        
        template = self.prompt_cache.get_template(template_id)
        hit = template is not None
        
        # Update metrics
        if hit:
            self.metrics["prompt_cache_hits"] += 1
        else:
            self.metrics["prompt_cache_misses"] += 1
        
        return hit, template
    
    def put_template(self, template_id: str, template: Any) -> None:
        """
        Add a prompt template to the cache.
        
        Args:
            template_id: Template identifier
            template: Template to cache
        """
        if not self.enable_prompt_cache or not self.prompt_cache:
            return
        
        self.prompt_cache.put_template(template_id, template)
    
    def get_rendered_template(self, template_id: str, params: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Get a rendered prompt template from the cache.
        
        Args:
            template_id: Template identifier
            params: Parameters used in rendering
            
        Returns:
            Tuple of (hit, rendered) where:
                hit: Whether the rendered template was found in the cache
                rendered: The cached rendered template if hit is True, otherwise None
        """
        if not self.enable_prompt_cache or not self.prompt_cache:
            return False, None
        
        # Create a stable key from the parameters
        params_key = self._create_params_key(params)
        
        rendered = self.prompt_cache.get_rendered(template_id, params_key)
        hit = rendered is not None
        
        # Update metrics
        if hit:
            self.metrics["prompt_cache_hits"] += 1
        else:
            self.metrics["prompt_cache_misses"] += 1
        
        return hit, rendered
    
    def put_rendered_template(self, template_id: str, params: Dict[str, Any], rendered: Any) -> None:
        """
        Add a rendered prompt template to the cache.
        
        Args:
            template_id: Template identifier
            params: Parameters used in rendering
            rendered: Rendered template to cache
        """
        if not self.enable_prompt_cache or not self.prompt_cache:
            return
        
        # Create a stable key from the parameters
        params_key = self._create_params_key(params)
        
        self.prompt_cache.put_rendered(template_id, params_key, rendered)
    
    def _create_params_key(self, params: Dict[str, Any]) -> str:
        """
        Create a stable key from template parameters.
        
        Args:
            params: Template parameters
            
        Returns:
            Stable string key
        """
        # Convert parameters to a stable string representation
        param_items = sorted(params.items())
        params_str = json.dumps(param_items, sort_keys=True)
        
        # Use hash for compact representation
        return hashlib.md5(params_str.encode()).hexdigest()
    
    # General Methods
    
    def persist(self) -> None:
        """Persist all caches to disk."""
        if not self.enable_disk_cache or not self.disk_cache:
            logger.warning("Disk cache is not enabled, cannot persist")
            return
        
        # Persist embedding cache
        if self.enable_embedding_cache and self.embedding_cache:
            for key in self.embedding_cache.keys():
                embedding = self.embedding_cache.get(key)
                self.disk_cache.put(f"embedding:{key}", embedding)
        
        # Persist result cache
        if self.enable_result_cache and self.result_cache:
            for key in self.result_cache.keys():
                result = self.result_cache.get(key)
                if result is not None:
                    self.disk_cache.put(f"result:{key}", result)
        
        logger.info("Cache persisted to disk")
    
    def load(self) -> None:
        """Load all caches from disk."""
        if not self.enable_disk_cache or not self.disk_cache:
            logger.warning("Disk cache is not enabled, cannot load")
            return
        
        # Load embedding cache
        if self.enable_embedding_cache and self.embedding_cache:
            prefix = "embedding:"
            for key in [k for k in self.disk_cache.keys() if k.startswith(prefix)]:
                embedding = self.disk_cache.get(key)
                if embedding is not None:
                    real_key = key[len(prefix):]
                    self.embedding_cache.put(real_key, embedding)
        
        # Load result cache
        if self.enable_result_cache and self.result_cache:
            prefix = "result:"
            for key in [k for k in self.disk_cache.keys() if k.startswith(prefix)]:
                result = self.disk_cache.get(key)
                if result is not None:
                    real_key = key[len(prefix):]
                    self.result_cache.put(real_key, result)
        
        logger.info("Cache loaded from disk")
    
    def clear(self) -> None:
        """Clear all caches."""
        if self.enable_embedding_cache and self.embedding_cache:
            self.embedding_cache.clear()
        
        if self.enable_semantic_cache and self.semantic_cache:
            self.semantic_cache.clear()
        
        if self.enable_result_cache and self.result_cache:
            self.result_cache.clear()
        
        if self.enable_prompt_cache and self.prompt_cache:
            self.prompt_cache.clear()
        
        if self.enable_disk_cache and self.disk_cache:
            self.disk_cache.clear()
        
        # Reset metrics
        for key in self.metrics:
            self.metrics[key] = 0
        
        logger.info("All caches cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        metrics = self.metrics.copy()
        
        # Add cache sizes
        if self.enable_embedding_cache and self.embedding_cache:
            metrics["embedding_cache_size"] = self.embedding_cache.size()
        
        if self.enable_semantic_cache and self.semantic_cache:
            metrics["semantic_cache_size"] = self.semantic_cache.size()
        
        if self.enable_result_cache and self.result_cache:
            metrics["result_cache_size"] = self.result_cache.size()
        
        if self.enable_prompt_cache and self.prompt_cache:
            metrics["prompt_cache_size"] = self.prompt_cache.size()
        
        if self.enable_disk_cache and self.disk_cache:
            metrics["disk_cache_size"] = self.disk_cache.size()
        
        # Calculate hit rates
        for cache_type in ["embedding_cache", "semantic_cache", "result_cache", "prompt_cache", "disk_cache"]:
            hits = metrics.get(f"{cache_type}_hits", 0)
            misses = metrics.get(f"{cache_type}_misses", 0)
            total = hits + misses
            
            if total > 0:
                metrics[f"{cache_type}_hit_rate"] = hits / total
            else:
                metrics[f"{cache_type}_hit_rate"] = 0.0
        
        return metrics 