"""
Result Cache Module

This module provides a result cache with time-based invalidation for storing
query results and other computed data with configurable expiration policies.
"""

import logging
import hashlib
import time
import json
import threading
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from enum import Enum

try:
    from .cache_manager import CacheManager, TimeBasedCache
except ImportError:
    from cache_manager import CacheManager, TimeBasedCache

# Configure logging
logger = logging.getLogger(__name__)

# Generic type for cache values
T = TypeVar('T')


class ExpirationPolicy(Enum):
    """Expiration policies for cached items."""
    
    NEVER = 0       # Never expire
    FIXED = 1       # Fixed time from creation
    ACCESS = 2      # Fixed time from last access
    SLIDING = 3     # Reset TTL on access
    ADAPTIVE = 4    # Adjust TTL based on access pattern


class CachedResult(Generic[T]):
    """Container for cached results with expiration handling."""
    
    def __init__(
        self,
        key: str,
        value: T,
        ttl_seconds: Optional[int] = 3600,  # 1 hour
        policy: ExpirationPolicy = ExpirationPolicy.FIXED,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1"
    ):
        """
        Initialize a cached result.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            policy: Expiration policy
            metadata: Additional metadata
            version: Version identifier for the cached data
        """
        self.key = key
        self.value = value
        self.ttl_seconds = ttl_seconds
        self.policy = policy
        self.metadata = metadata or {}
        self.version = version
        
        self.created_at = time.time()
        self.expires_at = self._calculate_expiry()
        self.access_count = 0
        self.last_access = self.created_at
    
    def _calculate_expiry(self) -> Optional[float]:
        """
        Calculate expiration timestamp based on policy.
        
        Returns:
            Expiration timestamp or None if no expiration
        """
        if self.policy == ExpirationPolicy.NEVER or self.ttl_seconds is None:
            return None
        
        if self.policy in (ExpirationPolicy.FIXED, ExpirationPolicy.ADAPTIVE):
            return self.created_at + self.ttl_seconds
        
        if self.policy in (ExpirationPolicy.ACCESS, ExpirationPolicy.SLIDING):
            return self.last_access + self.ttl_seconds
        
        # Default to fixed expiry
        return self.created_at + self.ttl_seconds
    
    def is_expired(self) -> bool:
        """
        Check if result is expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.expires_at is None:
            return False
        
        return time.time() > self.expires_at
    
    def access(self) -> None:
        """Record an access to this result and update expiry if needed."""
        self.access_count += 1
        self.last_access = time.time()
        
        # Update expiry for policies that depend on access time
        if self.policy == ExpirationPolicy.SLIDING:
            self.expires_at = self.last_access + self.ttl_seconds
        elif self.policy == ExpirationPolicy.ADAPTIVE:
            # Adjust TTL based on access pattern
            # More frequent access = longer TTL
            access_rate = self.access_count / max(1, (self.last_access - self.created_at) / 3600)
            adjustment_factor = min(2.0, 1.0 + (access_rate / 10))
            self.expires_at = self.last_access + (self.ttl_seconds * adjustment_factor)
    
    def remaining_ttl(self) -> Optional[float]:
        """
        Get remaining time to live in seconds.
        
        Returns:
            Remaining TTL or None if no expiration
        """
        if self.expires_at is None:
            return None
        
        return max(0, self.expires_at - time.time())
    
    def refresh(self) -> None:
        """Refresh expiration time."""
        if self.policy != ExpirationPolicy.FIXED:
            self.expires_at = time.time() + self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "key": self.key,
            "value": self.value,
            "ttl_seconds": self.ttl_seconds,
            "policy": self.policy.value,
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_access": self.last_access
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedResult':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            CachedResult instance
        """
        # Create instance with basic properties
        result = cls(
            key=data["key"],
            value=data["value"],
            ttl_seconds=data["ttl_seconds"],
            policy=ExpirationPolicy(data["policy"]),
            metadata=data["metadata"],
            version=data["version"]
        )
        
        # Set time-related properties
        result.created_at = data["created_at"]
        result.expires_at = data["expires_at"]
        result.access_count = data["access_count"]
        result.last_access = data["last_access"]
        
        return result


class ResultCache:
    """
    Cache for storing query results and other computed data with time-based invalidation.
    Supports different expiration policies for different types of data.
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        default_ttl: int = 3600,  # 1 hour
        default_policy: ExpirationPolicy = ExpirationPolicy.FIXED,
        max_size: int = 10000,
        namespace: str = "result_cache",
        cleanup_interval: int = 300,  # 5 minutes
        enable_metrics: bool = True,
        cache_tags: bool = True,
        ttl_by_type: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the result cache.
        
        Args:
            cache_manager: Optional central cache manager to use
            default_ttl: Default TTL in seconds
            default_policy: Default expiration policy
            max_size: Maximum cache size
            namespace: Namespace for the cache
            cleanup_interval: Interval for automatic cleanup in seconds
            enable_metrics: Whether to track cache metrics
            cache_tags: Whether to enable tag-based access
            ttl_by_type: Different TTLs for different result types
        """
        # External components
        self.cache_manager = cache_manager
        self.using_external_cache = cache_manager is not None
        
        # Internal cache if not using external cache manager
        if not self.using_external_cache:
            self.cache = TimeBasedCache(default_ttl)
            # Tag index for quick lookup by tag
            self.tag_index = {} if cache_tags else None
        
        # Configuration
        self.default_ttl = default_ttl
        self.default_policy = default_policy
        self.max_size = max_size
        self.namespace = namespace
        self.cleanup_interval = cleanup_interval
        self.enable_metrics = enable_metrics
        self.cache_tags = cache_tags
        self.ttl_by_type = ttl_by_type or {}
        
        # Metrics
        if self.enable_metrics:
            self.metrics = {
                "hits": 0,
                "misses": 0,
                "expirations": 0,
                "evictions": 0,
                "inserts": 0,
                "updates": 0
            }
            
            # Type-specific metrics
            self.type_metrics = {}
        
        # Start cleanup thread if using internal cache
        if not self.using_external_cache and self.cleanup_interval > 0:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self) -> None:
        """Start a background thread for periodic cache cleanup."""
        def cleanup_worker():
            while True:
                time.sleep(self.cleanup_interval)
                self.cleanup()
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
    
    def _format_key(self, key: str, result_type: Optional[str] = None) -> str:
        """
        Format a cache key with namespace and optional type.
        
        Args:
            key: Original key
            result_type: Optional result type
            
        Returns:
            Formatted cache key
        """
        if result_type:
            return f"{self.namespace}:{result_type}:{key}"
        else:
            return f"{self.namespace}:{key}"
    
    def _get_ttl_for_type(self, result_type: Optional[str]) -> int:
        """
        Get TTL for a given result type.
        
        Args:
            result_type: Result type
            
        Returns:
            TTL in seconds
        """
        if result_type and result_type in self.ttl_by_type:
            return self.ttl_by_type[result_type]
        
        return self.default_ttl
    
    def _update_metrics(self, event: str, result_type: Optional[str] = None) -> None:
        """
        Update cache metrics.
        
        Args:
            event: Event type (hit, miss, etc.)
            result_type: Optional result type for type-specific metrics
        """
        if not self.enable_metrics:
            return
        
        # Update global metrics
        if event in self.metrics:
            self.metrics[event] += 1
        
        # Update type-specific metrics
        if result_type:
            if result_type not in self.type_metrics:
                self.type_metrics[result_type] = {
                    "hits": 0,
                    "misses": 0,
                    "inserts": 0,
                    "updates": 0
                }
            
            if event in self.type_metrics[result_type]:
                self.type_metrics[result_type][event] += 1
    
    def _add_to_tag_index(self, key: str, tags: List[str]) -> None:
        """
        Add a key to the tag index.
        
        Args:
            key: Cache key
            tags: Tags to index by
        """
        if not self.cache_tags or not tags or self.using_external_cache:
            return
        
        for tag in tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            
            self.tag_index[tag].add(key)
    
    def _remove_from_tag_index(self, key: str, tags: Optional[List[str]] = None) -> None:
        """
        Remove a key from the tag index.
        
        Args:
            key: Cache key
            tags: Tags to remove from (all if None)
        """
        if not self.cache_tags or self.using_external_cache:
            return
        
        if tags:
            # Remove from specific tags
            for tag in tags:
                if tag in self.tag_index and key in self.tag_index[tag]:
                    self.tag_index[tag].remove(key)
                    
                    # Clean up empty tag sets
                    if not self.tag_index[tag]:
                        del self.tag_index[tag]
        else:
            # Remove from all tags
            for tag, keys in list(self.tag_index.items()):
                if key in keys:
                    keys.remove(key)
                    
                    # Clean up empty tag sets
                    if not keys:
                        del self.tag_index[tag]
    
    def get(self, key: str, result_type: Optional[str] = None) -> Tuple[bool, Optional[T]]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            result_type: Optional result type
            
        Returns:
            Tuple of (hit, value) where:
                hit: Whether the value was found in the cache
                value: The cached value if hit is True, otherwise None
        """
        formatted_key = self._format_key(key, result_type)
        
        if self.using_external_cache:
            hit, result = self.cache_manager.get_result(formatted_key)
            
            if hit and isinstance(result, CachedResult):
                # Check if expired
                if result.is_expired():
                    self._update_metrics("expirations", result_type)
                    return False, None
                
                # Record access
                result.access()
                
                # Update metrics
                self._update_metrics("hits", result_type)
                
                return True, result.value
        else:
            result_obj = self.cache.get(formatted_key)
            
            if result_obj is not None and isinstance(result_obj, CachedResult):
                # Check if expired
                if result_obj.is_expired():
                    self.cache.remove(formatted_key)
                    self._update_metrics("expirations", result_type)
                    return False, None
                
                # Record access
                result_obj.access()
                
                # Update metrics
                self._update_metrics("hits", result_type)
                
                return True, result_obj.value
        
        # Cache miss
        self._update_metrics("misses", result_type)
        return False, None
    
    def put(
        self,
        key: str,
        value: T,
        result_type: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        policy: Optional[ExpirationPolicy] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        version: str = "1"
    ) -> None:
        """
        Add a value to the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            result_type: Optional result type
            ttl_seconds: Time to live in seconds
            policy: Expiration policy
            metadata: Additional metadata
            tags: Tags for categorizing the cached item
            version: Version identifier for the cached data
        """
        formatted_key = self._format_key(key, result_type)
        
        # Use type-specific TTL if not specified
        if ttl_seconds is None:
            ttl_seconds = self._get_ttl_for_type(result_type)
        
        # Use default policy if not specified
        if policy is None:
            policy = self.default_policy
        
        # Create or update metadata
        merged_metadata = metadata or {}
        if result_type:
            merged_metadata["type"] = result_type
        if tags:
            merged_metadata["tags"] = tags
        
        # Create cached result
        result_obj = CachedResult(
            key=formatted_key,
            value=value,
            ttl_seconds=ttl_seconds,
            policy=policy,
            metadata=merged_metadata,
            version=version
        )
        
        # Check if this is an update or a new insert
        is_update = False
        if not self.using_external_cache:
            is_update = formatted_key in self.cache.cache
        
        # Add to cache
        if self.using_external_cache:
            self.cache_manager.put_result(formatted_key, result_obj, ttl_seconds)
        else:
            self.cache.put(formatted_key, result_obj, ttl_seconds)
            
            # Update tag index
            if tags:
                self._add_to_tag_index(formatted_key, tags)
        
        # Update metrics
        if is_update:
            self._update_metrics("updates", result_type)
        else:
            self._update_metrics("inserts", result_type)
    
    def refresh(self, key: str, result_type: Optional[str] = None) -> bool:
        """
        Refresh the expiry time of a cached item.
        
        Args:
            key: Cache key
            result_type: Optional result type
            
        Returns:
            True if refreshed, False if not found
        """
        formatted_key = self._format_key(key, result_type)
        
        if self.using_external_cache:
            hit, result = self.cache_manager.get_result(formatted_key)
            
            if hit and isinstance(result, CachedResult):
                result.refresh()
                self.cache_manager.put_result(formatted_key, result, result.ttl_seconds)
                return True
        else:
            result_obj = self.cache.get(formatted_key)
            
            if result_obj is not None and isinstance(result_obj, CachedResult):
                result_obj.refresh()
                self.cache.put(formatted_key, result_obj, result_obj.ttl_seconds)
                return True
        
        return False
    
    def remove(self, key: str, result_type: Optional[str] = None) -> bool:
        """
        Remove an item from the cache.
        
        Args:
            key: Cache key
            result_type: Optional result type
            
        Returns:
            True if removed, False if not found
        """
        formatted_key = self._format_key(key, result_type)
        
        # Get the item first to check if it has tags
        tags = None
        if not self.using_external_cache and self.cache_tags:
            result_obj = self.cache.get(formatted_key)
            if result_obj is not None and isinstance(result_obj, CachedResult):
                if result_obj.metadata and "tags" in result_obj.metadata:
                    tags = result_obj.metadata["tags"]
        
        # Remove from cache
        if self.using_external_cache:
            if hasattr(self.cache_manager, 'remove'):
                self.cache_manager.remove(formatted_key)
                return True
            return False
        else:
            if formatted_key in self.cache.cache:
                self.cache.remove(formatted_key)
                
                # Remove from tag index
                if self.cache_tags and tags:
                    self._remove_from_tag_index(formatted_key, tags)
                
                return True
            
            return False
    
    def clear(self, result_type: Optional[str] = None) -> None:
        """
        Clear the cache.
        
        Args:
            result_type: Optional result type to clear (None for all)
        """
        if self.using_external_cache:
            # Cannot selectively clear with external cache manager
            if result_type is None and hasattr(self.cache_manager, 'clear'):
                self.cache_manager.clear()
        else:
            if result_type is None:
                # Clear everything
                self.cache.clear()
                
                # Clear tag index
                if self.cache_tags:
                    self.tag_index.clear()
            else:
                # Clear only items of specified type
                prefix = f"{self.namespace}:{result_type}:"
                to_remove = [k for k in self.cache.keys() if k.startswith(prefix)]
                
                for k in to_remove:
                    self.remove(k.replace(prefix, ""), result_type)
        
        # Reset metrics for the cleared type
        if self.enable_metrics:
            if result_type is None:
                # Reset all metrics
                for key in self.metrics:
                    self.metrics[key] = 0
                self.type_metrics.clear()
            elif result_type in self.type_metrics:
                # Reset only metrics for this type
                for key in self.type_metrics[result_type]:
                    self.type_metrics[result_type][key] = 0
    
    def get_by_tag(self, tag: str, result_type: Optional[str] = None) -> List[Tuple[str, T]]:
        """
        Get all cached items with a specific tag.
        
        Args:
            tag: Tag to filter by
            result_type: Optional result type to filter by
            
        Returns:
            List of (key, value) tuples
        """
        if not self.cache_tags or self.using_external_cache:
            logger.warning("Tag-based access not supported with the current configuration")
            return []
        
        if tag not in self.tag_index:
            return []
        
        results = []
        
        for formatted_key in self.tag_index[tag]:
            result_obj = self.cache.get(formatted_key)
            
            if result_obj is None or result_obj.is_expired():
                continue
            
            # Check result type if specified
            if result_type and (
                not result_obj.metadata or
                "type" not in result_obj.metadata or
                result_obj.metadata["type"] != result_type
            ):
                continue
            
            # Extract original key from formatted key
            parts = formatted_key.split(":", 2)
            original_key = parts[2] if len(parts) > 2 else parts[1]
            
            results.append((original_key, result_obj.value))
        
        return results
    
    def invalidate_by_tag(self, tag: str, result_type: Optional[str] = None) -> int:
        """
        Invalidate all cached items with a specific tag.
        
        Args:
            tag: Tag to invalidate
            result_type: Optional result type to filter by
            
        Returns:
            Number of items invalidated
        """
        if not self.cache_tags or self.using_external_cache:
            logger.warning("Tag-based invalidation not supported with the current configuration")
            return 0
        
        if tag not in self.tag_index:
            return 0
        
        count = 0
        to_remove = set()
        
        for formatted_key in self.tag_index[tag]:
            result_obj = self.cache.get(formatted_key)
            
            if result_obj is None:
                to_remove.add(formatted_key)
                continue
            
            # Check result type if specified
            if result_type and (
                not result_obj.metadata or
                "type" not in result_obj.metadata or
                result_obj.metadata["type"] != result_type
            ):
                continue
            
            to_remove.add(formatted_key)
            count += 1
        
        # Remove all marked keys
        for formatted_key in to_remove:
            self.cache.remove(formatted_key)
        
        # Update tag index
        if tag in self.tag_index:
            self.tag_index[tag] -= to_remove
            if not self.tag_index[tag]:
                del self.tag_index[tag]
        
        return count
    
    def cleanup(self) -> int:
        """
        Remove expired items from the cache.
        
        Returns:
            Number of items removed
        """
        if self.using_external_cache:
            if hasattr(self.cache_manager, 'cleanup'):
                return self.cache_manager.cleanup()
            return 0
        
        # Use the TimeBasedCache's cleanup method
        removed = self.cache.cleanup()
        
        # Update metrics
        if self.enable_metrics:
            self.metrics["expirations"] += removed
        
        # Cleanup tag index if tags are enabled
        if self.cache_tags and removed > 0:
            # Find all cached keys to check against tag index
            cached_keys = set(self.cache.keys())
            
            # Clean up tag index
            for tag, keys in list(self.tag_index.items()):
                # Remove keys that are no longer in the cache
                invalid_keys = keys - cached_keys
                self.tag_index[tag] -= invalid_keys
                
                # Remove empty tag entries
                if not self.tag_index[tag]:
                    del self.tag_index[tag]
        
        return removed
    
    def size(self, result_type: Optional[str] = None) -> int:
        """
        Get the current size of the cache.
        
        Args:
            result_type: Optional result type to count
            
        Returns:
            Number of items in the cache
        """
        if self.using_external_cache:
            # Cannot filter by type with external cache manager
            if hasattr(self.cache_manager, 'size'):
                return self.cache_manager.size()
            return 0
        
        if result_type is None:
            return self.cache.size()
        
        # Count items of specified type
        prefix = f"{self.namespace}:{result_type}:"
        return sum(1 for k in self.cache.keys() if k.startswith(prefix))
    
    def get_metrics(self, result_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache metrics.
        
        Args:
            result_type: Optional result type to get metrics for
            
        Returns:
            Dictionary with cache metrics
        """
        if not self.enable_metrics:
            return {}
        
        if result_type is not None and result_type in self.type_metrics:
            # Return metrics for specific type
            metrics = self.type_metrics[result_type].copy()
            
            # Add hit rate
            total = metrics["hits"] + metrics["misses"]
            metrics["hit_rate"] = metrics["hits"] / total if total > 0 else 0.0
            
            return metrics
        
        # Return overall metrics
        metrics = self.metrics.copy()
        
        # Add cache size
        metrics["size"] = self.size()
        
        # Add hit rate
        total = metrics["hits"] + metrics["misses"]
        metrics["hit_rate"] = metrics["hits"] / total if total > 0 else 0.0
        
        # Add type-specific metrics
        metrics["types"] = self.type_metrics
        
        return metrics
    
    def get_all_tagged(self) -> Dict[str, List[str]]:
        """
        Get all tags and the keys associated with them.
        
        Returns:
            Dictionary mapping tags to lists of keys
        """
        if not self.cache_tags or self.using_external_cache:
            return {}
        
        result = {}
        
        for tag, keys in self.tag_index.items():
            # Extract original keys from formatted keys
            original_keys = []
            for formatted_key in keys:
                parts = formatted_key.split(":", 2)
                original_key = parts[2] if len(parts) > 2 else parts[1]
                original_keys.append(original_key)
            
            result[tag] = original_keys
        
        return result
    
    def get_tags_for_key(self, key: str, result_type: Optional[str] = None) -> List[str]:
        """
        Get all tags associated with a key.
        
        Args:
            key: Cache key
            result_type: Optional result type
            
        Returns:
            List of tags
        """
        formatted_key = self._format_key(key, result_type)
        
        if not self.cache_tags or self.using_external_cache:
            return []
        
        result_obj = self.cache.get(formatted_key)
        
        if result_obj is None or not isinstance(result_obj, CachedResult):
            return []
        
        if result_obj.metadata and "tags" in result_obj.metadata:
            return result_obj.metadata["tags"]
        
        return [] 