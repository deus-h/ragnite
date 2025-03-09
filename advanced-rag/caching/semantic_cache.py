"""
Semantic Cache Module

This module provides semantic caching capabilities for RAG queries,
enabling reuse of results for semantically similar questions.
"""

import logging
import hashlib
import time
import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from collections import OrderedDict
from datetime import datetime, timedelta

try:
    from .cache_manager import CacheManager, SemanticCache as BaseSemanticCache
except ImportError:
    from cache_manager import CacheManager, SemanticCache as BaseSemanticCache

# Configure logging
logger = logging.getLogger(__name__)

class QueryResult:
    """Container class for cached query results with metadata."""
    
    def __init__(
        self,
        query: str,
        embedding: List[float],
        result: Any,
        timestamp: float = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None
    ):
        """
        Initialize a query result.
        
        Args:
            query: Original query string
            embedding: Query embedding vector
            result: Query result data
            timestamp: Creation timestamp (defaults to current time)
            metadata: Additional metadata about the query/result
            ttl_seconds: Time-to-live in seconds (None means no expiration)
        """
        self.query = query
        self.embedding = embedding
        self.result = result
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}
        self.ttl_seconds = ttl_seconds
        self.access_count = 0
        self.last_access_time = self.timestamp
    
    def is_expired(self) -> bool:
        """
        Check if this result has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.ttl_seconds is None:
            return False
        return time.time() > (self.timestamp + self.ttl_seconds)
    
    def access(self) -> None:
        """Record an access to this result."""
        self.access_count += 1
        self.last_access_time = time.time()
    
    def age_seconds(self) -> float:
        """
        Get the age of this result in seconds.
        
        Returns:
            Age in seconds
        """
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            "query": self.query,
            "embedding": self.embedding,
            "result": self.result,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "ttl_seconds": self.ttl_seconds,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResult':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary representation
            
        Returns:
            QueryResult instance
        """
        instance = cls(
            query=data["query"],
            embedding=data["embedding"],
            result=data["result"],
            timestamp=data["timestamp"],
            metadata=data["metadata"],
            ttl_seconds=data["ttl_seconds"]
        )
        instance.access_count = data.get("access_count", 0)
        instance.last_access_time = data.get("last_access_time", instance.timestamp)
        return instance


class SemanticQueryCache:
    """
    Semantic cache for storing and retrieving results based on query similarity.
    This cache matches semantically similar queries to provide faster responses.
    """
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.92,
        similarity_function: Optional[Callable] = None,
        max_size: int = 1000,
        default_ttl: Optional[int] = 3600 * 24,  # 24 hours
        namespace: str = "semantic_cache",
        enable_result_scoring: bool = True,
        score_decay_rate: float = 0.5,  # Higher means faster decay with time
        rerank_cached_results: bool = True,
        max_rerank_results: int = 5,  # Maximum number of semantically similar results to rerank
        auto_update_query_embeddings: bool = True
    ):
        """
        Initialize the semantic query cache.
        
        Args:
            cache_manager: Optional central cache manager to use
            embedder: Function to generate embeddings for queries
            similarity_threshold: Threshold for considering queries similar
            similarity_function: Function to compute similarity between embeddings
            max_size: Maximum number of queries to cache
            default_ttl: Default time-to-live in seconds (None means no expiration)
            namespace: Namespace for the cache
            enable_result_scoring: Whether to score results based on similarity and freshness
            score_decay_rate: Rate at which scores decay with time
            rerank_cached_results: Whether to rerank multiple cached results
            max_rerank_results: Maximum number of semantically similar results to rerank
            auto_update_query_embeddings: Whether to update query embeddings when cache hits
        """
        # External components
        self.cache_manager = cache_manager
        self.embedder = embedder
        self.using_external_cache = cache_manager is not None
        
        # Internal cache if not using external cache manager
        if not self.using_external_cache:
            self.cache = OrderedDict()
            self.cache_by_id = {}
        else:
            if not hasattr(cache_manager, 'semantic_cache') or not cache_manager.semantic_cache:
                raise ValueError("Cache manager must have semantic_cache enabled")
        
        # Configuration
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.namespace = namespace
        self.enable_result_scoring = enable_result_scoring
        self.score_decay_rate = score_decay_rate
        self.rerank_cached_results = rerank_cached_results
        self.max_rerank_results = max_rerank_results
        self.auto_update_query_embeddings = auto_update_query_embeddings
        
        # Set similarity function
        if similarity_function:
            self.similarity_function = similarity_function
        else:
            self.similarity_function = self._cosine_similarity
        
        # Metrics
        self.metrics = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
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
    
    def _get_query_id(self, query: str) -> str:
        """
        Generate a stable ID for a query.
        
        Args:
            query: Query string
            
        Returns:
            Query ID
        """
        # Normalize the query by removing extra whitespace and lowercasing
        normalized_query = " ".join(query.lower().split())
        return f"{self.namespace}:{hashlib.md5(normalized_query.encode()).hexdigest()}"
    
    def _get_embedding(self, query: str) -> Optional[List[float]]:
        """
        Get the embedding for a query.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector or None if embedder is not available
        """
        if not self.embedder:
            return None
        
        try:
            return self.embedder(query)
        except Exception as e:
            logger.error(f"Error generating embedding for query: {str(e)}")
            return None
    
    def _calculate_result_score(self, result: QueryResult, similarity: float) -> float:
        """
        Calculate a score for a result based on similarity, freshness, and popularity.
        
        Args:
            result: Query result
            similarity: Similarity score between the query and the cached query
            
        Returns:
            Combined score
        """
        if not self.enable_result_scoring:
            return similarity
        
        # Age decay factor (newer is better)
        age_hours = result.age_seconds() / 3600
        age_factor = np.exp(-self.score_decay_rate * age_hours / 24)  # Decay based on days
        
        # Usage factor (more used is better)
        usage_factor = np.log1p(result.access_count) / 10  # Log to avoid too much weight
        
        # Combined score
        score = (
            0.6 * similarity +      # Similarity is most important
            0.3 * age_factor +      # Freshness is next
            0.1 * usage_factor      # Usage is least important
        )
        
        return score
    
    def _evict_if_needed(self) -> None:
        """Evict entries if the cache is over capacity."""
        if self.using_external_cache:
            return  # External cache handles eviction
        
        while len(self.cache) > self.max_size:
            # Remove least recently used item
            _, query_id = self.cache.popitem(last=False)
            if query_id in self.cache_by_id:
                del self.cache_by_id[query_id]
                self.metrics["evictions"] += 1
    
    def _update_lru(self, query_id: str) -> None:
        """
        Update the LRU status of a cache entry.
        
        Args:
            query_id: Query ID to mark as recently used
        """
        if self.using_external_cache:
            return  # External cache handles LRU
        
        if query_id in self.cache:
            # Move to the end (most recently used)
            self.cache.move_to_end(query_id)
    
    def get(self, query: str, embedding: Optional[List[float]] = None) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
        """
        Get a result from the cache based on query similarity.
        
        Args:
            query: Query string
            embedding: Query embedding vector (if None, will be generated if embedder available)
            
        Returns:
            Tuple of (hit, result, metadata) where:
                hit: Whether a match was found
                result: The cached result if hit is True, otherwise None
                metadata: Additional information about the cache lookup
        """
        query_id = self._get_query_id(query)
        
        # Get embedding if not provided
        if embedding is None and self.embedder is not None:
            embedding = self._get_embedding(query)
        
        metadata = {
            "query": query,
            "query_id": query_id,
            "match_type": None,
            "similarity": 0.0,
            "cached_query": None,
            "age_seconds": None,
            "access_count": None
        }
        
        # Try exact match first
        exact_hit = False
        
        if self.using_external_cache:
            if hasattr(self.cache_manager, 'get_result'):
                hit, result_obj = self.cache_manager.get_result(query_id)
                if hit and result_obj and not result_obj.is_expired():
                    exact_hit = True
                    result_data = result_obj
        else:
            if query_id in self.cache_by_id:
                result_data = self.cache_by_id[query_id]
                if not result_data.is_expired():
                    exact_hit = True
                else:
                    # Remove expired entry
                    self._remove(query_id)
                    self.metrics["expirations"] += 1
        
        if exact_hit:
            # Update metrics and LRU
            self.metrics["exact_hits"] += 1
            result_data.access()
            self._update_lru(query_id)
            
            # Update metadata
            metadata["match_type"] = "exact"
            metadata["similarity"] = 1.0
            metadata["cached_query"] = result_data.query
            metadata["age_seconds"] = result_data.age_seconds()
            metadata["access_count"] = result_data.access_count
            
            # Update query embeddings if needed
            if self.auto_update_query_embeddings and embedding is not None and not np.array_equal(embedding, result_data.embedding):
                # Average the new embedding with the existing one for continuous learning
                result_data.embedding = [
                    (a + b) / 2 for a, b in zip(result_data.embedding, embedding)
                ]
            
            return True, result_data.result, metadata
        
        # If no embedding or external cache without semantic support, we can't do semantic matching
        if embedding is None or (self.using_external_cache and not hasattr(self.cache_manager, 'get_semantic')):
            self.metrics["misses"] += 1
            return False, None, metadata
        
        # Try semantic matching
        if self.using_external_cache:
            hit, result_obj = self.cache_manager.get_semantic(query, embedding)
            
            if hit and isinstance(result_obj, QueryResult) and not result_obj.is_expired():
                similarity = self.similarity_function(embedding, result_obj.embedding)
                
                # Update metadata
                metadata["match_type"] = "semantic"
                metadata["similarity"] = similarity
                metadata["cached_query"] = result_obj.query
                metadata["age_seconds"] = result_obj.age_seconds()
                metadata["access_count"] = result_obj.access_count
                
                # Update metrics
                self.metrics["semantic_hits"] += 1
                result_obj.access()
                
                return True, result_obj.result, metadata
        else:
            # Find semantically similar queries in our internal cache
            if self.rerank_cached_results:
                # Get all candidates above threshold
                candidates = []
                
                for cached_id, result_data in self.cache_by_id.items():
                    if result_data.is_expired():
                        # Mark for removal but continue processing to avoid modifying during iteration
                        continue
                    
                    if not result_data.embedding:
                        continue
                    
                    similarity = self.similarity_function(embedding, result_data.embedding)
                    
                    if similarity >= self.similarity_threshold:
                        score = self._calculate_result_score(result_data, similarity)
                        candidates.append((cached_id, result_data, similarity, score))
                
                # Remove any expired entries found
                for candidate in candidates:
                    if candidate[1].is_expired():
                        self._remove(candidate[0])
                        self.metrics["expirations"] += 1
                
                # Sort by score and take top matches
                candidates.sort(key=lambda x: x[3], reverse=True)
                top_candidates = candidates[:self.max_rerank_results]
                
                if top_candidates:
                    # Use the highest scoring result
                    best_id, best_result, best_similarity, _ = top_candidates[0]
                    
                    # Update metadata
                    metadata["match_type"] = "semantic"
                    metadata["similarity"] = best_similarity
                    metadata["cached_query"] = best_result.query
                    metadata["age_seconds"] = best_result.age_seconds()
                    metadata["access_count"] = best_result.access_count
                    
                    # Update metrics and LRU
                    self.metrics["semantic_hits"] += 1
                    best_result.access()
                    self._update_lru(best_id)
                    
                    return True, best_result.result, metadata
            else:
                # Just find the best match
                best_match = None
                best_similarity = 0
                
                for cached_id, result_data in self.cache_by_id.items():
                    if result_data.is_expired():
                        # Mark for removal but continue processing to avoid modifying during iteration
                        continue
                    
                    if not result_data.embedding:
                        continue
                    
                    similarity = self.similarity_function(embedding, result_data.embedding)
                    
                    if similarity >= self.similarity_threshold and similarity > best_similarity:
                        best_match = (cached_id, result_data)
                        best_similarity = similarity
                
                # Remove any expired entries found during iteration
                to_remove = [cached_id for cached_id, result_data in self.cache_by_id.items() if result_data.is_expired()]
                for cached_id in to_remove:
                    self._remove(cached_id)
                    self.metrics["expirations"] += 1
                
                if best_match:
                    best_id, best_result = best_match
                    
                    # Update metadata
                    metadata["match_type"] = "semantic"
                    metadata["similarity"] = best_similarity
                    metadata["cached_query"] = best_result.query
                    metadata["age_seconds"] = best_result.age_seconds()
                    metadata["access_count"] = best_result.access_count
                    
                    # Update metrics and LRU
                    self.metrics["semantic_hits"] += 1
                    best_result.access()
                    self._update_lru(best_id)
                    
                    return True, best_result.result, metadata
        
        # No match found
        self.metrics["misses"] += 1
        return False, None, metadata
    
    def put(self, query: str, result: Any, embedding: Optional[List[float]] = None, ttl_seconds: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a result to the semantic cache.
        
        Args:
            query: Query string
            result: Result to cache
            embedding: Query embedding vector (if None, will be generated if embedder available)
            ttl_seconds: Time-to-live in seconds (None means use default)
            metadata: Additional metadata about the query/result
            
        Returns:
            Query ID
        """
        query_id = self._get_query_id(query)
        
        # Get embedding if not provided
        if embedding is None and self.embedder is not None:
            embedding = self._get_embedding(query)
        
        # Create result object
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        result_obj = QueryResult(
            query=query,
            embedding=embedding,
            result=result,
            metadata=metadata,
            ttl_seconds=ttl
        )
        
        if self.using_external_cache:
            if hasattr(self.cache_manager, 'put_result'):
                self.cache_manager.put_result(query_id, result_obj, ttl)
            
            if embedding is not None and hasattr(self.cache_manager, 'put_semantic'):
                self.cache_manager.put_semantic(query_id, embedding, result_obj)
        else:
            # Check capacity and evict if needed
            self._evict_if_needed()
            
            # Add to cache
            self.cache[query_id] = query_id  # For LRU tracking
            self.cache_by_id[query_id] = result_obj
        
        return query_id
    
    def _remove(self, query_id: str) -> None:
        """
        Remove an entry from the cache.
        
        Args:
            query_id: Query ID to remove
        """
        if self.using_external_cache:
            if hasattr(self.cache_manager, 'remove'):
                self.cache_manager.remove(query_id)
        else:
            if query_id in self.cache:
                del self.cache[query_id]
            
            if query_id in self.cache_by_id:
                del self.cache_by_id[query_id]
    
    def remove(self, query: str) -> None:
        """
        Remove a query from the cache.
        
        Args:
            query: Query string to remove
        """
        query_id = self._get_query_id(query)
        self._remove(query_id)
    
    def clear(self) -> None:
        """Clear the semantic cache."""
        if self.using_external_cache:
            if hasattr(self.cache_manager, 'clear'):
                self.cache_manager.clear()
        else:
            self.cache.clear()
            self.cache_by_id.clear()
        
        # Reset metrics
        for key in self.metrics:
            self.metrics[key] = 0
        
        logger.info("Semantic query cache cleared")
    
    def size(self) -> int:
        """
        Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        if self.using_external_cache:
            if hasattr(self.cache_manager, 'size'):
                return self.cache_manager.size()
            return 0
        return len(self.cache_by_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary with cache performance metrics
        """
        metrics = self.metrics.copy()
        
        # Add cache size
        metrics["size"] = self.size()
        
        # Calculate hit rates
        total_requests = metrics["exact_hits"] + metrics["semantic_hits"] + metrics["misses"]
        
        if total_requests > 0:
            metrics["hit_rate"] = (metrics["exact_hits"] + metrics["semantic_hits"]) / total_requests
            metrics["exact_hit_rate"] = metrics["exact_hits"] / total_requests
            metrics["semantic_hit_rate"] = metrics["semantic_hits"] / total_requests
        else:
            metrics["hit_rate"] = 0.0
            metrics["exact_hit_rate"] = 0.0
            metrics["semantic_hit_rate"] = 0.0
        
        return metrics
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        if self.using_external_cache:
            if hasattr(self.cache_manager, 'cleanup'):
                return self.cache_manager.cleanup()
            return 0
        
        # Find expired entries
        expired_ids = [
            query_id for query_id, result_data in self.cache_by_id.items()
            if result_data.is_expired()
        ]
        
        # Remove expired entries
        for query_id in expired_ids:
            self._remove(query_id)
        
        removed_count = len(expired_ids)
        self.metrics["expirations"] += removed_count
        
        return removed_count 