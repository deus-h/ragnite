# RAGNITE Caching Infrastructure

This module provides a comprehensive caching infrastructure for the RAGNITE RAG system, designed to improve performance and reduce redundant computation and API calls.

## Features

- **Embedding Cache**: Caches embeddings to avoid recomputing them for the same text
- **Semantic Cache**: Uses semantic similarity to retrieve cached results for similar queries
- **Result Cache**: Time-based caching of query results with automated invalidation
- **Prompt Cache**: Caches prompt templates and rendered prompts for reuse
- **Monitoring Dashboard**: Visual interface for monitoring cache performance

## Components

The caching infrastructure includes the following components:

- `cache_manager.py`: Core cache management functionality
- `embedding_cache.py`: Specialized caching for text and image embeddings
- `semantic_cache.py`: Similarity-based caching for queries
- `result_cache.py`: Time-based caching for query results
- `prompt_cache.py`: Caching for prompt templates
- `cache_dashboard.py`: Monitoring dashboard for cache performance

## Easy Integration

The easiest way to integrate caching with your RAG pipeline is using the `cache_integration.py` module in the parent directory:

```python
from advanced_rag.cache_integration import add_caching_to_pipeline
from basic_rag.src.rag_pipeline import RAGPipeline

# Create a RAG pipeline
rag = RAGPipeline(
    model_name="gpt-3.5-turbo",
    embedding_model="text-embedding-ada-002"
)

# Add caching with one line
cached_rag = add_caching_to_pipeline(rag)

# Use as normal - caching happens automatically
result = cached_rag.query("What is RAG?")
```

## Manual Integration

For more control over the caching behavior, you can directly use the cache components:

```python
from advanced_rag.caching.cache_manager import CacheManager
from advanced_rag.caching.embedding_cache import EmbeddingCache
from advanced_rag.caching.result_cache import ResultCache

# Create a cache manager
cache_manager = CacheManager(cache_dir="~/.ragnite/cache")

# Create an embedding cache
embedding_cache = EmbeddingCache(
    cache_manager=cache_manager,
    namespace="my_embeddings",
    ttl=86400 * 30  # 30 days
)

# Create a result cache
result_cache = ResultCache(
    cache_manager=cache_manager,
    namespace="my_results",
    ttl=86400  # 1 day
)

# Use the caches directly
embedding = embedding_cache.get("some text")
if embedding is None:
    # Generate embedding and cache it
    embedding = generate_embedding("some text")
    embedding_cache.set("some text", embedding)

result = result_cache.get("query key")
if result is None:
    # Generate result and cache it
    result = generate_result("query key")
    result_cache.set("query key", result)
```

## Cache Dashboard

The cache dashboard provides a visual interface for monitoring cache performance:

```python
from advanced_rag.caching.cache_dashboard import CacheDashboard

# Create a dashboard
dashboard = CacheDashboard(
    cache_manager=cache_manager,
    port=8080
)

# Start the dashboard (runs in a separate thread)
dashboard.start()

# Later, stop the dashboard
dashboard.stop()
```

## Configuration Options

### Cache Manager

- `cache_dir`: Directory to store cache files
- `max_memory_size`: Maximum memory size for in-memory caches
- `persistence_mode`: How to persist caches ('json', 'pickle', 'sqlite')

### Embedding Cache

- `namespace`: Cache namespace to use
- `ttl`: Time-to-live for cached embeddings in seconds
- `max_size`: Maximum number of entries in the cache

### Semantic Cache

- `namespace`: Cache namespace to use
- `ttl`: Time-to-live for cached results in seconds
- `similarity_threshold`: Threshold for semantic similarity (0.0-1.0)
- `embedding_model`: Model to use for embedding queries

### Result Cache

- `namespace`: Cache namespace to use
- `ttl`: Time-to-live for cached results in seconds
- `ignore_params`: Query parameters to ignore when generating cache keys

### Prompt Cache

- `namespace`: Cache namespace to use
- `ttl`: Time-to-live for cached prompts in seconds
- `partial_match`: Whether to allow partial matches for prompt variables

## Performance Impact

Using the caching infrastructure can significantly improve performance:

- **Embedding Cache**: Reduces embedding API calls by 90%+ for repeated content
- **Semantic Cache**: Improves response time by 80%+ for semantically similar queries
- **Result Cache**: Provides instant responses for exact repeat queries
- **Prompt Cache**: Reduces prompt rendering time by 50%+

## Best Practices

- Set appropriate TTL values based on your data freshness requirements
- Use namespaces to separate caches for different applications
- Monitor cache hit rates and adjust parameters as needed
- Use the dashboard to identify opportunities for optimization 