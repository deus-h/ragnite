# Hybrid Searchers

This directory contains hybrid searchers for the RAG Research project. Hybrid searchers combine multiple search strategies to improve retrieval performance.

## Overview

Hybrid search combines different search techniques to leverage the strengths of each approach and overcome their individual limitations. This approach can significantly improve both recall and precision in retrieval tasks.

## Available Hybrid Searchers

### VectorKeywordHybridSearcher

The `VectorKeywordHybridSearcher` combines vector similarity search with keyword search to improve retrieval performance. It runs both search methods in parallel and merges the results using configurable strategies.

Features:
- Combines dense vector embeddings with sparse keyword matching
- Configurable weights for each search component
- Multiple combination methods (linear combination or reciprocal rank fusion)
- Score normalization and result explanation capabilities
- See [vector_keyword_hybrid_searcher.md](./vector_keyword_hybrid_searcher.md) for detailed documentation

### BM25VectorHybridSearcher

The `BM25VectorHybridSearcher` combines BM25 keyword search with vector similarity search. BM25 is a sophisticated keyword search algorithm that provides better term frequency normalization and document length considerations than simple keyword matching.

Features:
- Uses BM25 (Okapi BM25, BM25+ or BM25L variants) for sophisticated keyword matching
- Flexible BM25 integration (external function or internal index)
- Configurable BM25 parameters (k1, b, delta)
- Multiple combination methods (linear combination or reciprocal rank fusion)
- See [bm25_vector_hybrid_searcher.md](./bm25_vector_hybrid_searcher.md) for detailed documentation

### MultiIndexHybridSearcher

The `MultiIndexHybridSearcher` allows searching across multiple indices, collections, or data sources and combining the results into a unified set of search results.

Features:
- Searches across multiple indices or collections simultaneously
- Configurable weights for each data source
- Flexible configuration for each search function
- Source tracking to identify where results came from
- See [multi_index_hybrid_searcher.md](./multi_index_hybrid_searcher.md) for detailed documentation

### WeightedHybridSearcher

The `WeightedHybridSearcher` combines multiple search strategies with automatic weight tuning based on performance metrics.

Features:
- Combines any number of search functions with weighted contributions
- Automatically adjusts weights based on performance metrics
- Tracks and analyzes search performance over time
- Configurable constraints for each strategy's weight
- See [weighted_hybrid_searcher.md](./weighted_hybrid_searcher.md) for detailed documentation

### Planned Hybrid Searchers

The following hybrid searchers are planned for future implementation:

#### MultiIndexHybridSearcher  

Will enable searching across multiple vector indices or collections, combining results from different data sources.

#### WeightedHybridSearcher

Will provide a general framework for weighted combination of multiple search techniques with automated weight tuning.

## Usage

To use a hybrid searcher, you need to provide functions that implement the individual search strategies:

```python
from tools.src.retrieval import get_hybrid_searcher

# Example for VectorKeywordHybridSearcher
def vector_search_func(query, limit, **kwargs):
    # Implement vector similarity search
    # Returns list of dicts with 'id' and 'score' keys
    ...

def keyword_search_func(query, limit, **kwargs):
    # Implement keyword-based search
    # Returns list of dicts with 'id' and 'score' keys
    ...

# Create a vector-keyword hybrid searcher
hybrid_searcher = get_hybrid_searcher(
    searcher_type="vector_keyword",
    vector_search_func=vector_search_func,
    keyword_search_func=keyword_search_func,
    config={
        "vector_weight": 0.7,
        "keyword_weight": 0.3,
        "combination_method": "linear_combination"
    }
)

# Example for BM25VectorHybridSearcher
# Option 1: With external BM25 search function
def bm25_search_func(query, limit, **kwargs):
    # Implement BM25 search
    # Returns list of dicts with 'id' and 'score' keys
    ...

bm25_hybrid_searcher = get_hybrid_searcher(
    searcher_type="bm25_vector",
    vector_search_func=vector_search_func,
    bm25_search_func=bm25_search_func
)

# Option 2: With internal BM25 index
corpus = ["document text 1", "document text 2", "document text 3"]
doc_ids = ["doc1", "doc2", "doc3"]

bm25_hybrid_searcher = get_hybrid_searcher(
    searcher_type="bm25_vector",
    vector_search_func=vector_search_func,
    corpus=corpus,
    doc_ids=doc_ids,
    config={
        "bm25_variant": "plus"  # 'okapi', 'plus', or 'l'
    }
)

# Search for documents
results = hybrid_searcher.search("your query here", limit=10)
```

## Examples

See the [examples directory](../../../examples/retrieval/hybrid_searchers/) for example scripts demonstrating the use of hybrid searchers. 