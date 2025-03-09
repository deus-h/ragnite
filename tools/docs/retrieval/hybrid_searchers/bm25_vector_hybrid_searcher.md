# BM25 Vector Hybrid Searcher

The `BM25VectorHybridSearcher` combines BM25 keyword search with vector similarity search to improve retrieval performance. BM25 is a sophisticated keyword search algorithm that provides better term frequency normalization and document length considerations than simple keyword matching.

## Features

- **Advanced Keyword Search**: Uses BM25 (Okapi BM25, BM25+ or BM25L variants) for sophisticated keyword matching
- **Semantic Understanding**: Combines BM25 with vector embeddings for semantic search
- **Flexible BM25 Integration**: Use an external BM25 search function or build an internal BM25 index
- **Configurable BM25 Parameters**: Fine-tune k1, b, and delta parameters for optimal performance
- **Multiple Combination Methods**: Linear weighted combination or reciprocal rank fusion
- **Score Normalization**: Automatically normalizes scores from different search strategies
- **Result Explanation**: Provides detailed explanations of how results were generated

## When to Use

- **Better Keyword Matching**: When you need more sophisticated keyword matching than simple term frequency
- **Domain-Specific Terminology**: When specialized terms require both exact matching and semantic understanding
- **Document Length Sensitivity**: When document length normalization is important for your corpus
- **Improved Precision**: When you want to balance semantic similarity with exact keyword relevance

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_hybrid_searcher

# Create a vector search function
def vector_search_func(query, limit, **kwargs):
    # Implement vector similarity search
    # Return list of dicts with 'id' and 'score' keys
    ...

# Option 1: Provide an external BM25 search function
def bm25_search_func(query, limit, **kwargs):
    # Implement BM25 search
    # Return list of dicts with 'id' and 'score' keys
    ...

# Create a hybrid searcher with external BM25 function
hybrid_searcher = get_hybrid_searcher(
    searcher_type="bm25_vector",
    vector_search_func=vector_search_func,
    bm25_search_func=bm25_search_func,
    config={
        "vector_weight": 0.5,
        "bm25_weight": 0.5,
        "combination_method": "linear_combination"
    }
)

# Option 2: Let the searcher build an internal BM25 index
corpus = ["document text 1", "document text 2", "document text 3"]
doc_ids = ["doc1", "doc2", "doc3"]

# Create a hybrid searcher with internal BM25 index
hybrid_searcher = get_hybrid_searcher(
    searcher_type="bm25_vector",
    vector_search_func=vector_search_func,
    corpus=corpus,
    doc_ids=doc_ids,
    config={
        "bm25_variant": "plus",  # 'okapi', 'plus', or 'l'
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "bm25_delta": 0.5  # only for BM25+ and BM25L
    }
)

# Search for documents
results = hybrid_searcher.search("machine learning algorithms", limit=5)
```

### Configuration Options

The `BM25VectorHybridSearcher` accepts the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `vector_weight` | Weight for vector search results | 0.5 |
| `bm25_weight` | Weight for BM25 search results | 0.5 |
| `combination_method` | Method to combine results (`linear_combination` or `reciprocal_rank_fusion`) | `linear_combination` |
| `min_score_threshold` | Minimum score for results to be included | 0.0 |
| `normalize_scores` | Whether to normalize scores before combining | `True` |
| `expand_results` | Whether to include all results from both methods (increases recall) | `True` |
| `bm25_variant` | BM25 variant to use (`okapi`, `plus`, or `l`) | `plus` |
| `bm25_k1` | BM25 k1 parameter (controls term frequency saturation) | 1.5 |
| `bm25_b` | BM25 b parameter (controls document length normalization) | 0.75 |
| `bm25_delta` | BM25L/BM25+ delta parameter (controls lower-bound on term frequency) | 0.5 |

### Adjusting Weights

You can dynamically adjust the weights of the vector and BM25 components:

```python
# Favor BM25 search more heavily
hybrid_searcher.set_component_weights({
    "vector": 0.3,
    "bm25": 0.7
})

# Check current weights
weights = hybrid_searcher.get_component_weights()
print(weights)  # {'vector': 0.3, 'bm25': 0.7}

# Search with new weights
results = hybrid_searcher.search("machine learning algorithms", limit=5)
```

### Changing the Combination Method

The searcher supports two methods for combining results:

1. **Linear Combination** (default): Weighted sum of the normalized scores
2. **Reciprocal Rank Fusion**: Combines results based on their ranks rather than scores

```python
# Change to Reciprocal Rank Fusion
hybrid_searcher.set_config({
    "combination_method": "reciprocal_rank_fusion"
})

# Search with new combination method
results = hybrid_searcher.search("machine learning algorithms", limit=5)
```

### Explaining Search Results

The searcher can provide detailed explanations of how the results were generated:

```python
# Get an explanation of the search results
explanation = hybrid_searcher.explain_search("neural networks", limit=5)

# Explanation contains:
# - Original query
# - Combined results
# - Search strategy description
# - Component results from vector and BM25 search
# - Weights used
# - BM25 parameters
# - Configuration details
```

## BM25 Variants

The searcher supports three BM25 variants:

1. **Okapi BM25**: The original BM25 algorithm
2. **BM25+**: An improved version that adds a small delta to term frequency to prevent zero scores
3. **BM25L**: A variant that better handles long documents

You can select the variant and tune its parameters:

```python
# Use BM25L with custom parameters
hybrid_searcher.set_config({
    "bm25_variant": "l",
    "bm25_k1": 1.2,  # Lower k1 for less term frequency saturation
    "bm25_b": 0.8,   # Higher b for more document length normalization
    "bm25_delta": 0.7  # Higher delta for stronger lower-bound on term frequency
})
```

## Example

See `tools/examples/retrieval/hybrid_searchers/bm25_vector_hybrid_searcher_example.py` for a complete example.

## Integration with Vector Databases

The `BM25VectorHybridSearcher` can be integrated with various vector databases. Here's an example with Chroma:

```python
import chromadb
from rank_bm25 import BM25Plus
from tools.src.retrieval import get_hybrid_searcher

# Create Chroma client
client = chromadb.Client()
collection = client.get_or_create_collection("documents")

# Define vector search function
def vector_search(query, limit, **kwargs):
    results = collection.query(
        query_texts=[query],
        n_results=limit,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results to expected structure
    return [
        {
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1.0 - results["distances"][0][i]  # Convert distance to similarity
        }
        for i in range(len(results["ids"][0]))
    ]

# Get all documents for BM25 index
all_docs = collection.get(include=["documents", "ids"])
corpus = all_docs["documents"]
doc_ids = all_docs["ids"]

# Create hybrid searcher with internal BM25 index
hybrid_searcher = get_hybrid_searcher(
    searcher_type="bm25_vector",
    vector_search_func=vector_search,
    corpus=corpus,
    doc_ids=doc_ids,
    config={
        "vector_weight": 0.6,
        "bm25_weight": 0.4,
        "bm25_variant": "plus"
    }
)
```

## Performance Considerations

- Building an internal BM25 index requires storing the entire corpus in memory, which may be memory-intensive for large document collections.
- The BM25 algorithm is generally faster than vector search for large collections, but the hybrid approach requires running both.
- Consider adjusting the weights based on the quality of your embeddings and the importance of keyword matches for your use case.
- For very large collections, consider providing an external BM25 search function that uses an optimized implementation.

## Limitations

- The internal BM25 implementation uses a simple tokenization approach (lowercase and split on whitespace), which may not be optimal for all languages or domains.
- The searcher does not handle document updates to the internal BM25 index - you need to recreate the searcher if your corpus changes.
- The BM25 algorithm does not understand semantics, so it may miss relevant documents that use different terminology. 