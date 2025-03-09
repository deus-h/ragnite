# Vector Keyword Hybrid Searcher

The `VectorKeywordHybridSearcher` combines vector similarity search with keyword search to improve retrieval performance. It runs both search methods in parallel and merges the results using configurable strategies.

## Features

- **Dual Search Strategy**: Combines dense vector embeddings with sparse keyword matching
- **Configurable Weights**: Adjust the importance of vector vs. keyword search results
- **Multiple Combination Methods**: Linear weighted combination or reciprocal rank fusion
- **Score Normalization**: Automatically normalizes scores from different search strategies
- **Result Explanation**: Provides detailed explanations of how results were generated

## When to Use

- **Semantic + Lexical Search**: When you need both semantic understanding and exact keyword matching
- **Cold Start Problems**: When you have little training data for vector embeddings
- **Domain-Specific Terminology**: When specialized terms may not be well-represented in embeddings
- **Improved Recall**: When you want to catch both semantically similar and keyword-matched documents

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_hybrid_searcher

# Create search functions for your vector and keyword search
def vector_search_func(query, limit, **kwargs):
    # Implement vector similarity search
    # Return list of dicts with 'id' and 'score' keys
    ...

def keyword_search_func(query, limit, **kwargs):
    # Implement keyword-based search
    # Return list of dicts with 'id' and 'score' keys
    ...

# Create a hybrid searcher
hybrid_searcher = get_hybrid_searcher(
    searcher_type="vector_keyword",
    vector_search_func=vector_search_func,
    keyword_search_func=keyword_search_func,
    config={
        "vector_weight": 0.7,
        "keyword_weight": 0.3,
        "combination_method": "linear_combination",
        "normalize_scores": True,
        "expand_results": True
    }
)

# Search for documents
results = hybrid_searcher.search("machine learning algorithms", limit=5)
```

### Configuration Options

The `VectorKeywordHybridSearcher` accepts the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `vector_weight` | Weight for vector search results | 0.7 |
| `keyword_weight` | Weight for keyword search results | 0.3 |
| `combination_method` | Method to combine results (`linear_combination` or `reciprocal_rank_fusion`) | `linear_combination` |
| `min_score_threshold` | Minimum score for results to be included | 0.0 |
| `normalize_scores` | Whether to normalize scores before combining | `True` |
| `expand_results` | Whether to include all results from both methods (increases recall) | `True` |

### Adjusting Weights

You can dynamically adjust the weights of the vector and keyword components:

```python
# Favor keyword search more heavily
hybrid_searcher.set_component_weights({
    "vector": 0.3,
    "keyword": 0.7
})

# Check current weights
weights = hybrid_searcher.get_component_weights()
print(weights)  # {'vector': 0.3, 'keyword': 0.7}

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
# - Component results from vector and keyword search
# - Weights used
# - Configuration details
```

## Example

See `tools/examples/retrieval/hybrid_searchers/vector_keyword_hybrid_searcher_example.py` for a complete example.

## Combination Methods

### Linear Combination

The linear combination method calculates the final score as:

```
final_score = (vector_weight × vector_score) + (keyword_weight × keyword_score)
```

This method is most effective when both search strategies return scores in similar ranges or when score normalization is enabled.

### Reciprocal Rank Fusion

The reciprocal rank fusion (RRF) method combines results based on their ranks rather than scores:

```
RRF_score = (vector_weight × 1/(k + vector_rank)) + (keyword_weight × 1/(k + keyword_rank))
```

Where:
- `vector_rank` is the rank of the document in vector search results
- `keyword_rank` is the rank of the document in keyword search results
- `k` is a constant (default: 60) that controls the balance between ranks

RRF is useful when the score distributions across different search strategies are not comparable.

## Integration with Vector Databases

The `VectorKeywordHybridSearcher` can be integrated with various vector databases:

### Chroma Example

```python
import chromadb
from tools.src.retrieval import get_hybrid_searcher

# Create Chroma client
client = chromadb.Client()
collection = client.get_or_create_collection("documents")

# Define search functions
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

def keyword_search(query, limit, **kwargs):
    # Use Chroma's where filter for keyword matching
    results = collection.query(
        query_texts=[""],  # Empty query for filtering only
        where_document={"$contains": query},
        n_results=limit,
        include=["documents", "metadatas"]
    )
    
    # Calculate simple TF score
    processed_results = []
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        content = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        
        # Simple term frequency scoring
        query_terms = query.lower().split()
        term_count = sum(1 for term in query_terms if term in content.lower())
        score = term_count / len(query_terms)
        
        processed_results.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "score": score
        })
    
    # Sort by score
    processed_results.sort(key=lambda x: x["score"], reverse=True)
    return processed_results

# Create hybrid searcher
hybrid_searcher = get_hybrid_searcher(
    searcher_type="vector_keyword",
    vector_search_func=vector_search,
    keyword_search_func=keyword_search
)
```

## Performance Considerations

- The hybrid search runs both search strategies in parallel, which may increase latency compared to a single search strategy.
- Setting `expand_results` to `True` increases recall but requires retrieving more documents from each search strategy.
- Consider adjusting the weights based on the quality of your embeddings and the importance of exact keyword matches for your use case.

## Limitations

- The searcher requires both vector and keyword search functions to be implemented and compatible.
- The linear combination method assumes that scores from different search strategies are comparable after normalization.
- The searcher does not optimize for performance or memory usage beyond what the underlying search functions provide. 