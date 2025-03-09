# Multi-Index Hybrid Searcher

The `MultiIndexHybridSearcher` allows searching across multiple indices, collections, or data sources and combining the results into a unified set of search results. This enables searching over heterogeneous data sources while providing a single, coherent interface to the user.

## Features

- **Cross-Index Search**: Search across multiple indices, collections, or data sources simultaneously
- **Weighted Contribution**: Configure custom weights for each data source based on its importance
- **Flexible Configuration**: Each data source can have its own search function and parameters
- **Multiple Combination Methods**: Linear weighted combination or reciprocal rank fusion
- **Source Tracking**: Includes information about which source provided each result
- **Detailed Explanations**: Get comprehensive explanations of how results were generated

## When to Use

The `MultiIndexHybridSearcher` is particularly valuable when:

- **Searching Across Multiple Data Sources**: When your information is spread across multiple indices or databases
- **Combining Different Document Types**: When you need to search across different types of documents (e.g., code, documentation, and support tickets)
- **Cross-Silo Information Retrieval**: When information is stored in different organizational silos
- **Multi-Domain Search**: When you need to search across different domains or specialized knowledge bases
- **A/B Testing Search Strategies**: When comparing different search approaches on the same query

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_hybrid_searcher

# Define search functions for different indices
def search_documentation(query, limit, **kwargs):
    # Search in documentation index
    # Return list of dicts with 'id', 'content', and 'score' keys
    ...

def search_code_repository(query, limit, **kwargs):
    # Search in code repository
    # Return list of dicts with 'id', 'content', and 'score' keys
    ...

def search_support_tickets(query, limit, **kwargs):
    # Search in support tickets
    # Return list of dicts with 'id', 'content', and 'score' keys
    ...

# Create a list of search function configurations
search_funcs = [
    {
        'func': search_documentation,
        'weight': 0.5,
        'name': 'documentation',
        'params': {'index_name': 'docs'}  # Additional parameters specific to this function
    },
    {
        'func': search_code_repository,
        'weight': 0.3,
        'name': 'code',
        'params': {'language': 'python'}
    },
    {
        'func': search_support_tickets,
        'weight': 0.2,
        'name': 'tickets',
        'params': {'status': 'resolved'}
    }
]

# Create the hybrid searcher
multi_index_searcher = get_hybrid_searcher(
    searcher_type="multi_index",
    search_funcs=search_funcs,
    config={
        'combination_method': 'linear_combination',
        'normalize_scores': True,
        'include_source': True
    }
)

# Search across all indices
results = multi_index_searcher.search("how to implement authentication", limit=10)

# Use the combined results
for i, result in enumerate(results):
    print(f"{i+1}. [{result['source']}] {result['id']}: {result['score']:.4f}")
    print(f"   {result['content'][:100]}...")
```

### Configuration Options

The `MultiIndexHybridSearcher` accepts the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `combination_method` | Method to combine results (`linear_combination` or `reciprocal_rank_fusion`) | `linear_combination` |
| `min_score_threshold` | Minimum score for results to be included | 0.0 |
| `normalize_scores` | Whether to normalize scores before combining | True |
| `expand_results` | Whether to request more results from each search function | True |
| `include_source` | Whether to include source information in the results | True |

### Search Function Configuration

Each search function configuration in the `search_funcs` list should be a dictionary with the following keys:

| Key | Description | Required? |
|-----|-------------|-----------|
| `func` | The search function to call | Required |
| `weight` | The weight for this function's results (0.0 to 1.0) | Optional (default: 1.0) |
| `name` | A name or identifier for this data source | Optional (default: "index_{i}") |
| `params` | Additional parameters to pass to the search function | Optional (default: {}) |

Search functions should have the signature:
```python
def search_func(query: str, limit: int, **kwargs) -> List[Dict[str, Any]]:
    # ...
```

And return a list of dictionaries, each with at least:
- `id`: A unique identifier for the document
- `score`: A relevance score (higher is better)
- `content` or `text`: The document content

### Adjusting Weights

You can dynamically adjust the weights for each index after initialization:

```python
# Get current weights
weights = multi_index_searcher.get_component_weights()
print(weights)  # {'documentation': 0.5, 'code': 0.3, 'tickets': 0.2}

# Update weights
multi_index_searcher.set_component_weights({
    'documentation': 0.4,
    'code': 0.4,
    'tickets': 0.2
})

# Verify new weights
weights = multi_index_searcher.get_component_weights()
print(weights)  # {'documentation': 0.4, 'code': 0.4, 'tickets': 0.2}
```

### Understanding Results

When `include_source` is set to `True`, each result includes:

```python
{
    'id': 'doc123',
    'content': 'Document content...',
    'score': 0.89,
    'source': 'documentation',  # The source that provided this result
    'sources': [
        {
            'name': 'documentation',
            'score': 0.92,      # Original score from this source
            'weight': 0.5,      # Weight applied to this source
            'weighted_score': 0.46  # Weighted contribution to final score
        }
    ]
}
```

If a document appears in multiple sources, the `sources` list will include entries for each occurrence.

### Explaining Search Results

For debugging or transparency, you can get detailed explanations of the search process:

```python
explanation = multi_index_searcher.explain_search(
    "authentication implementation", 
    limit=5
)

print(f"Query: {explanation['query']}")
print(f"Strategy: {explanation['search_strategy']}")
print(f"Indices searched:")
for index in explanation['indices']:
    print(f"  {index['name']} (weight: {index['weight']})")

print("Results from each component:")
for component in explanation['components']:
    print(f"\n{component['name']} (weight: {component['weight']})")
    for i, result in enumerate(component['results'][:3]):  # Show top 3
        print(f"  {i+1}. {result['id']}: {result['score']:.4f}")

print("\nFinal combined results:")
for i, result in enumerate(explanation['results'][:5]):
    print(f"{i+1}. [{result['source']}] {result['id']}: {result['score']:.4f}")
```

## Examples

### Integration with Vector Databases

Here's an example of combining results from multiple vector databases:

```python
import chromadb
from qdrant_client import QdrantClient

# Initialize Chroma and Qdrant clients
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection("primary_docs")

qdrant_client = QdrantClient(host="localhost", port=6333)
qdrant_collection_name = "secondary_docs"

# Create search functions for each database
def search_chroma(query, limit, **kwargs):
    results = chroma_collection.query(
        query_texts=[query],
        n_results=limit,
        include=["documents", "metadatas", "distances"]
    )
    
    return [
        {
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1.0 - results["distances"][0][i]  # Convert distance to similarity
        }
        for i in range(len(results["ids"][0]))
    ]

def search_qdrant(query, limit, **kwargs):
    results = qdrant_client.search(
        collection_name=qdrant_collection_name,
        query_vector=get_embedding(query),  # Function to get query embedding
        limit=limit
    )
    
    return [
        {
            "id": result.id,
            "content": result.payload.get("content", ""),
            "metadata": {k: v for k, v in result.payload.items() if k != "content"},
            "score": result.score
        }
        for result in results
    ]

# Configure multi-index search
search_funcs = [
    {"func": search_chroma, "weight": 0.6, "name": "primary_docs"},
    {"func": search_qdrant, "weight": 0.4, "name": "secondary_docs"}
]

# Create hybrid searcher
hybrid_searcher = get_hybrid_searcher(
    searcher_type="multi_index",
    search_funcs=search_funcs
)
```

### Specialized Search Domains

Example combining general, code, and scientific search:

```python
from tools.src.retrieval import get_hybrid_searcher

# Assume we have specialized search functions for different domains
def general_search(query, limit, **kwargs):
    # Generic vector search
    ...

def code_search(query, limit, **kwargs):
    # Specialized code search with language understanding
    ...

def scientific_search(query, limit, **kwargs):
    # Scientific paper search with formula understanding
    ...

# Create multi-index searcher with dynamic weighting based on query analysis
def get_domain_weights(query):
    # Simple keyword-based heuristic to determine weights
    if any(term in query.lower() for term in ["code", "function", "class", "algorithm"]):
        return {"general": 0.2, "code": 0.7, "scientific": 0.1}
    elif any(term in query.lower() for term in ["paper", "research", "study", "experiment"]):
        return {"general": 0.2, "code": 0.1, "scientific": 0.7}
    else:
        return {"general": 0.6, "code": 0.2, "scientific": 0.2}

# Initialize with default weights
multi_domain_searcher = get_hybrid_searcher(
    searcher_type="multi_index",
    search_funcs=[
        {"func": general_search, "weight": 0.6, "name": "general"},
        {"func": code_search, "weight": 0.2, "name": "code"},
        {"func": scientific_search, "weight": 0.2, "name": "scientific"}
    ]
)

# Search with dynamic weights adjusted for the query
def domain_aware_search(query, limit=10):
    # Get domain-specific weights for this query
    weights = get_domain_weights(query)
    
    # Update searcher weights
    multi_domain_searcher.set_component_weights(weights)
    
    # Perform search
    return multi_domain_searcher.search(query, limit=limit)
```

## See Also

For a complete example of using the `MultiIndexHybridSearcher`, see `tools/examples/retrieval/hybrid_searchers/multi_index_hybrid_searcher_example.py`. 