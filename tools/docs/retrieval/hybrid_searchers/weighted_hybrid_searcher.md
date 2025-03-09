# Weighted Hybrid Searcher

The `WeightedHybridSearcher` combines multiple search strategies with automatic weight tuning based on performance metrics. This enables creating adaptive search systems that improve over time based on relevance feedback.

## Features

- **Multiple Search Strategies**: Combine any number of search functions with weighted contributions
- **Automatic Weight Tuning**: Automatically adjust weights based on performance metrics
- **Performance Tracking**: Track and analyze search performance over time
- **Configurable Constraints**: Set minimum and maximum weights for each strategy
- **Multiple Combination Methods**: Linear weighted combination or reciprocal rank fusion
- **Detailed Explanations**: Get comprehensive explanations of the search process and results

## When to Use

The `WeightedHybridSearcher` is particularly valuable when:

- **Optimizing Search Performance**: When you want to automatically tune search weights based on relevance feedback
- **Combining Multiple Strategies**: When you need to combine more than two search strategies
- **A/B Testing Search Approaches**: When comparing different search approaches and want to automatically find the optimal weights
- **Adaptive Search Systems**: When building search systems that improve over time based on user feedback
- **Complex Retrieval Tasks**: When different search strategies excel at different aspects of retrieval

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_hybrid_searcher

# Define search functions for different strategies
def vector_search(query, limit, **kwargs):
    # Vector similarity search
    # Return list of dicts with 'id', 'content', and 'score' keys
    ...

def keyword_search(query, limit, **kwargs):
    # Keyword-based search
    # Return list of dicts with 'id', 'content', and 'score' keys
    ...

def bm25_search(query, limit, **kwargs):
    # BM25 search
    # Return list of dicts with 'id', 'content', and 'score' keys
    ...

# Create a list of search function configurations
search_funcs = [
    {
        'func': vector_search,
        'weight': 0.5,
        'name': 'vector',
        'min_weight': 0.2,  # Minimum weight for this strategy
        'max_weight': 0.8   # Maximum weight for this strategy
    },
    {
        'func': keyword_search,
        'weight': 0.3,
        'name': 'keyword',
        'min_weight': 0.1,
        'max_weight': 0.5
    },
    {
        'func': bm25_search,
        'weight': 0.2,
        'name': 'bm25',
        'min_weight': 0.1,
        'max_weight': 0.5
    }
]

# Create the weighted hybrid searcher
weighted_searcher = get_hybrid_searcher(
    searcher_type="weighted",
    search_funcs=search_funcs,
    config={
        'combination_method': 'linear_combination',
        'normalize_scores': True,
        'include_source': True,
        'auto_tune': False  # Start with auto-tuning disabled
    }
)

# Search with the current weights
results = weighted_searcher.search("machine learning algorithms", limit=10)

# Use the results
for i, result in enumerate(results):
    print(f"{i+1}. [{result['source']}] {result['id']}: {result['score']:.4f}")
    print(f"   {result['content'][:100]}...")
```

### Automatic Weight Tuning

To enable automatic weight tuning, you need to provide relevance feedback in the form of relevant document IDs:

```python
# Enable auto-tuning
weighted_searcher.set_config({
    'auto_tune': True,
    'tuning_metric': 'reciprocal_rank',  # Metric to optimize
    'tuning_strategy': 'gradient_descent',  # Tuning strategy
    'learning_rate': 0.01  # Learning rate for gradient descent
})

# Search with auto-tuning
query = "neural networks deep learning"
relevant_doc_ids = ["doc123", "doc456"]  # IDs of documents known to be relevant

# The searcher will automatically adjust weights based on performance
results = weighted_searcher.search(query, limit=10, relevant_doc_ids=relevant_doc_ids)

# Check the updated weights
updated_weights = weighted_searcher.get_component_weights()
print(f"Updated weights: {updated_weights}")
```

### Configuration Options

The `WeightedHybridSearcher` accepts the following configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `combination_method` | Method to combine results (`linear_combination` or `reciprocal_rank_fusion`) | `linear_combination` |
| `min_score_threshold` | Minimum score for results to be included | 0.0 |
| `normalize_scores` | Whether to normalize scores before combining | True |
| `expand_results` | Whether to request more results from each search function | True |
| `include_source` | Whether to include source information in results | True |
| `auto_tune` | Whether to automatically tune weights | False |
| `tuning_metric` | Metric to use for weight tuning (`reciprocal_rank`, `precision@k`, `ndcg`) | `reciprocal_rank` |
| `tuning_strategy` | Strategy for weight tuning (`gradient_descent`) | `gradient_descent` |
| `learning_rate` | Learning rate for gradient-based tuning | 0.01 |
| `tuning_iterations` | Number of iterations for weight tuning | 10 |
| `relevance_threshold` | Threshold for considering a result relevant | 0.5 |

### Search Function Configuration

Each search function configuration in the `search_funcs` list should be a dictionary with the following keys:

| Key | Description | Required? |
|-----|-------------|-----------|
| `func` | The search function to call | Required |
| `weight` | Initial weight for this function's results (0.0 to 1.0) | Optional (default: 1.0) |
| `name` | A name or identifier for this search strategy | Optional (default: "strategy_{i}") |
| `min_weight` | Minimum weight for this strategy | Optional (default: 0.0) |
| `max_weight` | Maximum weight for this strategy | Optional (default: 1.0) |
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

### Adjusting Weights Manually

You can manually adjust the weights for each strategy:

```python
# Get current weights
weights = weighted_searcher.get_component_weights()
print(weights)  # {'vector': 0.5, 'keyword': 0.3, 'bm25': 0.2}

# Update weights
weighted_searcher.set_component_weights({
    'vector': 0.6,
    'keyword': 0.2,
    'bm25': 0.2
})

# Verify new weights
weights = weighted_searcher.get_component_weights()
print(weights)  # {'vector': 0.6, 'keyword': 0.2, 'bm25': 0.2}
```

### Performance Metrics

The searcher calculates the following performance metrics when relevant document IDs are provided:

- **Precision@k**: The proportion of relevant documents in the top k results (for k=1, 3, 5, 10)
- **Reciprocal Rank**: The reciprocal of the rank of the first relevant document
- **nDCG**: Normalized Discounted Cumulative Gain, which measures the ranking quality

### Explaining Search Results

For debugging or transparency, you can get detailed explanations of the search process:

```python
explanation = weighted_searcher.explain_search(
    "deep learning", 
    limit=5,
    relevant_doc_ids=["doc123", "doc456"]
)

print(f"Query: {explanation['query']}")
print(f"Strategy: {explanation['search_strategy']}")

print("\nStrategy weights:")
for strategy in explanation['strategies']:
    print(f"  {strategy['name']}: {strategy['weight']:.2f} (min: {strategy['min_weight']}, max: {strategy['max_weight']})")

print("\nResults from each component:")
for component in explanation['components']:
    print(f"\n{component['name']} (weight: {component['weight']:.2f})")
    if 'metrics' in component:
        print(f"  Metrics: {component['metrics']}")
    for i, result in enumerate(component['results'][:2]):  # Show top 2 from each component
        print(f"  {i+1}. [Score: {result['score']:.4f}] {result['id']}")

print("\nFinal combined results:")
for i, result in enumerate(explanation['results'][:3]):  # Show top 3 combined results
    print(f"{i+1}. [{result['source']}] {result['id']}: {result['score']:.4f}")

if 'performance_metrics' in explanation:
    print("\nPerformance metrics:")
    for metric, value in explanation['performance_metrics'].items():
        print(f"  {metric}: {value:.4f}")

if 'weight_tuning' in explanation:
    print("\nWeight tuning history:")
    history = explanation['weight_tuning']['history']
    for i, query in enumerate(history['queries'][-3:]):  # Show last 3 queries
        print(f"  Query: {query}")
        print(f"  Weights: {history['weights'][i]}")
        for metric, values in history['metrics'].items():
            if i < len(values):
                print(f"  {metric}: {values[i]:.4f}")
        print()
```

## Weight Tuning Strategies

### Gradient Descent

The gradient descent strategy adjusts weights based on the performance of each component:

1. Calculate performance metrics for each search component
2. Increase weights for components with higher performance
3. Decrease weights for components with lower performance
4. Apply constraints (min/max weights)
5. Normalize weights to sum to 1.0

The learning rate controls how quickly weights are adjusted. A higher learning rate leads to faster adaptation but may cause instability.

## Examples

### Adaptive Search System

Example of an adaptive search system that improves over time:

```python
from tools.src.retrieval import get_hybrid_searcher

# Define search functions for different strategies
def vector_search(query, limit, **kwargs):
    # Vector similarity search
    ...

def keyword_search(query, limit, **kwargs):
    # Keyword-based search
    ...

def bm25_search(query, limit, **kwargs):
    # BM25 search
    ...

# Create weighted hybrid searcher with auto-tuning
adaptive_searcher = get_hybrid_searcher(
    searcher_type="weighted",
    search_funcs=[
        {'func': vector_search, 'name': 'vector', 'weight': 0.33},
        {'func': keyword_search, 'name': 'keyword', 'weight': 0.33},
        {'func': bm25_search, 'name': 'bm25', 'weight': 0.34}
    ],
    config={
        'auto_tune': True,
        'tuning_metric': 'ndcg',
        'learning_rate': 0.05
    }
)

# Simulate user feedback over time
queries_and_feedback = [
    ("machine learning", ["doc1", "doc5", "doc9"]),
    ("neural networks", ["doc3", "doc7", "doc12"]),
    ("deep learning", ["doc2", "doc8", "doc15"]),
    # ... more queries and feedback
]

# Process queries and adapt weights
for query, relevant_docs in queries_and_feedback:
    print(f"\nQuery: {query}")
    print(f"Current weights: {adaptive_searcher.get_component_weights()}")
    
    # Search with auto-tuning
    results = adaptive_searcher.search(query, limit=10, relevant_doc_ids=relevant_docs)
    
    print(f"Updated weights: {adaptive_searcher.get_component_weights()}")
    
    # Show top 3 results
    for i, result in enumerate(results[:3]):
        print(f"  {i+1}. [{result['source']}] {result['id']}: {result['score']:.4f}")

# Get performance history
history = adaptive_searcher.get_performance_history()
print("\nPerformance history:")
for i, query in enumerate(history['queries']):
    print(f"Query: {query}")
    print(f"Weights: {history['weights'][i]}")
    for metric, values in history['metrics'].items():
        if i < len(values):
            print(f"{metric}: {values[i]:.4f}")
    print()
```

## See Also

For a complete example of using the `WeightedHybridSearcher`, see `tools/examples/retrieval/hybrid_searchers/weighted_hybrid_searcher_example.py`. 