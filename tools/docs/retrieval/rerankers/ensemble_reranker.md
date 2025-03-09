# Ensemble Reranker

The `EnsembleReranker` combines multiple reranking strategies to improve relevance scoring. By leveraging the strengths of different rerankers, it can achieve more robust and accurate document ranking than any single reranker.

## Features

- Combines multiple rerankers with configurable weights
- Supports different combination methods (weighted average, max score, reciprocal rank fusion)
- Dynamically adjustable weights for each component reranker
- Detailed result explanations including source contributions
- Easy addition and removal of component rerankers

## When to Use

Ensemble rerankers are particularly effective when:

- You want to combine complementary reranking approaches
- Different reranking strategies excel at different aspects of relevance
- You need more robust rankings that are less susceptible to individual reranker weaknesses
- You want to experiment with different weighting schemes
- You need transparency into how different rerankers contribute to the final score

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_reranker

# First, create individual rerankers
cross_encoder_reranker = get_reranker(
    reranker_type="cross_encoder",
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

mono_t5_reranker = get_reranker(
    reranker_type="mono_t5",
    model_name="castorini/monot5-base-msmarco"
)

# Create an ensemble that combines both rerankers
ensemble = get_reranker(
    reranker_type="ensemble",
    rerankers=[cross_encoder_reranker, mono_t5_reranker],
    weights={"reranker_0": 0.7, "reranker_1": 0.3},  # Optional custom weights
    combination_method="weighted_average"  # Optional combination method
)

# Example documents
documents = [
    {"id": "doc1", "content": "Python is a programming language known for its readability."},
    {"id": "doc2", "content": "Java is a programming language used for enterprise applications."},
    {"id": "doc3", "content": "Python is widely used in data science and machine learning."}
]

# Rerank documents
query = "Python for machine learning"
reranked_documents = ensemble.rerank(query, documents)

# Use the reranked documents
for i, doc in enumerate(reranked_documents):
    print(f"{i+1}. [Score: {doc['score']:.4f}] {doc['content']}")
    
    # Optionally examine source contributions
    if 'sources' in doc:
        print("  Source contributions:")
        for source in doc['sources']:
            print(f"    - {source['name']}: {source['score']:.4f} (weight: {source['weight']:.2f})")
```

### Different Combination Methods

The `EnsembleReranker` supports three combination methods:

#### 1. Weighted Average (Default)

```python
ensemble = get_reranker(
    reranker_type="ensemble",
    rerankers=[reranker1, reranker2, reranker3],
    combination_method="weighted_average"
)
```

This method combines scores using a weighted average, where each reranker's output is weighted according to its configured weight.

#### 2. Max Score

```python
ensemble = get_reranker(
    reranker_type="ensemble",
    rerankers=[reranker1, reranker2, reranker3],
    combination_method="max_score"
)
```

This method takes the maximum score from any reranker for each document, ignoring the weights. It's useful when you want the most optimistic estimate of relevance.

#### 3. Reciprocal Rank Fusion

```python
ensemble = get_reranker(
    reranker_type="ensemble",
    rerankers=[reranker1, reranker2, reranker3],
    combination_method="reciprocal_rank_fusion",
    config={"rrf_k": 60}  # Optional RRF parameter
)
```

This method combines rankings (not scores) using the reciprocal rank fusion algorithm, which is particularly good at balancing different reranking approaches.

### Adjusting Weights Dynamically

You can adjust the weights of component rerankers after initialization:

```python
# Get information about the current rerankers and weights
reranker_info = ensemble.get_reranker_info()
for info in reranker_info:
    print(f"{info['name']} ({info['type']}): weight = {info['weight']}")

# Update the weights
ensemble.set_weights({
    "reranker_0": 0.8,
    "reranker_1": 0.2
})

# Change the combination method
ensemble.set_combination_method("reciprocal_rank_fusion")
```

### Adding and Removing Rerankers

You can modify the ensemble by adding or removing rerankers:

```python
# Add a new reranker
llm_reranker = get_reranker(
    reranker_type="llm",
    provider="openai"
)

ensemble.add_reranker(llm_reranker, weight=0.4)

# Remove a reranker by index
ensemble.remove_reranker(1)  # Removes the second reranker
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `rerankers` | List of reranker objects | Required |
| `weights` | Dictionary mapping reranker names to weights | Equal weights |
| `combination_method` | Method to combine scores (`weighted_average`, `max_score`, `reciprocal_rank_fusion`) | `weighted_average` |
| `rrf_k` | Constant for reciprocal rank fusion | 60 |

## Examples

### Combining Different Types of Rerankers

```python
from tools.src.retrieval import get_reranker

# Create different types of rerankers
cross_encoder = get_reranker(
    reranker_type="cross_encoder",
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

mono_t5 = get_reranker(
    reranker_type="mono_t5",
    model_name="castorini/monot5-base-msmarco"
)

llm = get_reranker(
    reranker_type="llm",
    provider="openai",
    model="gpt-3.5-turbo"
)

# Create an ensemble with all three rerankers
ensemble = get_reranker(
    reranker_type="ensemble",
    rerankers=[cross_encoder, mono_t5, llm],
    weights={
        "reranker_0": 0.5,  # Cross-encoder (faster, general purpose)
        "reranker_1": 0.3,  # MonoT5 (specialized for ranking)
        "reranker_2": 0.2   # LLM (most powerful but slowest)
    }
)

# The rest of your code remains the same
```

### Comparing Combination Methods

```python
from tools.src.retrieval import get_reranker
import time

# Create component rerankers
reranker1 = get_reranker(reranker_type="cross_encoder")
reranker2 = get_reranker(reranker_type="mono_t5")

# Create ensembles with different combination methods
weighted_avg = get_reranker(
    reranker_type="ensemble",
    rerankers=[reranker1, reranker2],
    combination_method="weighted_average"
)

max_score = get_reranker(
    reranker_type="ensemble",
    rerankers=[reranker1, reranker2],
    combination_method="max_score"
)

rrf = get_reranker(
    reranker_type="ensemble",
    rerankers=[reranker1, reranker2],
    combination_method="reciprocal_rank_fusion"
)

# Example documents
documents = [
    {"id": "doc1", "content": "Machine learning algorithms build a model based on sample data to make predictions."},
    {"id": "doc2", "content": "Python is a high-level programming language known for its readability."},
    {"id": "doc3", "content": "Deep learning is a subset of machine learning that uses neural networks."},
    {"id": "doc4", "content": "JavaScript is a scripting language that enables interactive web pages."}
]

query = "How does deep learning compare to traditional machine learning?"

# Compare results from different methods
print("Weighted Average Method:")
results1 = weighted_avg.rerank(query, documents.copy())
for i, doc in enumerate(results1):
    print(f"{i+1}. [Score: {doc['score']:.4f}] {doc['id']}")

print("\nMax Score Method:")
results2 = max_score.rerank(query, documents.copy())
for i, doc in enumerate(results2):
    print(f"{i+1}. [Score: {doc['score']:.4f}] {doc['id']}")

print("\nReciprocal Rank Fusion Method:")
results3 = rrf.rerank(query, documents.copy())
for i, doc in enumerate(results3):
    print(f"{i+1}. [Score: {doc['score']:.4f}] {doc['id']}")
```

## Notes

1. The `EnsembleReranker` works with any combination of rerankers that implement the `BaseReranker` interface.

2. When using rerankers with very different score distributions, consider using "reciprocal_rank_fusion" as it works with ranks rather than raw scores.

3. Component rerankers can be computationally expensive, especially LLM-based ones. The ensemble will be at least as slow as its slowest component.

4. The weights are automatically normalized to sum to 1.0.

5. For debugging purposes, examine the "sources" field in the results to see how each component reranker contributed to the final score. 