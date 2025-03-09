# MonoT5 Reranker

The `MonoT5Reranker` uses T5 models fine-tuned for the ranking task to rerank documents based on their relevance to a query. This implementation uses the MonoT5 approach, where a T5 model is fine-tuned to generate "true" or "false" based on query-document relevance. The probability of generating "true" is used as the relevance score.

## Features

- Uses powerful T5-based models fine-tuned specifically for ranking
- Supports batch processing for efficient inference
- Compatible with both CPU and CUDA for flexible deployment
- Handles large documents with configurable maximum length
- Normalizes scores for consistent interpretation

## When to Use

MonoT5 rerankers are particularly effective when:

- You need stronger relevance judgments than basic retrieval methods
- You've already retrieved a set of candidate documents and want to improve their ranking
- You have access to GPU resources (though CPU is supported, GPU is recommended for faster inference)
- You need a reranker that balances performance and efficiency

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_reranker

# Create a MonoT5Reranker with default model (castorini/monot5-base-msmarco)
reranker = get_reranker(
    reranker_type="mono_t5",
    device="cuda"  # or "cpu" for CPU-only inference
)

# Example documents retrieved by a first-stage retriever
documents = [
    {"id": "doc1", "content": "Python is a programming language known for its readability."},
    {"id": "doc2", "content": "Java is a programming language used for enterprise applications."},
    {"id": "doc3", "content": "Python is widely used in data science and machine learning."}
]

# Rerank the documents based on their relevance to the query
query = "Python for machine learning"
reranked_documents = reranker.rerank(query, documents, top_k=2)

# Use the reranked documents
for i, doc in enumerate(reranked_documents):
    print(f"{i+1}. [Score: {doc['score']:.4f}] {doc['content']}")
```

### Using a Different Model

```python
# Create a MonoT5Reranker with a different model
reranker = get_reranker(
    reranker_type="mono_t5",
    model_name="castorini/monot5-large-msmarco",  # Using a larger model
    device="cuda",
    batch_size=4,  # Smaller batch size for larger model
    max_length=384
)
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `model_name` | Name or path of the MonoT5 model to use | `castorini/monot5-base-msmarco` |
| `device` | Device to run the model on (`cuda` or `cpu`) | CUDA if available, otherwise CPU |
| `batch_size` | Batch size for inference | 8 |
| `max_length` | Maximum sequence length for the model | 512 |

## Available MonoT5 Models

Some popular MonoT5 models:

- `castorini/monot5-base-msmarco`: A T5-base model fine-tuned on MS MARCO
- `castorini/monot5-small-msmarco`: A smaller variant for faster inference
- `castorini/monot5-large-msmarco`: A larger variant with better performance
- `castorini/monot5-3b-msmarco`: A 3B parameter model for even better performance

## Notes

1. The reranker requires the `transformers` library. If it's not installed, the module will raise an `ImportError` with instructions to install it.

2. Longer sequences may require more memory. If you encounter memory issues:
   - Reduce the batch size
   - Reduce max_length
   - Use a smaller model
   - Move to a device with more memory

3. MonoT5 models are designed to judge the relevance of a document to a query, not to generate text. They are fine-tuned to output "true" for relevant documents and "false" for non-relevant ones.

## Example

```python
from tools.src.retrieval import get_reranker
import time

# Create a MonoT5Reranker
reranker = get_reranker(
    reranker_type="mono_t5",
    model_name="castorini/monot5-base-msmarco",
    device="cuda"
)

# Example documents
documents = [
    {"id": "doc1", "content": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."},
    {"id": "doc2", "content": "Python is a high-level programming language known for its readability and simplicity."},
    {"id": "doc3", "content": "Data science involves extracting knowledge and insights from structured and unstructured data."},
    {"id": "doc4", "content": "Neural networks are computing systems inspired by the biological neural networks in animal brains."},
    {"id": "doc5", "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."}
]

# Measure reranking time
start_time = time.time()
query = "How do neural networks work in machine learning?"
reranked_docs = reranker.rerank(query, documents)
end_time = time.time()

print(f"Reranking took {end_time - start_time:.2f} seconds\n")

# Display results
print("Reranked documents:")
for i, doc in enumerate(reranked_docs):
    print(f"{i+1}. [Score: {doc['score']:.4f}] {doc['id']}: {doc['content']}")
```

This will likely output the documents in an order that reflects their relevance to the query about neural networks, with documents about neural networks and machine learning ranked higher than general programming documents. 