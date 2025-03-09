# Cross-Encoder Reranker

The `CrossEncoderReranker` uses cross-encoder models to rerank retrieval results based on query-document relevance scoring. Cross-encoders directly score the relevance between a query and document by processing them together through a transformer model.

## What Are Cross-Encoders?

Cross-encoders differ from the bi-encoders typically used in initial vector retrieval:

- **Bi-encoders** (used in most vector databases) encode the query and documents separately, allowing for fast retrieval through vector similarity search.
- **Cross-encoders** process the query and document together as a pair, which provides higher quality relevance scoring but is more computationally intensive.

## Features

- **High-Quality Relevance Scoring**: Cross-encoders typically outperform bi-encoders for relevance ranking because they directly model the relationship between query and document.
- **Flexible Model Selection**: Use any Hugging Face cross-encoder model, with a default of `ms-marco-MiniLM-L-6-v2`.
- **Score Normalization**: Options for scaling scores to [0, 1] range and softmax normalization.
- **Batch Processing**: Configurable batch size for efficient processing of multiple documents.
- **GPU Acceleration**: Automatic GPU usage when available for faster inference.
- **Detailed Explanations**: Comprehensive explanations of reranking decisions and score changes.

## When to Use

Cross-encoder reranking is most valuable when:

- **Precision is Critical**: When you need the highest quality rankings, especially for the top results.
- **Computation Budget Allows**: When you can afford the additional computation time for reranking.
- **Result Set is Limited**: Typically applied to a subset of initial results (e.g., top 100) due to computational costs.
- **Complex Relevance Criteria**: When relevance depends on nuanced relationships between query and document.

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_reranker

# Create a cross-encoder reranker
reranker = get_reranker(
    reranker_type="cross_encoder",
    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default model
)

# Initial retrieval results from your vector database or search system
initial_results = [
    {"id": "doc1", "content": "Machine learning is a subfield of artificial intelligence...", "score": 0.85},
    {"id": "doc2", "content": "Deep learning uses neural networks with many layers...", "score": 0.82},
    {"id": "doc3", "content": "Natural language processing deals with interactions between computers and human language...", "score": 0.78}
]

# Rerank the results
query = "How do neural networks work?"
reranked_results = reranker.rerank(query, initial_results)

# Use the reranked results
for i, result in enumerate(reranked_results):
    print(f"{i+1}. [Score: {result['score']:.4f}] {result['id']}: {result['content'][:50]}...")
```

### Configuration Options

```python
reranker = get_reranker(
    reranker_type="cross_encoder",
    model_name_or_path="cross-encoder/ms-marco-TinyBERT-L-2-v2",  # Smaller, faster model
    config={
        'batch_size': 16,              # Process 16 documents at a time
        'scale_scores': True,          # Scale scores to [0, 1] range
        'content_field': 'text',       # Field containing document content
        'max_length': 256,             # Maximum sequence length
        'normalize_scores': True,      # Apply softmax to scores
        'use_gpu': True                # Use GPU if available
    }
)
```

Available configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| `batch_size` | Number of query-document pairs to process at once | 32 |
| `scale_scores` | Scale scores to [0, 1] range | True |
| `content_field` | Field name containing document content | 'content' |
| `max_length` | Maximum sequence length (longer sequences will be truncated) | None (model default) |
| `normalize_scores` | Apply softmax normalization to scores | False |
| `use_gpu` | Use GPU for inference if available | True |

### Explaining Reranking Decisions

You can get detailed explanations of the reranking process:

```python
explanation = reranker.explain_reranking(query, initial_results)

# Print the explanation
print(f"Query: {explanation['query']}")
print(f"Reranker: {explanation['reranking_method']} with model {explanation['model_name']}")

# Show rank changes
for change in explanation['detailed_changes']:
    original_rank = change['original_rank'] + 1  # Convert to 1-indexed for display
    new_rank = change['new_rank'] + 1            # Convert to 1-indexed for display
    
    if change['rank_change'] > 0:
        direction = "↑"  # Moved up
    elif change['rank_change'] < 0:
        direction = "↓"  # Moved down
    else:
        direction = "="  # No change
    
    print(f"Document {change['id']} moved from rank {original_rank} to {new_rank} {direction}")
    print(f"  Original score: {change['original_score']:.4f} → New score: {change['new_score']:.4f}")
    print(f"  Preview: {change['document_preview']}")
```

### Changing Configuration After Initialization

You can update the configuration after creating the reranker:

```python
# Change to using softmax normalization
reranker.set_config({'normalize_scores': True})

# Change the batch size and content field
reranker.set_config({
    'batch_size': 8,
    'content_field': 'body'
})

# Rerank with new configuration
reranked_results = reranker.rerank(query, initial_results)
```

## Recommended Models

The following cross-encoder models are recommended for different use cases:

**General Purpose**:
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (default, good balance of quality and speed)
- `cross-encoder/ms-marco-TinyBERT-L-2-v2` (smaller, faster, lower quality)
- `cross-encoder/ms-marco-electra-base` (higher quality, slower)

**Domain-Specific**:
- `cross-encoder/nli-deberta-v3-base` (good for natural language inference)
- `cross-encoder/stsb-roberta-base` (good for semantic textual similarity)
- `cross-encoder/qnli-distilroberta-base` (good for question answering)

## Practical Tips

1. **Two-Stage Retrieval**: For large document collections, use a vector database for initial retrieval of the top-k (e.g., 100-1000) results, then apply cross-encoder reranking to get the top-n (e.g., 10-20) final results.

2. **Performance Considerations**: Cross-encoder inference is computationally intensive. Consider these trade-offs:
   - Smaller models are faster but may have lower quality
   - Reducing max_length speeds up inference but may lose context
   - Increasing batch_size can improve throughput but requires more memory

3. **Integration with Vector Databases**:
   ```python
   # Example with Chroma
   import chromadb
   from tools.src.retrieval import get_reranker
   
   # Get reranker
   reranker = get_reranker("cross_encoder")
   
   # Create client and collection
   client = chromadb.Client()
   collection = client.get_or_create_collection("docs")
   
   # Initial retrieval with Chroma
   query = "neural networks in deep learning"
   initial_results = collection.query(
       query_texts=[query],
       n_results=100,  # Get more results for reranking
       include=["documents", "metadatas", "distances"]
   )
   
   # Format results for reranker
   formatted_results = [
       {
           "id": initial_results["ids"][0][i],
           "content": initial_results["documents"][0][i],
           "metadata": initial_results["metadatas"][0][i],
           "score": 1.0 - initial_results["distances"][0][i]  # Convert distance to similarity
       }
       for i in range(len(initial_results["ids"][0]))
   ]
   
   # Rerank
   reranked_results = reranker.rerank(query, formatted_results)
   
   # Use top 10 reranked results
   top_results = reranked_results[:10]
   ```

4. **Model Caching**: Cross-encoder models are loaded once when the reranker is initialized and kept in memory for reuse. If you're reranking multiple queries, create the reranker once and reuse it.

## Example

See `tools/examples/retrieval/rerankers/cross_encoder_reranker_example.py` for a complete example. 