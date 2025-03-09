# Rerankers

This directory contains rerankers for the RAG Research project. Rerankers are tools that refine the ordering of documents retrieved by a first-stage retriever, improving the relevance of the top results.

## Overview

Reranking is a critical stage in the retrieval pipeline that helps improve relevance by applying more sophisticated relevance judgments to a smaller set of candidate documents. This two-stage approach combines the efficiency of first-stage retrieval with the accuracy of more complex reranking models.

## Available Rerankers

### CrossEncoderReranker

The `CrossEncoderReranker` uses cross-encoder models to rerank documents based on their relevance to a query. Cross-encoders process query-document pairs together through the model, providing high-quality relevance judgments.

Features:
- Highly accurate relevance scoring
- Support for various cross-encoder models
- Batch processing for efficient inference
- Compatible with both CPU and CUDA
- See [cross_encoder_reranker.md](./cross_encoder_reranker.md) for detailed documentation

### MonoT5Reranker

The `MonoT5Reranker` uses T5 models fine-tuned for ranking to rerank documents. It follows the MonoT5 approach where a T5 model is trained to generate "true" or "false" based on query-document relevance, with the probability of generating "true" used as the relevance score.

Features:
- Powerful T5-based models for accurate relevance scoring
- Effective balance of performance and efficiency
- Batch processing with adjustable batch size
- Compatible with CPU and GPU
- See [mono_t5_reranker.md](./mono_t5_reranker.md) for detailed documentation

### LLMReranker

The `LLMReranker` uses large language models (LLMs) to evaluate document relevance. It can work with any LLM API that follows a standard interface, including OpenAI, Anthropic, or local models.

Features:
- Leverages powerful LLMs for nuanced relevance judgments
- Multiple scoring methods for extracting relevance scores
- Customizable prompt templates for different use cases
- Support for multiple LLM providers
- See [llm_reranker.md](./llm_reranker.md) for detailed documentation

### EnsembleReranker

The `EnsembleReranker` combines multiple reranking strategies to improve relevance scoring. By leveraging the strengths of different rerankers, it achieves more robust and accurate rankings.

Features:
- Combines multiple rerankers with configurable weights
- Multiple combination methods (weighted average, max score, reciprocal rank fusion)
- Dynamically adjustable weights and configurable components
- Transparent result explanations with source contributions
- See [ensemble_reranker.md](./ensemble_reranker.md) for detailed documentation

## Usage

To use a reranker, you can create an instance through the factory function:

```python
from tools.src.retrieval import get_reranker

# Create a CrossEncoderReranker
cross_encoder = get_reranker(
    reranker_type="cross_encoder",
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Create a MonoT5Reranker
mono_t5 = get_reranker(
    reranker_type="mono_t5",
    model_name="castorini/monot5-base-msmarco"
)

# Create an LLMReranker with OpenAI
llm_reranker = get_reranker(
    reranker_type="llm",
    provider="openai",
    model="gpt-3.5-turbo"
)

# Create an EnsembleReranker combining multiple strategies
ensemble = get_reranker(
    reranker_type="ensemble",
    rerankers=[cross_encoder, mono_t5],
    weights={"reranker_0": 0.7, "reranker_1": 0.3},
    combination_method="weighted_average"
)

# Example documents retrieved by a first-stage retriever
documents = [
    {"id": "doc1", "content": "Python is a programming language known for its readability."},
    {"id": "doc2", "content": "Java is a programming language used for enterprise applications."},
    {"id": "doc3", "content": "Python is widely used in data science and machine learning."}
]

# Rerank the documents based on their relevance to the query
query = "Python for machine learning"
reranked_documents = cross_encoder.rerank(query, documents, top_k=2)

# Use the reranked documents
for i, doc in enumerate(reranked_documents):
    print(f"{i+1}. [Score: {doc['score']:.4f}] {doc['content']}")
```

## Choosing a Reranker

Different rerankers have different strengths and tradeoffs:

- **CrossEncoderReranker**: Good balance of accuracy and efficiency. Recommended for general purpose reranking.
- **MonoT5Reranker**: Specialized for ranking tasks with strong performance. Good when you need slightly better relevance than cross-encoders.
- **LLMReranker**: Highest potential accuracy but more expensive and slower. Best for specialized domains or when top-tier relevance is critical.
- **EnsembleReranker**: Most robust approach that combines strengths of multiple rerankers. Use when you want to minimize weaknesses of individual rerankers.

## Examples

See the [examples directory](../../../examples/retrieval/rerankers/) for example scripts demonstrating the use of rerankers. 