# LLM Reranker

The `LLMReranker` leverages Large Language Models (LLMs) to evaluate the relevance of documents to a query. This reranker can work with any LLM API that follows a standard interface, including OpenAI, Anthropic, or local models. The LLM is prompted to rate the relevance of each document, and these ratings are used to rerank the documents.

## Features

- Leverages powerful LLMs for nuanced relevance judgments
- Supports multiple scoring methods to extract relevance scores from LLM outputs
- Compatible with various LLM providers (OpenAI, Anthropic, local models)
- Customizable prompt templates for different use cases
- Batch processing for efficiency
- Built-in rate limiting to prevent API throttling

## When to Use

LLM rerankers are particularly effective when:

- You need highly accurate, contextually aware relevance judgments
- You want to capture semantic nuances that simpler models might miss
- You need to evaluate relevance with complex or domain-specific criteria
- You have a relatively small number of candidate documents (due to API costs)
- Latency is not your primary concern

## Usage

### Basic Usage with OpenAI

```python
from tools.src.retrieval import get_reranker

# Create an LLMReranker using OpenAI
reranker = get_reranker(
    reranker_type="llm",
    provider="openai",
    api_key="your-openai-api-key",  # Optional, can use OPENAI_API_KEY env var
    model="gpt-3.5-turbo"
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

### Using Anthropic

```python
# Create an LLMReranker using Anthropic
reranker = get_reranker(
    reranker_type="llm",
    provider="anthropic",
    api_key="your-anthropic-api-key",  # Optional, can use ANTHROPIC_API_KEY env var
    model="claude-3-haiku-20240307"
)
```

### Using a Custom LLM Provider

```python
# Define a custom LLM provider function
def my_llm_provider(prompt, **kwargs):
    # Implement your own LLM API call here
    # This could be a local model, a custom API, etc.
    # The function should return a string response
    ...
    return response

# Create an LLMReranker with the custom provider
reranker = get_reranker(
    reranker_type="llm",
    llm_provider=my_llm_provider,
    scoring_method="direct",
    prompt_template="Rate the relevance of the document to the query on a scale of 0-10:\nQuery: {query}\nDocument: {document}\nRating:"
)
```

## Scoring Methods

The `LLMReranker` supports several methods for extracting relevance scores from LLM outputs:

- **direct**: Extracts a numerical score directly from the LLM's response.
- **json**: Extracts a score from a JSON structure in the LLM's response.
- **scale_1_10**: Extracts a score on a 1-10 scale and normalizes it.
- **scale_1_5**: Extracts a score on a 1-5 scale and normalizes it.

Example with JSON scoring:

```python
reranker = get_reranker(
    reranker_type="llm",
    provider="openai",
    scoring_method="json",
    prompt_template=(
        "Evaluate the relevance of the document to the query. "
        "Return a JSON object with a 'score' field between 0 and 10.\n\n"
        "Query: {query}\n\n"
        "Document: {document}\n\n"
        "JSON response:"
    )
)
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `llm_provider` | Function to call the LLM API | Required |
| `scoring_method` | Method to extract scores (`direct`, `json`, `scale_1_10`, `scale_1_5`) | `direct` |
| `prompt_template` | Template for prompts sent to the LLM | A default template asking for a 0-10 rating |
| `batch_size` | Number of documents to process in a batch | 4 |
| `rate_limit_delay` | Delay between API calls in seconds | 0 (no delay) |

For provider-specific implementations:

| Option | Description | Default |
|--------|-------------|---------|
| `api_key` | API key for the provider | From environment variable |
| `model` | Model to use | `gpt-3.5-turbo` (OpenAI) or `claude-3-haiku-20240307` (Anthropic) |
| `temperature` | Temperature for generation | 0.0 |

## Custom Prompt Templates

You can customize the prompt template to suit your specific use case:

```python
# Example of a custom prompt template for academic relevance
prompt_template = """
Evaluate the relevance of the following document to the research query.
Consider factors like methodology, findings, and theoretical framework.
Rate relevance from 0 (completely irrelevant) to 10 (perfectly relevant).

Research Query: {query}

Document: {document}

Academic Relevance Score (0-10):
"""

reranker = get_reranker(
    reranker_type="llm",
    provider="openai",
    prompt_template=prompt_template
)
```

## Examples

### Complete Example with Performance Metrics

```python
from tools.src.retrieval import get_reranker
import time
import os

# Set API key in environment variable
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Create an LLM reranker
reranker = get_reranker(
    reranker_type="llm",
    provider="openai",
    model="gpt-3.5-turbo",
    batch_size=2,
    config={"rate_limit_delay": 0.5}  # Add delay to avoid rate limits
)

# Example documents
documents = [
    {"id": "doc1", "content": "Machine learning algorithms build a model based on sample data to make predictions without being explicitly programmed."},
    {"id": "doc2", "content": "Python is a high-level programming language known for its readability and simplicity."},
    {"id": "doc3", "content": "Deep learning is a subset of machine learning that uses multi-layered neural networks to learn from data."},
    {"id": "doc4", "content": "JavaScript is a scripting language that enables interactive web pages and is an essential part of web applications."}
]

# Measure reranking time
start_time = time.time()
query = "How does deep learning differ from traditional machine learning?"
reranked_docs = reranker.rerank(query, documents)
end_time = time.time()

print(f"Reranking took {end_time - start_time:.2f} seconds\n")

# Display results
print("Reranked documents:")
for i, doc in enumerate(reranked_docs):
    print(f"{i+1}. [Score: {doc['score']:.4f}] {doc['id']}: {doc['content']}")
```

### Example with JSON Scoring

```python
from tools.src.retrieval import get_reranker

# Create a reranker that expects JSON output
json_reranker = get_reranker(
    reranker_type="llm",
    provider="openai",
    model="gpt-4",  # More capable model for structured output
    scoring_method="json",
    prompt_template=(
        "Analyze the relevance of the document to the query. "
        "Return your analysis as a JSON object with the following fields:\n"
        "- score: A number between 0 and 10 representing relevance\n"
        "- reasoning: A brief explanation of your score\n\n"
        "Query: {query}\n\n"
        "Document: {document}\n\n"
        "JSON response:"
    )
)

# The rest of your code remains the same
```

## Notes

1. LLM reranking can be expensive due to API costs, especially for large document sets. Consider using it as a final reranking stage after filtering candidates with faster methods.

2. The response format from LLMs can vary, especially when using different providers or models. If you're having trouble with score extraction, try using a more explicit prompt or a different scoring method.

3. For production use with high volume, consider implementing caching to avoid redundant API calls for the same query-document pairs.

4. When using OpenAI or other commercial APIs, be aware of rate limits and costs. Use batch_size and rate_limit_delay to avoid hitting rate limits. 