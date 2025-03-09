# Retrieval Debuggers

This directory contains various retrieval debugger components for analyzing, evaluating, and improving RAG (Retrieval-Augmented Generation) systems.

## Available Debuggers

### 1. RetrievalInspector

The `RetrievalInspector` provides detailed analysis of retrieval results to help identify issues and improve retrieval quality. It supports analysis of individual queries, batches of queries, and comparison between different retrieval systems.

[RetrievalInspector Documentation](./retrieval_inspector.md)

### 2. QueryAnalyzer

The `QueryAnalyzer` focuses on in-depth analysis of queries themselves, helping you understand query complexity, ambiguity, and other characteristics that may affect retrieval performance.

[QueryAnalyzer Documentation](./query_analyzer.md)

### 3. ContextAnalyzer

The `ContextAnalyzer` provides comprehensive analysis of retrieved context quality and characteristics, helping evaluate and improve the content that serves as input to generation models. It analyzes relevance, diversity, information density, and readability.

[ContextAnalyzer Documentation](./context_analyzer.md)

## Using Retrieval Debuggers

All retrieval debuggers can be instantiated using the `get_retrieval_debugger` factory function:

```python
from tools.src.retrieval import get_retrieval_debugger

# Create a retrieval inspector
inspector = get_retrieval_debugger(
    debugger_type="retrieval_inspector",
    # additional parameters
)

# Create a query analyzer
query_analyzer = get_retrieval_debugger(
    debugger_type="query_analyzer",
    # additional parameters
)

# Create a context analyzer
context_analyzer = get_retrieval_debugger(
    debugger_type="context_analyzer",
    # additional parameters
)
```

## Common Interface

All retrieval debuggers implement a common interface with the following methods:

- `analyze(query, results)`: Analyzes retrieval results for a given query
- `compare(query, result_sets, names=None)`: Compares multiple sets of retrieval results
- `evaluate(query, results, ground_truth)`: Evaluates retrieval results against ground truth
- `get_insights(query, results)`: Gets actionable insights from retrieval results

## Examples

See the [examples directory](../../../../examples/retrieval/retrieval_debuggers/) for complete usage examples for each debugger type. 