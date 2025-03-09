# Context Analyzer

The `ContextAnalyzer` is a retrieval debugger component that provides in-depth analysis of retrieved context quality and characteristics. It helps evaluate and improve RAG (Retrieval-Augmented Generation) systems by analyzing the relevance, diversity, information density, and readability of retrieved documents.

## Features

- **Content Relevance Analysis**: Evaluates how relevant the retrieved documents are to the query.
- **Content Diversity Analysis**: Measures the diversity of information within the retrieved set.
- **Information Density Analysis**: Quantifies the richness and density of information in retrieved content.
- **Readability Analysis**: Assesses the readability and complexity of retrieved content.
- **Sentiment Analysis**: (Optional) Analyzes the sentiment of retrieved content.

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_retrieval_debugger

# Create a ContextAnalyzer
analyzer = get_retrieval_debugger(
    debugger_type="context_analyzer",
    content_field="content",  # Field containing document text
    analyze_relevance=True,
    analyze_diversity=True,
    analyze_information=True,
    analyze_readability=True,
    similarity_threshold=0.7,  # Threshold for considering documents similar
    relevance_threshold=0.5,   # Threshold for considering content relevant
)

# Analyze retrieved context
analysis = analyzer.analyze(query, results)
```

### Available Analysis Types

You can enable or disable different types of analysis:

```python
analyzer = get_retrieval_debugger(
    debugger_type="context_analyzer",
    analyze_relevance=True,    # Analyze content relevance to query
    analyze_diversity=True,    # Analyze content diversity 
    analyze_information=True,  # Analyze information density
    analyze_readability=True,  # Analyze readability metrics
    analyze_sentiment=False,   # Analyze content sentiment (requires NLTK)
)
```

### Methods

#### `analyze(query, results)`

Analyzes retrieved context and returns detailed metrics.

```python
analysis = analyzer.analyze(query, results)
print(analysis)
```

Output example:
```json
{
  "result_count": 5,
  "relevance": {
    "average_relevance": 0.76,
    "relevant_count": 4,
    "relevant_percentage": 80.0
  },
  "diversity": {
    "diversity_score": 0.82,
    "duplicate_count": 1,
    "unique_information_percentage": 92.5
  },
  "information": {
    "average_information_density": 0.68,
    "average_word_count": 145.2,
    "average_lexical_diversity": 0.71,
    "total_potential_entities": 42
  },
  "readability": {
    "average_reading_ease": 48.3,
    "average_grade_level": 12.1,
    "most_common_difficulty": "Fairly difficult"
  },
  "insights": [
    "4 out of 5 documents are highly relevant to the query",
    "The retrieved context shows good diversity with minimal overlap",
    "Documents contain a high density of technical information",
    "Content requires college-level reading ability"
  ]
}
```

#### `compare(query, result_sets, names=None)`

Compares multiple sets of retrieval results.

```python
comparison = analyzer.compare(
    query,
    [results_set1, results_set2],
    names=["Default System", "Optimized System"]
)
```

#### `evaluate(query, results, ground_truth, content_overlap=False)`

Evaluates retrieved content against ground truth.

```python
evaluation = analyzer.evaluate(query, results, ground_truth)
```

#### `get_insights(query, results)`

Gets actionable insights about the context.

```python
insights = analyzer.get_insights(query, results)
```

## Configuration Options

- `content_field`: Field containing document text (default: "content")
- `similarity_threshold`: Threshold for considering documents similar (default: 0.7)
- `relevance_threshold`: Threshold for considering content relevant (default: 0.5)
- `similarity_method`: Method to calculate document similarity ("tfidf", "jaccard", or "custom")
- `readability_metrics`: Whether to analyze readability metrics (default: True)
- `sentiment_analysis`: Whether to analyze content sentiment (default: False)

## Example

See `tools/examples/retrieval/retrieval_debuggers/context_analyzer_example.py` for a complete usage example.

## Metrics Explained

### Relevance Metrics

- **Average Relevance**: Mean relevance score of all retrieved documents
- **Relevant Count**: Number of documents above the relevance threshold
- **Relevant Percentage**: Percentage of documents that are relevant to the query

### Diversity Metrics

- **Diversity Score**: Overall diversity score (higher is better)
- **Duplicate Count**: Number of document pairs with high similarity
- **Unique Information Percentage**: Percentage of unique (non-duplicated) content

### Information Density Metrics

- **Average Information Density**: Mean information density score
- **Average Word Count**: Mean word count per document
- **Average Lexical Diversity**: Mean lexical diversity score
- **Total Potential Entities**: Estimated number of entities across all documents

### Readability Metrics

- **Average Reading Ease**: Mean Flesch Reading Ease score
- **Average Grade Level**: Mean Flesch-Kincaid Grade Level
- **Most Common Difficulty**: Most frequent difficulty level across documents 