# Retrieval Visualizer

The `RetrievalVisualizer` is a retrieval debugger component that provides comprehensive visualization tools for retrieval results. It helps evaluate and improve RAG (Retrieval-Augmented Generation) systems by creating visual representations of various aspects of retrieval performance.

## Features

- **Score Visualizations**: Visualize score distributions and score decay patterns.
- **Content Similarity Visualizations**: Create content similarity matrices and maps.
- **Metadata Visualizations**: Analyze distribution and correlation of metadata fields.
- **Comparison Visualizations**: Compare multiple retrieval systems visually.
- **Interactive or Static**: Generate either static Matplotlib plots or interactive Plotly visualizations.

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_retrieval_debugger

# Create a RetrievalVisualizer
visualizer = get_retrieval_debugger(
    debugger_type="visualizer",
    content_field="content",
    score_field="score",
    metadata_fields=["category", "source", "date"],
    interactive=False,  # Set to True for interactive Plotly visualizations
)

# Use specific visualization methods
score_distribution = visualizer.plot_score_distribution(results)
similarity_map = visualizer.plot_content_similarity_map(query, results)

# Or use the analyze method for multiple visualizations
analysis = visualizer.analyze(query, results)
visualizations = analysis["visualizations"]
```

### Visualization Methods

#### Score Visualizations

```python
# Plot score distribution
fig = visualizer.plot_score_distribution(results)

# Plot rank vs score (score decay)
fig = visualizer.plot_rank_vs_score(results)
```

#### Content Visualizations

```python
# Plot content similarity matrix
fig = visualizer.plot_content_similarity_matrix(results)

# Plot 2D content similarity map with query
fig = visualizer.plot_content_similarity_map(query, results)
```

#### Metadata Visualizations

```python
# Plot distribution of a metadata field
fig = visualizer.plot_metadata_distribution(results, "category")

# Plot correlation between a numeric metadata field and scores
fig = visualizer.plot_metadata_correlation(results, "date_score")
```

#### Comparison Visualizations

```python
# Compare score distributions across systems
fig = visualizer.plot_comparison_scores(
    [results_system1, results_system2],
    names=["System 1", "System 2"]
)

# Compare top K results across systems
fig = visualizer.plot_top_k_comparison(
    query,
    [results_system1, results_system2],
    names=["System 1", "System 2"],
    k=5
)
```

### Interface Methods

As a subclass of `BaseRetrievalDebugger`, the visualizer also provides standard interface methods:

```python
# Analyze results and generate multiple visualizations
analysis = visualizer.analyze(query, results)

# Compare multiple result sets
comparison = visualizer.compare(query, [results_set1, results_set2])

# Get insights about retrieval results
insights = visualizer.get_insights(query, results)
```

## Configuration Options

- `content_field`: Field containing document text (default: "content")
- `score_field`: Field containing retrieval scores (default: "score")
- `metadata_fields`: List of metadata fields to include in visualizations (default: [])
- `interactive`: Whether to create interactive Plotly visualizations (default: False)
- `plot_style`: Seaborn style for plots (default: "whitegrid")
- `default_figsize`: Default figure size in inches (default: (10, 6))
- `color_palette`: Color palette for plots (default: "viridis")
- `embedding_dim_reduction`: Method for dimensionality reduction (default: "tsne")
- `output_dir`: Directory to save plots if save_plots is True (default: None)
- `save_plots`: Whether to save generated plots to disk (default: False)
- `plot_format`: Format to save plots (default: "png")
- `dpi`: Resolution for saved plots (default: 150)

## Example

See `tools/examples/retrieval/retrieval_debuggers/retrieval_visualizer_example.py` for a complete usage example.

## Dependencies

- Required: matplotlib, numpy, scikit-learn, seaborn
- Optional: plotly (for interactive visualizations), umap-learn (for UMAP dimensionality reduction)

## Visualization Types Explained

### Score Distribution

Shows the frequency distribution of retrieval scores, helping identify patterns in score ranges, potential thresholds, or biases.

### Rank vs Score

Visualizes how retrieval scores decay as rank increases, useful for determining optimal cutoff points and evaluating ranking quality.

### Content Similarity Matrix

Creates a heatmap showing similarity between all retrieved documents, helping identify redundancy or diverse information.

### Content Similarity Map

Plots documents and query in a 2D space based on content similarity, with distance representing semantic relationships.

### Metadata Distribution

Shows the distribution of values for a specific metadata field, useful for understanding result composition.

### Metadata Correlation

Plots correlation between numeric metadata fields and retrieval scores, helping identify if certain metadata characteristics influence ranking.

### Comparison Scores

Compares score distributions across multiple retrieval systems, using box plots to show statistical differences.

### Top-K Comparison

Compares the scores of top K results across different retrieval systems, showing how ranking quality differs. 