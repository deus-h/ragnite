# RAG Utility Tools

A comprehensive toolkit for building, debugging, and optimizing Retrieval-Augmented Generation (RAG) systems.

## Overview

RAG Utility Tools provides a collection of modular components for working with every aspect of RAG systems:

- **Data Processing**: Document loaders, text chunkers, metadata extractors, and data cleaners
- **Embeddings**: Embedding generators, visualizers, analyzers, model adapters, and dimensionality reduction
- **Vector Databases**: Database connectors, index optimizers, query benchmarkers, and schema managers
- **Retrieval**: Query processors, retrieval debuggers, filter builders, hybrid searchers, and re-rankers
- **Generation**: Prompt templates, context formatters, output parsers, hallucination detectors, and citation generators
- **Monitoring**: Performance trackers, usage analyzers, error loggers, cost estimators, and latency monitors

## Installation

You can install the package using pip:

```bash
# Install the core package
pip install rag-tools

# Install with specific components
pip install rag-tools[data,embeddings,visualization]

# Install all components
pip install rag-tools[all]
```

## Quick Start

Here's a simple example of using the embedding tools:

```python
from rag_tools.embeddings import get_embedding_generator, get_embedding_visualizer

# Generate embeddings
generator = get_embedding_generator(
    generator_type="sentence-transformers",
    model_name="all-MiniLM-L6-v2"
)

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science."
]

embeddings = generator.batch_generate(texts)

# Visualize embeddings
visualizer = get_embedding_visualizer(
    visualizer_type="matplotlib",
    n_components=2
)

fig = visualizer.visualize(
    embeddings=embeddings,
    labels=["fox", "ai", "python"]
)

visualizer.save("embedding_visualization.png")
```

## Components

### Data Processing

- **Document Loaders**: Load documents from various file formats (PDF, HTML, Markdown, etc.)
- **Text Chunkers**: Split documents into chunks for embedding and retrieval
- **Metadata Extractors**: Extract metadata from documents for filtering and context
- **Data Cleaners**: Clean and normalize text for better embedding quality
- **Data Augmentation**: Generate variations of text for improved retrieval

### Embeddings

- **Embedding Generators**: Generate embeddings using various models (Sentence Transformers, Hugging Face, OpenAI, etc.)
- **Embedding Visualizers**: Visualize embeddings in 2D/3D space
- **Embedding Analyzers**: Analyze embeddings for similarity, clustering, and outliers
- **Model Adapters**: Convert embeddings between different models
- **Dimensionality Reduction**: Reduce embedding dimensions for visualization and efficiency

### Vector Databases

- **Database Connectors**: Connect to various vector databases (Chroma, Qdrant, Pinecone, etc.)
- **Index Optimizers**: Optimize vector indices for better performance
- **Query Benchmarkers**: Benchmark query performance across different configurations
- **Data Migration**: Migrate data between vector databases
- **Schema Managers**: Manage vector database schemas

### Retrieval

- **Query Processors**: Process and expand queries for better retrieval
- **Retrieval Debuggers**: Debug retrieval results and identify issues
- **Filter Builders**: Build filters for metadata-based filtering
- **Hybrid Searchers**: Combine vector search with keyword search
- **Re-rankers**: Re-rank retrieval results for better relevance

### Generation

- **Prompt Templates**: Create and manage prompt templates
- **Context Formatters**: Format retrieved context for generation
- **Output Parsers**: Parse and validate generated outputs
- **Hallucination Detectors**: Detect and mitigate hallucinations
- **Citation Generators**: Generate citations for retrieved information

### Monitoring

- **Performance Trackers**: Track RAG system performance
- **Usage Analyzers**: Analyze usage patterns and identify bottlenecks
- **Error Loggers**: Log and analyze errors
- **Cost Estimators**: Estimate costs for API-based components
- **Latency Monitors**: Monitor and optimize latency

## Examples

Check out the `examples` directory for more detailed examples of using the various components.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 