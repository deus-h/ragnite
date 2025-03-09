# Context Formatters

This directory contains documentation for the context formatters in the RAG Research project. Context formatters transform retrieved documents into well-structured contexts for language models, improving the quality of generated responses.

## Overview

Context formatters handle the critical task of preparing retrieved documents for use in prompts. They transform raw document data into formatted strings that maximize the language model's ability to utilize the information effectively. This includes handling document content, metadata, and organizing information in a way that is most useful for the specific use case.

## Available Context Formatters

### BasicContextFormatter

The `BasicContextFormatter` provides simple formatting of document contents with minimal styling.

Features:
- Concatenates document contents with configurable separators
- Optional numbering of documents
- Configurable maximum length for each document
- Support for header and footer text
- See [basic_context_formatter.md](./basic_context_formatter.md) for detailed documentation

### MetadataEnrichedFormatter

The `MetadataEnrichedFormatter` enhances document content with its associated metadata.

Features:
- Includes selected metadata fields with each document
- Customizable metadata formatting and ordering
- Configurable metadata field selection
- Handles missing metadata fields gracefully
- See [metadata_enriched_formatter.md](./metadata_enriched_formatter.md) for detailed documentation

### SourceAttributionFormatter

The `SourceAttributionFormatter` adds source citations to documents and includes a references section.

Features:
- Adds citation markers to document content
- Generates a references section with source information
- Configurable citation placement (prefix, suffix, or both)
- Customizable citation and reference formats
- See [source_attribution_formatter.md](./source_attribution_formatter.md) for detailed documentation

### HierarchicalContextFormatter

The `HierarchicalContextFormatter` organizes documents into a hierarchical structure based on metadata fields.

Features:
- Groups documents by categories and subcategories
- Supports single-level or two-level hierarchies
- Customizable formatting for groups, subgroups, and items
- Advanced sorting and grouping options
- See [hierarchical_context_formatter.md](./hierarchical_context_formatter.md) for detailed documentation

## Usage

To use a context formatter, you can create an instance directly or use the factory function:

```python
from tools.src.retrieval import get_context_formatter, BasicContextFormatter

# Using the factory function
basic_formatter = get_context_formatter(
    formatter_type="basic",
    numbered=True,
    separator="\n\n"
)

# Or creating an instance directly
from tools.src.retrieval import BasicContextFormatter
basic_formatter = BasicContextFormatter(config={
    "numbered": True,
    "separator": "\n\n"
})

# Format retrieved documents
documents = [
    {"content": "Document 1 content", "title": "Title 1", "source": "Source 1"},
    {"content": "Document 2 content", "title": "Title 2", "source": "Source 2"}
]
formatted_context = basic_formatter.format_context(documents)
```

## Factory Function

The `get_context_formatter` factory function provides a unified interface for creating context formatters:

```python
from tools.src.retrieval import get_context_formatter

# Create a basic formatter
basic = get_context_formatter(
    formatter_type="basic",
    numbered=True,
    max_length=500
)

# Create a metadata enriched formatter
metadata = get_context_formatter(
    formatter_type="metadata",
    metadata_fields=["title", "source", "date"],
    metadata_separator="\n"
)

# Create a source attribution formatter
attribution = get_context_formatter(
    formatter_type="source_attribution",
    citation_format="[{index}]",
    reference_header="\n\nSources:"
)

# Create a hierarchical formatter
hierarchical = get_context_formatter(
    formatter_type="hierarchical",
    hierarchy_field="category",
    secondary_field="subcategory"
)
```

## Choosing a Context Formatter

Different context formatters are suitable for different use cases:

- **BasicContextFormatter**: Use when you just need to combine document contents with minimal formatting.
- **MetadataEnrichedFormatter**: Use when document metadata (title, source, date, etc.) is important for context.
- **SourceAttributionFormatter**: Use when you want to encourage the language model to cite sources in its responses.
- **HierarchicalContextFormatter**: Use when documents fall into natural categories and organization is important.

## Examples

See the [examples directory](../../../../examples/retrieval/generation/) for example scripts demonstrating the use of context formatters. 