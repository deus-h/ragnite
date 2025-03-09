# Filter Builders

This directory contains various filter builder components for constructing filters for vector database queries. Filter builders provide a unified interface for creating complex filter expressions, abstracting away the differences between various vector database syntaxes.

## Available Filter Builders

### 1. MetadataFilterBuilder

The `MetadataFilterBuilder` helps construct filters based on document metadata fields. It supports a wide range of filter conditions including equality, range comparisons, string operations, and existence checks.

[MetadataFilterBuilder Documentation](./metadata_filter_builder.md)

### 2. DateFilterBuilder (Coming Soon)

The `DateFilterBuilder` will provide specialized support for date and time-based filtering, including date ranges, relative dates, and date comparisons.

### 3. NumericFilterBuilder (Coming Soon)

The `NumericFilterBuilder` will offer optimized filtering for numeric fields, including range queries, statistical filters, and binning.

### 4. CompositeFilterBuilder (Coming Soon)

The `CompositeFilterBuilder` will enable the creation of complex filter compositions, combining multiple filter types with logical operators.

## Using Filter Builders

All filter builders can be instantiated using the `get_filter_builder` factory function:

```python
from tools.src.retrieval import get_filter_builder

# Create a metadata filter builder
metadata_filter = get_filter_builder(
    builder_type="metadata",
    target_format="chroma"  # Optional: specify target vector database
)

# Create a date filter builder (when implemented)
# date_filter = get_filter_builder(builder_type="date")

# Create a numeric filter builder (when implemented)
# numeric_filter = get_filter_builder(builder_type="numeric")
```

## Common Interface

All filter builders implement a common interface with the following methods:

- `build()`: Build and return the filter in the specified format
- `reset()`: Reset the filter to an empty state
- `to_dict()`: Convert the filter to a dictionary representation
- `from_dict(filter_dict)`: Load a filter from a dictionary representation

## Database Support

Filter builders support generating filters for various vector database systems:

- Generic (MongoDB-like syntax)
- Chroma
- Qdrant
- Pinecone
- Weaviate
- Milvus
- pgvector (PostgreSQL with pgvector extension)

The level of support varies by filter builder and database type.

## Examples

See the [examples directory](../../../../examples/retrieval/filter_builders/) for complete usage examples for each filter builder type. 