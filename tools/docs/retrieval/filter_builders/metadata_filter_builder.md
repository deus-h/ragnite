# Metadata Filter Builder

The `MetadataFilterBuilder` is a component that helps construct filters based on document metadata fields for vector database queries. It provides a fluent interface for creating filters with various conditions and operators, abstracting away the differences between vector database filter syntaxes.

## Features

- **Rich Filter Conditions**: Supports equality, range, list membership, string operations, and existence checks
- **Logical Operators**: Combines filters with AND, OR, and NOT operators
- **Database-Specific Formatting**: Formats filters for different vector databases (Chroma, Qdrant, Pinecone, etc.)
- **Fluent Interface**: Chain method calls for concise filter creation
- **JSON-Compatible Output**: Produces filters as JSON-compatible dictionaries

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_filter_builder

# Create a filter builder
filter_builder = get_filter_builder(builder_type="metadata")

# Create a simple filter
filter_builder.equals("category", "scientific_paper")
              .greater_than("publication_year", 2020)
              .in_list("author", ["Smith, J.", "Johnson, A."])

# Build the filter
filter_dict = filter_builder.build()
```

### Available Filter Conditions

#### Equality Operations

```python
# Equals
filter_builder.equals("category", "scientific_paper")

# Not equals
filter_builder.not_equals("document_type", "review")

# In list
filter_builder.in_list("tag", ["important", "verified", "peer-reviewed"])

# Not in list
filter_builder.not_in_list("status", ["draft", "retracted"])
```

#### Numeric Comparisons

```python
# Greater than
filter_builder.greater_than("citation_count", 10)

# Greater than or equal
filter_builder.greater_than_or_equal("publication_year", 2020)

# Less than
filter_builder.less_than("length", 5000)

# Less than or equal
filter_builder.less_than_or_equal("word_count", 1000)

# Between (inclusive by default)
filter_builder.between("publication_year", 2018, 2023)

# Between (exclusive)
filter_builder.between("publication_year", 2018, 2023, inclusive=False)
```

#### String Operations

```python
# Contains substring
filter_builder.contains("title", "neural network")

# Contains (case sensitive)
filter_builder.contains("title", "RNA", case_sensitive=True)

# Starts with
filter_builder.starts_with("title", "Introduction to")

# Ends with
filter_builder.ends_with("file_name", ".pdf")
```

#### Existence Checks

```python
# Field exists
filter_builder.exists("abstract")

# Field does not exist
filter_builder.not_exists("retraction_date")
```

### Logical Operators

#### AND Operator

```python
# Implicit AND (all conditions in a chain are combined with AND)
filter_builder.equals("category", "scientific_paper")
              .greater_than("publication_year", 2020)

# Explicit AND with multiple filter builders
recent_papers = get_filter_builder("metadata").greater_than("year", 2022)
ai_papers = get_filter_builder("metadata").contains("title", "AI")

combined = get_filter_builder("metadata").and_operator(recent_papers, ai_papers)
```

#### OR Operator

```python
# OR operator combining multiple conditions
filter_builder.or_operator(
    get_filter_builder("metadata").equals("category", "research"),
    get_filter_builder("metadata").equals("category", "review")
)

# Complex OR within a chain
filter_builder.equals("source", "journal")
              .or_operator(
                  get_filter_builder("metadata").greater_than("impact_factor", 5),
                  get_filter_builder("metadata").equals("peer_reviewed", True)
              )
```

#### NOT Operator

```python
# NOT operator to negate a condition
filter_builder.not_operator(
    get_filter_builder("metadata").contains("title", "survey")
)
```

### Database-Specific Formatting

```python
# Target a specific vector database format
filter_builder = get_filter_builder(
    builder_type="metadata",
    target_format="qdrant"  # Options: "generic", "chroma", "qdrant", "pinecone", "weaviate", "milvus", "pgvector"
)

# Create filter as usual
filter_builder.equals("category", "scientific_paper")

# Get filter formatted for the target database
qdrant_filter = filter_builder.build()
```

## Methods

### Builder Methods

| Method                        | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| `equals(field, value)`        | Field equals a value                                      |
| `not_equals(field, value)`    | Field does not equal a value                              |
| `in_list(field, values)`      | Field value is in a list                                  |
| `not_in_list(field, values)`  | Field value is not in a list                              |
| `greater_than(field, value)`  | Field is greater than a value                             |
| `greater_than_or_equal(field, value)` | Field is greater than or equal to a value         |
| `less_than(field, value)`     | Field is less than a value                                |
| `less_than_or_equal(field, value)` | Field is less than or equal to a value               |
| `between(field, min, max, inclusive)` | Field is between min and max values               |
| `contains(field, value, case_sensitive)` | Field contains a substring                     |
| `starts_with(field, value, case_sensitive)` | Field starts with a substring               |
| `ends_with(field, value, case_sensitive)` | Field ends with a substring                   |
| `exists(field)`               | Field exists                                              |
| `not_exists(field)`           | Field does not exist                                      |
| `and_operator(*filter_builders)` | Combine filter builders with AND logic                 |
| `or_operator(*filter_builders)` | Combine filter builders with OR logic                   |
| `not_operator(filter_builder)` | Negate a filter builder                                  |

### Utility Methods

| Method                      | Description                                               |
|-----------------------------|-----------------------------------------------------------|
| `build()`                   | Build and return the filter                               |
| `reset()`                   | Reset the filter to empty state                           |
| `to_dict()`                 | Convert the filter to a dictionary representation          |
| `from_dict(filter_dict)`    | Load a filter from a dictionary representation            |

## Filter Output Format

The filter is built in a generic format that follows these conventions:

```python
# Equality
{"field": {"$eq": value}}

# Logical operators
{"$and": [condition1, condition2, ...]}
{"$or": [condition1, condition2, ...]}
{"$not": condition}

# Comparison operators
{"field": {"$gt": value}}
{"field": {"$gte": value}}
{"field": {"$lt": value}}
{"field": {"$lte": value}}

# Other operators
{"field": {"$in": [value1, value2, ...]}}
{"field": {"$nin": [value1, value2, ...]}}
{"field": {"$exists": true_or_false}}
{"field": {"$contains": value, "$case_sensitive": true_or_false}}
```

This generic format is then converted to the target database format when `build()` is called.

## Example

See `tools/examples/retrieval/filter_builders/metadata_filter_builder_example.py` for a complete usage example.

## Database Support

Currently supports the following vector database formats:
- Generic (default)
- Chroma
- Qdrant (partial)
- Pinecone (partial)
- Weaviate (partial)
- Milvus (partial)
- pgvector (partial) 