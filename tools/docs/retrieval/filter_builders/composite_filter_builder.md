# Composite Filter Builder

The `CompositeFilterBuilder` is a powerful filter builder that enables construction of complex filters by combining different filter types (metadata, date, numeric) for vector database queries.

## Features

- **Multi-Type Filtering**: Combine metadata, date, and numeric filters in a single query
- **Logical Operations**: Support for AND, OR, and NOT operations across different filter types
- **Nested Filtering**: Create hierarchical filter structures for complex query conditions
- **Builder Convenience**: Helper methods to create and combine various filter types
- **Database Support**: Format filters for different vector database systems

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_filter_builder

# Create a composite filter builder
composite_filter = get_filter_builder(builder_type="composite")

# Create individual filters of different types
metadata_filter = get_filter_builder("metadata").equals("category", "electronics")
date_filter = get_filter_builder("date").after("created_at", "2023-01-01")
numeric_filter = get_filter_builder("numeric").greater_than("price", 500)

# Add filters to the composite filter
composite_filter.add_filter(metadata_filter)
composite_filter.add_filter(date_filter)
composite_filter.add_filter(numeric_filter)

# Build the filter
filter_dict = composite_filter.build()
```

### Logical Operations

#### AND Logic

```python
# Create individual filters of different types
metadata_filter = get_filter_builder("metadata").equals("category", "electronics")
date_filter = get_filter_builder("date").after("created_at", "2023-01-01")
numeric_filter = get_filter_builder("numeric").between("price", 500, 1000)

# Combine filters with AND logic (all conditions must be true)
composite_filter.and_filters(metadata_filter, date_filter, numeric_filter)
```

#### OR Logic

```python
# Create filters for different categories
electronics_filter = get_filter_builder("metadata").equals("category", "electronics")
books_filter = get_filter_builder("metadata").equals("category", "books")
clothing_filter = get_filter_builder("metadata").equals("category", "clothing")

# Combine filters with OR logic (any condition can be true)
composite_filter.or_filters(electronics_filter, books_filter, clothing_filter)
```

#### NOT Logic (Negation)

```python
# Create a filter to negate
high_price_filter = get_filter_builder("numeric").greater_than("price", 1000)

# Negate the filter (products NOT over $1000)
composite_filter.not_filter(high_price_filter)
```

### Helper Methods for Creating Filters

The `CompositeFilterBuilder` includes helper methods to create other filter types directly:

```python
# Create a composite filter builder
composite_filter = get_filter_builder(builder_type="composite")

# Create other filter types directly from the composite filter
metadata_filter = composite_filter.metadata_filter().equals("category", "electronics")
date_filter = composite_filter.date_filter().after("created_at", "2023-01-01")
numeric_filter = composite_filter.numeric_filter().greater_than("price", 500)

# Add these filters to the composite filter
composite_filter.and_filters(metadata_filter, date_filter, numeric_filter)
```

### Complex Nested Filters

You can create complex nested filters by combining composite filters:

```python
# Create a composite filter builder for the main filter
composite_filter = get_filter_builder(builder_type="composite")

# Create a composite filter for products on sale
sale_filter = get_filter_builder("composite").and_filters(
    get_filter_builder("metadata").exists("discount_percentage"),
    get_filter_builder("numeric").greater_than("discount_percentage", 0)
)

# Create a composite filter for product categories (electronics OR clothing)
category_filter = get_filter_builder("composite").or_filters(
    get_filter_builder("metadata").equals("category", "electronics"),
    get_filter_builder("metadata").equals("category", "clothing")
)

# Create a composite filter to exclude books
not_books_filter = get_filter_builder("composite").not_filter(
    get_filter_builder("metadata").equals("category", "books")
)

# Combine all filters with AND logic
composite_filter.and_filters(sale_filter, category_filter, not_books_filter)
```

### Combining Existing Filters

You can also combine existing filters using the special combine methods:

```python
# Create two separate filters
filter1 = get_filter_builder("metadata").equals("category", "electronics")
filter2 = get_filter_builder("numeric").between("price", 100, 500)

# Create a composite filter and combine the two filters with AND logic
composite_filter = get_filter_builder("composite").combine_with_and(filter1, filter2)

# Or with OR logic
composite_filter = get_filter_builder("composite").combine_with_or(filter1, filter2)
```

### Database-Specific Formatting

```python
# Target a specific vector database format
composite_filter = get_filter_builder(
    builder_type="composite",
    target_format="qdrant"  # Options: "generic", "chroma", "qdrant", "pinecone", "weaviate", "milvus", "pgvector"
)

# Create and add filters as usual
composite_filter.and_filters(
    get_filter_builder("metadata").equals("category", "electronics"),
    get_filter_builder("numeric").between("price", 100, 500)
)

# Get filter formatted for the target database
qdrant_filter = composite_filter.build()
```

## Methods

### Filter Building Methods

| Method                        | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| `add_filter(filter_builder)`  | Add a filter builder to the composite filter              |
| `and_filters(*filter_builders)` | Combine multiple filter builders with AND logic         |
| `or_filters(*filter_builders)` | Combine multiple filter builders with OR logic           |
| `not_filter(filter_builder)`  | Negate a filter builder                                   |

### Filter Creation Helper Methods

| Method                        | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| `metadata_filter()`           | Create and return a new metadata filter builder           |
| `date_filter()`               | Create and return a new date filter builder               |
| `numeric_filter()`            | Create and return a new numeric filter builder            |

### Filter Combination Methods

| Method                        | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| `combine_with_and(other_filter)` | Combine this filter with another filter using AND logic  |
| `combine_with_or(other_filter)` | Combine this filter with another filter using OR logic   |

### Utility Methods

| Method                        | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| `build()`                     | Build and return the filter                               |
| `reset()`                     | Reset the filter to empty state                           |
| `to_dict()`                   | Convert the filter to a dictionary representation         |
| `from_dict(filter_dict)`      | Load a filter from a dictionary representation            |

## Example Use Cases

The `CompositeFilterBuilder` is ideal for complex filtering scenarios such as:

### E-commerce Product Search

```python
# Create a composite filter for an e-commerce product search
composite_filter = get_filter_builder("composite").and_filters(
    get_filter_builder("metadata").equals("category", "electronics"),
    get_filter_builder("numeric").greater_than_or_equal("rating", 4),
    get_filter_builder("date").in_past("created_at", 30, "days"),
    get_filter_builder("numeric").greater_than("stock_count", 0),
    get_filter_builder("numeric").between("price", 100, 500)
)
```

### Real Estate Property Search

```python
# Create a composite filter for a real estate property search
composite_filter = get_filter_builder("composite").and_filters(
    get_filter_builder("composite").or_filters(
        get_filter_builder("metadata").equals("property_type", "house"),
        get_filter_builder("metadata").equals("property_type", "condo")
    ),
    get_filter_builder("numeric").between("price", 200000, 500000),
    get_filter_builder("numeric").greater_than_or_equal("bedrooms", 3),
    get_filter_builder("numeric").greater_than_or_equal("bathrooms", 2),
    get_filter_builder("numeric").greater_than("square_feet", 1500),
    get_filter_builder("date").in_past("listing_date", 3, "months")
)
```

### Academic Paper Search

```python
# Create a composite filter for an academic paper search
composite_filter = get_filter_builder("composite").and_filters(
    get_filter_builder("composite").or_filters(
        get_filter_builder("metadata").contains("topic", "machine learning"),
        get_filter_builder("metadata").contains("topic", "artificial intelligence")
    ),
    get_filter_builder("date").in_past("publication_date", 2, "years"),
    get_filter_builder("numeric").greater_than("citation_count", 50)
)
```

## Example

See `tools/examples/retrieval/filter_builders/composite_filter_builder_example.py` for a complete usage example. 