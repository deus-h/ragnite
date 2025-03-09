# Numeric Filter Builder

The `NumericFilterBuilder` is a specialized filter builder that helps construct numeric-based filters for vector database queries. It provides methods for filtering by specific numeric values, ranges, and special numeric properties.

## Features

- **Numeric Comparisons**: Filter by equals, not equals, greater than, less than
- **Range Operations**: Filter by between, not between, and approximate matching
- **List Operations**: Filter by inclusion or exclusion from a list of values
- **Special Numeric Operations**: Filter by integer/decimal type, divisibility, and sign
- **Logical Operators**: Combine numeric filters with AND, OR, and NOT operations
- **Database Support**: Format filters for different vector database systems

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_filter_builder

# Create a numeric filter builder
filter_builder = get_filter_builder(builder_type="numeric")

# Create a simple numeric filter
filter_builder.equals("price", 99.99)

# Build the filter
filter_dict = filter_builder.build()
```

### Available Numeric Filter Conditions

#### Basic Comparisons

```python
# Equals a specific value
filter_builder.equals("price", 99.99)

# Not equals a specific value
filter_builder.not_equals("price", 99.99)

# Greater than a value
filter_builder.greater_than("price", 100)

# Greater than or equal to a value
filter_builder.greater_than_or_equal("price", 100)

# Less than a value
filter_builder.less_than("price", 50)

# Less than or equal to a value
filter_builder.less_than_or_equal("price", 50)
```

#### Range Operations

```python
# Between two values (inclusive by default)
filter_builder.between("price", 10, 50)

# Between two values (exclusive)
filter_builder.between("price", 10, 50, inclusive=False)

# Not between two values (inclusive by default)
filter_builder.not_between("price", 100, 200)

# Not between two values (exclusive)
filter_builder.not_between("price", 100, 200, inclusive=False)

# Near a value (within a tolerance)
filter_builder.near("price", 100, 5)  # price between 95 and 105
```

#### List Operations

```python
# In a list of values
filter_builder.in_list("quantity", [1, 2, 3, 5, 8, 13])

# Not in a list of values
filter_builder.not_in_list("rating", [1, 2])
```

#### Special Numeric Operations

```python
# Is an integer
filter_builder.is_integer("quantity")

# Is a decimal (has fractional part)
filter_builder.is_decimal("price")

# Is divisible by a value
filter_builder.divisible_by("quantity", 6)

# Is positive (> 0)
filter_builder.is_positive("profit_margin")

# Is positive or zero (>= 0)
filter_builder.is_positive("profit_margin", include_zero=True)

# Is negative (< 0)
filter_builder.is_negative("price_change")

# Is negative or zero (<= 0)
filter_builder.is_negative("price_change", include_zero=True)
```

#### Existence Checks

```python
# Field exists
filter_builder.exists("discount")

# Field does not exist
filter_builder.not_exists("discount")
```

### Logical Operators

#### Implicit AND (Method Chaining)

```python
# Chain methods to create AND conditions
filter_builder.greater_than("price", 50).less_than("price", 100)
```

#### Explicit Logical Operators

```python
# AND operator
filter_builder.and_operator(
    get_filter_builder("numeric").greater_than("stock_count", 0),
    get_filter_builder("numeric").greater_than_or_equal("rating", 4)
)

# OR operator
filter_builder.or_operator(
    get_filter_builder("numeric").less_than_or_equal("price", 25),
    get_filter_builder("numeric").greater_than_or_equal("price", 100)
)

# NOT operator
filter_builder.not_operator(
    get_filter_builder("numeric").between("price", 40, 60)
)
```

### Database-Specific Formatting

```python
# Target a specific vector database format
filter_builder = get_filter_builder(
    builder_type="numeric",
    target_format="qdrant"  # Options: "generic", "chroma", "qdrant", "pinecone", "weaviate", "milvus", "pgvector"
)

# Create filter as usual
filter_builder.between("price", 10, 100)

# Get filter formatted for the target database
qdrant_filter = filter_builder.build()
```

## Methods

### Basic Comparison Methods

| Method                        | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| `equals(field, value)`        | Field equals a specific value                             |
| `not_equals(field, value)`    | Field does not equal a specific value                     |
| `greater_than(field, value)`  | Field is greater than a value                             |
| `greater_than_or_equal(field, value)` | Field is greater than or equal to a value         |
| `less_than(field, value)`     | Field is less than a value                                |
| `less_than_or_equal(field, value)` | Field is less than or equal to a value              |

### Range Methods

| Method                         | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| `between(field, min, max, inclusive)` | Field is between min and max values               |
| `not_between(field, min, max, inclusive)` | Field is not between min and max values       |
| `near(field, target, tolerance)` | Field is within tolerance of target value              |

### List Methods

| Method                         | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| `in_list(field, values)`       | Field value is in a list of values                        |
| `not_in_list(field, values)`   | Field value is not in a list of values                    |

### Special Numeric Methods

| Method                         | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| `is_integer(field)`            | Field value is an integer                                 |
| `is_decimal(field)`            | Field value has a fractional part                         |
| `divisible_by(field, divisor)` | Field value is divisible by divisor                       |
| `is_positive(field, include_zero)` | Field value is positive (optionally including zero)   |
| `is_negative(field, include_zero)` | Field value is negative (optionally including zero)   |

### Existence Methods

| Method                         | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| `exists(field)`                | Field exists                                              |
| `not_exists(field)`            | Field does not exist                                      |

### Logical Operators

| Method                         | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| `and_operator(*filter_builders)` | Combine filter builders with AND logic                  |
| `or_operator(*filter_builders)` | Combine filter builders with OR logic                    |
| `not_operator(filter_builder)` | Negate a filter builder                                   |

### Utility Methods

| Method                         | Description                                               |
|--------------------------------|-----------------------------------------------------------|
| `build()`                      | Build and return the filter                               |
| `reset()`                      | Reset the filter to empty state                           |
| `to_dict()`                    | Convert the filter to a dictionary representation         |
| `from_dict(filter_dict)`       | Load a filter from a dictionary representation            |

## Example

See `tools/examples/retrieval/filter_builders/numeric_filter_builder_example.py` for a complete usage example.

## Use Cases

The `NumericFilterBuilder` is particularly useful for:

- E-commerce applications filtering products by price ranges
- Analytics applications filtering metrics by thresholds
- Financial applications filtering transactions by amount
- Inventory systems filtering items by quantity
- Any application that needs to filter documents by numeric metadata 