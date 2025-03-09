# Date Filter Builder

The `DateFilterBuilder` is a specialized filter builder that helps construct date and time-based filters for vector database queries. It provides intuitive methods for filtering by exact dates, date ranges, relative dates, and common time periods.

## Features

- **Date Comparison**: Filter by dates before, after, or between specific dates
- **Relative Dates**: Filter by relative time periods like "last 30 days" or "next 2 weeks"
- **Common Time Periods**: Easily filter by today, this week, this month, or this year
- **Date Lists**: Filter by inclusion or exclusion from a list of dates
- **Multiple Date Formats**: Support for strings in ISO format, `datetime` objects, and `date` objects
- **Logical Operators**: Combine date filters with AND, OR, and NOT operations
- **Database Support**: Format filters for different vector database systems

## Usage

### Basic Usage

```python
from tools.src.retrieval import get_filter_builder

# Create a date filter builder
filter_builder = get_filter_builder(builder_type="date")

# Create a simple date filter
filter_builder.equals("publication_date", "2023-06-15")

# Build the filter
filter_dict = filter_builder.build()
```

### Available Date Filter Conditions

#### Date Equality

```python
# Equals a specific date
filter_builder.equals("publication_date", "2023-06-15")

# Not equals a specific date
filter_builder.not_equals("publication_date", "2023-06-15")
```

#### Date Comparisons

```python
# Before a date (exclusive)
filter_builder.before("publication_date", "2023-12-31")

# Before or on a date (inclusive)
filter_builder.before("publication_date", "2023-12-31", inclusive=True)

# After a date (exclusive)
filter_builder.after("publication_date", "2023-01-01")

# After or on a date (inclusive)
filter_builder.after("publication_date", "2023-01-01", inclusive=True)

# Between two dates (inclusive by default)
filter_builder.between("publication_date", "2023-01-01", "2023-12-31")

# Between two dates (exclusive)
filter_builder.between("publication_date", "2023-01-01", "2023-12-31", inclusive=False)
```

#### Date Lists

```python
# In a list of dates
filter_builder.in_date_list("event_date", ["2023-01-01", "2023-07-04", "2023-12-25"])

# Not in a list of dates
filter_builder.not_in_date_list("event_date", ["2023-01-01", "2023-07-04", "2023-12-25"])
```

#### Relative Dates

```python
# In the past N time units
filter_builder.in_past("created_at", 30, "days")

# In the future N time units
filter_builder.in_future("due_date", 2, "weeks")

# In the last N time units (alias for in_past)
filter_builder.in_last("created_at", 30, "days")

# In the next N time units (alias for in_future)
filter_builder.in_next("due_date", 2, "weeks")
```

Available time units: `"days"`, `"weeks"`, `"months"`, `"years"`

#### Common Time Periods

```python
# Today
filter_builder.today("event_date")

# Yesterday
filter_builder.yesterday("event_date")

# Tomorrow
filter_builder.tomorrow("event_date")

# This week
filter_builder.this_week("event_date")

# This month
filter_builder.this_month("event_date")

# This year
filter_builder.this_year("event_date")
```

#### Special Date Operations

```python
# On a specific date (ignoring time components)
filter_builder.on_date("event_date", "2023-07-04")

# Date exists
filter_builder.exists("event_date")

# Date does not exist
filter_builder.not_exists("event_date")
```

### Multiple Date Formats

The `DateFilterBuilder` accepts dates in multiple formats:

```python
# ISO format string
filter_builder.equals("publication_date", "2023-06-15")

# datetime object
import datetime
filter_builder.equals("publication_date", datetime.datetime(2023, 6, 15, 12, 30, 0))

# date object
filter_builder.equals("publication_date", datetime.date(2023, 6, 15))
```

### Logical Operators

#### AND Operator

```python
# Implicit AND (all conditions in a chain are combined with AND)
filter_builder.after("created_at", "2023-01-01")
              .before("updated_at", "2023-12-31")

# Explicit AND with multiple filter builders
first_half = get_filter_builder("date").between("event_date", "2023-01-01", "2023-06-30")
second_half = get_filter_builder("date").between("event_date", "2023-07-01", "2023-12-31")

combined = get_filter_builder("date").and_operator(first_half, second_half)
```

#### OR Operator

```python
# OR operator combining multiple conditions
filter_builder.or_operator(
    get_filter_builder("date").equals("event_date", "2023-01-01"),
    get_filter_builder("date").equals("event_date", "2023-12-31")
)
```

#### NOT Operator

```python
# NOT operator to negate a condition
filter_builder.not_operator(
    get_filter_builder("date").this_week("event_date")
)
```

### Database-Specific Formatting

```python
# Target a specific vector database format
filter_builder = get_filter_builder(
    builder_type="date",
    target_format="qdrant"  # Options: "generic", "chroma", "qdrant", "pinecone", "weaviate", "milvus", "pgvector"
)

# Create filter as usual
filter_builder.between("event_date", "2023-01-01", "2023-12-31")

# Get filter formatted for the target database
qdrant_filter = filter_builder.build()
```

## Methods

### Date Filter Methods

| Method                        | Description                                               |
|-------------------------------|-----------------------------------------------------------|
| `equals(field, date_value)`   | Field equals a specific date                              |
| `not_equals(field, date_value)` | Field does not equal a specific date                    |
| `before(field, date_value, inclusive)` | Field is before a specific date                  |
| `after(field, date_value, inclusive)` | Field is after a specific date                    |
| `between(field, start_date, end_date, inclusive)` | Field is between two dates            |
| `on_date(field, date_value)`  | Field matches a specific date (ignores time components)   |
| `in_date_list(field, date_values)` | Field value is in a list of dates                    |
| `not_in_date_list(field, date_values)` | Field value is not in a list of dates            |
| `in_past(field, amount, unit, inclusive)` | Field is in the past N time units             |
| `in_future(field, amount, unit, inclusive)` | Field is in the future N time units         |
| `in_last(field, amount, unit)` | Field is in the last N time units                        |
| `in_next(field, amount, unit)` | Field is in the next N time units                        |
| `today(field)`                | Field is today                                            |
| `yesterday(field)`           | Field is yesterday                                        |
| `tomorrow(field)`            | Field is tomorrow                                         |
| `this_week(field)`           | Field is in the current week                              |
| `this_month(field)`          | Field is in the current month                             |
| `this_year(field)`           | Field is in the current year                              |
| `exists(field)`              | Field exists                                              |
| `not_exists(field)`          | Field does not exist                                      |

### Logical Operators

| Method                        | Description                                               |
|-------------------------------|-----------------------------------------------------------|
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

## Example

See `tools/examples/retrieval/filter_builders/date_filter_builder_example.py` for a complete usage example.

## Dependencies

- Python's standard `datetime` module
- `dateutil` package for relative date calculations and string parsing 