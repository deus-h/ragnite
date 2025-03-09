# Usage Analyzers for RAG Systems

Usage analyzers are tools for tracking and analyzing how users interact with Retrieval-Augmented Generation (RAG) systems. They help identify patterns in queries, track user sessions, monitor feature usage, and analyze errors, providing valuable insights for system optimization and user experience improvement.

## Overview

Understanding how users interact with your RAG system is crucial for:

- Improving retrieval and generation quality
- Optimizing user experience
- Identifying popular features and usage patterns
- Detecting and resolving issues
- Making data-driven decisions for system enhancements

The usage analyzers in this module provide comprehensive tracking and analysis capabilities for various aspects of RAG system usage.

## Available Analyzers

### QueryAnalyzer

Tracks and analyzes query patterns, including:

- Most common query terms
- Query length distribution
- Query frequency over time
- Query complexity
- Query categories (factual, instructional, comparative, etc.)

### UserSessionAnalyzer

Tracks and analyzes user sessions, including:

- Session duration
- Session activity
- User engagement
- Session flow (event sequences and transitions)
- User retention

### FeatureUsageAnalyzer

Tracks and analyzes feature usage, including:

- Feature popularity
- Feature usage patterns
- Feature combinations
- Feature usage by user segment
- Feature usage over time

### ErrorAnalyzer

Tracks and analyzes errors, including:

- Error frequency
- Error types and categories
- Error patterns
- Error impact
- Error resolution

## Usage

### Basic Usage

```python
from tools.src.monitoring.usage_analyzers import QueryAnalyzer

# Create analyzer
analyzer = QueryAnalyzer(
    name="my_query_analyzer",
    data_dir="./usage_data",
    config={
        "min_term_length": 3,
        "max_common_terms": 20,
        "time_window": "day"
    }
)

# Track a query event
analyzer.track({
    "query": "How do vector databases work in RAG systems?",
    "user_id": "user_123",
    "session_id": "session_456",
    "timestamp": "2023-06-15T14:30:00"
})

# Analyze tracked data
analysis_results = analyzer.analyze()

# Print analysis results
print(analysis_results)

# Save tracked data
analyzer.save_data("query_analysis.json")
```

### Using the Factory Function

```python
from tools.src.monitoring.usage_analyzers import get_usage_analyzer

# Create analyzer using factory function
analyzer = get_usage_analyzer(
    analyzer_type="query",  # Options: "query", "user_session", "feature_usage", "error"
    name="my_analyzer",
    data_dir="./usage_data",
    config={"time_window": "week"}
)

# Use analyzer as usual
analyzer.track(event_data)
analysis_results = analyzer.analyze()
```

## Configuration Options

### QueryAnalyzer

- `min_term_length`: Minimum length of terms to include in analysis (default: 3)
- `max_common_terms`: Maximum number of common terms to return (default: 20)
- `time_window`: Time window for analysis ("hour", "day", "week", "month") (default: "day")
- `complexity_factors`: Weights for calculating query complexity
- `category_patterns`: Regex patterns for categorizing queries

### UserSessionAnalyzer

- `session_timeout`: Session timeout in minutes (default: 30)
- `min_session_events`: Minimum number of events for a valid session (default: 2)
- `retention_periods`: Periods for retention analysis (default: ["day", "week", "month"])

### FeatureUsageAnalyzer

- `time_window`: Time window for analysis (default: "day")
- `top_combinations`: Number of top feature combinations to return (default: 10)
- `user_segments`: User segment definitions

### ErrorAnalyzer

- `time_window`: Time window for analysis (default: "day")
- `severity_levels`: Error severity levels (default: ["critical", "high", "medium", "low"])
- `error_categories`: Regex patterns for categorizing errors
- `total_requests`: Total number of requests for calculating error rate

## Data Management

All analyzers provide methods for managing tracked data:

- `save_data(filename)`: Save tracked data to a file
- `load_data(filepath)`: Load tracked data from a file
- `clear_data()`: Clear all tracked data
- `get_data_summary()`: Get a summary of the tracked data

## Analysis Results

Each analyzer provides different analysis results:

### QueryAnalyzer

- `common_terms`: Most common terms in queries
- `length_distribution`: Distribution of query lengths
- `time_analysis`: Query frequency over time
- `complexity_stats`: Statistics on query complexity
- `category_distribution`: Distribution of query categories

### UserSessionAnalyzer

- `session_stats`: Statistics on session duration and activity
- `user_engagement`: Metrics on user engagement
- `session_flow`: Analysis of session flow
- `user_retention`: Metrics on user retention

### FeatureUsageAnalyzer

- `feature_popularity`: Metrics on feature popularity
- `usage_patterns`: Analysis of feature usage patterns
- `feature_combinations`: Analysis of feature combinations
- `segment_analysis`: Analysis by user segment
- `time_analysis`: Analysis of feature usage over time

### ErrorAnalyzer

- `error_frequency`: Metrics on error frequency
- `error_types`: Analysis of error types
- `error_patterns`: Analysis of error patterns
- `error_impact`: Analysis of error impact
- `error_resolution`: Analysis of error resolution

## Example

See the complete example script in `tools/examples/monitoring/usage_analyzers/usage_analyzers_example.py` for a comprehensive demonstration of all analyzers. 