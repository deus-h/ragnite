#!/usr/bin/env python3
"""
Example script demonstrating the usage of usage analyzers in RAG systems.

This script shows how to use the various usage analyzers to track and analyze
user interactions with a RAG system, including query patterns, user sessions,
feature usage, and errors.
"""

import sys
import os
import json
import datetime
import random
import uuid
from typing import Dict, List, Any

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import usage analyzers
from tools.src.monitoring.usage_analyzers import (
    QueryAnalyzer,
    UserSessionAnalyzer,
    FeatureUsageAnalyzer,
    ErrorAnalyzer,
    get_usage_analyzer
)


# Sample data generation functions
def generate_sample_queries(count: int = 100) -> List[Dict[str, Any]]:
    """Generate sample query events."""
    query_templates = [
        "What is {topic}?",
        "How does {topic} work?",
        "Compare {topic} and {topic2}",
        "Explain the difference between {topic} and {topic2}",
        "What are the best practices for {topic}?",
        "Can you provide examples of {topic}?",
        "What are the advantages of {topic}?",
        "What are the disadvantages of {topic}?",
        "How to implement {topic} in {language}?",
        "What's the history of {topic}?"
    ]
    
    topics = [
        "RAG", "vector databases", "embeddings", "LLMs", "prompt engineering",
        "fine-tuning", "retrieval", "generation", "context window", "tokens",
        "semantic search", "neural networks", "transformers", "attention mechanism",
        "BERT", "GPT", "T5", "zero-shot learning", "few-shot learning"
    ]
    
    languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]
    
    user_ids = [f"user_{i}" for i in range(1, 11)]
    session_ids = [f"session_{i}" for i in range(1, 21)]
    
    events = []
    
    for i in range(count):
        # Generate timestamp within the last 30 days
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        timestamp = (
            datetime.datetime.now() - 
            datetime.timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        ).isoformat()
        
        # Generate query
        topic = random.choice(topics)
        topic2 = random.choice([t for t in topics if t != topic])
        language = random.choice(languages)
        
        template = random.choice(query_templates)
        query = template.format(topic=topic, topic2=topic2, language=language)
        
        # Create event
        event = {
            "query": query,
            "user_id": random.choice(user_ids),
            "session_id": random.choice(session_ids),
            "timestamp": timestamp
        }
        
        events.append(event)
    
    return events


def generate_sample_sessions(count: int = 100) -> List[Dict[str, Any]]:
    """Generate sample session events."""
    event_types = [
        "session_start", "search", "view_result", "click_result",
        "expand_result", "save_result", "share_result", "feedback",
        "refine_query", "session_end"
    ]
    
    user_ids = [f"user_{i}" for i in range(1, 11)]
    
    events = []
    
    # Generate sessions
    for i in range(count // 10):  # Each session has ~10 events
        user_id = random.choice(user_ids)
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # Generate session start time
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        start_time = (
            datetime.datetime.now() - 
            datetime.timedelta(days=days_ago, hours=hours_ago)
        )
        
        # Generate session events
        session_events = []
        
        # Always start with session_start
        session_events.append({
            "event_type": "session_start",
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": start_time.isoformat()
        })
        
        # Generate random events in between
        num_events = random.randint(3, 8)
        for j in range(num_events):
            event_time = start_time + datetime.timedelta(minutes=random.randint(1, 30))
            event_type = random.choice(event_types[1:-1])  # Exclude start/end
            
            session_events.append({
                "event_type": event_type,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": event_time.isoformat()
            })
        
        # Always end with session_end
        end_time = session_events[-1]["timestamp"]
        end_time = datetime.datetime.fromisoformat(end_time) + datetime.timedelta(minutes=random.randint(1, 10))
        
        session_events.append({
            "event_type": "session_end",
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": end_time.isoformat()
        })
        
        events.extend(session_events)
    
    return events


def generate_sample_feature_usage(count: int = 100) -> List[Dict[str, Any]]:
    """Generate sample feature usage events."""
    features = [
        "semantic_search", "keyword_search", "hybrid_search",
        "document_retrieval", "passage_retrieval", "query_expansion",
        "reranking", "summarization", "question_answering",
        "citation_generation", "feedback_collection"
    ]
    
    user_ids = [f"user_{i}" for i in range(1, 11)]
    session_ids = [f"session_{i}" for i in range(1, 21)]
    
    events = []
    
    for i in range(count):
        # Generate timestamp within the last 30 days
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        timestamp = (
            datetime.datetime.now() - 
            datetime.timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        ).isoformat()
        
        # Select feature
        feature_id = random.choice(features)
        
        # Generate parameters based on feature
        parameters = {}
        if feature_id == "semantic_search":
            parameters = {
                "query": f"Sample query {random.randint(1, 100)}",
                "top_k": random.choice([5, 10, 20, 50]),
                "threshold": round(random.uniform(0.5, 0.9), 2)
            }
        elif feature_id == "document_retrieval":
            parameters = {
                "doc_id": f"doc_{random.randint(1, 1000)}",
                "include_metadata": random.choice([True, False])
            }
        elif feature_id == "summarization":
            parameters = {
                "max_length": random.choice([100, 200, 300, 500]),
                "min_length": random.choice([50, 100]),
                "format": random.choice(["bullet_points", "paragraph", "key_points"])
            }
        
        # Generate result
        result = {
            "status": random.choices(
                ["success", "partial_success", "error"],
                weights=[0.8, 0.15, 0.05]
            )[0]
        }
        
        # Create event
        event = {
            "feature_id": feature_id,
            "user_id": random.choice(user_ids),
            "session_id": random.choice(session_ids),
            "timestamp": timestamp,
            "parameters": parameters,
            "result": result
        }
        
        events.append(event)
    
    return events


def generate_sample_errors(count: int = 50) -> List[Dict[str, Any]]:
    """Generate sample error events."""
    error_types = [
        "ConnectionError", "TimeoutError", "AuthenticationError",
        "RateLimitError", "ValidationError", "ParseError",
        "ResourceNotFoundError", "ServerError", "ClientError"
    ]
    
    error_messages = [
        "Failed to connect to the server",
        "Request timed out after 30 seconds",
        "Authentication failed: Invalid API key",
        "Rate limit exceeded: 100 requests per minute",
        "Invalid query format: Missing required parameter",
        "Failed to parse response: Invalid JSON",
        "Resource not found: Document ID does not exist",
        "Internal server error: Database connection failed",
        "Client error: Invalid request parameters"
    ]
    
    components = [
        "retrieval", "embedding", "generation", "parsing",
        "authentication", "database", "api", "frontend"
    ]
    
    user_ids = [f"user_{i}" for i in range(1, 11)]
    session_ids = [f"session_{i}" for i in range(1, 21)]
    
    events = []
    
    for i in range(count):
        # Generate timestamp within the last 30 days
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        timestamp = (
            datetime.datetime.now() - 
            datetime.timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        ).isoformat()
        
        # Select error type and message
        error_type = random.choice(error_types)
        error_message = random.choice(error_messages)
        
        # Generate error code
        if "Connection" in error_type or "Timeout" in error_type:
            error_code = random.choice([408, 503, 504])
        elif "Authentication" in error_type:
            error_code = random.choice([401, 403])
        elif "RateLimit" in error_type:
            error_code = 429
        elif "Validation" in error_type or "Parse" in error_type:
            error_code = random.choice([400, 422])
        elif "NotFound" in error_type:
            error_code = 404
        elif "Server" in error_type:
            error_code = random.choice([500, 502, 503])
        else:
            error_code = random.choice([400, 500])
        
        # Determine if error is resolved
        is_resolved = random.random() < 0.7  # 70% of errors are resolved
        
        # Create resolution if resolved
        resolution = None
        if is_resolved:
            # Generate resolution time
            resolution_hours = random.randint(0, 48)
            resolution_minutes = random.randint(0, 59)
            resolved_at = (
                datetime.datetime.fromisoformat(timestamp) + 
                datetime.timedelta(hours=resolution_hours, minutes=resolution_minutes)
            ).isoformat()
            
            resolution_methods = [
                "automatic_retry", "manual_fix", "configuration_update",
                "code_patch", "service_restart", "user_action"
            ]
            
            resolution = {
                "resolved": True,
                "resolved_at": resolved_at,
                "method": random.choice(resolution_methods)
            }
        
        # Create event
        event = {
            "error_message": error_message,
            "error_type": error_type,
            "error_code": error_code,
            "component": random.choice(components),
            "user_id": random.choice(user_ids),
            "session_id": random.choice(session_ids),
            "timestamp": timestamp
        }
        
        if resolution:
            event["resolution"] = resolution
        
        events.append(event)
    
    return events


# Example functions for each analyzer
def query_analyzer_example():
    """Example usage of QueryAnalyzer."""
    print("\n=== Query Analyzer Example ===")
    
    # Create analyzer
    analyzer = QueryAnalyzer(
        name="example_query_analyzer",
        data_dir="./usage_data",
        config={
            "min_term_length": 3,
            "max_common_terms": 10,
            "time_window": "day"
        }
    )
    
    # Generate and track sample queries
    sample_queries = generate_sample_queries(100)
    for query_event in sample_queries:
        analyzer.track(query_event)
    
    print(f"Tracked {len(analyzer.data)} query events")
    
    # Analyze queries
    analysis_results = analyzer.analyze()
    
    # Print analysis results
    print("\nQuery Analysis Results:")
    print(f"- Common Terms: {list(analysis_results['common_terms'].keys())[:5]}...")
    
    length_stats = analysis_results['length_distribution']['character_length']
    print(f"- Query Length: Min={length_stats['min']}, Max={length_stats['max']}, Avg={length_stats['avg']:.2f}")
    
    word_stats = analysis_results['length_distribution']['word_count']
    print(f"- Word Count: Min={word_stats['min']}, Max={word_stats['max']}, Avg={word_stats['avg']:.2f}")
    
    print(f"- Query Categories: {list(analysis_results['category_distribution'].keys())}")
    
    # Save data
    data_file = analyzer.save_data("query_analysis_example.json")
    print(f"\nSaved analysis data to: {data_file}")


def user_session_analyzer_example():
    """Example usage of UserSessionAnalyzer."""
    print("\n=== User Session Analyzer Example ===")
    
    # Create analyzer
    analyzer = UserSessionAnalyzer(
        name="example_session_analyzer",
        data_dir="./usage_data",
        config={
            "session_timeout": 30,  # minutes
            "min_session_events": 2
        }
    )
    
    # Generate and track sample sessions
    sample_sessions = generate_sample_sessions(100)
    for session_event in sample_sessions:
        analyzer.track(session_event)
    
    print(f"Tracked {len(analyzer.data)} session events")
    
    # Analyze sessions
    analysis_results = analyzer.analyze()
    
    # Print analysis results
    print("\nSession Analysis Results:")
    
    session_stats = analysis_results['session_stats']
    print(f"- Session Count: {session_stats['session_count']}")
    
    duration = session_stats['duration_minutes']
    print(f"- Duration (minutes): Min={duration['min']:.2f}, Max={duration['max']:.2f}, Avg={duration['avg']:.2f}")
    
    events = session_stats['events_per_session']
    print(f"- Events Per Session: Min={events['min']}, Max={events['max']}, Avg={events['avg']:.2f}")
    
    engagement = analysis_results['user_engagement']
    print(f"- Unique Users: {engagement['unique_users']}")
    
    event_types = engagement['event_type_distribution']
    print(f"- Event Types: {list(event_types.keys())}")
    
    # Save data
    data_file = analyzer.save_data("session_analysis_example.json")
    print(f"\nSaved analysis data to: {data_file}")


def feature_usage_analyzer_example():
    """Example usage of FeatureUsageAnalyzer."""
    print("\n=== Feature Usage Analyzer Example ===")
    
    # Create analyzer
    analyzer = FeatureUsageAnalyzer(
        name="example_feature_analyzer",
        data_dir="./usage_data",
        config={
            "time_window": "day",
            "top_combinations": 5
        }
    )
    
    # Generate and track sample feature usage
    sample_features = generate_sample_feature_usage(100)
    for feature_event in sample_features:
        analyzer.track(feature_event)
    
    print(f"Tracked {len(analyzer.data)} feature usage events")
    
    # Analyze feature usage
    analysis_results = analyzer.analyze()
    
    # Print analysis results
    print("\nFeature Usage Analysis Results:")
    
    popularity = analysis_results['feature_popularity']
    top_features = sorted(
        popularity['total_usage'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    print(f"- Top Features: {[f[0] for f in top_features]}")
    
    patterns = analysis_results['usage_patterns']
    print(f"- Parameter Usage: {list(patterns['parameter_usage'].keys())}")
    
    combinations = analysis_results['feature_combinations']
    print(f"- Top Combinations: {list(combinations['top_combinations'].keys())[:3]}")
    
    time_analysis = analysis_results['time_analysis']
    print(f"- Time Analysis: {len(time_analysis)} features tracked over time")
    
    # Save data
    data_file = analyzer.save_data("feature_usage_example.json")
    print(f"\nSaved analysis data to: {data_file}")


def error_analyzer_example():
    """Example usage of ErrorAnalyzer."""
    print("\n=== Error Analyzer Example ===")
    
    # Create analyzer
    analyzer = ErrorAnalyzer(
        name="example_error_analyzer",
        data_dir="./usage_data",
        config={
            "time_window": "day",
            "total_requests": 1000  # For calculating error rate
        }
    )
    
    # Generate and track sample errors
    sample_errors = generate_sample_errors(50)
    for error_event in sample_errors:
        analyzer.track(error_event)
    
    print(f"Tracked {len(analyzer.data)} error events")
    
    # Analyze errors
    analysis_results = analyzer.analyze()
    
    # Print analysis results
    print("\nError Analysis Results:")
    
    frequency = analysis_results['error_frequency']
    print(f"- Total Errors: {frequency['total_errors']}")
    print(f"- Error Rate: {frequency['error_rate']:.2%}")
    
    types = analysis_results['error_types']
    print(f"- Error Types: {list(types['error_types'].keys())}")
    print(f"- Error Categories: {list(types['error_categories'].keys())}")
    
    impact = analysis_results['error_impact']
    print(f"- Affected Users: {impact['affected_users']}")
    print(f"- Severity Distribution: {impact['severity_distribution']}")
    
    resolution = analysis_results['error_resolution']
    print(f"- Resolved: {resolution['resolved_count']} ({resolution['resolution_rate']:.2%})")
    print(f"- Unresolved: {resolution['unresolved_count']}")
    
    # Save data
    data_file = analyzer.save_data("error_analysis_example.json")
    print(f"\nSaved analysis data to: {data_file}")


def factory_function_example():
    """Example usage of the factory function."""
    print("\n=== Factory Function Example ===")
    
    # Create analyzers using factory function
    analyzers = []
    
    # Query analyzer
    query_analyzer = get_usage_analyzer(
        analyzer_type="query",
        name="factory_query_analyzer",
        config={"time_window": "week"}
    )
    analyzers.append(("Query Analyzer", query_analyzer))
    
    # User session analyzer
    session_analyzer = get_usage_analyzer(
        analyzer_type="user_session",
        name="factory_session_analyzer"
    )
    analyzers.append(("User Session Analyzer", session_analyzer))
    
    # Feature usage analyzer
    feature_analyzer = get_usage_analyzer(
        analyzer_type="feature_usage",
        name="factory_feature_analyzer"
    )
    analyzers.append(("Feature Usage Analyzer", feature_analyzer))
    
    # Error analyzer
    error_analyzer = get_usage_analyzer(
        analyzer_type="error",
        name="factory_error_analyzer"
    )
    analyzers.append(("Error Analyzer", error_analyzer))
    
    # Print analyzer information
    for name, analyzer in analyzers:
        print(f"- Created {name}: {analyzer.__class__.__name__}")
        print(f"  - Name: {analyzer.name}")
        print(f"  - Data Directory: {analyzer.data_dir}")
        print(f"  - Config: {list(analyzer.config.keys())}")


def main():
    """Run all examples."""
    try:
        # Create usage_data directory if it doesn't exist
        os.makedirs("./usage_data", exist_ok=True)
        
        # Run examples
        query_analyzer_example()
        user_session_analyzer_example()
        feature_usage_analyzer_example()
        error_analyzer_example()
        factory_function_example()
        
        print("\nAll examples completed successfully!")
    
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 