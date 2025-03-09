#!/usr/bin/env python3
"""
Query Analyzer Example

This script demonstrates how to use the QueryAnalyzer to analyze query characteristics
and understand how to improve query performance.
"""

import sys
import os
import json
from pprint import pprint

# Add the 'tools' directory to the Python path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from tools.src.retrieval import get_retrieval_debugger

def main():
    """Demonstrate the QueryAnalyzer with various query examples."""
    
    print("🔍 Query Analyzer Example\n")
    
    # Create sample queries of different types and complexity
    sample_queries = [
        "vector databases",
        "What is retrieval-augmented generation?",
        "How does vector search compare to keyword search?",
        "What are the best practices for implementing RAG with vector databases?",
        "I need information about optimizing vector database performance but not for huge datasets",
        "Compare Chroma vs Qdrant vs Pinecone for RAG applications"
    ]
    
    # Sample retrieval results (normally these would come from your retrieval system)
    # We'll use a minimal structure for demonstration purposes
    result_counts = {
        "vector databases": 15,
        "What is retrieval-augmented generation?": 8,
        "How does vector search compare to keyword search?": 10,
        "What are the best practices for implementing RAG with vector databases?": 5,
        "I need information about optimizing vector database performance but not for huge datasets": 3,
        "Compare Chroma vs Qdrant vs Pinecone for RAG applications": 12
    }
    
    # Mock results with just enough structure for the analyzer to work with
    def get_mock_results(query: str):
        count = result_counts.get(query, 5)
        return [{"id": f"doc_{i}", "score": 0.9 - (i * 0.05)} for i in range(count)]
    
    # Create a QueryAnalyzer
    analyzer = get_retrieval_debugger(
        debugger_type="query_analyzer",
        complexity_threshold=10,
        min_term_length=3,
        include_entity_recognition=False,  # Set to True if you have spaCy installed
        query_reformulation=True,
        term_weighting_method="tfidf",
        max_insights=5
    )
    
    # Example 1: Basic Query Analysis
    print("Example 1: Basic Query Analysis")
    
    # Analyze each query
    for query in sample_queries:
        results = get_mock_results(query)
        
        print(f"\nQuery: \"{query}\"")
        
        # Perform analysis
        analysis = analyzer.analyze(query, results)
        
        # Display key term information
        print("Key Terms:")
        for term, score in analysis.get("key_terms", []):
            print(f"  - {term} (score: {score:.3f})")
        
        # Display complexity metrics
        complexity = analysis.get("complexity", {})
        print("\nComplexity:")
        print(f"  - Word Count: {complexity.get('word_count', 0)}")
        print(f"  - Is Complex: {complexity.get('is_complex', False)}")
        print(f"  - Question Type: {complexity.get('question_type', 'unknown')}")
        
        # Display insights
        print("\nInsights:")
        for i, insight in enumerate(analysis.get("insights", []), 1):
            print(f"{i}. {insight}")
        
        print("-" * 80)
    
    # Example 2: Comparing Query Performance
    print("\n\nExample 2: Comparing Query Performance Across Different Systems")
    
    # Define a complex query
    complex_query = "What are the differences between HNSW and IVF indexing methods for vector databases in terms of performance and accuracy?"
    
    # Create mock results for three different systems
    system1_results = [{"id": f"doc_1_{i}", "score": 0.9 - (i * 0.05)} for i in range(8)]
    system2_results = [{"id": f"doc_2_{i}", "score": 0.85 - (i * 0.06)} for i in range(12)]
    system3_results = [{"id": f"doc_3_{i}", "score": 0.95 - (i * 0.08)} for i in range(5)]
    
    # Compare query performance across systems
    comparison = analyzer.compare(
        complex_query,
        [system1_results, system2_results, system3_results],
        names=["Default System", "Optimized System", "Experimental System"]
    )
    
    # Display comparison results
    print(f"Query: \"{complex_query}\"\n")
    
    print("Performance Comparison:")
    for metrics in comparison.get("performance_comparison", []):
        print(f"\n{metrics['name']}:")
        print(f"  - Result Count: {metrics['result_count']}")
        print(f"  - Processing Time: {metrics['processing_time']:.4f}s")
    
    print(f"\nBest Performing System: {comparison.get('best_performing', 'None')}")
    
    print("\nInsights:")
    for i, insight in enumerate(comparison.get("insights", []), 1):
        print(f"{i}. {insight}")
    
    # Example 3: Query Reformulation Suggestions
    print("\n\nExample 3: Query Reformulation Suggestions")
    
    # Create a complex, potentially problematic query
    problematic_query = "I want to find information about vector databases but not about how they work internally and not about performance issues with large datasets, focusing mainly on integration with LLMs"
    
    # Analyze the query
    reformulation_analysis = analyzer.analyze(problematic_query, [])
    
    print(f"Original Query: \"{problematic_query}\"\n")
    
    # Display complexity information
    complexity = reformulation_analysis.get("complexity", {})
    print("Query Complexity:")
    print(f"  - Word Count: {complexity.get('word_count', 0)}")
    print(f"  - Has Negation: {complexity.get('has_negation', False)}")
    
    # Display reformulation suggestions
    print("\nReformulation Suggestions:")
    for i, suggestion in enumerate(reformulation_analysis.get("reformulations", []), 1):
        print(f"{i}. {suggestion}")
    
    # Example 4: Evaluating Query Effectiveness
    print("\n\nExample 4: Evaluating Query Effectiveness")
    
    # Create a test query
    test_query = "How to implement semantic search with vector databases"
    
    # Create mock results
    test_results = [
        {"id": "doc_1", "score": 0.92},
        {"id": "doc_2", "score": 0.85},
        {"id": "doc_3", "score": 0.78},
        {"id": "doc_4", "score": 0.72},
        {"id": "doc_5", "score": 0.65},
        {"id": "doc_6", "score": 0.61},
        {"id": "doc_7", "score": 0.58},
    ]
    
    # Define ground truth (relevant document IDs)
    ground_truth = ["doc_2", "doc_5", "doc_7", "doc_9"]
    
    # Evaluate query effectiveness
    evaluation = analyzer.evaluate(test_query, test_results, ground_truth)
    
    print(f"Query: \"{test_query}\"\n")
    
    print("Evaluation Metrics:")
    for metric, value in evaluation.items():
        print(f"  - {metric}: {value:.3f}")
    
    # Example 5: Custom Term Analysis with a Corpus
    print("\n\nExample 5: Custom Term Analysis with a Corpus")
    
    # Create a query
    domain_query = "implementing vector search for semantic retrieval"
    
    # Create a small corpus to provide context for term importance
    mini_corpus = [
        "Vector databases store embeddings for semantic search applications",
        "Semantic retrieval uses vector similarity to find relevant documents",
        "Implementing vector search requires choosing appropriate vector representations",
        "Neural search frameworks provide tools for semantic search implementation"
    ]
    
    # Analyze with custom corpus
    corpus_analysis = analyzer.analyze(
        domain_query, 
        [],
        corpus=mini_corpus,
        max_key_terms=8  # Request more terms
    )
    
    print(f"Query: \"{domain_query}\"\n")
    
    print("Key Terms (with domain corpus context):")
    for term, score in corpus_analysis.get("key_terms", []):
        print(f"  - {term} (score: {score:.3f})")
    
    print("\n🔮 Query analysis complete!")


if __name__ == "__main__":
    main() 