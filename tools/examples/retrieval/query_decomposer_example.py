#!/usr/bin/env python3
"""
Query Decomposer Example

This script demonstrates how to use the QueryDecomposer to break down
complex queries into simpler ones for improved retrieval.
"""

import sys
import os

# Add the 'tools' directory to the Python path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.retrieval import get_query_processor

def main():
    """Demonstrate query decomposition using various methods."""
    
    print("🔍 Query Decomposer Example\n")
    
    # Example 1: Rule-based decomposition
    print("Example 1: Rule-based Decomposition")
    
    # Create a rule-based decomposer
    rule_decomposer = get_query_processor(
        processor_type="decomposer",
        decomposition_method="rule",
        merge_original=True,
        max_subqueries=4
    )
    
    # Test complex queries
    complex_queries = [
        "What is machine learning and how does it differ from deep learning?",
        "How to install Python and set up a virtual environment",
        "What are the advantages and disadvantages of vector databases versus traditional databases?",
        "What is RAG and how can it be used in question answering systems?",
        "Explain the causes and effects of climate change"
    ]
    
    for query in complex_queries:
        subqueries = rule_decomposer.process_query(query)
        print(f"\nOriginal: {query}")
        print(f"Decomposed ({len(subqueries)} subqueries):")
        for i, subquery in enumerate(subqueries):
            if i == 0 and subquery == query:
                print(f"  - {subquery} (original)")
            else:
                print(f"  - {subquery}")
    
    # Example 2: Custom rules-based decomposition
    print("\n\nExample 2: Custom Rules-based Decomposition")
    
    # Create custom rules
    custom_rules = [
        # Rule for breaking down requirements
        {
            'pattern': r'I need to (.+?) that (.+)',
            'subqueries': [r'\1', r'\2', r'how to \1 that \2']
        },
        # Rule for breaking down comparison queries
        {
            'pattern': r'Compare (.+?) with (.+)',
            'subqueries': [r'\1', r'\2', r'difference between \1 and \2', r'similarities between \1 and \2']
        }
    ]
    
    # Create a custom rules decomposer
    custom_rule_decomposer = get_query_processor(
        processor_type="decomposer",
        decomposition_method="rule",
        decomposition_rules=custom_rules,
        merge_original=True
    )
    
    custom_complex_queries = [
        "I need to build a system that can understand natural language",
        "Compare relational databases with document databases",
        "I need to select a framework that supports real-time data processing"
    ]
    
    for query in custom_complex_queries:
        subqueries = custom_rule_decomposer.process_query(query)
        print(f"\nOriginal: {query}")
        print(f"Decomposed ({len(subqueries)} subqueries):")
        for i, subquery in enumerate(subqueries):
            if i == 0 and subquery == query:
                print(f"  - {subquery} (original)")
            else:
                print(f"  - {subquery}")

    # Example 3: Custom function decomposition
    print("\n\nExample 3: Custom Function Decomposition")
    
    # Define a custom decomposition function
    def custom_decomposer(query: str) -> list[str]:
        # This is a simplified example; in practice, you would use more sophisticated methods
        subqueries = []
        
        # Look for specific keywords and generate targeted subqueries
        if "best practices" in query.lower():
            subqueries.append(f"recommended approaches for {query.lower().replace('best practices', '').strip()}")
            subqueries.append(f"common mistakes in {query.lower().replace('best practices', '').strip()}")
            
        if "tutorial" in query.lower():
            subqueries.append(f"beginner guide to {query.lower().replace('tutorial', '').strip()}")
            subqueries.append(f"step by step {query.lower().replace('tutorial', '').strip()}")
            
        if "example" in query.lower() or "examples" in query.lower():
            clean_query = query.lower().replace("examples", "").replace("example", "").strip()
            subqueries.append(f"code sample for {clean_query}")
            subqueries.append(f"implementation of {clean_query}")
            
        # If no specific keywords were found, split by conjunctions
        if not subqueries and any(conj in query.lower() for conj in ["and", "or", "with"]):
            for conj in ["and", "or", "with"]:
                if conj in query.lower():
                    parts = query.lower().split(conj)
                    for part in parts:
                        if len(part.strip().split()) >= 3:  # Only include substantial parts
                            subqueries.append(part.strip())
        
        return subqueries
    
    # Create a custom function decomposer
    custom_func_decomposer = get_query_processor(
        processor_type="decomposer",
        decomposition_method="custom",
        custom_decomposer=custom_decomposer,
        merge_original=True
    )
    
    function_complex_queries = [
        "Python best practices for data science",
        "React tutorial for beginners",
        "Code examples for vector search implementation",
        "Performance optimization and memory management"
    ]
    
    for query in function_complex_queries:
        subqueries = custom_func_decomposer.process_query(query)
        print(f"\nOriginal: {query}")
        print(f"Decomposed ({len(subqueries)} subqueries):")
        for i, subquery in enumerate(subqueries):
            if i == 0 and subquery == query:
                print(f"  - {subquery} (original)")
            else:
                print(f"  - {subquery}")
    
    print("\n🔮 Query decomposition complete!")


if __name__ == "__main__":
    main() 