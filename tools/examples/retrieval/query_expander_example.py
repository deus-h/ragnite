#!/usr/bin/env python3
"""
Query Expander Example

This script demonstrates how to use the QueryExpander to expand queries
with synonyms and related terms.
"""

import sys
import os

# Add the 'tools' directory to the Python path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.retrieval import get_query_processor

def main():
    """Demonstrate query expansion using various methods."""
    
    print("🔍 Query Expansion Example\n")
    
    # Example 1: Basic thesaurus-based expansion
    print("Example 1: Thesaurus-based expansion")
    
    # Create a simple thesaurus
    thesaurus = {
        "query": ["search", "question", "lookup"],
        "expand": ["extend", "broaden", "enlarge"],
        "improve": ["enhance", "upgrade", "boost"],
        "retrieval": ["search", "recovery", "access"]
    }
    
    # Create an expander using the factory function
    expander = get_query_processor(
        processor_type="expander",
        expansion_method="synonym",
        max_expansions=3,
        thesaurus=thesaurus
    )
    
    # Test queries
    test_queries = [
        "how to improve query retrieval",
        "expand search results",
        "methods for improving retrieval performance"
    ]
    
    for query in test_queries:
        expanded_queries = expander.process_query(query)
        print(f"\nOriginal: {query}")
        print(f"Expanded ({len(expanded_queries)} variations):")
        for i, expanded in enumerate(expanded_queries):
            if i == 0:
                print(f"  - {expanded} (original)")
            else:
                print(f"  - {expanded}")
    
    # Example 2: WordNet-based expansion (if available)
    try:
        print("\n\nExample 2: WordNet-based expansion")
        
        # Create a WordNet-based expander
        wordnet_expander = get_query_processor(
            processor_type="expander",
            expansion_method="wordnet",
            max_expansions=5
        )
        
        # Test with the same queries
        for query in test_queries:
            expanded_queries = wordnet_expander.process_query(query)
            print(f"\nOriginal: {query}")
            print(f"Expanded ({len(expanded_queries)} variations):")
            for i, expanded in enumerate(expanded_queries):
                if i == 0:
                    print(f"  - {expanded} (original)")
                else:
                    print(f"  - {expanded}")
    except ImportError:
        print("\nWordNet expansion requires NLTK. Install with: pip install nltk")
    
    # Example 3: Custom expansion function
    print("\n\nExample 3: Custom expansion function")
    
    # Define a custom expansion function
    def custom_query_expander(query: str):
        # This is a very simplistic example; in practice, you would use more sophisticated methods
        expanded = []
        
        # Add a "how to" version if not already present
        if not query.startswith("how to"):
            expanded.append(f"how to {query}")
        
        # Add a question form
        if not query.endswith("?"):
            expanded.append(f"what is {query}?")
            
        # Add a definition form
        expanded.append(f"define {query}")
        
        return expanded
    
    # Create a custom expander
    custom_expander = get_query_processor(
        processor_type="expander",
        expansion_method="custom",
        custom_expander=custom_query_expander
    )
    
    # Test with the same queries
    for query in test_queries:
        expanded_queries = custom_expander.process_query(query)
        print(f"\nOriginal: {query}")
        print(f"Expanded ({len(expanded_queries)} variations):")
        for i, expanded in enumerate(expanded_queries):
            if i == 0:
                print(f"  - {expanded} (original)")
            else:
                print(f"  - {expanded}")
    
    print("\n🔮 Query expansion complete!")


if __name__ == "__main__":
    main() 