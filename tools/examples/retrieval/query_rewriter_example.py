#!/usr/bin/env python3
"""
Query Rewriter Example

This script demonstrates how to use the QueryRewriter to rewrite queries
for improved retrieval performance.
"""

import sys
import os

# Add the 'tools' directory to the Python path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.retrieval import get_query_processor

def main():
    """Demonstrate query rewriting using various methods."""
    
    print("🔍 Query Rewriter Example\n")
    
    # Example 1: Basic rewriting
    print("Example 1: Basic Rewriting")
    
    # Create a basic rewriter
    basic_rewriter = get_query_processor(
        processor_type="rewriter",
        rewrite_method="basic",
        add_prefix="I need information about",
        remove_stopwords=False,
        fix_spelling=False
    )
    
    # Test queries
    test_queries = [
        "machine learning",
        "python programming tutorial",
        "quantum computing basics",
        "misspeled woords in querys"
    ]
    
    for query in test_queries:
        rewritten_query = basic_rewriter.process_query(query)
        print(f"\nOriginal: {query}")
        print(f"Rewritten: {rewritten_query}")
    
    # Example 2: Template-based rewriting
    print("\n\nExample 2: Template-based Rewriting")
    
    # Create a template-based rewriter
    templates = [
        "What is {}?",
        "How do I use {}?",
        "Tell me about {}"
    ]
    
    template_rewriter = get_query_processor(
        processor_type="rewriter",
        rewrite_method="template",
        templates=templates
    )
    
    for query in test_queries:
        rewritten_query = template_rewriter.process_query(query)
        print(f"\nOriginal: {query}")
        print(f"Rewritten: {rewritten_query}")
    
    # Example 3: Spell correction (if available)
    try:
        print("\n\nExample 3: Spell Correction")
        
        spell_rewriter = get_query_processor(
            processor_type="rewriter",
            rewrite_method="basic",
            fix_spelling=True
        )
        
        misspelled_queries = [
            "artifcial intelligance",
            "pythom programing",
            "retrival augmneted generation",
            "natral languag procesing"
        ]
        
        for query in misspelled_queries:
            rewritten_query = spell_rewriter.process_query(query)
            print(f"\nOriginal: {query}")
            print(f"Rewritten: {rewritten_query}")
    except ImportError:
        print("\nSpell checking requires pyspellchecker. Install with: pip install pyspellchecker")
    
    # Example 4: Custom rewriting function
    print("\n\nExample 4: Custom Rewriting Function")
    
    # Define a custom rewriting function
    def custom_query_rewriter(query: str) -> str:
        # This is a very simplistic example; in practice, you would use more sophisticated methods
        
        # Convert to lowercase except for proper nouns (simplified)
        words = query.split()
        processed_words = []
        
        for word in words:
            # Preserve capitalization for proper nouns
            if word and word[0].isupper() and len(word) > 1:
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        
        # Join words
        query = ' '.join(processed_words)
        
        # Add quotes for multi-word queries
        if len(processed_words) > 1:
            query = f'"{query}"'
            
        # Add search operator for more precise results
        if "how to" not in query.lower() and "what is" not in query.lower():
            query = f"exact:{query}"
        
        return query
    
    # Create a custom rewriter
    custom_rewriter = get_query_processor(
        processor_type="rewriter",
        rewrite_method="custom",
        custom_rewriter=custom_query_rewriter
    )
    
    advanced_queries = [
        "Python vs JavaScript",
        "TensorFlow examples",
        "How to implement RAG",
        "New York City"
    ]
    
    for query in advanced_queries:
        rewritten_query = custom_rewriter.process_query(query)
        print(f"\nOriginal: {query}")
        print(f"Rewritten: {rewritten_query}")
    
    print("\n🔮 Query rewriting complete!")


if __name__ == "__main__":
    main() 