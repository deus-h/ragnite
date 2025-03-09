#!/usr/bin/env python3
"""
Multi-Index Hybrid Searcher Example

This script demonstrates how to use the MultiIndexHybridSearcher to search
across multiple indices or collections and combine the results.

The example shows:
1. Creating search functions for multiple sources
2. Configuring the MultiIndexHybridSearcher with different weights for each source
3. Searching across all sources and combining results
4. Adjusting weights dynamically
5. Explaining the search process and results
"""

import sys
import os
import json
import pprint
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tools.src.retrieval import get_hybrid_searcher

# Sample documents for demonstration
GENERAL_DOCS = [
    {
        "id": "general_1",
        "content": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed."
    },
    {
        "id": "general_2",
        "content": "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by humans or animals."
    },
    {
        "id": "general_3",
        "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge from data."
    },
    {
        "id": "general_4",
        "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."
    }
]

CODE_DOCS = [
    {
        "id": "code_1",
        "content": "def train_model(data, labels, epochs=10):\n    model = create_model()\n    model.fit(data, labels, epochs=epochs)\n    return model"
    },
    {
        "id": "code_2",
        "content": "class NeuralNetwork:\n    def __init__(self, layers):\n        self.layers = layers\n        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]"
    },
    {
        "id": "code_3",
        "content": "def preprocess_data(data):\n    # Normalize data\n    data = (data - data.mean(axis=0)) / data.std(axis=0)\n    return data"
    }
]

SCIENCE_DOCS = [
    {
        "id": "science_1",
        "content": "The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated."
    },
    {
        "id": "science_2",
        "content": "Backpropagation is a method used to calculate the gradient of the loss function with respect to the weights in an artificial neural network."
    },
    {
        "id": "science_3",
        "content": "The accuracy of machine learning models is highly dependent on the quality and quantity of the training data."
    },
    {
        "id": "science_4",
        "content": "Overfitting occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data."
    },
    {
        "id": "science_5",
        "content": "Feature selection is the process of selecting a subset of relevant features for use in model construction."
    }
]


def simple_search(query: str, docs: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    """
    Simple search function that matches query terms to documents.
    
    Args:
        query: The search query
        docs: List of documents to search in
        limit: Maximum number of results to return
    
    Returns:
        List of matching documents with scores
    """
    query_terms = set(query.lower().split())
    results = []
    
    for doc in docs:
        content = doc["content"].lower()
        # Count matching terms
        term_count = sum(1 for term in query_terms if term in content)
        
        if term_count > 0:
            # Calculate a simple score based on term matches
            score = term_count / len(query_terms)
            
            # Create a result with the required fields
            result = {
                "id": doc["id"],
                "content": doc["content"],
                "score": score
            }
            results.append(result)
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top results up to the limit
    return results[:limit]


def search_general(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """Search function for general documents."""
    return simple_search(query, GENERAL_DOCS, limit)


def search_code(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """Search function for code documents."""
    return simple_search(query, CODE_DOCS, limit)


def search_science(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """Search function for science documents."""
    return simple_search(query, SCIENCE_DOCS, limit)


def print_results(results: List[Dict[str, Any]], title: str = "Results") -> None:
    """
    Print formatted results.
    
    Args:
        results: List of result dictionaries
        title: Title for the results section
    """
    print(f"\n{title}")
    print("-" * 80)
    
    if not results:
        print("No results found.")
        return
    
    for i, result in enumerate(results):
        # Check if source information is available
        source_info = f"[{result['source']}] " if 'source' in result else ""
        
        print(f"{i+1}. {source_info}[Score: {result['score']:.4f}] {result['id']}")
        
        # Print a snippet of the content
        content = result.get('content', '')
        print(f"   {content[:100]}{'...' if len(content) > 100 else ''}")
        
        # If sources detail is available, print it
        if 'sources' in result:
            print("   Sources:")
            for source in result['sources']:
                print(f"     - {source['name']}: original score={source['score']:.4f}, weight={source['weight']}")
        
        print()


def main():
    """
    Main function demonstrating the MultiIndexHybridSearcher.
    """
    print("\n" + "="*80)
    print("Multi-Index Hybrid Searcher Example")
    print("="*80)
    
    # Example 1: Basic usage with equal weights
    print("\n1. Basic usage with equal weights")
    print("-" * 60)
    
    # Create a list of search function configurations with equal weights
    search_funcs_equal = [
        {'func': search_general, 'name': 'general', 'weight': 1.0},
        {'func': search_code, 'name': 'code', 'weight': 1.0},
        {'func': search_science, 'name': 'science', 'weight': 1.0}
    ]
    
    # Create the hybrid searcher with equal weights
    equal_weights_searcher = get_hybrid_searcher(
        searcher_type="multi_index",
        search_funcs=search_funcs_equal,
        config={
            'combination_method': 'linear_combination',
            'normalize_scores': True,
            'include_source': True
        }
    )
    
    # Search with a query that should match documents in all collections
    query = "machine learning model"
    print(f"\nSearching for: '{query}' with equal weights")
    
    results = equal_weights_searcher.search(query, limit=5)
    print_results(results)
    
    # Example 2: Custom weights for each source
    print("\n\n2. Custom weights for each source")
    print("-" * 60)
    
    # Create a list of search function configurations with custom weights
    search_funcs_weighted = [
        {'func': search_general, 'name': 'general', 'weight': 0.5},
        {'func': search_code, 'name': 'code', 'weight': 0.2},
        {'func': search_science, 'name': 'science', 'weight': 0.3}
    ]
    
    # Create the hybrid searcher with custom weights
    weighted_searcher = get_hybrid_searcher(
        searcher_type="multi_index",
        search_funcs=search_funcs_weighted,
        config={
            'combination_method': 'linear_combination',
            'normalize_scores': True,
            'include_source': True
        }
    )
    
    # Check the weights
    weights = weighted_searcher.get_component_weights()
    print(f"Component weights: {weights}")
    
    # Search with the same query
    print(f"\nSearching for: '{query}' with custom weights")
    
    results = weighted_searcher.search(query, limit=5)
    print_results(results)
    
    # Example 3: Changing weights dynamically
    print("\n\n3. Changing weights dynamically")
    print("-" * 60)
    
    # Update weights to favor code documents
    new_weights = {
        'general': 0.2,
        'code': 0.7,
        'science': 0.1
    }
    
    weighted_searcher.set_component_weights(new_weights)
    
    # Check the updated weights
    updated_weights = weighted_searcher.get_component_weights()
    print(f"Updated component weights: {updated_weights}")
    
    # Search with the same query but new weights
    print(f"\nSearching for: '{query}' with updated weights")
    
    results = weighted_searcher.search(query, limit=5)
    print_results(results)
    
    # Example 4: Using reciprocal rank fusion
    print("\n\n4. Using reciprocal rank fusion")
    print("-" * 60)
    
    # Create a searcher using reciprocal rank fusion
    rrf_searcher = get_hybrid_searcher(
        searcher_type="multi_index",
        search_funcs=search_funcs_weighted,  # Reuse the weighted search functions
        config={
            'combination_method': 'reciprocal_rank_fusion',
            'normalize_scores': True,
            'include_source': True
        }
    )
    
    # Search with reciprocal rank fusion
    print(f"\nSearching for: '{query}' with reciprocal rank fusion")
    
    results = rrf_searcher.search(query, limit=5)
    print_results(results)
    
    # Example 5: Getting search explanations
    print("\n\n5. Getting search explanations")
    print("-" * 60)
    
    # Get an explanation of the search results
    explanation = weighted_searcher.explain_search(query, limit=5)
    
    print(f"Query: {explanation['query']}")
    print(f"Search strategy: {explanation['search_strategy']}")
    print(f"Description: {explanation['description']}")
    
    print("\nIndices:")
    for index in explanation['indices']:
        print(f"  - {index['name']} (weight: {index['weight']})")
    
    print("\nComponent results:")
    for component in explanation['components']:
        print(f"\n  {component['name']} (weight: {component['weight']})")
        for i, result in enumerate(component['results'][:2]):  # Show top 2 from each component
            print(f"    {i+1}. [Score: {result['score']:.4f}] {result['id']}")
    
    print("\nCombined results:")
    for i, result in enumerate(explanation['results'][:3]):  # Show top 3 combined results
        source_info = f"[{result['source']}] " if 'source' in result else ""
        print(f"  {i+1}. {source_info}[Score: {result['score']:.4f}] {result['id']}")
    
    print("\n" + "="*80)
    print("End of Multi-Index Hybrid Searcher Example")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 