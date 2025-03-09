#!/usr/bin/env python3
"""
Weighted Hybrid Searcher Example

This script demonstrates how to use the WeightedHybridSearcher to combine multiple
search strategies with automatic weight tuning based on performance metrics.

The example shows:
1. Creating search functions for different strategies
2. Configuring the WeightedHybridSearcher with different weights for each strategy
3. Searching with manual weights
4. Enabling automatic weight tuning based on relevance feedback
5. Tracking performance metrics and weight changes over time
"""

import sys
import os
import json
import pprint
import time
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tools.src.retrieval import get_hybrid_searcher

# Sample documents for demonstration
DOCUMENTS = [
    {
        "id": "doc1",
        "content": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
        "keywords": ["machine learning", "computers", "programming"],
        "category": "ai"
    },
    {
        "id": "doc2",
        "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        "keywords": ["deep learning", "machine learning", "neural networks"],
        "category": "ai"
    },
    {
        "id": "doc3",
        "content": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
        "keywords": ["neural networks", "computing", "brain"],
        "category": "ai"
    },
    {
        "id": "doc4",
        "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.",
        "keywords": ["python", "programming", "language"],
        "category": "programming"
    },
    {
        "id": "doc5",
        "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence.",
        "keywords": ["nlp", "natural language", "ai"],
        "category": "ai"
    },
    {
        "id": "doc6",
        "content": "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment.",
        "keywords": ["reinforcement learning", "machine learning", "agents"],
        "category": "ai"
    },
    {
        "id": "doc7",
        "content": "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos.",
        "keywords": ["computer vision", "images", "videos"],
        "category": "ai"
    },
    {
        "id": "doc8",
        "content": "JavaScript is a programming language that is one of the core technologies of the World Wide Web, alongside HTML and CSS.",
        "keywords": ["javascript", "programming", "web"],
        "category": "programming"
    },
    {
        "id": "doc9",
        "content": "TensorFlow is an open-source software library for machine learning and artificial intelligence.",
        "keywords": ["tensorflow", "machine learning", "library"],
        "category": "ai"
    },
    {
        "id": "doc10",
        "content": "PyTorch is an open-source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing.",
        "keywords": ["pytorch", "machine learning", "library"],
        "category": "ai"
    }
]


def vector_search(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """
    Simulated vector search function.
    
    In a real application, this would use embeddings and vector similarity.
    For this example, we'll simulate vector search with a simple word overlap approach.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        **kwargs: Additional arguments
    
    Returns:
        List of dictionaries with search results
    """
    query_words = set(query.lower().split())
    results = []
    
    for doc in DOCUMENTS:
        content = doc["content"].lower()
        # Count word overlap (simulating vector similarity)
        overlap = sum(1 for word in query_words if word in content)
        
        if overlap > 0:
            # Calculate a score based on word overlap
            score = overlap / len(query_words)
            
            # Add some variation to simulate vector search
            if "neural" in query.lower() and "neural" in content:
                score += 0.2
            if "learning" in query.lower() and "learning" in content:
                score += 0.1
            
            # Cap at 1.0
            score = min(1.0, score)
            
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "score": score,
                "metadata": {"category": doc["category"]}
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top results up to the limit
    return results[:limit]


def keyword_search(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """
    Simulated keyword search function.
    
    This function searches for exact keyword matches in the document keywords.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        **kwargs: Additional arguments
    
    Returns:
        List of dictionaries with search results
    """
    query_terms = query.lower().split()
    results = []
    
    for doc in DOCUMENTS:
        # Check for keyword matches
        matches = sum(1 for term in query_terms if any(term in kw.lower() for kw in doc["keywords"]))
        
        if matches > 0:
            # Calculate a score based on keyword matches
            score = matches / len(query_terms)
            
            # Exact matches get a boost
            if any(query.lower() == kw.lower() for kw in doc["keywords"]):
                score += 0.3
            
            # Cap at 1.0
            score = min(1.0, score)
            
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "score": score,
                "metadata": {"category": doc["category"]}
            })
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top results up to the limit
    return results[:limit]


def category_search(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """
    Simulated category-based search function.
    
    This function prioritizes documents in the 'ai' category for AI-related queries.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        **kwargs: Additional arguments
    
    Returns:
        List of dictionaries with search results
    """
    # Determine if this is an AI-related query
    ai_terms = ["ai", "artificial intelligence", "machine learning", "neural", "deep learning"]
    is_ai_query = any(term in query.lower() for term in ai_terms)
    
    results = []
    
    for doc in DOCUMENTS:
        # Base score on category match
        if is_ai_query and doc["category"] == "ai":
            score = 0.8
        elif not is_ai_query and doc["category"] != "ai":
            score = 0.7
        else:
            score = 0.3
        
        # Add some content matching
        query_words = set(query.lower().split())
        content = doc["content"].lower()
        overlap = sum(1 for word in query_words if word in content)
        
        if overlap > 0:
            score += 0.1 * (overlap / len(query_words))
        
        # Cap at 1.0
        score = min(1.0, score)
        
        results.append({
            "id": doc["id"],
            "content": doc["content"],
            "score": score,
            "metadata": {"category": doc["category"]}
        })
    
    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top results up to the limit
    return results[:limit]


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
        print(f"   {result['content']}")
        
        # If sources detail is available, print it
        if 'sources' in result:
            print("   Sources:")
            for source in result['sources']:
                print(f"     - {source['name']}: original score={source['score']:.4f}, weight={source['weight']}")
        
        print()


def main():
    """
    Main function demonstrating the WeightedHybridSearcher.
    """
    print("\n" + "="*80)
    print("Weighted Hybrid Searcher Example")
    print("="*80)
    
    # Example 1: Basic usage with manual weights
    print("\n1. Basic usage with manual weights")
    print("-" * 60)
    
    # Create a list of search function configurations with manual weights
    search_funcs = [
        {'func': vector_search, 'name': 'vector', 'weight': 0.5},
        {'func': keyword_search, 'name': 'keyword', 'weight': 0.3},
        {'func': category_search, 'name': 'category', 'weight': 0.2}
    ]
    
    # Create the weighted hybrid searcher
    weighted_searcher = get_hybrid_searcher(
        searcher_type="weighted",
        search_funcs=search_funcs,
        config={
            'combination_method': 'linear_combination',
            'normalize_scores': True,
            'include_source': True,
            'auto_tune': False  # Start with auto-tuning disabled
        }
    )
    
    # Check the weights
    weights = weighted_searcher.get_component_weights()
    print(f"Component weights: {weights}")
    
    # Search with a query
    query = "neural networks machine learning"
    print(f"\nSearching for: '{query}' with manual weights")
    
    results = weighted_searcher.search(query, limit=5)
    print_results(results)
    
    # Example 2: Adjusting weights manually
    print("\n\n2. Adjusting weights manually")
    print("-" * 60)
    
    # Update weights to favor keyword search
    new_weights = {
        'vector': 0.2,
        'keyword': 0.7,
        'category': 0.1
    }
    
    weighted_searcher.set_component_weights(new_weights)
    
    # Check the updated weights
    updated_weights = weighted_searcher.get_component_weights()
    print(f"Updated component weights: {updated_weights}")
    
    # Search with the same query but new weights
    print(f"\nSearching for: '{query}' with updated weights")
    
    results = weighted_searcher.search(query, limit=5)
    print_results(results)
    
    # Example 3: Using constraints on weights
    print("\n\n3. Using constraints on weights")
    print("-" * 60)
    
    # Create a searcher with weight constraints
    constrained_searcher = get_hybrid_searcher(
        searcher_type="weighted",
        search_funcs=[
            {'func': vector_search, 'name': 'vector', 'weight': 0.4, 'min_weight': 0.3, 'max_weight': 0.6},
            {'func': keyword_search, 'name': 'keyword', 'weight': 0.4, 'min_weight': 0.3, 'max_weight': 0.6},
            {'func': category_search, 'name': 'category', 'weight': 0.2, 'min_weight': 0.1, 'max_weight': 0.3}
        ],
        config={
            'normalize_scores': True,
            'include_source': True
        }
    )
    
    # Try to set weights outside the constraints
    print("Attempting to set weights outside constraints:")
    try:
        constrained_searcher.set_component_weights({
            'vector': 0.1,  # Below min_weight of 0.3
            'keyword': 0.8,  # Above max_weight of 0.6
            'category': 0.1
        })
        
        # Check what weights were actually applied (should be constrained)
        constrained_weights = constrained_searcher.get_component_weights()
        print(f"Constrained weights: {constrained_weights}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Example 4: Enabling automatic weight tuning
    print("\n\n4. Enabling automatic weight tuning")
    print("-" * 60)
    
    # Create a searcher with auto-tuning enabled
    auto_tune_searcher = get_hybrid_searcher(
        searcher_type="weighted",
        search_funcs=[
            {'func': vector_search, 'name': 'vector', 'weight': 0.33},
            {'func': keyword_search, 'name': 'keyword', 'weight': 0.33},
            {'func': category_search, 'name': 'category', 'weight': 0.34}
        ],
        config={
            'auto_tune': True,
            'tuning_metric': 'reciprocal_rank',
            'learning_rate': 0.05,
            'include_source': True
        }
    )
    
    # Define some queries and relevant documents for tuning
    tuning_data = [
        {
            'query': "neural networks",
            'relevant_docs': ["doc2", "doc3"]  # Documents about neural networks
        },
        {
            'query': "machine learning libraries",
            'relevant_docs': ["doc9", "doc10"]  # Documents about ML libraries
        },
        {
            'query': "programming languages",
            'relevant_docs': ["doc4", "doc8"]  # Documents about programming languages
        }
    ]
    
    # Run through the tuning data
    print("Training the searcher with relevance feedback:")
    for i, data in enumerate(tuning_data):
        query = data['query']
        relevant_docs = data['relevant_docs']
        
        print(f"\nIteration {i+1}:")
        print(f"Query: '{query}'")
        print(f"Relevant documents: {relevant_docs}")
        print(f"Current weights: {auto_tune_searcher.get_component_weights()}")
        
        # Search with auto-tuning
        results = auto_tune_searcher.search(query, limit=5, relevant_doc_ids=relevant_docs)
        
        # Print top 3 results
        for j, result in enumerate(results[:3]):
            print(f"  {j+1}. [{result['source']}] {result['id']}: {result['score']:.4f}")
        
        print(f"Updated weights: {auto_tune_searcher.get_component_weights()}")
    
    # Example 5: Getting search explanations
    print("\n\n5. Getting search explanations")
    print("-" * 60)
    
    # Get an explanation of the search results
    query = "deep learning neural networks"
    relevant_docs = ["doc2", "doc3"]
    
    print(f"Explaining search for: '{query}'")
    explanation = auto_tune_searcher.explain_search(query, limit=5, relevant_doc_ids=relevant_docs)
    
    print(f"\nQuery: {explanation['query']}")
    print(f"Search strategy: {explanation['search_strategy']}")
    
    print("\nStrategy weights:")
    for strategy in explanation['strategies']:
        print(f"  {strategy['name']}: {strategy['weight']:.4f}")
    
    if 'performance_metrics' in explanation:
        print("\nPerformance metrics:")
        for metric, value in explanation['performance_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    # Example 6: Viewing performance history
    print("\n\n6. Viewing performance history")
    print("-" * 60)
    
    history = auto_tune_searcher.get_performance_history()
    
    print("Performance history:")
    for i, query in enumerate(history['queries']):
        print(f"\nQuery: {query}")
        print(f"Weights: {history['weights'][i]}")
        
        for metric, values in history['metrics'].items():
            if i < len(values):
                print(f"{metric}: {values[i]:.4f}")
    
    print("\n" + "="*80)
    print("End of Weighted Hybrid Searcher Example")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 