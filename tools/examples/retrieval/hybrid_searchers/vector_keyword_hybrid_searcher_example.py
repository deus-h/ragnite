#!/usr/bin/env python3
"""
Vector Keyword Hybrid Searcher Example

This script demonstrates how to use the VectorKeywordHybridSearcher to combine
vector similarity search with keyword search to improve retrieval performance.
"""

import sys
import os
import json
import numpy as np
from pprint import pprint
from typing import List, Dict, Any

# Add the project root to the path so we can import the tools package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
sys.path.insert(0, project_root)

from tools.src.retrieval import get_hybrid_searcher


# Mock vector database with sample documents
mock_documents = {
    "doc1": {
        "id": "doc1",
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a type of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "vector": np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    },
    "doc2": {
        "id": "doc2",
        "title": "Natural Language Processing",
        "content": "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        "vector": np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    },
    "doc3": {
        "id": "doc3",
        "title": "Deep Learning Techniques",
        "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to analyze various factors of data.",
        "vector": np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    },
    "doc4": {
        "id": "doc4",
        "title": "Python Programming for Data Science",
        "content": "Python is a popular programming language for data science because of its readability, libraries, and versatility.",
        "vector": np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    },
    "doc5": {
        "id": "doc5",
        "title": "Introduction to Neural Networks",
        "content": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
        "vector": np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    },
    "doc6": {
        "id": "doc6",
        "title": "Data Visualization with Python",
        "content": "Data visualization is the graphical representation of information and data using visual elements like charts, graphs, and maps.",
        "vector": np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    },
    "doc7": {
        "id": "doc7",
        "title": "Statistical Methods for Machine Learning",
        "content": "Statistics plays a crucial role in machine learning by providing the mathematical foundation for many algorithms.",
        "vector": np.array([0.7, 0.8, 0.9, 1.0, 0.1])
    },
    "doc8": {
        "id": "doc8",
        "title": "Reinforcement Learning Algorithms",
        "content": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize the notion of cumulative reward.",
        "vector": np.array([0.8, 0.9, 1.0, 0.1, 0.2])
    },
    "doc9": {
        "id": "doc9",
        "title": "Feature Engineering for Machine Learning",
        "content": "Feature engineering is the process of using domain knowledge to extract features from raw data that make machine learning algorithms work better.",
        "vector": np.array([0.9, 1.0, 0.1, 0.2, 0.3])
    },
    "doc10": {
        "id": "doc10",
        "title": "Ethics in Artificial Intelligence",
        "content": "AI ethics is a set of values, principles, and techniques that employ widely accepted standards of right and wrong to guide moral conduct in the development and use of AI technologies.",
        "vector": np.array([1.0, 0.1, 0.2, 0.3, 0.4])
    }
}


# Mock vector search function
def mock_vector_search(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """
    Perform a mock vector search based on a simple similarity metric.
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        **kwargs: Additional search parameters
        
    Returns:
        List[Dict[str, Any]]: Search results
    """
    # Simple query embedding simulation (not a real embedding)
    query_terms = query.lower().split()
    query_vector = np.array([
        sum([0.1, 0.2, 0.3, 0.4, 0.5][i % 5] * (i + 1) for i, term in enumerate(query_terms) if term)
    ] * 5)
    
    # Calculate cosine similarity
    results = []
    for doc_id, doc in mock_documents.items():
        # Cosine similarity
        similarity = np.dot(query_vector, doc["vector"]) / (
            np.linalg.norm(query_vector) * np.linalg.norm(doc["vector"])
        )
        
        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "score": float(similarity)
        })
    
    # Sort by score and return top results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


# Mock keyword search function
def mock_keyword_search(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """
    Perform a mock keyword search based on term frequency.
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        **kwargs: Additional search parameters
        
    Returns:
        List[Dict[str, Any]]: Search results
    """
    query_terms = set(query.lower().split())
    
    results = []
    for doc_id, doc in mock_documents.items():
        # Calculate term frequency in title and content
        text = (doc["title"] + " " + doc["content"]).lower()
        term_count = sum(1 for term in query_terms if term in text)
        
        # Calculate score based on term frequency
        score = term_count / (len(query_terms) or 1)
        
        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "score": score
        })
    
    # Sort by score and return top results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


def main():
    """Run the vector keyword hybrid searcher example."""
    print("\n" + "="*80)
    print("Vector Keyword Hybrid Searcher Example")
    print("="*80 + "\n")
    
    # Create a hybrid searcher
    hybrid_searcher = get_hybrid_searcher(
        searcher_type="vector_keyword",
        vector_search_func=mock_vector_search,
        keyword_search_func=mock_keyword_search,
        config={
            "vector_weight": 0.7,
            "keyword_weight": 0.3,
            "combination_method": "linear_combination",
            "normalize_scores": True,
            "expand_results": True
        }
    )
    
    # Example 1: Basic hybrid search
    print("\n1. Basic Hybrid Search")
    print("-" * 50)
    
    query = "machine learning algorithms"
    print(f"Query: '{query}'\n")
    
    # Perform a hybrid search
    results = hybrid_searcher.search(query, limit=5)
    
    # Print the results
    print("Top 5 results using hybrid search:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Vector Score: {result['vector_score']:.4f}, Keyword Score: {result['keyword_score']:.4f}")
        print(f"   {result['content'][:100]}...")
    
    # Example 2: Adjust component weights
    print("\n2. Adjusting Component Weights")
    print("-" * 50)
    
    print("Original weights:", hybrid_searcher.get_component_weights())
    
    # Favor keyword search more heavily
    hybrid_searcher.set_component_weights({
        "vector": 0.3,
        "keyword": 0.7
    })
    
    print("Updated weights:", hybrid_searcher.get_component_weights())
    
    # Perform a hybrid search with updated weights
    results = hybrid_searcher.search(query, limit=5)
    
    # Print the results
    print("\nTop 5 results with updated weights (favoring keyword search):")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Vector Score: {result['vector_score']:.4f}, Keyword Score: {result['keyword_score']:.4f}")
        print(f"   {result['content'][:100]}...")
    
    # Example 3: Change combination method
    print("\n3. Using a Different Combination Method")
    print("-" * 50)
    
    # Reset weights to default
    hybrid_searcher.set_component_weights({
        "vector": 0.7,
        "keyword": 0.3
    })
    
    # Change combination method to Reciprocal Rank Fusion
    hybrid_searcher.set_config({
        "combination_method": "reciprocal_rank_fusion"
    })
    
    # Perform a hybrid search with RRF
    results = hybrid_searcher.search(query, limit=5)
    
    # Print the results
    print("Top 5 results using Reciprocal Rank Fusion:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Vector Rank: {result.get('vector_rank', 'N/A')}, Keyword Rank: {result.get('keyword_rank', 'N/A')}")
        print(f"   {result['content'][:100]}...")
    
    # Example 4: Explain search results
    print("\n4. Explaining Search Results")
    print("-" * 50)
    
    # Explain search results
    query = "neural networks deep learning"
    print(f"Query: '{query}'\n")
    
    explanation = hybrid_searcher.explain_search(query, limit=3)
    
    # Print the explanation summary
    print("Search Explanation:")
    print(f"Strategy: {explanation['strategy']}")
    print(f"Weights: Vector={explanation['weights']['vector']:.2f}, Keyword={explanation['weights']['keyword']:.2f}")
    print(f"Configuration: {json.dumps(explanation['configuration'], indent=2)}")
    
    # Print combined results
    print("\nCombined Results:")
    for i, result in enumerate(explanation['results'], 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
    
    # Print component results summary
    print("\nVector Search Results:")
    for i, result in enumerate(explanation['component_results']['vector'][:3], 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
    
    print("\nKeyword Search Results:")
    for i, result in enumerate(explanation['component_results']['keyword'][:3], 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")


if __name__ == "__main__":
    main()
    print("\nVector Keyword Hybrid Searcher Example completed successfully!") 