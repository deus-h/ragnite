#!/usr/bin/env python3
"""
BM25 Vector Hybrid Searcher Example

This script demonstrates how to use the BM25VectorHybridSearcher to combine
BM25 keyword search with vector similarity search for improved retrieval performance.

The example shows:
1. Creating a hybrid searcher with an internal BM25 index
2. Creating a hybrid searcher with an external BM25 search function
3. Configuring different BM25 variants and parameters
4. Adjusting component weights
5. Using different combination methods
6. Getting search explanations
"""

import sys
import os
import json
import pprint
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tools.src.retrieval import get_hybrid_searcher


# Sample corpus for demonstration
SAMPLE_CORPUS = [
    "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
    "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
    "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos.",
    "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment.",
    "Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.",
    "Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels.",
    "Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.",
    "Generative adversarial networks are a class of machine learning frameworks designed by Ian Goodfellow and his colleagues in 2014.",
    "Recurrent neural networks are a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence."
]

# Document IDs for the sample corpus
SAMPLE_DOC_IDS = [f"doc_{i}" for i in range(len(SAMPLE_CORPUS))]


def mock_vector_search(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """
    Mock function for vector similarity search.
    
    In a real application, this would call a vector database or embedding model.
    For this example, we'll simulate vector search results based on simple word overlap.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        **kwargs: Additional arguments
    
    Returns:
        List of dictionaries with search results
    """
    # Simple word-based similarity (not a real vector search, just for demonstration)
    query_words = set(query.lower().split())
    results = []
    
    for i, doc in enumerate(SAMPLE_CORPUS):
        # Count word overlap (very simplified simulation of vector similarity)
        doc_words = set(doc.lower().split())
        overlap = len(query_words.intersection(doc_words))
        
        # Add some variation to simulate vector search
        # In a real application, this would be the cosine similarity or other vector distance
        similarity = min(0.95, (overlap * 0.2) + (0.1 if "machine" in doc.lower() else 0))
        
        if similarity > 0:
            results.append({
                "id": SAMPLE_DOC_IDS[i],
                "content": doc,
                "score": similarity
            })
    
    # Sort by score and limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


def mock_external_bm25_search(query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
    """
    Mock function for external BM25 search.
    
    In a real application, this would call an external BM25 implementation.
    For this example, we'll simulate BM25 results based on term frequency.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        **kwargs: Additional arguments
    
    Returns:
        List of dictionaries with search results
    """
    # Simple term frequency (not a real BM25, just for demonstration)
    query_terms = query.lower().split()
    results = []
    
    for i, doc in enumerate(SAMPLE_CORPUS):
        doc_lower = doc.lower()
        
        # Count term frequency (very simplified simulation of BM25)
        term_count = sum(1 for term in query_terms if term in doc_lower)
        score = term_count / len(query_terms) if query_terms else 0
        
        # Add BM25-like behavior: favor shorter documents with exact matches
        doc_length_factor = 1.0 - (len(doc.split()) / 30)  # Normalize by expected max length
        score = score * (0.8 + 0.2 * doc_length_factor)
        
        if score > 0:
            results.append({
                "id": SAMPLE_DOC_IDS[i],
                "content": doc,
                "score": min(0.99, score)  # Cap at 0.99
            })
    
    # Sort by score and limit results
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


def main():
    """
    Main function demonstrating the BM25VectorHybridSearcher.
    """
    print("\n" + "="*80)
    print("BM25 Vector Hybrid Searcher Example")
    print("="*80)
    
    # Example 1: Create a hybrid searcher with an internal BM25 index
    print("\n1. Creating a hybrid searcher with an internal BM25 index")
    print("-" * 60)
    
    hybrid_searcher = get_hybrid_searcher(
        searcher_type="bm25_vector",
        vector_search_func=mock_vector_search,
        corpus=SAMPLE_CORPUS,
        doc_ids=SAMPLE_DOC_IDS,
        config={
            "vector_weight": 0.6,
            "bm25_weight": 0.4,
            "bm25_variant": "plus",  # Using BM25+ variant
            "combination_method": "linear_combination",
            "normalize_scores": True
        }
    )
    
    # Perform a search
    query = "machine learning neural networks"
    print(f"\nSearching for: '{query}'")
    results = hybrid_searcher.search(query, limit=5)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['id']}: {result['content'][:80]}...")
    
    # Example 2: Create a hybrid searcher with an external BM25 search function
    print("\n\n2. Creating a hybrid searcher with an external BM25 search function")
    print("-" * 60)
    
    external_hybrid_searcher = get_hybrid_searcher(
        searcher_type="bm25_vector",
        vector_search_func=mock_vector_search,
        bm25_search_func=mock_external_bm25_search,
        config={
            "vector_weight": 0.5,
            "bm25_weight": 0.5,
            "combination_method": "linear_combination"
        }
    )
    
    # Perform a search
    query = "deep learning artificial intelligence"
    print(f"\nSearching for: '{query}'")
    results = external_hybrid_searcher.search(query, limit=5)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['id']}: {result['content'][:80]}...")
    
    # Example 3: Configuring different BM25 variants and parameters
    print("\n\n3. Configuring different BM25 variants and parameters")
    print("-" * 60)
    
    # Create a searcher with BM25L variant and custom parameters
    bm25l_searcher = get_hybrid_searcher(
        searcher_type="bm25_vector",
        vector_search_func=mock_vector_search,
        corpus=SAMPLE_CORPUS,
        doc_ids=SAMPLE_DOC_IDS,
        config={
            "bm25_variant": "l",  # Using BM25L variant
            "bm25_k1": 1.2,       # Lower k1 for less term frequency saturation
            "bm25_b": 0.8,        # Higher b for more document length normalization
            "bm25_delta": 0.7     # Higher delta for stronger lower-bound on term frequency
        }
    )
    
    # Perform a search
    query = "reinforcement learning agents"
    print(f"\nSearching for: '{query}' with BM25L variant")
    results = bm25l_searcher.search(query, limit=3)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['id']}: {result['content'][:80]}...")
    
    # Example 4: Adjusting component weights
    print("\n\n4. Adjusting component weights")
    print("-" * 60)
    
    # Get current weights
    weights = hybrid_searcher.get_component_weights()
    print(f"Current weights: {weights}")
    
    # Adjust weights to favor BM25 more heavily
    hybrid_searcher.set_component_weights({
        "vector": 0.3,
        "bm25": 0.7
    })
    
    # Get updated weights
    weights = hybrid_searcher.get_component_weights()
    print(f"Updated weights: {weights}")
    
    # Perform a search with new weights
    query = "machine learning without programming"
    print(f"\nSearching for: '{query}' with updated weights")
    results = hybrid_searcher.search(query, limit=3)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['id']}: {result['content'][:80]}...")
    
    # Example 5: Using different combination methods
    print("\n\n5. Using different combination methods")
    print("-" * 60)
    
    # Change to reciprocal rank fusion
    hybrid_searcher.set_config({
        "combination_method": "reciprocal_rank_fusion"
    })
    
    # Perform a search with reciprocal rank fusion
    query = "neural networks deep learning"
    print(f"\nSearching for: '{query}' with reciprocal rank fusion")
    results = hybrid_searcher.search(query, limit=3)
    
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['id']}: {result['content'][:80]}...")
    
    # Example 6: Getting search explanations
    print("\n\n6. Getting search explanations")
    print("-" * 60)
    
    # Get an explanation of the search results
    query = "computer vision image processing"
    print(f"\nExplaining search for: '{query}'")
    explanation = hybrid_searcher.explain_search(query, limit=3)
    
    print("\nExplanation:")
    print(f"Query: {explanation['query']}")
    print(f"Search strategy: {explanation['search_strategy']}")
    print(f"Description: {explanation['description']}")
    print(f"Configuration:")
    pprint.pprint(explanation['configuration'])
    
    print("\nComponent weights:")
    pprint.pprint({
        "vector": explanation['components']['vector']['weight'],
        "bm25": explanation['components']['bm25']['weight']
    })
    
    print("\nTop combined results:")
    for i, result in enumerate(explanation['results'][:2]):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['id']}")
    
    print("\nBM25 parameters:")
    pprint.pprint(explanation['configuration']['bm25_parameters'])
    
    print("\n" + "="*80)
    print("End of BM25 Vector Hybrid Searcher Example")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 