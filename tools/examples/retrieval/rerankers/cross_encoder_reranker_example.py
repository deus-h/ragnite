#!/usr/bin/env python3
"""
Cross-Encoder Reranker Example

This script demonstrates how to use the CrossEncoderReranker to improve 
retrieval results by re-scoring documents using a cross-encoder model.

The example shows:
1. Creating a cross-encoder reranker
2. Reranking retrieval results
3. Configuring the reranker with different options
4. Getting detailed explanations of the reranking process
5. Comparing different cross-encoder models
"""

import sys
import os
import json
import time
import pprint
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from tools.src.retrieval import get_reranker

# Sample retrieval results for demonstration
SAMPLE_RESULTS = [
    {
        "id": "doc1",
        "content": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.",
        "score": 0.82
    },
    {
        "id": "doc2",
        "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
        "score": 0.85
    },
    {
        "id": "doc3",
        "content": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains. The neural network itself is not an algorithm, but rather a framework for many different machine learning algorithms to work together.",
        "score": 0.78
    },
    {
        "id": "doc4",
        "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically-typed and garbage-collected.",
        "score": 0.65
    },
    {
        "id": "doc5",
        "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
        "score": 0.75
    },
    {
        "id": "doc6",
        "content": "Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
        "score": 0.72
    },
    {
        "id": "doc7",
        "content": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.",
        "score": 0.68
    }
]


def print_results(results: List[Dict[str, Any]], title: str = "Results") -> None:
    """
    Print formatted results.
    
    Args:
        results: List of result dictionaries
        title: Title to print above results
    """
    print(f"\n{title}")
    print("-" * 80)
    for i, result in enumerate(results):
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['id']}: {result['content'][:80]}...")


def print_comparison(original_results: List[Dict[str, Any]], 
                    reranked_results: List[Dict[str, Any]]) -> None:
    """
    Print a comparison of original and reranked results.
    
    Args:
        original_results: Original retrieval results
        reranked_results: Reranked results
    """
    print("\nComparison (Original vs. Reranked)")
    print("-" * 80)
    
    # Create dictionaries mapping document IDs to their positions
    original_positions = {result['id']: i for i, result in enumerate(original_results)}
    reranked_positions = {result['id']: i for i, result in enumerate(reranked_results)}
    
    # Print side-by-side comparison
    print(f"{'#':>3} | {'Original ID':>10} | {'Original Score':>13} | {'Reranked ID':>10} | {'Reranked Score':>13}")
    print("-" * 70)
    
    for i in range(min(len(original_results), len(reranked_results))):
        orig_id = original_results[i]['id']
        orig_score = original_results[i]['score']
        
        reranked_id = reranked_results[i]['id']
        reranked_score = reranked_results[i]['score']
        
        # Determine if position changed
        position_change = ""
        if orig_id != reranked_id:
            orig_pos_in_reranked = reranked_positions.get(orig_id, -1)
            if orig_pos_in_reranked > i:
                position_change = "↓"  # Moved down
            elif orig_pos_in_reranked >= 0:
                position_change = "↑"  # Moved up
        
        print(f"{i+1:>3} | {orig_id:>10} | {orig_score:>13.4f} | {reranked_id:>10} {position_change} | {reranked_score:>13.4f}")


def time_reranking(reranker, query: str, results: List[Dict[str, Any]], n_runs: int = 3) -> float:
    """
    Time the reranking process over multiple runs.
    
    Args:
        reranker: Reranker instance
        query: Query string
        results: List of results to rerank
        n_runs: Number of runs to average over
    
    Returns:
        Average time in seconds
    """
    times = []
    
    for _ in range(n_runs):
        # Create a deep copy of results to avoid any side effects
        results_copy = [{k: v for k, v in result.items()} for result in results]
        
        start = time.time()
        reranker.rerank(query, results_copy)
        end = time.time()
        
        times.append(end - start)
    
    return sum(times) / len(times)


def main():
    """
    Main function demonstrating the CrossEncoderReranker.
    """
    print("\n" + "="*80)
    print("Cross-Encoder Reranker Example")
    print("="*80)
    
    # Example 1: Basic usage with default model
    print("\n1. Basic usage with default model")
    print("-" * 60)
    
    # Create a reranker with the default model
    reranker = get_reranker(
        reranker_type="cross_encoder"
    )
    
    # Sample query
    query = "How do neural networks work?"
    
    # Print original results
    print_results(SAMPLE_RESULTS, "Original Results")
    
    # Rerank the results
    reranked_results = reranker.rerank(query, SAMPLE_RESULTS.copy())
    
    # Print reranked results
    print_results(reranked_results, "Reranked Results")
    
    # Compare original and reranked results
    print_comparison(SAMPLE_RESULTS, reranked_results)
    
    # Example 2: Configuring the reranker
    print("\n\n2. Configuring the reranker")
    print("-" * 60)
    
    # Create a reranker with custom configuration
    custom_reranker = get_reranker(
        reranker_type="cross_encoder",
        config={
            'batch_size': 16,
            'scale_scores': True,
            'normalize_scores': True,  # Apply softmax normalization
            'max_length': 256
        }
    )
    
    # Rerank with custom configuration
    custom_reranked_results = custom_reranker.rerank(query, SAMPLE_RESULTS.copy())
    
    # Print results with custom configuration
    print_results(custom_reranked_results, "Results with Custom Configuration")
    
    # Example 3: Getting reranking explanations
    print("\n\n3. Getting reranking explanations")
    print("-" * 60)
    
    # Get explanation
    explanation = reranker.explain_reranking(query, SAMPLE_RESULTS.copy())
    
    # Print basic explanation information
    print(f"Query: {explanation['query']}")
    print(f"Reranking method: {explanation['reranking_method']}")
    print(f"Model: {explanation['model_name']}")
    
    # Print rank changes
    print("\nRank changes:")
    for change in explanation['detailed_changes']:
        original_rank = change['original_rank'] + 1  # Convert to 1-indexed for display
        new_rank = change['new_rank'] + 1            # Convert to 1-indexed for display
        
        if change['rank_change'] > 0:
            direction = f"↑ (improved by {change['rank_change']} positions)"
        elif change['rank_change'] < 0:
            direction = f"↓ (worsened by {abs(change['rank_change'])} positions)"
        else:
            direction = "= (unchanged)"
        
        print(f"Document {change['id']}: {original_rank} → {new_rank} {direction}")
        print(f"  Score: {change['original_score']:.4f} → {change['new_score']:.4f} (change: {change['score_change']:.4f})")
    
    # Example 4: Performance comparison
    print("\n\n4. Performance comparison")
    print("-" * 60)
    
    try:
        # Try to create rerankers with different models (might fail if models aren't available)
        models = [
            ("Default (ms-marco-MiniLM-L-6-v2)", None),  # Default model
            ("TinyBERT", "cross-encoder/ms-marco-TinyBERT-L-2-v2"),  # Smaller, faster model
        ]
        
        print(f"Timing reranking for {len(SAMPLE_RESULTS)} documents:")
        
        for model_name, model_path in models:
            try:
                model_reranker = get_reranker(
                    reranker_type="cross_encoder",
                    model_name_or_path=model_path
                )
                
                avg_time = time_reranking(model_reranker, query, SAMPLE_RESULTS.copy())
                print(f"{model_name}: {avg_time:.4f} seconds")
                
            except Exception as e:
                print(f"{model_name}: Error - {str(e)}")
                
    except Exception as e:
        print(f"Performance comparison failed: {str(e)}")
        print("This example requires the sentence-transformers package and internet access to download models.")
    
    # Example 5: Updating configuration after initialization
    print("\n\n5. Updating configuration after initialization")
    print("-" * 60)
    
    # Update configuration
    reranker.set_config({
        'content_field': 'content',  # Field to use for document content
        'normalize_scores': True,    # Apply softmax normalization
        'batch_size': 8              # Smaller batch size
    })
    
    # Rerank with updated configuration
    updated_reranked_results = reranker.rerank(query, SAMPLE_RESULTS.copy())
    
    # Print results with updated configuration
    print_results(updated_reranked_results, "Results with Updated Configuration")
    
    print("\n" + "="*80)
    print("End of Cross-Encoder Reranker Example")
    print("="*80 + "\n")


if __name__ == "__main__":
    main() 