#!/usr/bin/env python3
"""
Rerankers Comparison Example

This script demonstrates how to use and compare different rerankers:
1. CrossEncoderReranker
2. MonoT5Reranker
3. LLMReranker (with a mock provider for demonstration)
4. EnsembleReranker (combining the above rerankers)

It retrieves a set of documents and reranks them using each reranker, then
compares the results and performance.
"""

import sys
import os
import time
import pprint
from typing import List, Dict, Any, Callable

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Conditionally import required libraries
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Some rerankers will be skipped.")

from tools.src.retrieval import get_reranker

# Sample documents for demonstration
DOCUMENTS = [
    {
        "id": "doc1",
        "content": "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."
    },
    {
        "id": "doc2",
        "content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected."
    },
    {
        "id": "doc3",
        "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised."
    },
    {
        "id": "doc4",
        "content": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains. They consist of artificial neurons connected to form layers that can learn patterns from data."
    },
    {
        "id": "doc5",
        "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."
    },
    {
        "id": "doc6",
        "content": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward."
    },
    {
        "id": "doc7",
        "content": "JavaScript is a high-level, interpreted programming language that conforms to the ECMAScript specification. It is multi-paradigm, supporting event-driven, functional, and imperative programming styles."
    },
    {
        "id": "doc8",
        "content": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. It seeks to automate tasks that the human visual system can do."
    }
]


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
        print(f"{i+1}. [Score: {result['score']:.4f}] {result['id']}")
        print(f"   {result['content'][:100]}...")
        
        # Print original score if available
        if "original_score" in result:
            print(f"   Original score: {result['original_score']:.4f}")
        
        # Print source contributions if available
        if "sources" in result:
            print("   Sources:")
            for source in result["sources"]:
                print(f"     - {source['name']}: {source['score']:.4f} (weight: {source['weight']:.2f})")
        
        print()


def create_mock_llm_provider() -> Callable:
    """
    Create a mock LLM provider for demonstration purposes.
    This avoids needing actual API keys for the example.
    
    Returns:
        A callable that mimics an LLM provider
    """
    # Hard-coded responses based on keywords in the query and document
    keyword_scores = {
        "machine learning": 9.5,
        "deep learning": 9.0,
        "neural network": 8.5,
        "python": 7.0,
        "javascript": 3.0,
        "programming": 5.0,
        "natural language": 8.0,
        "computer vision": 7.5,
        "reinforcement learning": 8.0
    }
    
    def mock_llm_provider(prompt: str, **kwargs) -> str:
        """
        Mock LLM provider that returns a relevance score based on keyword matching.
        
        Args:
            prompt: The prompt string containing query and document
            **kwargs: Additional arguments (ignored in this mock)
            
        Returns:
            A string representing a relevance score (0-10)
        """
        # Extract query and document from the prompt
        query_start = prompt.find("Query:") + 7
        query_end = prompt.find("Document:")
        document_start = prompt.find("Document:") + 10
        
        if query_start > 6 and query_end > 0 and document_start > 9:
            query = prompt[query_start:query_end].strip().lower()
            document = prompt[document_start:].strip().lower()
            
            # Calculate a score based on keyword presence
            score = 5.0  # Default middle score
            
            # Increase score for relevant keywords in both query and document
            for keyword, keyword_score in keyword_scores.items():
                if keyword in query and keyword in document:
                    score = max(score, keyword_score)
                elif keyword in query:  # Keyword in query but not in document
                    # Slightly reduce the score if keyword is important but missing
                    if keyword_score > 7:
                        score = min(score, 4.0)
            
            # Add some variability
            import random
            score += random.uniform(-0.5, 0.5)
            score = max(0, min(10, score))  # Ensure it's between 0 and 10
            
            # Return the score as a string
            return f"{score:.1f}"
        
        return "5.0"  # Default score if parsing fails
    
    return mock_llm_provider


def run_cross_encoder_example(query: str, documents: List[Dict[str, Any]]) -> None:
    """
    Run example using the CrossEncoderReranker.
    
    Args:
        query: The search query
        documents: List of documents to rerank
    """
    if not TRANSFORMERS_AVAILABLE:
        print("\nSkipping CrossEncoderReranker example (transformers not available)")
        return
    
    print("\n" + "="*80)
    print("CrossEncoderReranker Example")
    print("="*80)
    
    try:
        # Create a CrossEncoderReranker
        reranker = get_reranker(
            reranker_type="cross_encoder",
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        # Measure reranking time
        start_time = time.time()
        results = reranker.rerank(query, documents.copy())
        end_time = time.time()
        
        # Print results and timing
        print(f"Reranking with CrossEncoderReranker took {end_time - start_time:.4f} seconds")
        print_results(results, "CrossEncoderReranker Results")
        
    except Exception as e:
        print(f"Error in CrossEncoderReranker example: {e}")


def run_mono_t5_example(query: str, documents: List[Dict[str, Any]]) -> None:
    """
    Run example using the MonoT5Reranker.
    
    Args:
        query: The search query
        documents: List of documents to rerank
    """
    if not TRANSFORMERS_AVAILABLE:
        print("\nSkipping MonoT5Reranker example (transformers not available)")
        return
    
    print("\n" + "="*80)
    print("MonoT5Reranker Example")
    print("="*80)
    
    try:
        # Create a MonoT5Reranker
        reranker = get_reranker(
            reranker_type="mono_t5",
            model_name="castorini/monot5-base-msmarco"
        )
        
        # Measure reranking time
        start_time = time.time()
        results = reranker.rerank(query, documents.copy())
        end_time = time.time()
        
        # Print results and timing
        print(f"Reranking with MonoT5Reranker took {end_time - start_time:.4f} seconds")
        print_results(results, "MonoT5Reranker Results")
        
    except Exception as e:
        print(f"Error in MonoT5Reranker example: {e}")


def run_llm_reranker_example(query: str, documents: List[Dict[str, Any]]) -> None:
    """
    Run example using the LLMReranker with a mock provider.
    
    Args:
        query: The search query
        documents: List of documents to rerank
    """
    print("\n" + "="*80)
    print("LLMReranker Example (using mock provider)")
    print("="*80)
    
    try:
        # Create a mock LLM provider
        mock_provider = create_mock_llm_provider()
        
        # Create an LLMReranker with the mock provider
        reranker = get_reranker(
            reranker_type="llm",
            llm_provider=mock_provider,
            scoring_method="scale_1_10",
            batch_size=4
        )
        
        # Measure reranking time
        start_time = time.time()
        results = reranker.rerank(query, documents.copy())
        end_time = time.time()
        
        # Print results and timing
        print(f"Reranking with LLMReranker took {end_time - start_time:.4f} seconds")
        print_results(results, "LLMReranker Results")
        
    except Exception as e:
        print(f"Error in LLMReranker example: {e}")


def run_ensemble_reranker_example(query: str, documents: List[Dict[str, Any]]) -> None:
    """
    Run example using the EnsembleReranker combining multiple rerankers.
    
    Args:
        query: The search query
        documents: List of documents to rerank
    """
    if not TRANSFORMERS_AVAILABLE:
        print("\nSkipping EnsembleReranker example (transformers not available)")
        return
    
    print("\n" + "="*80)
    print("EnsembleReranker Example")
    print("="*80)
    
    try:
        # Create individual rerankers
        cross_encoder = get_reranker(
            reranker_type="cross_encoder",
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        mono_t5 = get_reranker(
            reranker_type="mono_t5",
            model_name="castorini/monot5-base-msmarco"
        )
        
        mock_provider = create_mock_llm_provider()
        llm_reranker = get_reranker(
            reranker_type="llm",
            llm_provider=mock_provider,
            scoring_method="scale_1_10"
        )
        
        # Create an ensemble combining all three rerankers
        ensemble = get_reranker(
            reranker_type="ensemble",
            rerankers=[cross_encoder, mono_t5, llm_reranker],
            weights={
                "reranker_0": 0.5,  # Cross-encoder
                "reranker_1": 0.3,  # MonoT5
                "reranker_2": 0.2   # LLM
            },
            combination_method="weighted_average"
        )
        
        # Print ensemble information
        print("Ensemble Components:")
        for info in ensemble.get_reranker_info():
            print(f"- {info['name']} ({info['type']}): weight = {info['weight']}")
        
        # Measure reranking time
        start_time = time.time()
        results = ensemble.rerank(query, documents.copy())
        end_time = time.time()
        
        # Print results and timing
        print(f"Reranking with EnsembleReranker took {end_time - start_time:.4f} seconds")
        print_results(results, "EnsembleReranker Results (Weighted Average)")
        
        # Try with different combination methods
        print("\nTrying different combination methods:")
        
        # Max score method
        ensemble.set_combination_method("max_score")
        results = ensemble.rerank(query, documents.copy())
        print_results(results, "EnsembleReranker Results (Max Score)")
        
        # Reciprocal rank fusion method
        ensemble.set_combination_method("reciprocal_rank_fusion")
        results = ensemble.rerank(query, documents.copy())
        print_results(results, "EnsembleReranker Results (Reciprocal Rank Fusion)")
        
    except Exception as e:
        print(f"Error in EnsembleReranker example: {e}")


def main():
    """
    Main function running the reranker examples.
    """
    print("\nRerankers Comparison Example")
    print("===========================")
    print("This example demonstrates different rerankers and compares their results.")
    
    # Define the query
    query = "How do neural networks work in machine learning?"
    print(f"\nQuery: '{query}'")
    
    # Print the original documents (unsorted)
    print_results(DOCUMENTS, "Original Documents (Unsorted)")
    
    # Run examples for each reranker
    run_cross_encoder_example(query, DOCUMENTS)
    run_mono_t5_example(query, DOCUMENTS)
    run_llm_reranker_example(query, DOCUMENTS)
    run_ensemble_reranker_example(query, DOCUMENTS)
    
    print("\nRerankers comparison complete!")


if __name__ == "__main__":
    main() 