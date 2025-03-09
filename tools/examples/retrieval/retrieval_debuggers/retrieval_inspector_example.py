#!/usr/bin/env python3
"""
Retrieval Inspector Example

This script demonstrates how to use the RetrievalInspector to analyze
and debug retrieval results.
"""

import sys
import os
import json
from pprint import pprint

# Add the 'tools' directory to the Python path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from tools.src.retrieval import get_retrieval_debugger

def main():
    """Demonstrate the RetrievalInspector with sample retrieval results."""
    
    print("🔍 Retrieval Inspector Example\n")
    
    # Create sample query and retrieval results
    query = "What are the best practices for implementing RAG with vector databases?"
    
    # Sample retrieval results (normally these would come from your retrieval system)
    results = [
        {
            "id": "doc_1",
            "score": 0.89,
            "content": """
            Best practices for RAG with vector databases include:
            1. Proper document chunking and preprocessing
            2. Using high-quality embedding models
            3. Optimizing vector search parameters
            4. Implementing efficient metadata filtering
            5. Regularly evaluating and fine-tuning retrieval performance
            """,
            "metadata": {
                "source": "vector_db_guide.pdf",
                "page": 24,
                "category": "best_practices",
                "date": "2023-09-15"
            }
        },
        {
            "id": "doc_2",
            "score": 0.76,
            "content": """
            When using vector databases for RAG applications, ensure you:
            - Choose the appropriate index type (HNSW, IVF, etc.)
            - Configure the index parameters based on your data size and query patterns
            - Monitor query latency and throughput
            - Scale your vector database as your data grows
            """,
            "metadata": {
                "source": "vector_db_guide.pdf",
                "page": 25,
                "category": "best_practices",
                "date": "2023-09-15"
            }
        },
        {
            "id": "doc_3",
            "score": 0.72,
            "content": """
            Vector databases store vector embeddings and enable efficient similarity search.
            Popular vector databases include:
            - Qdrant
            - Chroma
            - Pinecone
            - Weaviate
            - Milvus
            - pgvector (PostgreSQL extension)
            """,
            "metadata": {
                "source": "vector_db_comparison.pdf",
                "page": 5,
                "category": "introduction",
                "date": "2023-08-10"
            }
        },
        {
            "id": "doc_4",
            "score": 0.68,
            "content": """
            RAG (Retrieval-Augmented Generation) combines the power of large language models
            with information retrieval systems. It allows language models to access external
            knowledge, reducing hallucinations and improving factual accuracy.
            """,
            "metadata": {
                "source": "rag_fundamentals.pdf",
                "page": 3,
                "category": "introduction",
                "date": "2023-07-20"
            }
        },
        {
            "id": "doc_5",
            "score": 0.65,
            "content": """
            Best practices for implementing RAG with vector databases include:
            1. Choose appropriate chunking strategies
            2. Use high-quality embedding models
            3. Optimize vector database parameters
            4. Implement effective reranking strategies
            """,
            "metadata": {
                "source": "rag_tutorial.pdf",
                "page": 12,
                "category": "best_practices",
                "date": "2023-10-05"
            }
        }
    ]
    
    # Create a RetrievalInspector
    inspector = get_retrieval_debugger(
        debugger_type="inspector",
        similarity_threshold=0.7,  # Threshold for considering documents similar
        relevance_threshold=0.5,   # Threshold for considering a document relevant
        analyze_content=True,
        analyze_metadata=True,
        similarity_method="tfidf",  # Options: "tfidf", "jaccard", or "custom"
        include_charts=True
    )
    
    # Example 1: Basic Analysis
    print("Example 1: Basic Analysis of Retrieval Results")
    
    # Analyze the retrieval results
    analysis = inspector.analyze(query, results)
    
    # Print selected analysis results
    print(f"\nQuery: {query}")
    print(f"Number of results: {analysis['result_count']}")
    print(f"Average query-result similarity: {analysis['metrics']['average_similarity']:.3f}")
    print(f"Result diversity score: {analysis['diversity']['diversity_score']:.3f}")
    
    print("\nInsights:")
    for i, insight in enumerate(analysis['insights'], 1):
        print(f"{i}. {insight}")
    
    # Example 2: Compare Different Retrieval Results
    print("\n\nExample 2: Comparing Different Retrieval Results")
    
    # Create a second set of results (simulating a different retrieval method)
    results2 = [
        {
            "id": "doc_6",
            "score": 0.92,
            "content": """
            Expert tips for RAG with vector databases:
            1. Use semantic chunking instead of fixed-size chunking
            2. Apply embedding models specifically fine-tuned for your domain
            3. Implement hybrid search combining vector, keyword, and metadata filters
            4. Use multiple queries for complex questions
            5. Apply reranking to improve result quality
            """,
            "metadata": {
                "source": "rag_best_practices.pdf",
                "page": 8,
                "category": "best_practices",
                "date": "2023-11-12"
            }
        },
        {
            "id": "doc_7",
            "score": 0.88,
            "content": """
            When optimizing vector databases for RAG:
            - Tune the index for your specific use case (recall vs. latency)
            - Consider using multiple vector representations for different aspects
            - Implement proper caching strategies
            - Use filters to narrow down the search space before vector search
            """,
            "metadata": {
                "source": "vector_search_optimization.pdf",
                "page": 15,
                "category": "optimization",
                "date": "2023-10-25"
            }
        },
        {
            "id": "doc_1",  # Duplicate from first set
            "score": 0.85,
            "content": """
            Best practices for RAG with vector databases include:
            1. Proper document chunking and preprocessing
            2. Using high-quality embedding models
            3. Optimizing vector search parameters
            4. Implementing efficient metadata filtering
            5. Regularly evaluating and fine-tuning retrieval performance
            """,
            "metadata": {
                "source": "vector_db_guide.pdf",
                "page": 24,
                "category": "best_practices",
                "date": "2023-09-15"
            }
        }
    ]
    
    # Compare the two result sets
    comparison = inspector.compare(
        query,
        [results, results2],
        names=["Baseline System", "Experimental System"]
    )
    
    # Print comparison results
    print("\nComparison Metrics:")
    for system in comparison['comparison']:
        print(f"\n{system['name']}:")
        print(f"  - Result Count: {system['result_count']}")
        print(f"  - Average Similarity: {system['average_similarity']:.3f}")
        print(f"  - Diversity Score: {system['diversity_score']:.3f}")
    
    print(f"\nBest Performing System: {comparison['best_performing']}")
    
    print("\nComparative Insights:")
    for i, insight in enumerate(comparison['insights'], 1):
        print(f"{i}. {insight}")
    
    # Example 3: Evaluation Against Ground Truth
    print("\n\nExample 3: Evaluation Against Ground Truth")
    
    # Define ground truth relevant document IDs
    ground_truth = ["doc_1", "doc_6", "doc_7", "doc_9"]
    
    # Evaluate retrieval results against ground truth
    evaluation = inspector.evaluate(query, results, ground_truth)
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    for metric, value in evaluation.items():
        print(f"  - {metric}: {value:.3f}")
    
    # Example 4: Custom Analysis
    print("\n\nExample 4: Custom Analysis")
    
    # Define a custom similarity function (optional)
    def custom_similarity(text1, text2):
        # This is a very simplistic example; in practice, you would use a more sophisticated method
        common_words = set(text1.lower().split()) & set(text2.lower().split())
        return len(common_words) / (len(set(text1.lower().split())) + len(set(text2.lower().split())) + 0.001)
    
    # Create a custom inspector
    custom_inspector = get_retrieval_debugger(
        debugger_type="inspector",
        similarity_method="custom",
        custom_similarity_fn=custom_similarity,
        min_token_overlap=2,
        max_insights=3
    )
    
    # Get focused insights
    insights = custom_inspector.get_insights(query, results)
    
    print("\nTop Insights from Custom Analysis:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print("\n🔮 Retrieval inspection complete!")


if __name__ == "__main__":
    main() 