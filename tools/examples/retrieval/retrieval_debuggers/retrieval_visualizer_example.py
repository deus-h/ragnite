#!/usr/bin/env python3
"""
Retrieval Visualizer Example

This script demonstrates how to use the RetrievalVisualizer to create
visualizations of retrieval results.
"""

import sys
import os
import json
from pprint import pprint
import matplotlib.pyplot as plt

# Add the 'tools' directory to the Python path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from tools.src.retrieval import get_retrieval_debugger

def main():
    """Demonstrate the RetrievalVisualizer with sample retrieval results."""
    
    print("📊 Retrieval Visualizer Example\n")
    
    # Create sample query and retrieval results
    query = "What are the benefits of using vector databases for RAG applications?"
    
    # Sample retrieval results (normally these would come from your retrieval system)
    results_system1 = [
        {
            "id": "doc_1",
            "score": 0.92,
            "content": """
            Vector databases provide several key benefits for Retrieval Augmented Generation (RAG) applications:
            
            1. Semantic search capabilities that understand the meaning behind queries rather than just keywords
            
            2. High-performance similarity search, enabling fast retrieval of relevant context even with millions of documents
            
            3. Efficient storage and indexing of high-dimensional embeddings with specialized index structures like HNSW or IVF
            
            4. Support for hybrid search combining vector similarity with metadata filtering
            
            5. Scalability to handle large document collections with minimal latency degradation
            """,
            "metadata": {
                "source": "vector_db_guide.pdf",
                "page": 15,
                "category": "technical",
                "recency": 0.9,
                "word_count": 112
            }
        },
        {
            "id": "doc_2",
            "score": 0.87,
            "content": """
            When implementing RAG applications, vector databases offer substantial advantages:
            
            - They enable semantic search that captures the meaning of text rather than just keywords
            - They provide millisecond query times even when searching through millions of documents
            - They support efficient updating and maintenance of the knowledge base
            - Their specialized indexing algorithms like HNSW provide an optimal balance of recall and performance
            - They often include features like metadata filtering to narrow search scope and improve precision
            """,
            "metadata": {
                "source": "rag_implementation_guide.pdf",
                "page": 28,
                "category": "technical",
                "recency": 0.85,
                "word_count": 98
            }
        },
        {
            "id": "doc_3",
            "score": 0.81,
            "content": """
            Vector databases excel at powering RAG applications by:
            
            1. Enabling retrieval based on semantic similarity rather than lexical matching
            2. Supporting efficient storage and query of high-dimensional embeddings
            3. Providing specialized indexing algorithms to balance speed and recall
            4. Offering APIs and integrations with popular ML frameworks and LLM platforms
            5. Supporting multi-modal vector embeddings (text, images, audio) in many cases
            """,
            "metadata": {
                "source": "vector_db_comparison.pdf",
                "page": 7,
                "category": "technical",
                "recency": 0.95,
                "word_count": 85
            }
        },
        {
            "id": "doc_4",
            "score": 0.76,
            "content": """
            Performance benchmarks of vector databases for RAG applications:
            
            | Database   | Query Time (ms) | Indexing Time | Memory Usage | Precision@10 |
            |------------|-----------------|---------------|--------------|--------------|
            | Chroma     | 15              | Fast          | Medium       | 0.87         |
            | Qdrant     | 12              | Medium        | Low          | 0.92         |
            | Pinecone   | 8               | N/A (cloud)   | N/A (cloud)  | 0.89         |
            | Weaviate   | 20              | Medium        | Medium       | 0.85         |
            | Milvus     | 5               | Slow          | High         | 0.94         |
            | pgvector   | 40              | Fast          | Low          | 0.82         |
            
            These benchmarks demonstrate the performance advantages of specialized vector databases for RAG retrieval tasks.
            """,
            "metadata": {
                "source": "vector_db_benchmarks.pdf",
                "page": 42,
                "category": "benchmark",
                "recency": 0.7,
                "word_count": 132
            }
        },
        {
            "id": "doc_5",
            "score": 0.68,
            "content": """
            Vector databases are specialized database systems designed to store and search vector embeddings efficiently. Unlike traditional databases that excel at exact matching, vector databases are optimized for similarity search in high-dimensional spaces, making them ideal for machine learning applications including RAG systems.
            
            The core capability of these databases is approximate nearest neighbor (ANN) search, which finds the most similar vectors to a query vector without exhaustively comparing against every vector in the database. This is achieved through specialized indexing structures like HNSW (Hierarchical Navigable Small World), IVF (Inverted File Index), or PQ (Product Quantization).
            """,
            "metadata": {
                "source": "vector_db_fundamentals.pdf",
                "page": 3,
                "category": "introduction",
                "recency": 0.8,
                "word_count": 115
            }
        }
    ]
    
    # Second set of results from a different system for comparison
    results_system2 = [
        {
            "id": "doc_6",
            "score": 0.95,
            "content": """
            RAG applications benefit from vector databases in several ways:
            
            1. Superior retrieval quality through semantic search: Vector databases understand the meaning behind queries, not just keywords.
            
            2. Fast similarity search: Even with millions of documents, vector databases can retrieve relevant information in milliseconds.
            
            3. Efficient embedding storage: Specialized data structures enable compact storage and fast querying of high-dimensional vectors.
            
            4. Hybrid search capabilities: Combine vector similarity with traditional filtering for more precise results.
            
            5. Scalability: Designed to handle growing document collections with minimal performance degradation.
            """,
            "metadata": {
                "source": "advanced_rag_guide.pdf",
                "page": 22,
                "category": "technical",
                "recency": 0.98,
                "word_count": 120
            }
        },
        {
            "id": "doc_7",
            "score": 0.89,
            "content": """
            Cost-benefit analysis of vector databases for RAG:
            
            Benefits:
            - Improved retrieval quality through semantic understanding
            - Faster query times with specialized indexing
            - Better user experience with more relevant responses
            - Reduced LLM token usage by providing more relevant context
            
            Costs:
            - Additional infrastructure for vector database deployment
            - Embedding generation computational overhead
            - Integration complexity with existing systems
            - Potential vendor lock-in with cloud-based solutions
            
            Despite the costs, most organizations find that the improved quality and performance of RAG systems using vector databases justifies the investment.
            """,
            "metadata": {
                "source": "rag_economics.pdf",
                "page": 15,
                "category": "business",
                "recency": 0.9,
                "word_count": 130
            }
        },
        {
            "id": "doc_8",
            "score": 0.82,
            "content": """
            A case study of implementing RAG with vector databases at a Fortune 500 company:
            
            The company integrated a vector database into their customer support system to enhance their RAG application. Key findings:
            
            - 78% improvement in answer relevance
            - 45% reduction in response time
            - 23% decrease in escalations to human agents
            - 92% user satisfaction (up from 67%)
            
            Technical implementation:
            - Used HNSW indexing for optimal performance
            - Combined vector search with metadata filtering
            - Implemented caching for frequent queries
            - Deployed monitoring for ongoing optimization
            
            The vector database was a critical component that enabled the RAG system to succeed at enterprise scale.
            """,
            "metadata": {
                "source": "rag_case_studies.pdf",
                "page": 34,
                "category": "case_study",
                "recency": 0.95,
                "word_count": 145
            }
        }
    ]
    
    # Create a RetrievalVisualizer
    visualizer = get_retrieval_debugger(
        debugger_type="visualizer",
        content_field="content",
        score_field="score",
        metadata_fields=["category", "recency", "word_count"],
        interactive=False,  # Set to True if you have Plotly installed
        plot_style="whitegrid",
        default_figsize=(10, 6),
        color_palette="viridis",
        embedding_dim_reduction="tsne",
        save_plots=False  # Set to True and provide output_dir to save plots
    )
    
    # Example 1: Basic Score Distribution Visualization
    print("Example 1: Basic Score Distribution Visualization")
    score_dist_fig = visualizer.plot_score_distribution(results_system1)
    plt.show()
    
    # Example 2: Rank vs Score Visualization
    print("\nExample 2: Rank vs Score Visualization")
    rank_score_fig = visualizer.plot_rank_vs_score(results_system1)
    plt.show()
    
    # Example 3: Content Similarity Matrix
    print("\nExample 3: Content Similarity Matrix")
    sim_matrix_fig = visualizer.plot_content_similarity_matrix(results_system1)
    plt.show()
    
    # Example 4: Content Similarity Map
    print("\nExample 4: Content Similarity Map")
    sim_map_fig = visualizer.plot_content_similarity_map(query, results_system1)
    plt.show()
    
    # Example 5: Metadata Distribution
    print("\nExample 5: Metadata Distribution")
    metadata_dist_fig = visualizer.plot_metadata_distribution(results_system1, "category")
    plt.show()
    
    # Example 6: Metadata Correlation
    print("\nExample 6: Metadata Correlation")
    metadata_corr_fig = visualizer.plot_metadata_correlation(results_system1, "recency")
    plt.show()
    
    # Example 7: Compare Multiple Retrieval Systems
    print("\nExample 7: Compare Multiple Retrieval Systems")
    comparison_fig = visualizer.plot_comparison_scores(
        [results_system1, results_system2],
        names=["System 1", "System 2"]
    )
    plt.show()
    
    # Example 8: Top-K Comparison
    print("\nExample 8: Top-K Comparison")
    top_k_fig = visualizer.plot_top_k_comparison(
        query,
        [results_system1, results_system2],
        names=["System 1", "System 2"],
        k=3
    )
    plt.show()
    
    # Example 9: Using the BaseRetrievalDebugger Interface
    print("\nExample 9: Using the BaseRetrievalDebugger Interface")
    
    # Analyze a single result set
    analysis = visualizer.analyze(query, results_system1)
    print("Analysis Results:")
    print(f"Number of results: {analysis['result_count']}")
    print(f"Score statistics: {json.dumps(analysis['score_stats'], indent=2)}")
    
    # Compare multiple result sets
    comparison = visualizer.compare(
        query, 
        [results_system1, results_system2],
        names=["System 1", "System 2"]
    )
    print("\nComparison Results:")
    print(f"Best system: {comparison['best_system']}")
    
    # Get insights
    insights = visualizer.get_insights(query, results_system1)
    print("\nRetrievalVisualizer Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print("\n📊 Visualization example complete!")

if __name__ == "__main__":
    main() 