#!/usr/bin/env python3
"""
Context Analyzer Example

This script demonstrates how to use the ContextAnalyzer to analyze
retrieved context quality and characteristics.
"""

import sys
import os
import json
from pprint import pprint

# Add the 'tools' directory to the Python path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from tools.src.retrieval import get_retrieval_debugger

def main():
    """Demonstrate the ContextAnalyzer with sample retrieval results."""
    
    print("🔍 Context Analyzer Example\n")
    
    # Create sample query and retrieval results
    query = "What are the best practices for implementing RAG with vector databases?"
    
    # Sample retrieval results (normally these would come from your retrieval system)
    results = [
        {
            "id": "doc_1",
            "score": 0.89,
            "content": """
            Best practices for implementing RAG with vector databases include:
            
            1. Proper document chunking: Break documents into semantically meaningful chunks rather than arbitrary splits.
            
            2. High-quality embeddings: Use domain-specific embedding models when possible to better capture semantic relationships.
            
            3. Vector database optimization: Configure index parameters (like HNSW or IVF) based on your specific retrieval needs.
            
            4. Metadata filtering: Store and utilize metadata to enable more precise filtering during retrieval.
            
            5. Regular performance evaluation: Continuously assess retrieval quality using metrics like precision, recall, and relevance.
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
            For optimal RAG performance with vector databases, consider the following technical optimizations:
            
            - Select the appropriate index type based on your data size and query patterns
            - For smaller datasets (<1M vectors), HNSW often provides better recall
            - For larger datasets, IVF with proper clustering can be more efficient
            - Monitor query latency and adjust parameters accordingly
            - Use metadata filtering to narrow search space before vector similarity search
            - Configure distance metrics based on your embedding model (cosine vs euclidean)
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
            Common pitfalls to avoid when implementing RAG with vector databases:
            
            1. Neglecting proper chunking strategies
            2. Using generic embeddings for domain-specific tasks
            3. Ignoring the importance of metadata
            4. Overlooking performance optimization
            5. Failing to implement caching for frequent queries
            6. Not handling edge cases (like empty results)
            7. Insufficient monitoring and evaluation
            """,
            "metadata": {
                "source": "rag_implementation_guide.pdf",
                "page": 42,
                "category": "best_practices",
                "date": "2023-10-05"
            }
        },
        {
            "id": "doc_4",
            "score": 0.68,
            "content": """
            Vector databases are specialized database systems designed to store and efficiently search vector embeddings. These embeddings are high-dimensional numerical representations of data objects, created by machine learning models. Popular vector databases include Chroma, Qdrant, Pinecone, Weaviate, Milvus, and pgvector (PostgreSQL extension).

            Their key capabilities include fast similarity search using algorithms like HNSW and IVF, metadata filtering, hybrid search combining vector and keyword matching, and clustering for better organization of vector data. 
            """,
            "metadata": {
                "source": "vector_db_comparison.pdf",
                "page": 5,
                "category": "introduction",
                "date": "2023-08-10"
            }
        },
        {
            "id": "doc_5",
            "score": 0.65,
            "content": """
            The RAG (Retrieval-Augmented Generation) approach enhances large language models by integrating an information retrieval component that supplies relevant external knowledge. This significantly reduces hallucinations and improves factual accuracy by grounding the model's outputs in retrieved information.

            The RAG architecture consists of three main components:
            1. The retriever, which searches for relevant information from a knowledge base
            2. The contextual augmentation mechanism, which integrates retrieved information
            3. The generator, which produces the final output using the augmented context
            """,
            "metadata": {
                "source": "rag_fundamentals.pdf",
                "page": 3,
                "category": "introduction",
                "date": "2023-07-20"
            }
        }
    ]
    
    # Create a ContextAnalyzer
    analyzer = get_retrieval_debugger(
        debugger_type="context_analyzer",
        content_field="content",
        analyze_relevance=True,
        analyze_diversity=True,
        analyze_information=True,
        analyze_readability=True,
        similarity_threshold=0.7,  # Threshold for considering documents similar
        relevance_threshold=0.5,   # Threshold for considering content relevant
        similarity_method="tfidf",  # Options: "tfidf", "jaccard", or "custom"
        readability_metrics=True,
        sentiment_analysis=False    # Set to True if NLTK is installed
    )
    
    # Example 1: Basic Context Analysis
    print("Example 1: Basic Context Analysis")
    
    # Analyze the retrieved context
    analysis = analyzer.analyze(query, results)
    
    # Print summary of the analysis
    print(f"\nQuery: {query}")
    print(f"Number of results: {analysis['result_count']}")
    
    # Relevance analysis
    if "relevance" in analysis:
        relevance = analysis["relevance"]
        print(f"\nRelevance Analysis:")
        print(f"  - Average relevance: {relevance['average_relevance']:.3f}")
        print(f"  - Relevant documents: {relevance['relevant_count']}/{len(results)} ({relevance['relevant_percentage']:.1f}%)")
    
    # Diversity analysis
    if "diversity" in analysis:
        diversity = analysis["diversity"]
        print(f"\nDiversity Analysis:")
        print(f"  - Diversity score: {diversity['diversity_score']:.3f}")
        print(f"  - Similar content pairs: {diversity['duplicate_count']}")
        print(f"  - Unique information: {diversity['unique_information_percentage']:.1f}%")
    
    # Information density analysis
    if "information" in analysis:
        information = analysis["information"]
        print(f"\nInformation Analysis:")
        print(f"  - Average information density: {information['average_information_density']:.3f}")
        print(f"  - Average word count: {information['average_word_count']:.1f}")
        print(f"  - Average lexical diversity: {information['average_lexical_diversity']:.3f}")
        print(f"  - Total potential entities: {information['total_potential_entities']}")
    
    # Readability analysis
    if "readability" in analysis:
        readability = analysis["readability"]
        print(f"\nReadability Analysis:")
        print(f"  - Average reading ease: {readability['average_reading_ease']:.1f}")
        print(f"  - Average grade level: {readability['average_grade_level']:.1f}")
        print(f"  - Most common difficulty: {readability['most_common_difficulty']}")
    
    # Insights
    print("\nContext Insights:")
    for i, insight in enumerate(analysis.get("insights", []), 1):
        print(f"{i}. {insight}")
    
    # Example 2: Comparing Context from Different Retrieval Methods
    print("\n\nExample 2: Comparing Context from Different Retrieval Methods")
    
    # Create a second set of results (simulating a different retrieval method)
    results2 = [
        {
            "id": "doc_6",
            "score": 0.92,
            "content": """
            Advanced RAG implementations with vector databases typically include these key optimizations:
            
            1. Semantic chunking: Instead of fixed-size chunking, segment documents based on semantic boundaries.
            
            2. Embedding fine-tuning: Fine-tune embedding models on your specific domain for better representation.
            
            3. Hybrid search: Combine vector similarity with keyword and metadata filtering for more precise results.
            
            4. Multi-query approach: Generate multiple variations of the query to capture different aspects.
            
            5. Re-ranking: Apply a secondary ranking mechanism after initial retrieval to improve precision.
            """,
            "metadata": {
                "source": "advanced_rag_techniques.pdf",
                "page": 12,
                "category": "best_practices",
                "date": "2023-11-10"
            }
        },
        {
            "id": "doc_7",
            "score": 0.88,
            "content": """
            Vector index optimization is crucial for RAG performance. Key considerations include:
            
            - HNSW (Hierarchical Navigable Small World) indexes offer an excellent balance of speed and recall
            - Adjust M parameter (maximum connections per node) based on recall requirements
            - Increase ef_construction for better index quality at the cost of build time
            - For larger datasets, consider IVF (Inverted File Index) with appropriate cluster counts
            - Monitor query latency across different vector dimensions and dataset sizes
            - Balance index build time, search speed, and recall based on application needs
            """,
            "metadata": {
                "source": "vector_index_optimization.pdf",
                "page": 8,
                "category": "technical",
                "date": "2023-10-22"
            }
        },
        {
            "id": "doc_8",
            "score": 0.84,
            "content": """
            Effective chunking strategies significantly impact RAG quality with vector databases:
            
            - Semantic chunking preserves context better than fixed-size chunking
            - Overlap between chunks (typically 10-20%) reduces information loss
            - Consider document structure (headings, paragraphs, sections) when chunking
            - Different content types may require different chunking strategies
            - Experiment with chunk sizes to find the optimal balance for your specific use case
            - Store metadata with each chunk to maintain provenance and enable filtering
            """,
            "metadata": {
                "source": "document_chunking_guide.pdf",
                "page": 15,
                "category": "preprocessing",
                "date": "2023-09-05"
            }
        }
    ]
    
    # Compare the two sets of retrieval results
    comparison = analyzer.compare(
        query,
        [results, results2],
        names=["Default System", "Optimized System"]
    )
    
    # Print comparison results
    print("\nContext Quality Comparison:")
    for metrics in comparison['comparison']:
        print(f"\n{metrics['name']}:")
        if "average_relevance" in metrics:
            print(f"  - Relevance: {metrics['average_relevance']:.3f}")
        if "diversity_score" in metrics:
            print(f"  - Diversity: {metrics['diversity_score']:.3f}")
        if "information_density" in metrics:
            print(f"  - Information Density: {metrics['information_density']:.3f}")
        if "grade_level" in metrics:
            print(f"  - Grade Level: {metrics['grade_level']:.1f}")
    
    # Print best systems
    print("\nBest Systems by Category:")
    for category, system in comparison.get('best_systems', {}).items():
        print(f"  - {category.capitalize()}: {system}")
    
    if "best_overall" in comparison:
        print(f"\nBest Overall System: {comparison['best_overall']}")
    
    # Print insights
    print("\nComparative Insights:")
    for i, insight in enumerate(comparison.get('insights', []), 1):
        print(f"{i}. {insight}")
    
    # Example 3: Context Evaluation Against Ground Truth
    print("\n\nExample 3: Context Evaluation Against Ground Truth")
    
    # Define ground truth (relevant content or document IDs)
    ground_truth = [
        """
        Best practices for RAG with vector databases include proper document chunking, 
        using high-quality embeddings, optimizing vector database configuration, 
        implementing metadata filtering, and continuously evaluating performance.
        """,
        """
        Advanced RAG techniques include semantic chunking, embedding fine-tuning,
        hybrid search combining vectors and keywords, multi-query approaches,
        and re-ranking mechanisms.
        """,
        "doc_1",
        "doc_6"
    ]
    
    # Evaluate retrieved context against ground truth
    evaluation = analyzer.evaluate(query, results, ground_truth, content_overlap=True)
    
    # Print evaluation metrics
    print("\nContext Evaluation Metrics:")
    for metric, value in evaluation.items():
        print(f"  - {metric}: {value:.3f}")
    
    # Example 4: Getting Specific Insights
    print("\n\nExample 4: Getting Specific Insights")
    
    # Get actionable insights about the context
    insights = analyzer.get_insights(query, results)
    
    print("\nActionable Context Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print("\n🔮 Context analysis complete!")


if __name__ == "__main__":
    main() 