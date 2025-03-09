"""
Self-RAG Example

This script demonstrates how to use the Self-RAG (Retrieval-Augmented Generation with Self-Reflection) implementation.
It shows how to:
1. Initialize the Self-RAG pipeline
2. Decide whether to retrieve external information
3. Generate and critique responses
4. Revise responses based on feedback
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from self_rag_pipeline import SelfRAG


def run_self_rag_example():
    """
    Run an example of the Self-RAG pipeline.
    """
    # Path to the vector store
    vector_store_path = "../../../basic-rag/data/processed/vector_store"
    
    # Initialize the Self-RAG pipeline
    print("Initializing Self-RAG pipeline...")
    self_rag = SelfRAG(
        vector_store_path=vector_store_path,
        model_name="gpt-3.5-turbo",
        confidence_threshold=0.7
    )
    
    # Example queries
    example_queries = [
        "What is RAG and how does it work?",
        "Who is the current CEO of OpenAI?",  # Requires external knowledge retrieval
        "What are the advantages of HyDE over basic RAG?",
        "Can you explain how document chunking affects retrieval quality?",
        "How many planets are in the solar system?",  # Well-known fact, might not need retrieval
    ]
    
    # Run queries
    print("\n=== Running Example Queries with Self-RAG ===\n")
    for query in example_queries:
        print(f"\nQuery: {query}")
        
        # Run the full Self-RAG pipeline
        result = self_rag.query(query)
        
        # Print the results
        print("\nRetrieval Decision:")
        print(result["retrieval_decision"])
        
        if result["retrieved_documents"]:
            print("\nRetrieved Documents:")
            for i, doc in enumerate(result["retrieved_documents"]):
                print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):")
                print(doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content)
        else:
            print("\nNo documents retrieved. Using parametric knowledge.")
        
        print("\nInitial Response:")
        print(result["initial_response"])
        
        print("\nCritique:")
        print(result["critique"])
        
        print(f"\nConfidence Score: {result['confidence_score']}")
        
        print("\nFinal Response:")
        print(result["final_response"])
        
        print("\n" + "="*100)
    
    print("\nSelf-RAG example completed successfully!")


def demonstrate_confidence_impact():
    """
    Demonstrate how different confidence thresholds affect Self-RAG behavior.
    """
    # Path to the vector store
    vector_store_path = "../../../basic-rag/data/processed/vector_store"
    
    # Test query that might be answerable with parametric knowledge
    query = "What are the main components of a RAG system?"
    
    print(f"\nQuery: {query}")
    
    # Try with different confidence thresholds
    thresholds = [0.3, 0.7, 0.9]
    
    for threshold in thresholds:
        print(f"\n=== Using Confidence Threshold: {threshold} ===")
        
        # Initialize Self-RAG with this threshold
        self_rag = SelfRAG(
            vector_store_path=vector_store_path,
            model_name="gpt-3.5-turbo",
            confidence_threshold=threshold
        )
        
        # Run the query
        result = self_rag.query(query)
        
        # Print decision and whether documents were retrieved
        print("\nRetrieval Decision:")
        print(result["retrieval_decision"])
        
        print(f"\nDocuments Retrieved: {len(result['retrieved_documents']) > 0}")
        
        print(f"\nInitial Confidence: {result['confidence_score']}")
        
        print("\nFinal Response:")
        print(result["final_response"])
        
        print("\n" + "="*50)


def compare_with_basic_rag():
    """
    Compare Self-RAG with basic RAG on a challenging query.
    """
    # Import the basic RAG pipeline
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../basic-rag/src"))
    from rag_pipeline import RAGPipeline
    
    # Path to the vector store
    vector_store_path = "../../../basic-rag/data/processed/vector_store"
    
    # Test query that might lead to hallucinations
    query = "What are the limitations of RAG systems and how can they be addressed?"
    
    print(f"\nQuery: {query}")
    
    # Basic RAG
    print("\n=== Basic RAG Approach ===")
    basic_rag = RAGPipeline(vector_store_path=vector_store_path)
    basic_result = basic_rag.query(query)
    
    print("\nBasic RAG Response:")
    print(basic_result["response"])
    
    # Self-RAG
    print("\n=== Self-RAG Approach ===")
    self_rag = SelfRAG(vector_store_path=vector_store_path)
    self_result = self_rag.query(query)
    
    print("\nSelf-RAG Initial Response:")
    print(self_result["initial_response"])
    
    print("\nSelf-RAG Critique:")
    print(self_result["critique"])
    
    print("\nSelf-RAG Final Response:")
    print(self_result["final_response"])
    
    print("\nComparison complete!")


if __name__ == "__main__":
    # Choose which example to run
    run_self_rag_example()
    # demonstrate_confidence_impact()
    # compare_with_basic_rag() 