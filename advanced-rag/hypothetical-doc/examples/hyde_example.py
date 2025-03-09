"""
HyDE RAG Example

This script demonstrates how to use the Hypothetical Document Embeddings (HyDE) RAG implementation.
It shows how to:
1. Initialize the HyDE RAG pipeline
2. Generate a hypothetical document
3. Retrieve documents using the hypothetical document
4. Generate a response
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from hyde_rag_pipeline import HyDERAG


def run_hyde_example():
    """
    Run an example of the HyDE RAG pipeline.
    """
    # Path to the vector store
    vector_store_path = "../../../basic-rag/data/processed/vector_store"
    
    # Initialize the HyDE RAG pipeline
    print("Initializing HyDE RAG pipeline...")
    hyde_rag = HyDERAG(
        vector_store_path=vector_store_path,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Example queries
    example_queries = [
        "What are the benefits of using RAG over traditional language models?",
        "How does document chunking affect RAG performance?",
        "What are the main challenges in implementing RAG systems?",
        "How can RAG be applied to scientific research?",
        "What's the difference between HyDE and Multi-Query RAG techniques?",
    ]
    
    # Run queries
    print("\n=== Running Example Queries with HyDE ===\n")
    for query in example_queries:
        print(f"\nQuery: {query}")
        
        # Generate hypothetical document
        print("\nGenerating hypothetical document...")
        hypothetical_doc = hyde_rag.generate_hypothetical_document(query)
        print(hypothetical_doc)
        
        # Retrieve documents using the hypothetical document
        print("\nRetrieving documents using the hypothetical document...")
        retrieved_docs = hyde_rag.retrieve_with_hyde(hypothetical_doc)
        
        print("\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):")
            print(doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content)
        
        # Generate response
        print("\nGenerating response...")
        response = hyde_rag.generate(query, retrieved_docs)
        
        print("\nResponse:")
        print(response)
        print("\n" + "="*50)
    
    print("\nHyDE RAG example completed successfully!")


def compare_hyde_vs_direct():
    """
    Compare HyDE retrieval against direct query retrieval.
    """
    # Path to the vector store
    vector_store_path = "../../../basic-rag/data/processed/vector_store"
    
    # Initialize the HyDE RAG pipeline
    print("Initializing HyDE RAG pipeline...")
    hyde_rag = HyDERAG(
        vector_store_path=vector_store_path,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Choose a query with potential lexical gap
    query = "How can language models remember up-to-date information?"
    
    print(f"\nQuery: {query}")
    
    # Generate hypothetical document
    print("\n=== HyDE Approach ===")
    print("\nGenerating hypothetical document...")
    hypothetical_doc = hyde_rag.generate_hypothetical_document(query)
    print(hypothetical_doc)
    
    # Retrieve documents using the hypothetical document
    print("\nRetrieving documents using the hypothetical document...")
    hyde_docs = hyde_rag.retrieve_with_hyde(hypothetical_doc)
    
    print("\nHyDE Retrieved Documents:")
    for i, doc in enumerate(hyde_docs):
        print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):")
        print(doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content)
    
    # Direct retrieval for comparison
    print("\n=== Direct Query Approach ===")
    print("\nRetrieving documents using the original query...")
    direct_docs = hyde_rag.vector_store.similarity_search(query, k=hyde_rag.top_k)
    
    print("\nDirect Retrieved Documents:")
    for i, doc in enumerate(direct_docs):
        print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):")
        print(doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content)
    
    # Compare overlap
    hyde_content = [doc.page_content for doc in hyde_docs]
    direct_content = [doc.page_content for doc in direct_docs]
    
    overlap = sum(1 for content in hyde_content if content in direct_content)
    
    print(f"\nDocument Overlap: {overlap} of {len(hyde_docs)}")
    print(f"Unique documents from HyDE: {len(hyde_docs) - overlap}")
    print("\nComparison completed!")


if __name__ == "__main__":
    # Choose which example to run
    # run_hyde_example()
    compare_hyde_vs_direct() 