"""
Basic RAG Example

This script demonstrates how to use the basic RAG implementation.
It shows how to:
1. Ingest documents
2. Create a vector store
3. Query the RAG system
"""

import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from langchain.docstore.document import Document
from rag_pipeline import RAGPipeline


def create_sample_documents():
    """
    Create sample documents for demonstration.
    
    Returns:
        List of Document objects.
    """
    # Sample documents about different programming languages
    docs = [
        Document(
            page_content="""
            Python is a high-level, interpreted programming language known for its readability and versatility.
            It supports multiple programming paradigms, including procedural, object-oriented, and functional programming.
            Python is widely used in web development, data analysis, artificial intelligence, scientific computing, and more.
            Key features include dynamic typing, automatic memory management, and a comprehensive standard library.
            """,
            metadata={"source": "python_info.txt", "topic": "programming language"}
        ),
        Document(
            page_content="""
            JavaScript is a high-level, interpreted programming language that conforms to the ECMAScript specification.
            It is a core technology of the World Wide Web and is used to make web pages interactive.
            JavaScript supports event-driven, functional, and imperative programming styles.
            It has APIs for working with text, arrays, dates, regular expressions, and the DOM.
            """,
            metadata={"source": "javascript_info.txt", "topic": "programming language"}
        ),
        Document(
            page_content="""
            Rust is a multi-paradigm, high-level, general-purpose programming language designed for performance and safety.
            It enforces memory safety without requiring garbage collection, making it suitable for systems programming.
            Rust's ownership system guarantees memory safety and thread safety at compile time.
            It is used in performance-critical applications, embedded systems, and WebAssembly.
            """,
            metadata={"source": "rust_info.txt", "topic": "programming language"}
        ),
        Document(
            page_content="""
            Machine learning is a subset of artificial intelligence that focuses on developing systems that learn from data.
            It involves algorithms that improve automatically through experience and data.
            Common machine learning techniques include supervised learning, unsupervised learning, and reinforcement learning.
            Applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles.
            """,
            metadata={"source": "ml_info.txt", "topic": "technology"}
        ),
    ]
    
    return docs


def main():
    """
    Main function to demonstrate the RAG pipeline.
    """
    # Create a temporary directory for the vector store
    vector_store_path = "../data/processed/example_vector_store"
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    
    # Initialize the RAG pipeline
    rag = RAGPipeline()
    
    # Ingest documents
    print("Ingesting documents...")
    rag.ingest_documents(documents, chunk_size=500, chunk_overlap=50)
    
    # Save the vector store
    print(f"Saving vector store to {vector_store_path}")
    rag.save_vector_store(vector_store_path)
    
    # Example queries
    example_queries = [
        "What is Python used for?",
        "How does Rust ensure memory safety?",
        "What are the main programming paradigms supported by JavaScript?",
        "What is machine learning and what are its applications?",
        "What are the differences between Python and JavaScript?",
    ]
    
    # Run queries
    print("\n=== Running Example Queries ===\n")
    for query in example_queries:
        print(f"\nQuery: {query}")
        result = rag.query(query)
        
        print("\nRetrieved Documents:")
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"\nDocument {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):")
            print(doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content)
        
        print("\nResponse:")
        print(result["response"])
        print("\n" + "="*50)
    
    print("\nRAG example completed successfully!")


if __name__ == "__main__":
    main() 