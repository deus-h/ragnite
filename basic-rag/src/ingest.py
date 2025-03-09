"""
Document Ingestion Script for RAG

This script processes documents from a directory and creates a vector store for retrieval.
It supports various document formats including PDF, TXT, DOCX, and Markdown.
"""

import os
import argparse
from typing import List, Optional
import glob

from langchain.docstore.document import Document
from langchain.document_loaders import (
    TextLoader,
    PDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_pipeline import RAGPipeline


def load_documents(data_dir: str) -> List[Document]:
    """
    Load documents from a directory.

    Args:
        data_dir: Directory containing documents.

    Returns:
        List of loaded documents.
    """
    documents = []
    
    # Define supported file types and their loaders
    loaders = {
        "*.txt": TextLoader,
        "*.pdf": PDFLoader,
        "*.docx": Docx2txtLoader,
        "*.md": UnstructuredMarkdownLoader,
    }
    
    # Load documents for each supported file type
    for glob_pattern, loader_cls in loaders.items():
        for file_path in glob.glob(os.path.join(data_dir, glob_pattern)):
            try:
                loader = loader_cls(file_path)
                documents.extend(loader.load())
                print(f"Loaded {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return documents


def main():
    """
    Main function to run the document ingestion process.
    """
    parser = argparse.ArgumentParser(description="Ingest documents for RAG")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--vector_store_path",
        type=str,
        default="../data/processed/vector_store",
        help="Path to save the vector store",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Size of document chunks",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-ada-002",
        help="Embedding model to use",
    )
    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.vector_store_path), exist_ok=True)
    
    # Load documents
    print(f"Loading documents from {args.data_dir}")
    documents = load_documents(args.data_dir)
    print(f"Loaded {len(documents)} documents")
    
    if not documents:
        print("No documents found. Exiting.")
        return
    
    # Initialize RAG pipeline
    rag = RAGPipeline(embedding_model=args.embedding_model)
    
    # Ingest documents
    print(f"Ingesting documents with chunk size {args.chunk_size} and overlap {args.chunk_overlap}")
    rag.ingest_documents(
        documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    
    # Save vector store
    print(f"Saving vector store to {args.vector_store_path}")
    rag.save_vector_store(args.vector_store_path)
    
    print("Document ingestion complete!")


if __name__ == "__main__":
    main() 