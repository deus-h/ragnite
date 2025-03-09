#!/usr/bin/env python3
"""
Scientific RAG Demo

This script demonstrates how to use the Scientific RAG system for
various scientific research use cases.
"""

import os
import argparse
import sys
from pathlib import Path

# Add the parent directory to the Python path to import the scientific_rag package
sys.path.append(str(Path(__file__).parent.parent))

from src.scientific_rag import ScientificRAG


def setup_argparse():
    """Set up argument parsing for the demo script."""
    parser = argparse.ArgumentParser(description="Scientific RAG Demo")
    parser.add_argument(
        "--ingest", "-i",
        help="Path to a PDF file or directory of PDF files to ingest",
    )
    parser.add_argument(
        "--query", "-q",
        help="Scientific query to process",
    )
    parser.add_argument(
        "--section", "-s",
        help="Specific section to search in (e.g., methods, results)",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI for embeddings and generation",
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (if not set in environment variable)",
    )
    parser.add_argument(
        "--db-dir",
        default="./scientific_rag_db",
        help="Directory to store the vector database",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Generate and save ingestion statistics",
    )
    return parser


def ingest_documents(rag, path):
    """Ingest documents into the RAG system."""
    print(f"Ingesting documents from: {path}")
    
    if os.path.isfile(path):
        # Ingest a single file
        if not path.lower().endswith(".pdf"):
            print(f"Error: File must be a PDF: {path}")
            return
        
        try:
            num_chunks = rag.ingest_document(path)
            print(f"Successfully ingested {path}: {num_chunks} chunks")
        except Exception as e:
            print(f"Error ingesting {path}: {str(e)}")
    
    elif os.path.isdir(path):
        # Ingest all PDFs in the directory
        results = rag.ingest_documents(path)
        total_chunks = sum(results.values())
        total_files = len(results)
        successful_files = sum(1 for count in results.values() if count > 0)
        
        print(f"Ingestion complete:")
        print(f"  - Total files processed: {total_files}")
        print(f"  - Successfully ingested: {successful_files}")
        print(f"  - Total chunks created: {total_chunks}")
    
    else:
        print(f"Error: Path does not exist: {path}")


def process_query(rag, query, section=None):
    """Process a scientific query using the RAG system."""
    print(f"\nQuery: {query}")
    
    if section:
        print(f"Filtering by section: {section}")
        response = rag.query_by_section(query, section)
    else:
        response = rag.query(query)
    
    print("\nAnswer:")
    print(response["answer"])
    
    print("\nSources:")
    for i, source in enumerate(response["sources"], 1):
        print(f"\n[{i}] From: {source['metadata'].get('source', 'Unknown')}")
        print(f"    Section: {source['metadata'].get('section', 'Unknown')}")
        if 'page' in source['metadata']:
            print(f"    Page: {source['metadata']['page']}")
        print(f"    Content: {source['content'][:150]}...")


def main():
    """Main function for the demo script."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not any([args.ingest, args.query, args.stats]):
        parser.print_help()
        return
    
    # Initialize the Scientific RAG system
    print("Initializing Scientific RAG system...")
    rag = ScientificRAG(
        use_openai=args.openai,
        openai_api_key=args.api_key,
        db_directory=args.db_dir,
    )
    
    # Handle ingestion
    if args.ingest:
        ingest_documents(rag, args.ingest)
    
    # Handle query
    if args.query:
        process_query(rag, args.query, args.section)
    
    # Generate stats if requested
    if args.stats:
        print("\nGenerating ingestion statistics...")
        rag.save_ingestion_stats("ingestion_stats.json")
        print("Statistics saved to ingestion_stats.json")


if __name__ == "__main__":
    main() 