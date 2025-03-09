"""
Cached RAG Example

This example demonstrates how to integrate caching with the basic RAG pipeline
to improve performance and reduce redundant API calls.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add project root to path so we can import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import the basic RAG pipeline
from basic_rag.src.rag_pipeline import RAGPipeline

# Import the cache integration
from advanced_rag.cache_integration import add_caching_to_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the cached RAG example"""
    
    # Create a cache directory
    cache_dir = Path("~/.ragnite/cache").expanduser()
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a basic RAG pipeline
    logger.info("Creating RAG pipeline...")
    rag = RAGPipeline(
        model_name="gpt-3.5-turbo",
        embedding_model="text-embedding-ada-002",
        top_k=3
    )
    
    # Ingest some documents if needed
    rag.ingest_documents([
        "Retrieval-Augmented Generation (RAG) is an AI framework that enhances Large Language Models' responses by incorporating relevant knowledge from external sources.",
        "RAG combines the strengths of retrieval-based and generation-based approaches to provide more accurate, up-to-date, and verifiable responses.",
        "RAG systems typically consist of a retrieval component that finds relevant documents and a generation component that creates responses based on those documents.",
        "By using RAG, AI systems can overcome limitations of standard LLMs, such as outdated knowledge or hallucinations.",
        "RAG is particularly useful for domain-specific applications where accurate and verifiable information is critical."
    ])
    
    # Add caching to the pipeline
    logger.info("Adding caching to RAG pipeline...")
    cached_rag = add_caching_to_pipeline(
        rag, 
        cache_dir=str(cache_dir),
        dashboard_port=None  # Set to e.g. 8080 to enable dashboard
    )
    
    # Define some example queries
    queries = [
        "What is retrieval-augmented generation?",
        "What are the benefits of RAG?",
        "How does RAG improve AI responses?",
        "What is retrieval-augmented generation?",  # Repeated query to demonstrate caching
        "What are the benefits of RAG?",  # Repeated query to demonstrate caching
    ]
    
    # Run queries and measure performance
    for i, query in enumerate(queries):
        logger.info(f"\nQuery {i+1}: {query}")
        
        # Time the query
        start_time = time.time()
        response = cached_rag.query(query)
        elapsed = time.time() - start_time
        
        logger.info(f"Response: {response}")
        logger.info(f"Query completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 