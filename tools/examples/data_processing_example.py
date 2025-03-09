#!/usr/bin/env python3
"""
Data Processing Example

This script demonstrates how to use the RAG data processing tools with sample text data.
"""

import os
import logging
import sys
from pathlib import Path

# Add the parent directory to the Python path to import the tools
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.src.data_processing.document_loaders import Document
from tools.src.data_processing.text_chunkers import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTextSplitter
)
from tools.src.data_processing.metadata_extractors import (
    BasicMetadataExtractor,
    TitleExtractor,
    KeywordExtractor,
    CompositeMetadataExtractor
)
from tools.src.data_processing.data_cleaners import (
    WhitespaceNormalizer,
    HTMLCleaner,
    NoiseRemover,
    CompositeCleaner
)
from tools.src.data_processing.data_augmentation import (
    WordReplacementAugmenter,
    RandomSwapAugmenter,
    CompositeAugmenter
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample texts for demonstration
SAMPLE_TEXTS = {
    "research_paper": """
    # Machine Learning Applications in RAG Systems
    
    ## Abstract
    
    Retrieval-Augmented Generation (RAG) systems have gained significant traction in recent years.
    This paper explores various machine learning techniques applied to RAG systems, focusing on
    improving retrieval accuracy and generation quality. We demonstrate that combining dense retrieval
    with re-ranking methods yields superior results compared to traditional approaches.
    
    ## Introduction
    
    Large Language Models (LLMs) have demonstrated impressive capabilities in generating coherent and
    contextually relevant text. However, they often lack access to specific or up-to-date knowledge.
    Retrieval-Augmented Generation (RAG) addresses this limitation by augmenting LLMs with external
    knowledge sources.
    
    Email the author at researcher@example.com for more information.
    
    ## Methodology
    
    Our approach involves the following steps:
    1. Document processing and chunking
    2. Embedding generation using transformer models
    3. Vector database indexing and retrieval
    4. Re-ranking of retrieved documents
    5. Context-aware generation using LLMs
    
    Visit https://example.com/rag-research for the code repository.
    """,
    
    "news_article": """
    <h1>New Advancements in AI Research Announced</h1>
    
    <p>SAN FRANCISCO, June 15, 2023 - A team of researchers at Tech Innovations Inc. has announced 
    breakthrough advancements in artificial intelligence technology that could revolutionize how
    computers process natural language.</p>
    
    <p>The new approach, dubbed "DynamicRAG," combines retrieval-augmented generation with dynamic
    memory systems to create more accurate and contextually aware AI responses.</p>
    
    <p>"This represents a significant step forward in our ability to create AI systems that can
    access and reason over large knowledge bases," said Dr. Jane Smith, lead researcher on the project.</p>
    
    <p>The technology is expected to be implemented in various applications, from customer service
    chatbots to advanced research tools.</p>
    
    <p>For more information, contact press@techinnovations.example.com or call (555) 123-4567.</p>
    """,
    
    "technical_document": """
    # RAG System Implementation Guide
    
    This technical document outlines the steps required to implement a Retrieval-Augmented
    Generation (RAG) system using the latest tools and libraries.
    
    ## System Requirements
    
    - Python 3.9+
    - PyTorch 2.0+
    - Hugging Face Transformers 4.30+
    - 16GB+ RAM
    - CUDA-compatible GPU (recommended)
    
    ## Installation
    
    ```bash
    pip install torch transformers faiss-gpu langchain
    ```
    
    ## Implementation Steps
    
    1. **Data Preparation**
       - Collect and clean your corpus
       - Split documents into appropriate chunks
       - Remove irrelevant content and normalize text
    
    2. **Embedding Generation**
       - Select an embedding model (e.g., all-MiniLM-L6-v2)
       - Generate embeddings for all document chunks
       - Store embeddings in a vector database
    
    3. **Retrieval Setup**
       - Configure similarity search parameters
       - Implement filtering capabilities
       - Add re-ranking for improved relevance
    
    4. **Integration with LLM**
       - Select and configure your LLM
       - Design effective prompts
       - Implement context window management
    
    ## Troubleshooting
    
    If you encounter "Out of memory" errors, try reducing batch sizes or chunk lengths.
    """
}

def main():
    """Main function to demonstrate data processing tools."""
    logger.info("Starting data processing example")
    
    # Create sample documents
    documents = []
    for doc_type, text in SAMPLE_TEXTS.items():
        doc = Document(
            content=text,
            metadata={"id": f"doc_{doc_type}", "type": doc_type, "source": f"sample_{doc_type}.txt"}
        )
        documents.append(doc)
    
    logger.info(f"Created {len(documents)} sample documents")
    
    # ---------------------------------
    # Data Cleaning Demonstration
    # ---------------------------------
    logger.info("\n=== DEMONSTRATION 1: DATA CLEANING ===")
    
    # Create a composite cleaner
    cleaner = CompositeCleaner([
        HTMLCleaner(decode_html_entities=True, keep_important_tags=True),
        WhitespaceNormalizer(max_consecutive_newlines=2),
        NoiseRemover(remove_urls=True, remove_emails=True, remove_phone_numbers=True)
    ])
    
    # Clean the documents
    cleaned_documents = cleaner.clean_documents(documents)
    
    # Show an example
    news_doc = next(doc for doc in cleaned_documents if doc.metadata["type"] == "news_article")
    logger.info("Original news article first 100 chars: " + documents[1].content[:100].replace("\n", " "))
    logger.info("Cleaned news article first 100 chars: " + news_doc.content[:100].replace("\n", " "))
    
    # ---------------------------------
    # Metadata Extraction Demonstration
    # ---------------------------------
    logger.info("\n=== DEMONSTRATION 2: METADATA EXTRACTION ===")
    
    # Create a composite metadata extractor
    metadata_extractor = CompositeMetadataExtractor([
        BasicMetadataExtractor(),
        TitleExtractor(use_first_line=True),
        KeywordExtractor(max_keywords=10)
    ])
    
    # Extract metadata
    docs_with_metadata = metadata_extractor.extract_from_documents(cleaned_documents)
    
    # Show an example
    tech_doc = next(doc for doc in docs_with_metadata if doc.metadata["type"] == "technical_document")
    logger.info(f"Document title: {tech_doc.metadata.get('title', 'No title extracted')}")
    logger.info(f"Character count: {tech_doc.metadata.get('char_count')}")
    logger.info(f"Word count: {tech_doc.metadata.get('word_count')}")
    logger.info(f"Keywords: {tech_doc.metadata.get('keywords', [])[:5]}")
    
    # ---------------------------------
    # Text Chunking Demonstration
    # ---------------------------------
    logger.info("\n=== DEMONSTRATION 3: TEXT CHUNKING ===")
    
    # Create different types of chunkers
    chunkers = {
        "recursive": RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50),
        "token": TokenTextSplitter(chunk_size=50, chunk_overlap=10),
        "sentence": SentenceTextSplitter(chunk_size=3, chunk_overlap=1)
    }
    
    # Apply different chunkers to the research paper
    research_doc = next(doc for doc in docs_with_metadata if doc.metadata["type"] == "research_paper")
    
    for chunker_name, chunker in chunkers.items():
        chunks = chunker.split_documents([research_doc])
        logger.info(f"{chunker_name.capitalize()} chunker created {len(chunks)} chunks")
        
        # Show the first chunk
        if chunks:
            first_chunk = chunks[0]
            logger.info(f"First chunk ({len(first_chunk.content)} chars): {first_chunk.content[:100]}...")
            logger.info(f"Chunk metadata: chunk_index={first_chunk.metadata.get('chunk_index')}, "
                       f"chunk_count={first_chunk.metadata.get('chunk_count')}")
    
    # ---------------------------------
    # Data Augmentation Demonstration
    # ---------------------------------
    logger.info("\n=== DEMONSTRATION 4: DATA AUGMENTATION ===")
    
    try:
        # Create a composite augmenter
        augmenter = CompositeAugmenter(
            augmenters=[
                WordReplacementAugmenter(replace_fraction=0.2, num_variations=2),
                RandomSwapAugmenter(swap_fraction=0.1, num_variations=2)
            ],
            max_variations=3,
            combine_augmenters=True
        )
        
        # Get a small chunk to augment
        sample_chunk = chunkers["sentence"].split_documents([tech_doc])[0]
        
        # Augment the chunk
        augmented_docs = augmenter.augment_document(sample_chunk, max_variations=3)
        
        # Show the results
        logger.info(f"Original text: {sample_chunk.content}")
        for i, aug_doc in enumerate(augmented_docs):
            logger.info(f"Augmentation {i+1}: {aug_doc.content}")
    except ImportError as e:
        logger.warning(f"Could not demonstrate augmentation: {e}")
        logger.warning("Make sure NLTK is installed for augmentation features")
    
    # ---------------------------------
    # Complete Pipeline Demonstration
    # ---------------------------------
    logger.info("\n=== DEMONSTRATION 5: COMPLETE PIPELINE ===")
    
    # Process one document through a complete pipeline
    doc = documents[2]  # Technical document
    
    # 1. Clean
    cleaned_doc = cleaner.clean_document(doc)
    logger.info("1. Cleaned document")
    
    # 2. Extract metadata
    doc_with_metadata = metadata_extractor.extract(cleaned_doc)
    logger.info(f"2. Extracted metadata: {list(doc_with_metadata.metadata.keys())}")
    
    # 3. Chunk
    chunker = chunkers["recursive"]
    chunks = chunker.split_documents([cleaned_doc])
    logger.info(f"3. Split into {len(chunks)} chunks")
    
    # 4. Process chunks (example: count total tokens)
    total_words = sum(len(chunk.content.split()) for chunk in chunks)
    logger.info(f"4. Processed chunks: {len(chunks)} chunks with {total_words} total words")
    
    logger.info("\nData processing example completed successfully!")

if __name__ == "__main__":
    main() 