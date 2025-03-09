#!/usr/bin/env python3
"""
Data Processing Command Line Interface

This module provides a command-line interface for executing data processing tasks.
"""

import os
import argparse
import logging
import json
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# Local imports
from .document_loaders import DirectoryLoader, PDFLoader, TextLoader, HTMLLoader, MarkdownLoader
from .text_chunkers import get_chunker
from .metadata_extractors import create_comprehensive_extractor
from .data_cleaners import create_standard_cleaner, create_comprehensive_cleaner
from .data_augmentation import create_standard_augmentation_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_loader(file_path: str):
    """
    Get an appropriate loader based on the file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        An appropriate loader for the file type
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if os.path.isdir(file_path):
        return DirectoryLoader()
    elif ext == '.pdf':
        return PDFLoader()
    elif ext in ['.txt', '.log', '.csv']:
        return TextLoader()
    elif ext in ['.html', '.htm']:
        return HTMLLoader()
    elif ext in ['.md', '.markdown']:
        return MarkdownLoader()
    else:
        # Default to text loader
        logger.warning(f"No specific loader for extension {ext}, using TextLoader")
        return TextLoader()


def process_files(args):
    """
    Process files according to the command-line arguments.
    
    Args:
        args: Command-line arguments
    """
    input_path = args.input
    output_path = args.output
    
    logger.info(f"Processing input: {input_path}")
    
    # Get appropriate loader
    loader = get_loader(input_path)
    logger.info(f"Using loader: {loader.__class__.__name__}")
    
    # Load documents
    documents = loader.load(input_path)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Metadata extraction (optional)
    if args.extract_metadata:
        logger.info("Extracting metadata...")
        metadata_extractor = create_comprehensive_extractor()
        documents = metadata_extractor.extract_from_documents(documents)
        logger.info("Metadata extraction completed")
    
    # Cleaning (optional)
    if args.clean:
        logger.info("Cleaning documents...")
        if args.comprehensive_cleaning:
            cleaner = create_comprehensive_cleaner()
        else:
            cleaner = create_standard_cleaner()
        documents = cleaner.clean_documents(documents)
        logger.info("Document cleaning completed")
    
    # Chunking (optional)
    if args.chunk:
        logger.info(f"Chunking documents with strategy: {args.chunk_strategy}")
        chunker = get_chunker(
            strategy=args.chunk_strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        documents = chunker.split_documents(documents)
        logger.info(f"Chunking completed, resulting in {len(documents)} chunks")
    
    # Augmentation (optional)
    if args.augment:
        try:
            logger.info("Augmenting documents...")
            augmenter = create_standard_augmentation_pipeline(
                num_variations=args.augment_variations
            )
            augmented_docs = augmenter.augment_documents(
                documents,
                max_variations_per_doc=args.augment_variations,
                max_total_variations=args.augment_max_total
            )
            logger.info(f"Created {len(augmented_docs)} augmented documents")
            
            # Combine original and augmented documents if requested
            if args.augment_combine:
                documents = documents + augmented_docs
                logger.info(f"Combined original and augmented documents: {len(documents)} total")
            else:
                documents = augmented_docs
                logger.info(f"Using only augmented documents: {len(documents)} total")
        except ImportError as e:
            logger.error(f"Augmentation failed: {e}")
            logger.error("Make sure all required dependencies are installed")
    
    # Write output
    if output_path:
        logger.info(f"Writing output to: {output_path}")
        write_output(documents, output_path, args.output_format)
    else:
        # Print sample of processed documents
        print_sample(documents, args.sample_size)
    
    logger.info("Processing completed successfully")


def write_output(documents, output_path: str, output_format: str):
    """
    Write processed documents to the output location.
    
    Args:
        documents: List of processed documents
        output_path: Path to write output to
        output_format: Format of the output (json, text, csv)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if output_format == 'json':
        # Convert documents to JSON
        docs_data = []
        for doc in documents:
            doc_dict = {
                "content": doc.content,
                "metadata": doc.metadata
            }
            docs_data.append(doc_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, indent=2, ensure_ascii=False)
    
    elif output_format == 'csv':
        import csv
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Determine all possible metadata fields
            all_metadata_fields = set()
            for doc in documents:
                all_metadata_fields.update(doc.metadata.keys())
            
            # Create CSV writer
            fieldnames = ['content'] + sorted(list(all_metadata_fields))
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write documents
            for doc in documents:
                row = {'content': doc.content}
                row.update(doc.metadata)
                writer.writerow(row)
    
    elif output_format == 'text':
        # Write documents as plain text with metadata as comments
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(documents):
                f.write(f"# Document {i+1}\n")
                f.write(f"# Metadata: {json.dumps(doc.metadata)}\n")
                f.write(f"{doc.content}\n\n")
                f.write("#" + "-" * 79 + "\n\n")
    
    elif output_format == 'dir':
        # Write each document as a separate file in a directory
        os.makedirs(output_path, exist_ok=True)
        
        for i, doc in enumerate(documents):
            doc_id = doc.metadata.get('id', f"doc_{i+1}")
            file_path = os.path.join(output_path, f"{doc_id}.txt")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Metadata: {json.dumps(doc.metadata)}\n\n")
                f.write(doc.content)
    
    else:
        logger.error(f"Unsupported output format: {output_format}")
        return


def print_sample(documents, sample_size: int):
    """
    Print a sample of the processed documents.
    
    Args:
        documents: List of processed documents
        sample_size: Number of documents to sample
    """
    sample_size = min(sample_size, len(documents))
    
    print(f"\nSample of {sample_size} processed documents:")
    print("=" * 80)
    
    for i in range(sample_size):
        doc = documents[i]
        print(f"Document {i+1}:")
        print(f"Metadata: {json.dumps(doc.metadata)}")
        
        # Truncate content if it's too long
        content = doc.content
        if len(content) > 200:
            content = content[:200] + "..."
        
        print(f"Content: {content}")
        print("-" * 80)


def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(
        description='Process documents for RAG systems.'
    )
    
    # Input and output options
    parser.add_argument('--input', '-i', required=True, help='Input file or directory')
    parser.add_argument('--output', '-o', help='Output file or directory')
    parser.add_argument('--output-format', '-f', default='json',
                      choices=['json', 'text', 'csv', 'dir'],
                      help='Output format')
    parser.add_argument('--sample-size', type=int, default=3,
                      help='Number of documents to show in sample output')
    
    # Processing options
    parser.add_argument('--extract-metadata', '-m', action='store_true',
                      help='Extract metadata from documents')
    parser.add_argument('--clean', '-c', action='store_true',
                      help='Clean documents')
    parser.add_argument('--comprehensive-cleaning', action='store_true',
                      help='Use comprehensive cleaning (slower but more thorough)')
    parser.add_argument('--chunk', '-k', action='store_true',
                      help='Chunk documents')
    parser.add_argument('--chunk-strategy', default='recursive',
                      choices=['character', 'recursive', 'token', 'sentence', 'semantic'],
                      help='Chunking strategy')
    parser.add_argument('--chunk-size', type=int, default=1000,
                      help='Maximum chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                      help='Chunk overlap size')
    
    # Augmentation options
    parser.add_argument('--augment', '-a', action='store_true',
                      help='Augment documents')
    parser.add_argument('--augment-variations', type=int, default=2,
                      help='Number of variations to generate per document')
    parser.add_argument('--augment-max-total', type=int, default=None,
                      help='Maximum total augmented documents')
    parser.add_argument('--augment-combine', action='store_true',
                      help='Combine original and augmented documents')
    
    args = parser.parse_args()
    
    try:
        process_files(args)
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 