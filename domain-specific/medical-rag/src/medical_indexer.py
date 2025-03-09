"""
Medical Document Indexer

This script indexes medical documents, research papers, and clinical notes
to build a vector store for the Medical RAG system.
"""

import os
import argparse
import glob
from typing import List, Optional, Set

from medical_rag_pipeline import MedicalRAG


def get_medical_files(
    doc_dir: str, 
    extensions: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Get all medical document files in a directory.
    
    Args:
        doc_dir: The directory path containing medical documents.
        extensions: Specific file extensions to include. If None, defaults to common medical document formats.
        exclude_patterns: Glob patterns for paths to exclude.
        
    Returns:
        List of file paths.
    """
    # Set default extensions if not provided
    if extensions is None:
        extensions = [
            # Common document formats
            '.pdf', '.txt', '.md', '.html', '.xml', '.json',
            # Medical-specific formats
            '.dcm', '.dicom', '.cda', '.fhir', '.hl7',
            # Other formats
            '.csv', '.xlsx', '.docx'
        ]
    
    # Convert extensions to lowercase
    extensions = [ext.lower() if not ext.startswith('.') else ext.lower() for ext in extensions]
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    # Set default exclude patterns if not provided
    if exclude_patterns is None:
        exclude_patterns = [
            # Temporary files
            "*/.tmp/*", "*~", "*.bak",
            # System directories
            "*/.git/*", "*/.svn/*", "*/.DS_Store",
            # Cache directories
            "*/__pycache__/*", "*/.cache/*",
            # Other patterns to exclude
            "*/tmp/*", "*/temp/*", "*/logs/*"
        ]
    
    # Find all files recursively
    all_files = []
    for root, _, files in os.walk(doc_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Check if the file has a matching extension
            if extensions and file_ext not in extensions:
                continue
            
            # Check if the file matches any exclude pattern
            if any(glob.fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns):
                continue
            
            all_files.append(file_path)
    
    return all_files


def categorize_medical_documents(files: List[str]) -> Dict[str, List[str]]:
    """
    Categorize medical documents by type.
    
    Args:
        files: List of file paths.
        
    Returns:
        Dictionary mapping document categories to lists of file paths.
    """
    categories = {
        "research_papers": [],
        "clinical_notes": [],
        "medical_textbooks": [],
        "guidelines": [],
        "other": []
    }
    
    # Simple categorization based on file names and paths
    for file_path in files:
        filename = os.path.basename(file_path).lower()
        path = os.path.dirname(file_path).lower()
        
        # Research papers often have specific patterns in names
        if any(term in filename for term in ['paper', 'research', 'study', 'journal', 'article']) or \
           any(term in path for term in ['papers', 'research', 'publications', 'articles']):
            categories["research_papers"].append(file_path)
        
        # Clinical notes often have specific patterns
        elif any(term in filename for term in ['note', 'patient', 'clinical', 'ehr', 'emr', 'chart']) or \
             any(term in path for term in ['notes', 'patients', 'clinical', 'ehr', 'emr']):
            categories["clinical_notes"].append(file_path)
        
        # Medical textbooks
        elif any(term in filename for term in ['textbook', 'handbook', 'manual', 'guide']) or \
             any(term in path for term in ['textbooks', 'handbooks', 'manuals']):
            categories["medical_textbooks"].append(file_path)
        
        # Guidelines
        elif any(term in filename for term in ['guideline', 'protocol', 'standard', 'recommendation']) or \
             any(term in path for term in ['guidelines', 'protocols', 'standards']):
            categories["guidelines"].append(file_path)
        
        # Other medical documents
        else:
            categories["other"].append(file_path)
    
    return categories


def index_medical_documents(
    doc_dir: str,
    vector_store_path: str,
    extensions: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    batch_size: int = 10,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
    verify_facts: bool = False,  # Turn off for indexing to save time
    entity_recognition: bool = False  # Turn off for indexing to save time
) -> None:
    """
    Index all medical documents in a directory.
    
    Args:
        doc_dir: The directory path containing medical documents.
        vector_store_path: Path to save the vector store.
        extensions: Specific file extensions to include. If None, defaults to common medical document formats.
        exclude_patterns: Glob patterns for paths to exclude.
        batch_size: Number of files to process in each batch.
        min_chunk_size: Minimum chunk size for medical chunking.
        max_chunk_size: Maximum chunk size for medical chunking.
        verify_facts: Whether to verify medical facts during indexing (usually disabled for performance).
        entity_recognition: Whether to perform entity recognition during indexing (usually disabled for performance).
    """
    # Get all medical document files
    medical_files = get_medical_files(doc_dir, extensions, exclude_patterns)
    total_files = len(medical_files)
    
    if total_files == 0:
        print(f"No medical documents found in {doc_dir} with the specified criteria.")
        return
    
    # Categorize the files
    categorized_files = categorize_medical_documents(medical_files)
    
    print(f"Found {total_files} medical documents to index:")
    for category, files in categorized_files.items():
        print(f"- {category}: {len(files)} files")
    
    # Initialize the Medical RAG pipeline
    medical_rag = MedicalRAG(
        vector_store_path=vector_store_path if os.path.exists(vector_store_path) else None,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size,
        verify_facts=verify_facts,  # Usually disabled during indexing for performance
        entity_recognition=entity_recognition  # Usually disabled during indexing for performance
    )
    
    # Process files by category and in batches
    for category, files in categorized_files.items():
        print(f"\nProcessing {category} ({len(files)} files):")
        
        # Process files in batches
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}/{(len(files)+batch_size-1)//batch_size}: {len(batch)} files")
            
            successful = 0
            for file_path in batch:
                rel_path = os.path.relpath(file_path, doc_dir)
                try:
                    print(f"    Processing {rel_path}...")
                    medical_rag.ingest_medical_document(file_path)
                    successful += 1
                except Exception as e:
                    print(f"    Error processing {rel_path}: {e}")
            
            print(f"  Successfully processed {successful}/{len(batch)} files in the batch.")
            
            # Save the vector store after each batch
            try:
                medical_rag.save_vector_store(vector_store_path)
                print(f"  Saved vector store to {vector_store_path}")
            except Exception as e:
                print(f"  Error saving vector store: {e}")
    
    print(f"\nIndexing completed. Processed {total_files} files into {vector_store_path}.")


def main():
    """
    Main function to run the medical document indexer from the command line.
    """
    parser = argparse.ArgumentParser(description="Index medical documents for RAG")
    parser.add_argument("--doc_dir", type=str, required=True, help="Directory containing medical documents")
    parser.add_argument("--vector_store_path", type=str, default="../data/processed/medical_vector_store", help="Path to save the vector store")
    parser.add_argument("--extensions", type=str, nargs="+", help="Specific file extensions to include")
    parser.add_argument("--exclude", type=str, nargs="+", help="Glob patterns for paths to exclude")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of files to process in each batch")
    parser.add_argument("--min_chunk_size", type=int, default=100, help="Minimum chunk size for medical chunking")
    parser.add_argument("--max_chunk_size", type=int, default=2000, help="Maximum chunk size for medical chunking")
    args = parser.parse_args()
    
    # Index the medical documents
    index_medical_documents(
        doc_dir=args.doc_dir,
        vector_store_path=args.vector_store_path,
        extensions=args.extensions,
        exclude_patterns=args.exclude,
        batch_size=args.batch_size,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size
    )


if __name__ == "__main__":
    main() 