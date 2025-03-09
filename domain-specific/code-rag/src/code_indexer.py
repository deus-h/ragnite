"""
Code Indexer

This script indexes code files from a directory or repository to build a vector store
for the Code RAG system. It supports various programming languages and can filter
files by extensions or patterns.
"""

import os
import argparse
import glob
from typing import List, Optional, Set

from code_rag_pipeline import CodeRAG


def get_code_files(
    repo_dir: str, 
    language: str = "python", 
    extensions: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Get all code files in a repository directory.
    
    Args:
        repo_dir: The repository directory path.
        language: The programming language to focus on.
        extensions: Specific file extensions to include. If None, defaults based on language.
        exclude_patterns: Glob patterns for paths to exclude.
        
    Returns:
        List of file paths.
    """
    # Set default extensions based on language if not provided
    if extensions is None:
        if language.lower() == "python":
            extensions = ['.py']
        elif language.lower() == "javascript":
            extensions = ['.js', '.jsx']
        elif language.lower() == "typescript":
            extensions = ['.ts', '.tsx']
        elif language.lower() == "java":
            extensions = ['.java']
        elif language.lower() == "csharp":
            extensions = ['.cs']
        elif language.lower() == "go":
            extensions = ['.go']
        elif language.lower() == "rust":
            extensions = ['.rs']
        elif language.lower() == "cpp":
            extensions = ['.cpp', '.cc', '.cxx', '.h', '.hpp']
        else:
            extensions = []
    
    # Convert extensions to lowercase
    extensions = [ext.lower() if not ext.startswith('.') else ext.lower() for ext in extensions]
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    # Set default exclude patterns if not provided
    if exclude_patterns is None:
        exclude_patterns = [
            # Common version control directories
            "*/.git/*", "*/.svn/*", "*/.hg/*",
            # Common build directories
            "*/node_modules/*", "*/venv/*", "*/env/*", "*/build/*", "*/dist/*",
            # Common cache directories
            "*/__pycache__/*", "*/.pytest_cache/*", "*/.mypy_cache/*",
            # Common IDE directories
            "*/.vscode/*", "*/.idea/*", "*/.vs/*",
            # Other common patterns to exclude
            "*/vendor/*", "*/third_party/*", "*/external/*"
        ]
    
    # Find all files recursively
    all_files = []
    for root, _, files in os.walk(repo_dir):
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


def index_repository(
    repo_dir: str,
    vector_store_path: str,
    language: str = "python",
    extensions: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    batch_size: int = 100,
    min_chunk_size: int = 50,
    max_chunk_size: int = 1500
) -> None:
    """
    Index all code files in a repository.
    
    Args:
        repo_dir: The repository directory path.
        vector_store_path: Path to save the vector store.
        language: The programming language to focus on.
        extensions: Specific file extensions to include. If None, defaults based on language.
        exclude_patterns: Glob patterns for paths to exclude.
        batch_size: Number of files to process in each batch.
        min_chunk_size: Minimum chunk size for code chunking.
        max_chunk_size: Maximum chunk size for code chunking.
    """
    # Get all code files
    code_files = get_code_files(repo_dir, language, extensions, exclude_patterns)
    total_files = len(code_files)
    
    if total_files == 0:
        print(f"No code files found in {repo_dir} with the specified criteria.")
        return
    
    print(f"Found {total_files} code files to index.")
    
    # Initialize the Code RAG pipeline
    code_rag = CodeRAG(
        vector_store_path=vector_store_path if os.path.exists(vector_store_path) else None,
        language=language,
        min_chunk_size=min_chunk_size,
        max_chunk_size=max_chunk_size
    )
    
    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch = code_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_files+batch_size-1)//batch_size}: {len(batch)} files")
        
        successful = 0
        for file_path in batch:
            rel_path = os.path.relpath(file_path, repo_dir)
            try:
                code_rag.ingest_code_file(file_path)
                successful += 1
            except Exception as e:
                print(f"Error processing {rel_path}: {e}")
        
        print(f"Successfully processed {successful}/{len(batch)} files in the batch.")
        
        # Save the vector store after each batch
        try:
            code_rag.save_vector_store(vector_store_path)
            print(f"Saved vector store to {vector_store_path}")
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    print(f"Indexing completed. Processed {total_files} files into {vector_store_path}.")


def main():
    """
    Main function to run the code indexer from the command line.
    """
    parser = argparse.ArgumentParser(description="Index code files from a repository")
    parser.add_argument("--repo_dir", type=str, required=True, help="Repository directory path")
    parser.add_argument("--vector_store_path", type=str, default="../data/processed/vector_store", help="Path to save the vector store")
    parser.add_argument("--language", type=str, default="python", help="Programming language to focus on")
    parser.add_argument("--extensions", type=str, nargs="+", help="Specific file extensions to include")
    parser.add_argument("--exclude", type=str, nargs="+", help="Glob patterns for paths to exclude")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of files to process in each batch")
    parser.add_argument("--min_chunk_size", type=int, default=50, help="Minimum chunk size for code chunking")
    parser.add_argument("--max_chunk_size", type=int, default=1500, help="Maximum chunk size for code chunking")
    args = parser.parse_args()
    
    # Index the repository
    index_repository(
        repo_dir=args.repo_dir,
        vector_store_path=args.vector_store_path,
        language=args.language,
        extensions=args.extensions,
        exclude_patterns=args.exclude,
        batch_size=args.batch_size,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size
    )


if __name__ == "__main__":
    main() 