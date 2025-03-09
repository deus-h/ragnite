"""
Code Chunker

This module provides code-aware chunking functionality for different programming languages.
Instead of splitting code based on arbitrary token counts, it uses Abstract Syntax Trees (ASTs)
to identify logical units like functions, classes, and methods.
"""

import os
import ast
import re
import tokenize
from typing import List, Dict, Any, Optional, Tuple, Union
import io

from langchain.docstore.document import Document


class CodeChunker:
    """
    A base class for language-specific code chunkers.
    """
    
    def __init__(self, min_chunk_size: int = 50, max_chunk_size: int = 1500):
        """
        Initialize the code chunker.
        
        Args:
            min_chunk_size: Minimum chunk size in characters.
            max_chunk_size: Maximum chunk size in characters.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def split(self, code: str, filename: str) -> List[Document]:
        """
        Split code into chunks based on logical units.
        
        Args:
            code: The code to split.
            filename: The name of the file containing the code.
            
        Returns:
            List of Document objects representing the chunks.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def detect_language(self, filename: str) -> str:
        """
        Detect the programming language based on the file extension.
        
        Args:
            filename: The name of the file.
            
        Returns:
            The detected programming language.
        """
        ext = os.path.splitext(filename)[1].lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.rs': 'rust',
        }
        
        return language_map.get(ext, 'unknown')


class PythonCodeChunker(CodeChunker):
    """
    A code chunker for Python that uses the ast module to identify logical units.
    """
    
    def split(self, code: str, filename: str) -> List[Document]:
        """
        Split Python code into chunks based on classes and functions.
        
        Args:
            code: The Python code to split.
            filename: The name of the file containing the code.
            
        Returns:
            List of Document objects representing the chunks.
        """
        chunks = []
        
        try:
            # Parse the AST
            tree = ast.parse(code)
            
            # Extract imports and global variables
            imports = []
            global_vars = []
            
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_start = node.lineno
                    import_end = getattr(node, 'end_lineno', node.lineno)
                    import_code = self._get_code_lines(code, import_start, import_end)
                    imports.append(import_code)
                elif isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
                    global_var_start = node.lineno
                    global_var_end = getattr(node, 'end_lineno', node.lineno)
                    global_var_code = self._get_code_lines(code, global_var_start, global_var_end)
                    global_vars.append(global_var_code)
            
            if imports or global_vars:
                # Create a chunk for imports and global variables
                imports_and_globals = "\n".join(imports + global_vars)
                chunks.append(Document(
                    page_content=imports_and_globals,
                    metadata={
                        "source": filename,
                        "chunk_type": "imports_and_globals",
                        "language": "python"
                    }
                ))
            
            # Extract classes and functions
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    # Get the class code including its docstring and methods
                    class_start = node.lineno
                    class_end = max([getattr(n, 'end_lineno', node.lineno) for n in ast.walk(node) if hasattr(n, 'lineno')])
                    class_code = self._get_code_lines(code, class_start, class_end)
                    
                    # Extract class docstring
                    class_docstring = ast.get_docstring(node) or ""
                    
                    # Create a chunk for the class
                    chunks.append(Document(
                        page_content=class_code,
                        metadata={
                            "source": filename,
                            "chunk_type": "class",
                            "class_name": node.name,
                            "docstring": class_docstring,
                            "language": "python"
                        }
                    ))
                    
                    # Also extract methods separately if the class is large
                    if len(class_code) > self.max_chunk_size:
                        for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                            method_start = method.lineno
                            method_end = max([getattr(n, 'end_lineno', method.lineno) for n in ast.walk(method) if hasattr(n, 'lineno')])
                            method_code = self._get_code_lines(code, method_start, method_end)
                            
                            # Extract method docstring
                            method_docstring = ast.get_docstring(method) or ""
                            
                            # Create a chunk for the method
                            chunks.append(Document(
                                page_content=method_code,
                                metadata={
                                    "source": filename,
                                    "chunk_type": "method",
                                    "class_name": node.name,
                                    "method_name": method.name,
                                    "docstring": method_docstring,
                                    "language": "python"
                                }
                            ))
                
                elif isinstance(node, ast.FunctionDef):
                    # Get the function code including its docstring
                    func_start = node.lineno
                    func_end = max([getattr(n, 'end_lineno', node.lineno) for n in ast.walk(node) if hasattr(n, 'lineno')])
                    func_code = self._get_code_lines(code, func_start, func_end)
                    
                    # Extract function docstring
                    func_docstring = ast.get_docstring(node) or ""
                    
                    # Create a chunk for the function
                    chunks.append(Document(
                        page_content=func_code,
                        metadata={
                            "source": filename,
                            "chunk_type": "function",
                            "function_name": node.name,
                            "docstring": func_docstring,
                            "language": "python"
                        }
                    ))
            
            # Check if there are any logical units
            if not chunks:
                # If no logical units were found, add the entire file as a chunk
                chunks.append(Document(
                    page_content=code,
                    metadata={
                        "source": filename,
                        "chunk_type": "file",
                        "language": "python"
                    }
                ))
        
        except SyntaxError:
            # If there's a syntax error, fall back to line-based chunking
            chunks = self._fallback_chunking(code, filename)
        
        return chunks
    
    def _get_code_lines(self, code: str, start_line: int, end_line: int) -> str:
        """
        Extract lines from the code string.
        
        Args:
            code: The full code string.
            start_line: The starting line number (1-indexed).
            end_line: The ending line number (1-indexed).
            
        Returns:
            The extracted code as a string.
        """
        lines = code.split('\n')
        return '\n'.join(lines[start_line-1:end_line])
    
    def _fallback_chunking(self, code: str, filename: str) -> List[Document]:
        """
        Fallback method for chunking when AST parsing fails.
        
        Args:
            code: The code to chunk.
            filename: The name of the file.
            
        Returns:
            List of Document objects.
        """
        # Simple chunking based on lines and maximum chunk size
        chunks = []
        lines = code.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for the newline character
            if current_size + line_size > self.max_chunk_size and current_chunk:
                # Create a new chunk with the accumulated lines
                chunks.append(Document(
                    page_content='\n'.join(current_chunk),
                    metadata={
                        "source": filename,
                        "chunk_type": "fallback",
                        "language": "python"
                    }
                ))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(Document(
                page_content='\n'.join(current_chunk),
                metadata={
                    "source": filename,
                    "chunk_type": "fallback",
                    "language": "python"
                }
            ))
        
        return chunks


class JavaScriptCodeChunker(CodeChunker):
    """
    A code chunker for JavaScript that uses regular expressions to identify logical units.
    """
    
    def split(self, code: str, filename: str) -> List[Document]:
        """
        Split JavaScript code into chunks based on functions, classes, and modules.
        
        Args:
            code: The JavaScript code to split.
            filename: The name of the file containing the code.
            
        Returns:
            List of Document objects representing the chunks.
        """
        chunks = []
        
        # Extract imports and exports
        import_export_pattern = r'(import\s+.*?;|export\s+.*?;|require\s*\(.*?\))'
        imports_exports = re.findall(import_export_pattern, code, re.DOTALL)
        
        if imports_exports:
            imports_exports_code = '\n'.join(imports_exports)
            chunks.append(Document(
                page_content=imports_exports_code,
                metadata={
                    "source": filename,
                    "chunk_type": "imports_exports",
                    "language": "javascript"
                }
            ))
        
        # Extract classes
        class_pattern = r'(class\s+\w+(?:\s+extends\s+\w+)?\s*\{[\s\S]*?\n\})'
        classes = re.findall(class_pattern, code)
        
        for class_code in classes:
            # Extract class name
            class_name_match = re.search(r'class\s+(\w+)', class_code)
            class_name = class_name_match.group(1) if class_name_match else "Unknown"
            
            # Extract JSDoc comment if present
            jsdoc_match = re.search(r'/\*\*[\s\S]*?\*/', class_code)
            jsdoc = jsdoc_match.group(0) if jsdoc_match else ""
            
            chunks.append(Document(
                page_content=class_code,
                metadata={
                    "source": filename,
                    "chunk_type": "class",
                    "class_name": class_name,
                    "jsdoc": jsdoc,
                    "language": "javascript"
                }
            ))
            
            # Extract methods if the class is large
            if len(class_code) > self.max_chunk_size:
                method_pattern = r'((?:async\s+)?(?:[\w$]+\s*\([^)]*\)|get\s+[\w$]+\s*\(\)|set\s+[\w$]+\s*\([^)]*\))\s*\{[\s\S]*?\n\s*\})'
                methods = re.findall(method_pattern, class_code)
                
                for method_code in methods:
                    # Extract method name
                    method_name_match = re.search(r'(?:async\s+)?((?:get|set)\s+)?([\w$]+)\s*\(', method_code)
                    method_prefix = method_name_match.group(1) if method_name_match and method_name_match.group(1) else ""
                    method_name = method_name_match.group(2) if method_name_match else "Unknown"
                    
                    # Extract JSDoc comment if present
                    method_jsdoc_match = re.search(r'/\*\*[\s\S]*?\*/', method_code)
                    method_jsdoc = method_jsdoc_match.group(0) if method_jsdoc_match else ""
                    
                    chunks.append(Document(
                        page_content=method_code,
                        metadata={
                            "source": filename,
                            "chunk_type": "method",
                            "class_name": class_name,
                            "method_name": f"{method_prefix}{method_name}",
                            "jsdoc": method_jsdoc,
                            "language": "javascript"
                        }
                    ))
        
        # Extract functions (both declaration and expression)
        function_pattern = r'((?:function\s+[\w$]+|const\s+[\w$]+\s*=\s*(?:async\s+)?function|const\s+[\w$]+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>|let\s+[\w$]+\s*=\s*(?:async\s+)?function|let\s+[\w$]+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>|var\s+[\w$]+\s*=\s*(?:async\s+)?function|var\s+[\w$]+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)[\s\S]*?\{[\s\S]*?\n\})'
        functions = re.findall(function_pattern, code)
        
        for function_code in functions:
            # Extract function name
            function_name_match = re.search(r'(?:function\s+|const\s+|let\s+|var\s+)([\w$]+)', function_code)
            function_name = function_name_match.group(1) if function_name_match else "Unknown"
            
            # Extract JSDoc comment if present
            function_jsdoc_match = re.search(r'/\*\*[\s\S]*?\*/', function_code)
            function_jsdoc = function_jsdoc_match.group(0) if function_jsdoc_match else ""
            
            chunks.append(Document(
                page_content=function_code,
                metadata={
                    "source": filename,
                    "chunk_type": "function",
                    "function_name": function_name,
                    "jsdoc": function_jsdoc,
                    "language": "javascript"
                }
            ))
        
        # Check if there are any logical units
        if not chunks:
            # If no logical units were found, add the entire file as a chunk
            chunks.append(Document(
                page_content=code,
                metadata={
                    "source": filename,
                    "chunk_type": "file",
                    "language": "javascript"
                }
            ))
        
        return chunks


def create_code_chunker(language: str, min_chunk_size: int = 50, max_chunk_size: int = 1500) -> CodeChunker:
    """
    Factory function to create a language-specific code chunker.
    
    Args:
        language: The programming language.
        min_chunk_size: Minimum chunk size in characters.
        max_chunk_size: Maximum chunk size in characters.
        
    Returns:
        An instance of a CodeChunker subclass for the specified language.
    """
    language = language.lower()
    
    if language == 'python':
        return PythonCodeChunker(min_chunk_size, max_chunk_size)
    elif language in ['javascript', 'typescript']:
        return JavaScriptCodeChunker(min_chunk_size, max_chunk_size)
    else:
        # Default to Python chunker for now
        return PythonCodeChunker(min_chunk_size, max_chunk_size)


def split_code_file(file_path: str, min_chunk_size: int = 50, max_chunk_size: int = 1500) -> List[Document]:
    """
    Split a code file into chunks based on its language.
    
    Args:
        file_path: The path to the code file.
        min_chunk_size: Minimum chunk size in characters.
        max_chunk_size: Maximum chunk size in characters.
        
    Returns:
        List of Document objects representing the chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        code = f.read()
    
    # Detect the language
    chunker = CodeChunker()
    language = chunker.detect_language(file_path)
    
    # Create a language-specific chunker
    chunker = create_code_chunker(language, min_chunk_size, max_chunk_size)
    
    # Split the code
    return chunker.split(code, os.path.basename(file_path))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split code files into chunks based on logical units.")
    parser.add_argument("--file", type=str, required=True, help="Path to the code file.")
    args = parser.parse_args()
    
    chunks = split_code_file(args.file)
    
    print(f"Split the file into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({chunk.metadata['chunk_type']}):")
        print(f"Source: {chunk.metadata['source']}")
        if 'class_name' in chunk.metadata:
            print(f"Class: {chunk.metadata['class_name']}")
        if 'function_name' in chunk.metadata:
            print(f"Function: {chunk.metadata['function_name']}")
        if 'method_name' in chunk.metadata:
            print(f"Method: {chunk.metadata['method_name']}")
        
        # Print the first few lines of the chunk
        content_preview = "\n".join(chunk.page_content.split("\n")[:5])
        if len(chunk.page_content.split("\n")) > 5:
            content_preview += "\n..."
        print(f"\n{content_preview}") 