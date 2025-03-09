"""
Scientific Paper Chunker

This module provides specialized chunking strategies for scientific papers,
respecting the document structure like abstract, introduction, methods, results,
discussion, and conclusion.
"""

import re
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ScientificPaperChunker:
    """Specialized chunker for scientific papers that respects document structure."""
    
    # Common section headers in scientific papers
    SECTION_PATTERNS = {
        'abstract': r'abstract',
        'introduction': r'introduction|background',
        'methods': r'methods|methodology|materials\s+and\s+methods|experimental',
        'results': r'results',
        'discussion': r'discussion',
        'conclusion': r'conclusion|conclusions|concluding\s+remarks',
        'references': r'references|bibliography|literature\s+cited',
        'appendix': r'appendix|supplementary|supporting\s+information',
    }
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_sections: bool = True,
        handle_math: bool = True,
        handle_tables: bool = True,
        handle_figures: bool = True,
    ):
        """
        Initialize the scientific paper chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            respect_sections: Whether to preserve section boundaries
            handle_math: Whether to handle mathematical content specially
            handle_tables: Whether to handle tables specially
            handle_figures: Whether to handle figures specially
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sections = respect_sections
        self.handle_math = handle_math
        self.handle_tables = handle_tables
        self.handle_figures = handle_figures
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file with special handling for scientific content.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text with structural information preserved
        """
        doc = fitz.open(file_path)
        text = ""
        
        for page_num, page in enumerate(doc):
            # Extract plain text
            page_text = page.get_text()
            
            # Handle mathematical content if needed
            if self.handle_math:
                # This is a placeholder for more sophisticated math extraction
                # In a full implementation, we would use specialized tools for LaTeX extraction
                pass
                
            # Add page number reference for citation purposes
            text += f"[Page {page_num + 1}]\n{page_text}\n\n"
        
        return text
    
    def identify_sections(self, text: str) -> Dict[str, str]:
        """
        Identify and extract sections from the scientific paper.
        
        Args:
            text: The full text of the paper
            
        Returns:
            Dictionary mapping section names to section content
        """
        sections = {}
        current_section = 'preamble'
        sections[current_section] = ""
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if this line is a section header
            section_match = False
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if re.search(pattern, line.lower()):
                    current_section = section_name
                    sections[current_section] = line + '\n'
                    section_match = True
                    break
            
            if not section_match:
                # Add the line to the current section
                sections[current_section] += line + '\n'
        
        return sections
    
    def chunk_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a scientific paper and split it into semantically meaningful chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of chunks with metadata
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(file_path)
        
        if self.respect_sections:
            # Split by sections
            sections = self.identify_sections(text)
            chunks = []
            
            for section_name, section_content in sections.items():
                # Skip empty sections
                if not section_content.strip():
                    continue
                
                # Split each section into chunks
                section_chunks = self.splitter.split_text(section_content)
                
                # Add metadata to each chunk
                for i, chunk_text in enumerate(section_chunks):
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'source': file_path,
                            'section': section_name,
                            'chunk_index': i,
                            'total_chunks_in_section': len(section_chunks),
                        }
                    })
        else:
            # Simple chunking without respecting sections
            text_chunks = self.splitter.split_text(text)
            chunks = [
                {
                    'text': chunk,
                    'metadata': {
                        'source': file_path,
                        'chunk_index': i,
                        'total_chunks': len(text_chunks),
                    }
                }
                for i, chunk in enumerate(text_chunks)
            ]
        
        return chunks
    
    def process_multiple_papers(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple scientific papers.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            List of chunks from all papers with metadata
        """
        all_chunks = []
        
        for file_path in file_paths:
            paper_chunks = self.chunk_document(file_path)
            all_chunks.extend(paper_chunks)
        
        return all_chunks 