"""
Legal Document Chunker for RAG

This module provides intelligent chunking of legal documents based on the document structure
and legal content. It can identify different types of legal documents (statutes, cases, contracts)
and split them according to their logical sections.
"""

import re
import os
from typing import List, Dict, Any, Tuple, Optional
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)

class LegalChunker:
    """
    Specialized chunker for legal documents that can identify and split documents
    based on their legal structure and content.
    """

    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 2000):
        """
        Initialize the LegalChunker.

        Args:
            min_chunk_size (int): Minimum chunk size in characters
            max_chunk_size (int): Maximum chunk size in characters
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Patterns for different legal document types
        self.statute_patterns = [
            r"(?i)chapter\s+\d+",
            r"(?i)section\s+\d+",
            r"(?i)§\s*\d+",
            r"(?i)title\s+\d+",
            r"(?i)article\s+\d+"
        ]
        
        self.case_patterns = [
            r"(?i)v\.\s",
            r"(?i)plaintiff",
            r"(?i)defendant",
            r"(?i)court of appeals",
            r"(?i)supreme court",
            r"(?i)district court",
            r"(?i)opinion of",
            r"(?i)held:",
            r"(?i)\\d+ F\\.\\d+",    # Federal Reporter citation
            r"(?i)\\d+ U\\.S\\.",    # US Reports citation
            r"(?i)\\d+ S\\.Ct\\."    # Supreme Court Reporter citation
        ]
        
        self.contract_patterns = [
            r"(?i)agreement",
            r"(?i)between.*and",
            r"(?i)parties",
            r"(?i)whereas",
            r"(?i)term of",
            r"(?i)section \d+(\.\d+)*",
            r"(?i)article \d+(\.\d+)*",
            r"(?i)in witness whereof",
            r"(?i)signatures"
        ]
        
        # Patterns for citations
        self.citation_patterns = [
            r"\d+ U\.S\. \d+",                 # US Reports
            r"\d+ S\.Ct\. \d+",                # Supreme Court Reporter
            r"\d+ F\.\d[d] \d+",               # Federal Reporter
            r"\d+ F\. Supp\. \d+",             # Federal Supplement
            r"\d+ [A-Za-z]\.[A-Za-z]\. \d+",   # State/Regional Reporter
            r"(?i)[A-Za-z]+ v\. [A-Za-z]+,\s+\d+.*\d+"  # Case name with citation
        ]

    def split(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split a legal document into chunks based on its structure and content.

        Args:
            text (str): The text of the legal document
            metadata (Dict[str, Any], optional): Metadata associated with the document

        Returns:
            List[Document]: A list of document chunks with appropriate metadata
        """
        if metadata is None:
            metadata = {}
            
        # Determine document type
        if self._is_statute(text):
            metadata["doc_type"] = "statute"
            return self._split_statute(text, metadata)
        elif self._is_case(text):
            metadata["doc_type"] = "case_law"
            return self._split_case(text, metadata)
        elif self._is_contract(text):
            metadata["doc_type"] = "contract"
            return self._split_contract(text, metadata)
        else:
            # Use section-based splitting for general legal documents
            metadata["doc_type"] = "legal_document"
            return self._split_by_sections(text, metadata)

    def _is_statute(self, text: str) -> bool:
        """
        Determine if the text is likely a statute or regulation.

        Args:
            text (str): The text to analyze

        Returns:
            bool: True if the text appears to be a statute
        """
        # Check for patterns common in statutes
        statute_score = 0
        
        for pattern in self.statute_patterns:
            if re.search(pattern, text):
                statute_score += 1
                
        # Check for section numbering patterns typical in statutes
        if re.search(r"(?i)(\(a\)|\(1\)|\(i\))", text):
            statute_score += 1
            
        # Check for common statute language
        if any(term in text.lower() for term in ["shall", "pursuant to", "notwithstanding", 
                                              "hereinafter", "provision", "subsection"]):
            statute_score += 1
            
        return statute_score >= 3

    def _is_case(self, text: str) -> bool:
        """
        Determine if the text is likely a legal case.

        Args:
            text (str): The text to analyze

        Returns:
            bool: True if the text appears to be a legal case
        """
        # Check for patterns common in case law
        case_score = 0
        
        for pattern in self.case_patterns:
            if re.search(pattern, text):
                case_score += 1
                
        # Check for case-specific sections
        case_sections = ["facts", "procedural history", "issue", "holding", "reasoning"]
        if any(section in text.lower() for section in case_sections):
            case_score += 1
            
        # Check for citations to other cases
        if any(re.search(pattern, text) for pattern in self.citation_patterns):
            case_score += 2
            
        return case_score >= 3

    def _is_contract(self, text: str) -> bool:
        """
        Determine if the text is likely a legal contract.

        Args:
            text (str): The text to analyze

        Returns:
            bool: True if the text appears to be a contract
        """
        # Check for patterns common in contracts
        contract_score = 0
        
        for pattern in self.contract_patterns:
            if re.search(pattern, text):
                contract_score += 1
                
        # Check for contract-specific terms
        contract_terms = ["terms and conditions", "payment", "termination", "breach", 
                         "warranty", "liability", "intellectual property"]
        if any(term in text.lower() for term in contract_terms):
            contract_score += 1
            
        return contract_score >= 3

    def _split_statute(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split a statute into logical chunks based on sections.

        Args:
            text (str): The statute text
            metadata (Dict[str, Any]): Document metadata

        Returns:
            List[Document]: A list of document chunks
        """
        # Look for patterns like "Section 123", "§ 123", etc.
        section_pattern = r"(?i)(section\s+\d+|\§\s*\d+|article\s+\d+|title\s+\d+|chapter\s+\d+)"
        sections = re.split(section_pattern, text)
        
        if len(sections) <= 2:  # No clear sections found
            return self._fallback_chunking(text, metadata)
            
        documents = []
        for i in range(1, len(sections), 2):
            if i < len(sections) - 1:
                section_header = sections[i].strip()
                section_content = sections[i+1].strip()
                
                if len(section_content) < self.min_chunk_size:
                    continue
                    
                section_metadata = metadata.copy()
                section_metadata["section"] = section_header
                
                if len(section_content) > self.max_chunk_size:
                    # Further split large sections by paragraphs
                    paragraphs = self._split_into_paragraphs(section_content)
                    paragraph_chunks = []
                    current_chunk = ""
                    
                    for paragraph in paragraphs:
                        if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                            paragraph_metadata = section_metadata.copy()
                            paragraph_metadata["chunk_type"] = "paragraph_group"
                            paragraph_documents = Document(page_content=current_chunk, metadata=paragraph_metadata)
                            documents.append(paragraph_documents)
                            current_chunk = paragraph
                        else:
                            if current_chunk:
                                current_chunk += "\n\n"
                            current_chunk += paragraph
                    
                    if current_chunk:
                        paragraph_metadata = section_metadata.copy()
                        paragraph_metadata["chunk_type"] = "paragraph_group"
                        paragraph_documents = Document(page_content=current_chunk, metadata=paragraph_metadata)
                        documents.append(paragraph_documents)
                else:
                    section_metadata["chunk_type"] = "section"
                    documents.append(Document(page_content=section_content, metadata=section_metadata))
                    
        if not documents:  # Fallback if no valid documents were created
            return self._fallback_chunking(text, metadata)
            
        return documents

    def _split_case(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split a legal case into logical chunks based on case sections.

        Args:
            text (str): The case text
            metadata (Dict[str, Any]): Document metadata

        Returns:
            List[Document]: A list of document chunks
        """
        # Try to extract case citation and name
        case_citation_match = re.search(r"(?i)([A-Za-z\s]+v\.\s+[A-Za-z\s]+),?\s+(\d+\s+[A-Za-z\.]+\s+\d+)", text)
        if case_citation_match:
            metadata["case_name"] = case_citation_match.group(1).strip()
            metadata["citation"] = case_citation_match.group(2).strip()
            
        # Identify case sections
        section_patterns = [
            r"(?i)FACTS|BACKGROUND",
            r"(?i)PROCEDURAL HISTORY",
            r"(?i)ISSUE|QUESTION PRESENTED",
            r"(?i)HOLDING",
            r"(?i)REASONING|ANALYSIS|DISCUSSION",
            r"(?i)CONCLUSION|DISPOSITION"
        ]
        
        # Try to split by sections
        found_sections = []
        for pattern in section_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                found_sections.append((match.start(), match.group()))
                
        found_sections.sort()
        
        if len(found_sections) < 2:  # Not enough sections found
            # Try to split by paragraphs or use fallback
            paragraphs = self._split_into_paragraphs(text)
            if len(paragraphs) > 1:
                return self._chunk_paragraphs(paragraphs, metadata)
            else:
                return self._fallback_chunking(text, metadata)
                
        # Create chunks based on found sections
        documents = []
        for i in range(len(found_sections)):
            start_idx = found_sections[i][0]
            section_name = found_sections[i][1]
            
            # Determine end of this section
            if i < len(found_sections) - 1:
                end_idx = found_sections[i+1][0]
            else:
                end_idx = len(text)
                
            section_content = text[start_idx:end_idx].strip()
            if len(section_content) < self.min_chunk_size:
                continue
                
            section_metadata = metadata.copy()
            section_metadata["section"] = section_name
            section_metadata["chunk_type"] = "case_section"
            
            if len(section_content) > self.max_chunk_size:
                # Further split large sections
                sub_paragraphs = self._split_into_paragraphs(section_content)
                for sub_chunks in self._chunk_paragraphs(sub_paragraphs, section_metadata):
                    documents.append(sub_chunks)
            else:
                documents.append(Document(page_content=section_content, metadata=section_metadata))
                
        if not documents:  # Fallback if no valid documents were created
            return self._fallback_chunking(text, metadata)
            
        return documents

    def _split_contract(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split a contract into logical chunks based on sections and clauses.

        Args:
            text (str): The contract text
            metadata (Dict[str, Any]): Document metadata

        Returns:
            List[Document]: A list of document chunks
        """
        # Look for article or section patterns in contracts
        section_patterns = [
            r"(?i)ARTICLE\s+[IVX\d]+[\.\s]",
            r"(?i)SECTION\s+\d+[\.\s]",
            r"(?i)\d+\.\s+[A-Z]"
        ]
        
        # Try to get parties to the contract
        parties_match = re.search(r"(?i)between\s+([^,]+)(?:,\s+[^,]+)*\s+and\s+([^,]+)", text)
        if parties_match:
            metadata["party1"] = parties_match.group(1).strip()
            metadata["party2"] = parties_match.group(2).strip()
            
        # Try different section patterns
        for pattern in section_patterns:
            sections = re.split(pattern, text)
            if len(sections) > 2:  # Found sections with this pattern
                documents = []
                
                for i in range(1, len(sections)):
                    section_content = sections[i].strip()
                    if len(section_content) < self.min_chunk_size:
                        continue
                        
                    # Try to get section title from first line
                    lines = section_content.split('\n', 1)
                    section_title = lines[0].strip()
                    if len(lines) > 1:
                        section_content = lines[1].strip()
                        
                    section_metadata = metadata.copy()
                    if i < len(sections) - 1:  # Pattern before this section
                        matches = re.finditer(pattern, text)
                        match_idx = 0
                        for match in matches:
                            if match_idx == i - 1:
                                section_metadata["section"] = match.group().strip()
                                break
                            match_idx += 1
                    else:
                        section_metadata["section"] = "Final Section"
                        
                    section_metadata["section_title"] = section_title
                    section_metadata["chunk_type"] = "contract_section"
                    
                    if len(section_content) > self.max_chunk_size:
                        # Further split large sections
                        paragraphs = self._split_into_paragraphs(section_content)
                        current_chunk = ""
                        for paragraph in paragraphs:
                            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                                chunk_metadata = section_metadata.copy()
                                documents.append(Document(page_content=current_chunk, metadata=chunk_metadata))
                                current_chunk = paragraph
                            else:
                                if current_chunk:
                                    current_chunk += "\n\n"
                                current_chunk += paragraph
                                
                        if current_chunk:
                            chunk_metadata = section_metadata.copy()
                            documents.append(Document(page_content=current_chunk, metadata=chunk_metadata))
                    else:
                        documents.append(Document(page_content=section_content, metadata=section_metadata))
                        
                if documents:
                    return documents
                    
        # If no sections were found, try to split by paragraphs
        paragraphs = self._split_into_paragraphs(text)
        if len(paragraphs) > 1:
            return self._chunk_paragraphs(paragraphs, metadata)
        else:
            return self._fallback_chunking(text, metadata)

    def _split_by_sections(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split a general legal document by identified sections.

        Args:
            text (str): The document text
            metadata (Dict[str, Any]): Document metadata

        Returns:
            List[Document]: A list of document chunks
        """
        # Try to identify sections in the document
        sections = self._identify_sections(text)
        
        if not sections:
            # No clear sections found, fall back to paragraph splitting
            paragraphs = self._split_into_paragraphs(text)
            if len(paragraphs) > 1:
                return self._chunk_paragraphs(paragraphs, metadata)
            else:
                return self._fallback_chunking(text, metadata)
                
        # Process each section
        documents = []
        for section_title, section_content in sections:
            if len(section_content.strip()) < self.min_chunk_size:
                continue
                
            section_metadata = metadata.copy()
            section_metadata["section"] = section_title
            section_metadata["chunk_type"] = "legal_section"
            
            if len(section_content) > self.max_chunk_size:
                # Further split large sections
                paragraphs = self._split_into_paragraphs(section_content)
                for chunk in self._chunk_paragraphs(paragraphs, section_metadata):
                    documents.append(chunk)
            else:
                documents.append(Document(page_content=section_content, metadata=section_metadata))
                
        if not documents:  # Fallback if no valid documents were created
            return self._fallback_chunking(text, metadata)
            
        return documents

    def _identify_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Identify sections in a legal document.

        Args:
            text (str): The document text

        Returns:
            List[Tuple[str, str]]: A list of (section_title, section_content) tuples
        """
        # Common section header patterns in legal documents
        section_patterns = [
            r"(?i)^\s*([IVX]+\.\s+.+)$",  # Roman numerals with titles
            r"(?i)^\s*(\d+\.\s+[A-Z].+)$",  # Numbered sections with titles
            r"(?i)^\s*([A-Z][A-Z\s]+)$"      # ALL CAPS titles
        ]
        
        # Try to find section headers
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = ""
        
        for line in lines:
            matched = False
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    # Found a new section
                    if current_section is not None:
                        sections.append((current_section, current_content))
                    current_section = match.group(1).strip()
                    current_content = ""
                    matched = True
                    break
                    
            if not matched and current_section is not None:
                current_content += line + "\n"
                
        # Add the last section
        if current_section is not None:
            sections.append((current_section, current_content))
            
        return sections

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.

        Args:
            text (str): The text to split

        Returns:
            List[str]: A list of paragraphs
        """
        # Split by double newlines, which typically separate paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _chunk_paragraphs(self, paragraphs: List[str], metadata: Dict[str, Any]) -> List[Document]:
        """
        Chunk a list of paragraphs into appropriately sized documents.

        Args:
            paragraphs (List[str]): The paragraphs to chunk
            metadata (Dict[str, Any]): Document metadata

        Returns:
            List[Document]: A list of document chunks
        """
        documents = []
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
                
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed max size, create a new chunk
            if current_size + paragraph_size > self.max_chunk_size and current_chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_type"] = "paragraph_group"
                documents.append(Document(page_content=current_chunk, metadata=chunk_metadata))
                current_chunk = paragraph
                current_size = paragraph_size
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                    current_size += 2
                current_chunk += paragraph
                current_size += paragraph_size
                
        # Add the last chunk if it exists
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_type"] = "paragraph_group"
            documents.append(Document(page_content=current_chunk, metadata=chunk_metadata))
            
        return documents

    def _fallback_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Fallback method when other chunking strategies fail.

        Args:
            text (str): The text to chunk
            metadata (Dict[str, Any]): Document metadata

        Returns:
            List[Document]: A list of document chunks
        """
        # Use LangChain's text splitter as a fallback
        metadata = metadata.copy()
        metadata["chunk_type"] = "fallback"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]


def process_legal_document(file_path: str, min_chunk_size: int = 100, max_chunk_size: int = 2000) -> List[Document]:
    """
    Process a legal document file and return chunked documents.

    Args:
        file_path (str): Path to the document file
        min_chunk_size (int): Minimum chunk size
        max_chunk_size (int): Maximum chunk size

    Returns:
        List[Document]: A list of document chunks with metadata
    """
    # Get file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # Basic metadata from the file
    metadata = {
        "source": file_path,
        "filename": os.path.basename(file_path)
    }
    
    # Load the document based on file type
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
    elif ext == ".txt":
        loader = TextLoader(file_path)
        docs = loader.load()
        text = docs[0].page_content
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        text = docs[0].page_content
    else:
        # Try the unstructured loader for other formats
        try:
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load()
            text = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            raise ValueError(f"Could not load document {file_path}: {str(e)}")
            
    # Chunk the text
    chunker = LegalChunker(min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)
    return chunker.split(text, metadata) 