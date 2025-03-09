"""
Medical Document Chunker

This module provides medical-aware chunking functionality for medical documents.
It splits documents based on logical sections commonly found in medical literature
(abstract, introduction, methods, results, discussion, conclusion) rather than arbitrary token counts.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain.docstore.document import Document


class MedicalChunker:
    """
    A chunker for medical documents that understands medical document structure.
    """
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 2000):
        """
        Initialize the medical chunker.
        
        Args:
            min_chunk_size: Minimum chunk size in characters.
            max_chunk_size: Maximum chunk size in characters.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Common section headers in medical literature
        self.section_patterns = [
            # Common medical paper sections
            r'^(Abstract|ABSTRACT):?',
            r'^(Introduction|INTRODUCTION):?',
            r'^(Background|BACKGROUND):?',
            r'^(Methods|METHODS|Materials and Methods|MATERIALS AND METHODS):?',
            r'^(Results|RESULTS):?',
            r'^(Discussion|DISCUSSION):?',
            r'^(Conclusion|CONCLUSION|Conclusions|CONCLUSIONS):?',
            r'^(References|REFERENCES):?',
            r'^(Acknowledgements|ACKNOWLEDGEMENTS):?',
            
            # Clinical document sections
            r'^(Patient Information|PATIENT INFORMATION):?',
            r'^(Chief Complaint|CHIEF COMPLAINT):?',
            r'^(History of Present Illness|HISTORY OF PRESENT ILLNESS|HPI):?',
            r'^(Past Medical History|PAST MEDICAL HISTORY|PMH):?',
            r'^(Social History|SOCIAL HISTORY):?',
            r'^(Family History|FAMILY HISTORY):?',
            r'^(Medications|MEDICATIONS|Current Medications|CURRENT MEDICATIONS):?',
            r'^(Allergies|ALLERGIES):?',
            r'^(Review of Systems|REVIEW OF SYSTEMS|ROS):?',
            r'^(Physical Examination|PHYSICAL EXAMINATION|Physical Exam|PHYSICAL EXAM):?',
            r'^(Assessment|ASSESSMENT):?',
            r'^(Plan|PLAN|Treatment Plan|TREATMENT PLAN):?',
            r'^(Diagnosis|DIAGNOSIS|Diagnoses|DIAGNOSES):?',
            r'^(Follow-up|FOLLOW-UP):?',
            
            # Additional research paper sections
            r'^(Objectives|OBJECTIVES):?',
            r'^(Study Design|STUDY DESIGN):?',
            r'^(Setting|SETTING):?',
            r'^(Participants|PARTICIPANTS|Subjects|SUBJECTS|Patients|PATIENTS):?',
            r'^(Interventions|INTERVENTIONS):?',
            r'^(Main Outcome Measures|MAIN OUTCOME MEASURES):?',
            r'^(Statistical Analysis|STATISTICAL ANALYSIS):?',
            r'^(Ethics|ETHICS):?',
            r'^(Limitations|LIMITATIONS):?',
            r'^(Future Directions|FUTURE DIRECTIONS):?',
            r'^(Conflict of Interest|CONFLICT OF INTEREST):?',
            r'^(Funding|FUNDING):?',
            
            # Numbered section headers
            r'^\d+\.\s+([\w\s]+)\s*$',  # Like "1. Introduction"
            r'^\d+\.\d+\.\s+([\w\s]+)\s*$',  # Like "1.1. Study Design"
        ]
        
        # Compile all patterns
        self.section_patterns = [re.compile(pattern, re.MULTILINE) for pattern in self.section_patterns]

    def split(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split a medical document into chunks based on logical sections.
        
        Args:
            text: The medical document text.
            metadata: Optional metadata for the document.
            
        Returns:
            List of Document objects representing the chunks.
        """
        if metadata is None:
            metadata = {}
            
        chunks = []
        
        # Detect document type
        if self._is_research_paper(text):
            chunks = self._split_research_paper(text, metadata)
        elif self._is_clinical_note(text):
            chunks = self._split_clinical_note(text, metadata)
        else:
            # Fallback to a generic section-based splitting
            chunks = self._split_by_sections(text, metadata)
            
        # If no chunks were created (no clear sections found), fall back to simple chunking
        if not chunks:
            chunks = self._fallback_chunking(text, metadata)
            
        return chunks
    
    def _is_research_paper(self, text: str) -> bool:
        """
        Detect if the document is a research paper based on its content.
        
        Args:
            text: The document text.
            
        Returns:
            Boolean indicating if it's a research paper.
        """
        # Check for common patterns in research papers
        research_indicators = [
            r'\babstract\b', r'\bintroduction\b', r'\bmethods\b', r'\bresults\b', 
            r'\bdiscussion\b', r'\bconclusion\b', r'\breferences\b', 
            r'\bcite\b', r'\bet al\b', r'\bp\s*[<>=]\s*0\.\d+\b'
        ]
        
        # Count how many indicators are found
        count = sum(1 for pattern in research_indicators if re.search(pattern, text.lower()))
        
        # If at least 3 indicators are found, consider it a research paper
        return count >= 3
    
    def _is_clinical_note(self, text: str) -> bool:
        """
        Detect if the document is a clinical note based on its content.
        
        Args:
            text: The document text.
            
        Returns:
            Boolean indicating if it's a clinical note.
        """
        # Check for common patterns in clinical notes
        clinical_indicators = [
            r'\bpatient\b', r'\bchief complaint\b', r'\bhistory of present illness\b',
            r'\bpast medical history\b', r'\bmedications\b', r'\ballergies\b',
            r'\breview of systems\b', r'\bphysical exam\b', r'\bassessment\b', r'\bplan\b',
            r'\bdiagnosis\b', r'\bvital signs\b', r'\bfollowup\b', r'\bfollow-up\b'
        ]
        
        # Count how many indicators are found
        count = sum(1 for pattern in clinical_indicators if re.search(pattern, text.lower()))
        
        # If at least 3 indicators are found, consider it a clinical note
        return count >= 3
    
    def _split_research_paper(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split a research paper into logical sections.
        
        Args:
            text: The research paper text.
            metadata: Metadata for the document.
            
        Returns:
            List of Document objects.
        """
        sections = self._identify_sections(text)
        
        chunks = []
        
        # Process each section
        for i, (section_name, section_text) in enumerate(sections):
            # Update metadata with section information
            section_metadata = metadata.copy()
            section_metadata.update({
                "chunk_type": "section",
                "section_name": section_name,
                "section_index": i,
                "document_type": "research_paper"
            })
            
            # Create chunk
            chunk = Document(page_content=section_text, metadata=section_metadata)
            chunks.append(chunk)
            
            # If the section is too large, split it further
            if len(section_text) > self.max_chunk_size:
                # Split into paragraphs
                paragraphs = self._split_into_paragraphs(section_text)
                
                # Process each paragraph as a separate chunk
                for j, paragraph in enumerate(paragraphs):
                    if len(paragraph) < self.min_chunk_size:
                        continue
                        
                    paragraph_metadata = section_metadata.copy()
                    paragraph_metadata.update({
                        "chunk_type": "paragraph",
                        "section_name": section_name,
                        "paragraph_index": j
                    })
                    
                    paragraph_chunk = Document(page_content=paragraph, metadata=paragraph_metadata)
                    chunks.append(paragraph_chunk)
                    
        return chunks
    
    def _split_clinical_note(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split a clinical note into logical sections.
        
        Args:
            text: The clinical note text.
            metadata: Metadata for the document.
            
        Returns:
            List of Document objects.
        """
        sections = self._identify_sections(text)
        
        chunks = []
        
        # Process each section
        for i, (section_name, section_text) in enumerate(sections):
            # Update metadata with section information
            section_metadata = metadata.copy()
            section_metadata.update({
                "chunk_type": "section",
                "section_name": section_name,
                "section_index": i,
                "document_type": "clinical_note"
            })
            
            # Create chunk
            chunk = Document(page_content=section_text, metadata=section_metadata)
            chunks.append(chunk)
            
            # Clinical notes typically have shorter sections, so we may not need to split further
            # But if a section is very long, split it into paragraphs
            if len(section_text) > self.max_chunk_size:
                paragraphs = self._split_into_paragraphs(section_text)
                
                for j, paragraph in enumerate(paragraphs):
                    if len(paragraph) < self.min_chunk_size:
                        continue
                        
                    paragraph_metadata = section_metadata.copy()
                    paragraph_metadata.update({
                        "chunk_type": "paragraph",
                        "section_name": section_name,
                        "paragraph_index": j
                    })
                    
                    paragraph_chunk = Document(page_content=paragraph, metadata=paragraph_metadata)
                    chunks.append(paragraph_chunk)
                    
        return chunks
    
    def _split_by_sections(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Generic section-based splitting for medical documents.
        
        Args:
            text: The document text.
            metadata: Metadata for the document.
            
        Returns:
            List of Document objects.
        """
        sections = self._identify_sections(text)
        
        chunks = []
        
        # Process each section
        for i, (section_name, section_text) in enumerate(sections):
            # Update metadata with section information
            section_metadata = metadata.copy()
            section_metadata.update({
                "chunk_type": "section",
                "section_name": section_name,
                "section_index": i
            })
            
            # Create chunk
            chunk = Document(page_content=section_text, metadata=section_metadata)
            chunks.append(chunk)
            
            # If the section is too large, split it further into paragraphs
            if len(section_text) > self.max_chunk_size:
                paragraphs = self._split_into_paragraphs(section_text)
                
                for j, paragraph in enumerate(paragraphs):
                    if len(paragraph) < self.min_chunk_size:
                        continue
                        
                    paragraph_metadata = section_metadata.copy()
                    paragraph_metadata.update({
                        "chunk_type": "paragraph",
                        "paragraph_index": j
                    })
                    
                    paragraph_chunk = Document(page_content=paragraph, metadata=paragraph_metadata)
                    chunks.append(paragraph_chunk)
                    
        return chunks
    
    def _identify_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Identify sections in the document based on section headers.
        
        Args:
            text: The document text.
            
        Returns:
            List of (section_name, section_text) tuples.
        """
        # Find all potential section headers
        section_boundaries = []
        
        for pattern in self.section_patterns:
            for match in pattern.finditer(text):
                section_name = match.group(1) if match.lastindex else match.group(0).strip()
                start_pos = match.start()
                section_boundaries.append((start_pos, section_name))
        
        # Sort by position in the document
        section_boundaries.sort(key=lambda x: x[0])
        
        # If no sections were found, return the whole document as one section
        if not section_boundaries:
            return [("Document", text)]
        
        # Extract section text
        sections = []
        for i, (start_pos, section_name) in enumerate(section_boundaries):
            # Determine end position (next section start or end of document)
            end_pos = section_boundaries[i+1][0] if i+1 < len(section_boundaries) else len(text)
            
            # Extract section text
            section_text = text[start_pos:end_pos].strip()
            
            # Extract only the content (remove the header)
            header_end = section_text.find('\n')
            if header_end != -1:
                section_text = section_text[header_end:].strip()
            
            sections.append((section_name, section_text))
        
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on newlines.
        
        Args:
            text: The text to split.
            
        Returns:
            List of paragraph strings.
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Handle paragraphs that are still too long
        result = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.max_chunk_size:
                result.append(paragraph)
            else:
                # Split by single newlines
                sentences = paragraph.split('\n')
                
                current_chunk = []
                current_size = 0
                
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    if current_size + sentence_size > self.max_chunk_size and current_chunk:
                        # Save the current chunk
                        result.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    
                    current_chunk.append(sentence)
                    current_size += sentence_size
                
                # Add the last chunk if it's not empty
                if current_chunk:
                    result.append('\n'.join(current_chunk))
        
        return result
    
    def _fallback_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Fallback chunking method when no clear sections are identified.
        
        Args:
            text: The text to chunk.
            metadata: Metadata for the document.
            
        Returns:
            List of Document objects.
        """
        chunks = []
        
        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = []
        current_size = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed the max size, save the current chunk
            if current_size + paragraph_size > self.max_chunk_size and current_chunk:
                # Create a document from the current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_type": "fallback",
                    "chunk_index": len(chunks)
                })
                
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
                
                # Reset for the next chunk
                current_chunk = []
                current_size = 0
            
            # Add paragraph to the current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_type": "fallback",
                "chunk_index": len(chunks)
            })
            
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
        
        return chunks


def process_medical_document(file_path: str, min_chunk_size: int = 100, max_chunk_size: int = 2000) -> List[Document]:
    """
    Process a medical document file into chunks.
    
    Args:
        file_path: The path to the document file.
        min_chunk_size: Minimum chunk size in characters.
        max_chunk_size: Maximum chunk size in characters.
        
    Returns:
        List of Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Create metadata
    metadata = {
        "source": os.path.basename(file_path),
        "file_path": file_path,
        "file_type": os.path.splitext(file_path)[1].lower()
    }
    
    # Create a medical chunker and process the document
    chunker = MedicalChunker(min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)
    return chunker.split(text, metadata)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process a medical document into chunks.")
    parser.add_argument("--file", type=str, required=True, help="Path to the medical document file.")
    parser.add_argument("--min_chunk_size", type=int, default=100, help="Minimum chunk size in characters.")
    parser.add_argument("--max_chunk_size", type=int, default=2000, help="Maximum chunk size in characters.")
    args = parser.parse_args()
    
    chunks = process_medical_document(args.file, args.min_chunk_size, args.max_chunk_size)
    
    print(f"Processing {args.file}...")
    print(f"Split into {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Type: {chunk.metadata.get('chunk_type', 'unknown')}")
        if 'section_name' in chunk.metadata:
            print(f"Section: {chunk.metadata['section_name']}")
        
        # Print a preview of the content
        content_preview = chunk.page_content[:150] + "..." if len(chunk.page_content) > 150 else chunk.page_content
        print(f"\n{content_preview}") 