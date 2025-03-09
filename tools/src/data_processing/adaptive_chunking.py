"""
Adaptive Chunking

This module provides advanced chunking strategies that adapt to the content
based on semantic boundaries rather than using fixed-size chunks.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union, Callable, Iterator, Tuple
import numpy as np
from collections import deque

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    NLTK_AVAILABLE = False
    
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import other necessary components
from .text_chunkers import BaseChunker
from .document_loaders import Document

# Set up logging
logger = logging.getLogger(__name__)


class SemanticTextSplitter(BaseChunker):
    """
    Split text into chunks based on semantic boundaries using sentence embeddings.
    
    This splitter aims to keep semantically related content together in the same chunk,
    rather than breaking on arbitrary character counts. It uses embeddings to measure
    semantic similarity between sentences.
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        target_chunk_size: int = 1000,
        chunk_overlap: int = 100,
        threshold: float = 0.5,
        separator: str = "\n\n",
        hard_max_chunk_size: int = 2000,
        strip_whitespace: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize a semantic text splitter.
        
        Args:
            embedding_model: Model name for generating sentence embeddings
            target_chunk_size: Target size of each chunk in characters
            chunk_overlap: Minimum overlap between chunks in characters
            threshold: Similarity threshold for determining semantic boundaries
            separator: Separator to use when joining sentences
            hard_max_chunk_size: Hard maximum size for any chunk
            strip_whitespace: Whether to strip whitespace from chunks
            device: Device to use for model inference ('cpu', 'cuda', etc.)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The transformers package is required for SemanticTextSplitter. "
                "Install it with 'pip install transformers'"
            )
        
        if not NLTK_AVAILABLE:
            raise ImportError(
                "The nltk package is required for SemanticTextSplitter. "
                "Install it with 'pip install nltk'"
            )
        
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.threshold = threshold
        self.separator = separator
        self.hard_max_chunk_size = hard_max_chunk_size
        self.strip_whitespace = strip_whitespace
        
        # Set up the device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize the embedding model
        logger.info(f"Initializing embedding model: {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model).to(self.device)
        self.model.eval()
    
    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of sentences.
        
        Args:
            sentences: List of sentences to embed
            
        Returns:
            Array of sentence embeddings
        """
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 16
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use mean pooling to get sentence embeddings
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sentence_sums = torch.sum(token_embeddings * mask, 1)
                mask_sums = torch.clamp(mask.sum(1), min=1e-9)
                batch_embeddings = sentence_sums / mask_sums
                
                # Move to CPU and convert to numpy
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([])
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def _find_semantic_chunks(self, sentences: List[str], sentence_embeddings: np.ndarray) -> List[List[int]]:
        """
        Group sentences into semantic chunks.
        
        Args:
            sentences: List of sentences
            sentence_embeddings: Array of sentence embeddings
            
        Returns:
            List of lists where each inner list contains indices of sentences in a chunk
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = [0]  # Start with the first sentence
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed the hard max size
            if current_length + len(self.separator) + sentence_length > self.hard_max_chunk_size:
                # Start a new chunk
                chunks.append(current_chunk)
                current_chunk = [i]
                current_length = sentence_length
                continue
            
            # Calculate similarity between this sentence and the previous one
            similarity = self._calculate_similarity(
                sentence_embeddings[i], 
                sentence_embeddings[i-1]
            )
            
            # Check if we should start a new chunk based on semantic boundary
            if (similarity < self.threshold and 
                current_length >= self.target_chunk_size - self.chunk_overlap):
                chunks.append(current_chunk)
                current_chunk = [i]
                current_length = sentence_length
            # Or if we've reached the target chunk size
            elif current_length + len(self.separator) + sentence_length > self.target_chunk_size + self.chunk_overlap:
                chunks.append(current_chunk)
                
                # Find a good overlap point - go back a few sentences for overlap
                overlap_start = max(0, len(current_chunk) - max(1, int(self.chunk_overlap / 100)))
                current_chunk = current_chunk[overlap_start:]
                current_chunk.append(i)
                current_length = sum(len(sentences[j]) for j in current_chunk) + len(self.separator) * (len(current_chunk) - 1)
            else:
                # Add to the current chunk
                current_chunk.append(i)
                current_length += len(self.separator) + sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return [text] if text else []
        
        # Generate embeddings for all sentences
        sentence_embeddings = self._get_sentence_embeddings(sentences)
        
        # Group sentences into semantic chunks
        chunk_indices = self._find_semantic_chunks(sentences, sentence_embeddings)
        
        # Create chunks from sentence groups
        chunks = []
        for indices in chunk_indices:
            chunk = self.separator.join(sentences[i] for i in indices)
            if self.strip_whitespace:
                chunk = chunk.strip()
            chunks.append(chunk)
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantic chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            doc_chunks = self.split_text(doc.content)
            
            for i, chunk in enumerate(doc_chunks):
                # Create metadata for chunk
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["chunk_count"] = len(doc_chunks)
                chunk_metadata["is_semantic_chunk"] = True
                
                chunked_docs.append(Document(content=chunk, metadata=chunk_metadata))
        
        return chunked_docs


class HierarchicalTextSplitter(BaseChunker):
    """
    Split text hierarchically into chunks with awareness of document structure.
    
    This splitter creates chunks with awareness of hierarchy (document → section → paragraph)
    and maintains the hierarchical relationships in the metadata.
    """
    
    def __init__(
        self,
        section_pattern: str = r"(?<!\n\n)(\n#{1,5} .*?\n)",
        paragraph_pattern: str = r"\n\n+",
        max_section_length: int = 2000,
        max_paragraph_length: int = 500,
        overlap: int = 50,
        strip_whitespace: bool = True,
    ):
        """
        Initialize a hierarchical text splitter.
        
        Args:
            section_pattern: Regex pattern to identify sections
            paragraph_pattern: Regex pattern to identify paragraphs
            max_section_length: Maximum length of a section in characters
            max_paragraph_length: Maximum length of a paragraph in characters
            overlap: Overlap between chunks in characters
            strip_whitespace: Whether to strip whitespace from chunks
        """
        self.section_pattern = section_pattern
        self.paragraph_pattern = paragraph_pattern
        self.max_section_length = max_section_length
        self.max_paragraph_length = max_paragraph_length
        self.overlap = overlap
        self.strip_whitespace = strip_whitespace
    
    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into sections based on section_pattern.
        
        Args:
            text: Text to split
            
        Returns:
            List of section dictionaries with title and content
        """
        # Add a dummy section header if none exists to handle text without explicit sections
        if not re.search(self.section_pattern, "\n" + text):
            text = "# Document\n" + text
        
        # Split by section headers
        section_splits = re.split(self.section_pattern, "\n" + text)
        
        # Process splits into sections with titles
        sections = []
        current_title = "Document"
        
        for i, split in enumerate(section_splits):
            if i == 0 and not split.strip():
                # Skip empty first split
                continue
                
            if i % 2 == 1:  # This is a section header
                current_title = split.strip()
                # Extract the title from the header (remove # symbols)
                current_title = re.sub(r'^#+\s+', '', current_title)
            else:  # This is section content
                sections.append({
                    "title": current_title,
                    "content": split.strip(),
                    "level": len(re.match(r'^#+', current_title).group()) if re.match(r'^#+', current_title) else 1
                })
        
        return sections
    
    def _split_section_into_paragraphs(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a section into paragraphs.
        
        Args:
            section: Section dictionary with title and content
            
        Returns:
            List of paragraph dictionaries with section info and content
        """
        content = section["content"]
        
        # Split by paragraphs
        paragraphs = re.split(self.paragraph_pattern, content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Create paragraph objects
        paragraph_objects = []
        for i, paragraph in enumerate(paragraphs):
            paragraph_objects.append({
                "section_title": section["title"],
                "section_level": section["level"],
                "paragraph_index": i,
                "content": paragraph
            })
        
        return paragraph_objects
    
    def _chunk_paragraph(self, paragraph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a long paragraph into smaller chunks.
        
        Args:
            paragraph: Paragraph dictionary
            
        Returns:
            List of chunk dictionaries
        """
        content = paragraph["content"]
        
        # If paragraph is short enough, return as is
        if len(content) <= self.max_paragraph_length:
            return [paragraph]
        
        # Otherwise, split into sentences and chunk
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(content)
        else:
            # Fallback to simple sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > self.max_paragraph_length and current_chunk:
                # Create chunk
                chunk_obj = paragraph.copy()
                chunk_obj["content"] = current_chunk.strip() if self.strip_whitespace else current_chunk
                chunks.append(chunk_obj)
                
                # Start new chunk with overlap
                overlap_chars = min(self.overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_chars:] + " " + sentence
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_obj = paragraph.copy()
            chunk_obj["content"] = current_chunk.strip() if self.strip_whitespace else current_chunk
            chunks.append(chunk_obj)
        
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text hierarchically into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split into sections
        sections = self._split_into_sections(text)
        
        # Collect all chunks
        all_chunks = []
        
        for section in sections:
            # Check if section is short enough to be a single chunk
            if len(section["content"]) <= self.max_section_length:
                all_chunks.append(section["content"])
            else:
                # Split section into paragraphs
                paragraphs = self._split_section_into_paragraphs(section)
                
                # Split paragraphs into chunks if needed
                for paragraph in paragraphs:
                    paragraph_chunks = self._chunk_paragraph(paragraph)
                    all_chunks.extend([chunk["content"] for chunk in paragraph_chunks])
        
        return all_chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents hierarchically into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents with hierarchical metadata
        """
        chunked_docs = []
        
        for doc in documents:
            # Split into sections
            sections = self._split_into_sections(doc.content)
            
            # Process each section
            for section_idx, section in enumerate(sections):
                # Split section into paragraphs
                paragraphs = self._split_section_into_paragraphs(section)
                
                # Process each paragraph
                for paragraph_idx, paragraph in enumerate(paragraphs):
                    # Split paragraph into chunks if needed
                    paragraph_chunks = self._chunk_paragraph(paragraph)
                    
                    # Create documents for each chunk
                    for chunk_idx, chunk in enumerate(paragraph_chunks):
                        # Create metadata
                        chunk_metadata = doc.metadata.copy()
                        chunk_metadata["section_title"] = section["title"]
                        chunk_metadata["section_level"] = section["level"]
                        chunk_metadata["section_index"] = section_idx
                        chunk_metadata["paragraph_index"] = paragraph_idx
                        chunk_metadata["chunk_index"] = chunk_idx
                        chunk_metadata["total_sections"] = len(sections)
                        chunk_metadata["total_paragraphs_in_section"] = len(paragraphs)
                        chunk_metadata["total_chunks_in_paragraph"] = len(paragraph_chunks)
                        chunk_metadata["is_hierarchical_chunk"] = True
                        
                        chunked_docs.append(Document(
                            content=chunk["content"],
                            metadata=chunk_metadata
                        ))
        
        return chunked_docs


class TokenAwareChunker(BaseChunker):
    """
    Split text into chunks with awareness of token counts for specific models.
    
    This chunker is designed to optimize chunks for specific models and their token limits,
    ensuring that chunks don't exceed the model's context window.
    """
    
    def __init__(
        self,
        tokenizer_name_or_path: str = "gpt-3.5-turbo",
        target_tokens_per_chunk: int = 1000,
        overlap_tokens: int = 100,
        hard_max_tokens: int = 2000,
        separator: str = "\n\n",
        chunk_by: str = "paragraph",  # "paragraph", "sentence", or "character"
    ):
        """
        Initialize a token-aware chunker.
        
        Args:
            tokenizer_name_or_path: Name of the tokenizer to use (e.g., "gpt-3.5-turbo", "gpt-4", "claude-3-opus")
                or a path to a HuggingFace tokenizer
            target_tokens_per_chunk: Target number of tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            hard_max_tokens: Hard maximum number of tokens per chunk
            separator: Separator to use when joining text units
            chunk_by: How to split the text ("paragraph", "sentence", or "character")
        """
        self.target_tokens_per_chunk = target_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.hard_max_tokens = hard_max_tokens
        self.separator = separator
        self.chunk_by = chunk_by.lower()
        
        # Initialize the appropriate tokenizer
        if tokenizer_name_or_path.startswith("gpt-") or tokenizer_name_or_path.startswith("text-"):
            # Use tiktoken for OpenAI models
            try:
                import tiktoken
                
                # Map model names to encodings
                if tokenizer_name_or_path.startswith("gpt-4"):
                    self.tokenizer = tiktoken.encoding_for_model("gpt-4")
                elif tokenizer_name_or_path.startswith("gpt-3.5-turbo"):
                    self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
                else:
                    # Default to cl100k_base for newer models
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                
                self.tokenize = lambda text: self.tokenizer.encode(text)
            except ImportError:
                raise ImportError(
                    "The tiktoken package is required for OpenAI tokenizers. "
                    "Install it with 'pip install tiktoken'"
                )
        elif tokenizer_name_or_path.startswith("claude-"):
            # For Anthropic models, use a simple approximation
            # Claude uses a byte-pair encoding similar to GPT models
            # This is a rough approximation
            try:
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.tokenize = lambda text: self.tokenizer.encode(text)
            except ImportError:
                # Fallback to a very simple approximation
                self.tokenize = lambda text: text.split()
        else:
            # Try to use a HuggingFace tokenizer
            if TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
                self.tokenize = lambda text: self.tokenizer.encode(text)
            else:
                raise ImportError(
                    "The transformers package is required for custom tokenizers. "
                    "Install it with 'pip install transformers'"
                )
    
    def _split_into_units(self, text: str) -> List[str]:
        """
        Split text into units based on chunk_by setting.
        
        Args:
            text: Text to split
            
        Returns:
            List of text units
        """
        if self.chunk_by == "paragraph":
            units = text.split(self.separator)
            units = [unit.strip() for unit in units if unit.strip()]
            return units
        elif self.chunk_by == "sentence":
            if NLTK_AVAILABLE:
                return sent_tokenize(text)
            else:
                # Fallback to simple sentence splitting
                return re.split(r'(?<=[.!?])\s+', text)
        else:  # "character"
            # Split by character is just used for fallback
            # It's not efficient to split by character directly
            return [text]
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.tokenize(text))
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into token-optimized chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Split the text into units
        units = self._split_into_units(text)
        if not units:
            return []
        
        # If we're using character-level chunking
        if self.chunk_by == "character":
            return self._character_level_chunking(text)
        
        # Create chunks from units
        chunks = []
        current_chunk_units = []
        current_chunk_tokens = 0
        
        for unit in units:
            unit_tokens = self._count_tokens(unit)
            
            # If a single unit exceeds the hard max, handle it specially
            if unit_tokens > self.hard_max_tokens:
                if current_chunk_units:
                    chunks.append(self.separator.join(current_chunk_units))
                    current_chunk_units = []
                    current_chunk_tokens = 0
                
                # For very long units, fall back to character chunking
                unit_chunks = self._character_level_chunking(unit)
                chunks.extend(unit_chunks)
                continue
            
            # Check if adding this unit would exceed the target
            if current_chunk_tokens + unit_tokens > self.target_tokens_per_chunk and current_chunk_units:
                # Save the current chunk
                chunks.append(self.separator.join(current_chunk_units))
                
                # Start a new chunk with overlap
                overlap_start_idx = max(0, len(current_chunk_units) - 1)
                current_chunk_units = current_chunk_units[overlap_start_idx:]
                current_chunk_tokens = sum(self._count_tokens(u) for u in current_chunk_units)
                
                # Add separator tokens
                if current_chunk_units:
                    current_chunk_tokens += self._count_tokens(self.separator) * (len(current_chunk_units) - 1)
            
            # Add the unit to the current chunk
            current_chunk_units.append(unit)
            current_chunk_tokens = current_chunk_tokens + unit_tokens
            if current_chunk_units:
                current_chunk_tokens += self._count_tokens(self.separator) * (len(current_chunk_units) - 1)
        
        # Don't forget the last chunk
        if current_chunk_units:
            chunks.append(self.separator.join(current_chunk_units))
        
        return chunks
    
    def _character_level_chunking(self, text: str) -> List[str]:
        """
        Perform character-level chunking for very long text units.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Initialize chunks
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        # Calculate approximate characters per token for this text
        total_tokens = self._count_tokens(text)
        chars_per_token = len(text) / max(1, total_tokens)
        
        # Approximate chunk size in characters
        target_chars = int(self.target_tokens_per_chunk * chars_per_token)
        overlap_chars = int(self.overlap_tokens * chars_per_token)
        
        # Create chunks
        for i in range(0, len(text), target_chars):
            # Calculate end position with overlap
            end_pos = min(i + target_chars, len(text))
            
            # Extract chunk
            chunk = text[i:end_pos]
            
            # Add to chunks
            chunks.append(chunk)
            
            # Add overlap for next chunk
            if end_pos < len(text) and overlap_chars > 0:
                i -= overlap_chars
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into token-optimized chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            doc_chunks = self.split_text(doc.content)
            
            for i, chunk in enumerate(doc_chunks):
                # Get token count for metadata
                token_count = self._count_tokens(chunk)
                
                # Create metadata for chunk
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["chunk_count"] = len(doc_chunks)
                chunk_metadata["token_count"] = token_count
                chunk_metadata["is_token_optimized"] = True
                
                chunked_docs.append(Document(content=chunk, metadata=chunk_metadata))
        
        return chunked_docs 