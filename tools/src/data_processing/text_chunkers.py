"""
Text Chunkers

This module provides various text chunking strategies for splitting documents
into smaller pieces for embedding and retrieval.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable, Iterator, Tuple
import logging

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

# Set up logging
logger = logging.getLogger(__name__)

# Import Document class from document_loaders
from .document_loaders import Document


class BaseChunker(ABC):
    """
    Abstract base class for text chunkers.
    """
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        pass
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
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
                
                chunked_docs.append(Document(content=chunk, metadata=chunk_metadata))
        
        return chunked_docs


class CharacterTextSplitter(BaseChunker):
    """
    Split text into chunks based on a maximum character length.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        strip_whitespace: bool = True,
    ):
        """
        Initialize a character text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Amount of overlap between chunks
            separator: Separator to split on
            strip_whitespace: Whether to strip whitespace from chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.strip_whitespace = strip_whitespace
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on character length.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            text = text.strip()
        
        # Initialize variables
        chunks = []
        current_chunk = ""
        
        # Split the text on the separator
        splits = text.split(self.separator)
        
        for split in splits:
            # If adding this split would exceed the chunk size, start a new chunk
            if len(current_chunk) + len(split) + len(self.separator) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Include overlap from end of previous chunk
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + self.separator + split
            else:
                # Add the split to the current chunk
                if current_chunk:
                    current_chunk += self.separator + split
                else:
                    current_chunk = split
            
            # If the current chunk is now larger than the chunk size, add it and start a new one
            while len(current_chunk) > self.chunk_size:
                chunk_to_add = current_chunk[:self.chunk_size]
                chunks.append(chunk_to_add)
                # Include overlap from end of previous chunk
                current_chunk = current_chunk[self.chunk_size - self.chunk_overlap:]
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            chunks = [chunk.strip() for chunk in chunks]
        
        return chunks


class RecursiveCharacterTextSplitter(BaseChunker):
    """
    Split text recursively using a list of separators.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        strip_whitespace: bool = True,
    ):
        """
        Initialize a recursive character text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Amount of overlap between chunks
            separators: List of separators to use, in order of priority
            strip_whitespace: Whether to strip whitespace from chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.strip_whitespace = strip_whitespace
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text recursively using a list of separators.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            text = text.strip()
        
        # If the text is already small enough, return it as is
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator in order
        for separator in self.separators:
            if separator == "":
                # If we've reached the empty separator, split by character
                return self._split_by_character(text)
            
            if separator in text:
                # Split by this separator
                chunks = self._split_by_separator(text, separator)
                if chunks and len(chunks) > 1:
                    return chunks
        
        # If no separator worked, fall back to splitting by character
        return self._split_by_character(text)
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """
        Split text by a separator and merge resulting chunks as needed.
        
        Args:
            text: Text to split
            separator: Separator to use
            
        Returns:
            List of text chunks
        """
        # Split the text by the separator
        splits = text.split(separator)
        
        # If the separator is empty or not present, return the text as is
        if len(splits) == 1:
            return [text]
        
        # Initialize variables
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for split in splits:
            # Skip empty splits
            if not split and not separator:
                continue
            
            split_with_separator = split
            if current_chunk:
                split_with_separator = separator + split
            
            # If adding this split would exceed the chunk size, start a new chunk
            if current_chunk_size + len(split_with_separator) > self.chunk_size:
                # Only start a new chunk if the current one is not empty
                if current_chunk:
                    chunk_text = "".join(current_chunk)
                    chunks.append(chunk_text)
                
                # If the split itself is longer than the chunk size, recursively split it
                if len(split) > self.chunk_size:
                    subsplits = self._recursive_split(split)
                    chunks.extend(subsplits)
                    
                    # Start fresh with an empty chunk
                    current_chunk = []
                    current_chunk_size = 0
                else:
                    # Start a new chunk with this split
                    current_chunk = [split]
                    current_chunk_size = len(split)
            else:
                # Add the split to the current chunk
                if current_chunk:
                    current_chunk.append(separator + split)
                    current_chunk_size += len(separator) + len(split)
                else:
                    current_chunk.append(split)
                    current_chunk_size += len(split)
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append(chunk_text)
        
        # Apply overlap between chunks
        if self.chunk_overlap > 0:
            chunks = self._apply_overlap(chunks)
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            chunks = [chunk.strip() for chunk in chunks]
        
        return chunks
    
    def _split_by_character(self, text: str) -> List[str]:
        """
        Split text by character.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Initialize variables
        chunks = []
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            # Calculate the end of the chunk
            chunk_end = min(i + self.chunk_size, len(text))
            # Add the chunk
            chunks.append(text[i:chunk_end])
            # If we've reached the end of the text, break
            if chunk_end == len(text):
                break
        
        return chunks
    
    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text using the next separator in the list.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Find the index of the current separator in the list
        for i, separator in enumerate(self.separators):
            if separator == "":
                # If we've reached the empty separator, split by character
                return self._split_by_character(text)
            
            if separator in text:
                # Create a new splitter with the remaining separators
                recursive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separators=self.separators[i:],
                    strip_whitespace=self.strip_whitespace,
                )
                # Split using the new splitter
                return recursive_splitter.split_text(text)
        
        # If no separator worked, return the text as is
        return [text]
    
    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of chunks with overlap
        """
        if not chunks or len(chunks) <= 1 or self.chunk_overlap == 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                # No overlap for the first chunk
                overlapped_chunks.append(chunk)
            else:
                # Calculate the overlap from the previous chunk
                prev_chunk = chunks[i - 1]
                overlap_size = min(self.chunk_overlap, len(prev_chunk))
                
                # Add the overlap to the current chunk
                overlap = prev_chunk[-overlap_size:]
                overlapped_chunks.append(overlap + chunk)
        
        return overlapped_chunks


class TokenTextSplitter(BaseChunker):
    """
    Split text into chunks based on a maximum token count.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        strip_whitespace: bool = True,
    ):
        """
        Initialize a token text splitter.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            tokenizer: Function to tokenize text (defaults to NLTK word_tokenize)
            strip_whitespace: Whether to strip whitespace from chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_whitespace = strip_whitespace
        
        # Set up tokenizer
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            if not NLTK_AVAILABLE:
                raise ImportError(
                    "NLTK is required for TokenTextSplitter. "
                    "Install it with `pip install nltk`."
                )
            self.tokenizer = word_tokenize
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            text = text.strip()
        
        # Tokenize the text
        tokens = self.tokenizer(text)
        
        # Initialize variables
        chunks = []
        current_chunk_tokens = []
        
        # Process tokens
        i = 0
        while i < len(tokens):
            # Add tokens to the current chunk
            current_chunk_tokens.append(tokens[i])
            i += 1
            
            # If we've reached the chunk size, add the chunk and start a new one
            if len(current_chunk_tokens) >= self.chunk_size:
                # Join the tokens to create the chunk
                chunk_text = self._join_tokens(current_chunk_tokens)
                chunks.append(chunk_text)
                
                # Include overlap from end of previous chunk
                overlap_start = max(0, len(current_chunk_tokens) - self.chunk_overlap)
                current_chunk_tokens = current_chunk_tokens[overlap_start:]
        
        # Add the final chunk if it's not empty
        if current_chunk_tokens:
            chunk_text = self._join_tokens(current_chunk_tokens)
            chunks.append(chunk_text)
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            chunks = [chunk.strip() for chunk in chunks]
        
        return chunks
    
    def _join_tokens(self, tokens: List[str]) -> str:
        """
        Join tokens back into text.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Joined text
        """
        # Simple space-joining for now
        # A more sophisticated implementation could handle punctuation better
        return " ".join(tokens)


class SentenceTextSplitter(BaseChunker):
    """
    Split text into chunks based on sentences.
    """
    
    def __init__(
        self,
        chunk_size: int = 10,
        chunk_overlap: int = 2,
        sentence_tokenizer: Optional[Callable[[str], List[str]]] = None,
        strip_whitespace: bool = True,
    ):
        """
        Initialize a sentence text splitter.
        
        Args:
            chunk_size: Maximum number of sentences per chunk
            chunk_overlap: Number of overlapping sentences between chunks
            sentence_tokenizer: Function to tokenize text into sentences (defaults to NLTK sent_tokenize)
            strip_whitespace: Whether to strip whitespace from chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_whitespace = strip_whitespace
        
        # Set up sentence tokenizer
        if sentence_tokenizer:
            self.sentence_tokenizer = sentence_tokenizer
        else:
            if not NLTK_AVAILABLE:
                raise ImportError(
                    "NLTK is required for SentenceTextSplitter. "
                    "Install it with `pip install nltk`."
                )
            self.sentence_tokenizer = sent_tokenize
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            text = text.strip()
        
        # Tokenize the text into sentences
        sentences = self.sentence_tokenizer(text)
        
        # Initialize variables
        chunks = []
        current_chunk_sentences = []
        
        # Process sentences
        i = 0
        while i < len(sentences):
            # Add sentences to the current chunk
            current_chunk_sentences.append(sentences[i])
            i += 1
            
            # If we've reached the chunk size, add the chunk and start a new one
            if len(current_chunk_sentences) >= self.chunk_size:
                # Join the sentences to create the chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(chunk_text)
                
                # Include overlap from end of previous chunk
                overlap_start = max(0, len(current_chunk_sentences) - self.chunk_overlap)
                current_chunk_sentences = current_chunk_sentences[overlap_start:]
        
        # Add the final chunk if it's not empty
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(chunk_text)
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            chunks = [chunk.strip() for chunk in chunks]
        
        return chunks


class SemanticTextSplitter(BaseChunker):
    """
    Split text into semantically meaningful chunks using more advanced techniques.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        strip_whitespace: bool = True,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize a semantic text splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Amount of overlap between chunks
            separator: Separator to use for final chunk connection
            strip_whitespace: Whether to strip whitespace from chunks
            use_spacy: Whether to use spaCy for semantic chunking
            spacy_model: spaCy model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.strip_whitespace = strip_whitespace
        self.use_spacy = use_spacy
        
        if use_spacy:
            if not SPACY_AVAILABLE:
                raise ImportError(
                    "spaCy is required for semantic chunking. "
                    "Install it with `pip install spacy` and "
                    "download a model with `python -m spacy download en_core_web_sm`."
                )
            
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                logger.warning(f"spaCy model {spacy_model} not found. Downloading...")
                try:
                    spacy.cli.download(spacy_model)
                    self.nlp = spacy.load(spacy_model)
                except Exception as e:
                    logger.error(f"Error downloading spaCy model: {e}")
                    self.use_spacy = False
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into semantically meaningful chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            text = text.strip()
        
        # If spaCy is available, use semantic chunking
        if self.use_spacy:
            return self._split_with_spacy(text)
        
        # Fall back to recursive character splitting
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            strip_whitespace=self.strip_whitespace,
        )
        return fallback_splitter.split_text(text)
    
    def _split_with_spacy(self, text: str) -> List[str]:
        """
        Split text using spaCy's semantic understanding.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract sentences with their semantic information
        sentences = []
        for sent in doc.sents:
            sentences.append({
                "text": sent.text,
                "entities": [ent.text for ent in sent.ents],
                "root": sent.root.text if sent.root else None,
            })
        
        # Group sentences into semantically coherent chunks
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        current_entities = set()
        
        for sent_info in sentences:
            sent_text = sent_info["text"]
            sent_entities = set(sent_info["entities"])
            
            # If the sentence would make the chunk too large, start a new chunk
            if current_chunk_size + len(sent_text) > self.chunk_size and current_chunk:
                # Join the current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Calculate overlap
                overlap_sentences = []
                overlap_size = 0
                
                # Add sentences from the end of the current chunk for overlap
                for sentence in reversed(current_chunk):
                    if overlap_size + len(sentence) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sentence)
                        overlap_size += len(sentence)
                    else:
                        break
                
                # Start a new chunk with the overlap
                current_chunk = overlap_sentences
                current_chunk_size = overlap_size
                current_entities = set()
                for sentence in current_chunk:
                    # Approximate entity extraction for overlap sentences
                    for sent_info in sentences:
                        if sent_info["text"] == sentence:
                            current_entities.update(sent_info["entities"])
                            break
            
            # Add the sentence to the current chunk
            current_chunk.append(sent_text)
            current_chunk_size += len(sent_text)
            current_entities.update(sent_entities)
        
        # Add the final chunk if it's not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
        
        # Strip whitespace if requested
        if self.strip_whitespace:
            chunks = [chunk.strip() for chunk in chunks]
        
        return chunks


class TopicTextSplitter(BaseChunker):
    """
    Split text into chunks based on topic changes.
    Requires a more advanced model to detect topic changes.
    This is a placeholder implementation that falls back to semantic splitting.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize a topic text splitter.
        Falls back to semantic splitting for now.
        """
        self.semantic_splitter = SemanticTextSplitter(**kwargs)
        
        logger.warning(
            "TopicTextSplitter is a placeholder implementation. "
            "It currently falls back to semantic splitting. "
            "A full implementation would require a more advanced topic detection model."
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text based on topic changes (falls back to semantic splitting).
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self.semantic_splitter.split_text(text)


# Factory function to get an appropriate chunker based on strategy
def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> BaseChunker:
    """
    Get a chunker based on the specified strategy.
    
    Args:
        strategy: Chunking strategy to use
        chunk_size: Maximum size of each chunk
        chunk_overlap: Amount of overlap between chunks
        **kwargs: Additional arguments for specific chunkers
        
    Returns:
        Chunker instance
        
    Raises:
        ValueError: If an unsupported strategy is specified
    """
    if strategy == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    elif strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    elif strategy == "token":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    elif strategy == "sentence":
        return SentenceTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    elif strategy == "semantic":
        return SemanticTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    elif strategy == "topic":
        return TopicTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported chunking strategy: {strategy}. "
            "Supported strategies are: character, recursive, token, sentence, semantic, topic."
        ) 