"""
Metadata Extractors

This module provides tools for extracting metadata from documents,
including title extraction, author extraction, date extraction,
entity extraction, and keyword extraction.
"""

import re
import os
import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Set
import logging
from collections import Counter

# Try to import optional dependencies
try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Import Document class from document_loaders
from .document_loaders import Document


class BaseMetadataExtractor(ABC):
    """
    Abstract base class for metadata extractors.
    """
    
    @abstractmethod
    def extract(self, document: Document) -> Dict[str, Any]:
        """
        Extract metadata from a document.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata
        """
        pass
    
    def extract_from_documents(self, documents: List[Document]) -> List[Document]:
        """
        Extract metadata from a list of documents.
        
        Args:
            documents: List of documents to extract metadata from
            
        Returns:
            List of documents with updated metadata
        """
        updated_documents = []
        
        for doc in documents:
            metadata = self.extract(doc)
            
            # Update document metadata
            updated_metadata = doc.metadata.copy()
            updated_metadata.update(metadata)
            
            # Create new document with updated metadata
            updated_doc = Document(content=doc.content, metadata=updated_metadata)
            updated_documents.append(updated_doc)
        
        return updated_documents


class BasicMetadataExtractor(BaseMetadataExtractor):
    """
    Extract basic metadata from document content and file information.
    """
    
    def __init__(self, extract_language: bool = True):
        """
        Initialize a basic metadata extractor.
        
        Args:
            extract_language: Whether to extract language information
        """
        self.extract_language = extract_language
        
        if extract_language and not LANGDETECT_AVAILABLE:
            logger.warning(
                "Language detection requested but langdetect is not installed. "
                "Install it with `pip install langdetect`."
            )
            self.extract_language = False
    
    def extract(self, document: Document) -> Dict[str, Any]:
        """
        Extract basic metadata from a document.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {}
        
        # Extract metadata from document content
        content = document.content
        
        # Document statistics
        metadata["char_count"] = len(content)
        metadata["word_count"] = len(content.split())
        metadata["line_count"] = len(content.splitlines())
        
        # Calculate approximate reading time (avg reading speed: 250 words/min)
        words_per_minute = 250
        reading_time_minutes = metadata["word_count"] / words_per_minute
        metadata["reading_time_minutes"] = round(reading_time_minutes, 2)
        
        # Extract language if requested
        if self.extract_language and LANGDETECT_AVAILABLE:
            try:
                language = detect(content[:1000])  # Use first 1000 chars for efficiency
                metadata["language"] = language
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
        
        # Extract information from file path if available
        if "source" in document.metadata:
            source = document.metadata["source"]
            if isinstance(source, str) and os.path.exists(source):
                file_stats = os.stat(source)
                
                # File information
                metadata["file_size_bytes"] = file_stats.st_size
                metadata["file_created_at"] = datetime.datetime.fromtimestamp(
                    file_stats.st_ctime
                ).isoformat()
                metadata["file_modified_at"] = datetime.datetime.fromtimestamp(
                    file_stats.st_mtime
                ).isoformat()
                metadata["file_extension"] = os.path.splitext(source)[1].lower()
        
        return metadata


class TitleExtractor(BaseMetadataExtractor):
    """
    Extract title from document content.
    """
    
    def __init__(
        self,
        use_first_line: bool = True,
        min_title_length: int = 3,
        max_title_length: int = 100,
        use_heuristics: bool = True,
    ):
        """
        Initialize a title extractor.
        
        Args:
            use_first_line: Whether to use the first line as a title candidate
            min_title_length: Minimum title length
            max_title_length: Maximum title length
            use_heuristics: Whether to use additional heuristics to find titles
        """
        self.use_first_line = use_first_line
        self.min_title_length = min_title_length
        self.max_title_length = max_title_length
        self.use_heuristics = use_heuristics
    
    def extract(self, document: Document) -> Dict[str, Any]:
        """
        Extract title from a document.
        
        Args:
            document: Document to extract title from
            
        Returns:
            Dictionary containing extracted title
        """
        metadata = {}
        content = document.content
        
        # Use existing title if available in metadata
        if "title" in document.metadata and document.metadata["title"]:
            return {"title": document.metadata["title"]}
        
        # Try to extract title
        title = self._extract_title(content, document.metadata)
        
        if title:
            metadata["title"] = title
        
        return metadata
    
    def _extract_title(self, content: str, existing_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract title from document content using various heuristics.
        
        Args:
            content: Document content
            existing_metadata: Existing document metadata
            
        Returns:
            Extracted title or None
        """
        title_candidates = []
        
        # Use first line if requested
        if self.use_first_line:
            lines = content.splitlines()
            if lines:
                first_line = lines[0].strip()
                if self.min_title_length <= len(first_line) <= self.max_title_length:
                    title_candidates.append(first_line)
        
        # Use HTML title if available for HTML documents
        if "source" in existing_metadata and BS4_AVAILABLE:
            source = existing_metadata["source"]
            if isinstance(source, str) and source.lower().endswith((".html", ".htm")):
                try:
                    with open(source, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                        title_tag = soup.find("title")
                        if title_tag and title_tag.string:
                            title = title_tag.string.strip()
                            if self.min_title_length <= len(title) <= self.max_title_length:
                                title_candidates.append(title)
                except Exception as e:
                    logger.warning(f"Error extracting HTML title: {e}")
        
        # Use PDF title if available for PDF documents
        if "source" in existing_metadata and PYPDF2_AVAILABLE:
            source = existing_metadata["source"]
            if isinstance(source, str) and source.lower().endswith(".pdf"):
                try:
                    with open(source, "rb") as f:
                        pdf_reader = PyPDF2.PdfFileReader(f)
                        info = pdf_reader.getDocumentInfo()
                        if info and info.title:
                            title = info.title.strip()
                            if self.min_title_length <= len(title) <= self.max_title_length:
                                title_candidates.append(title)
                except Exception as e:
                    logger.warning(f"Error extracting PDF title: {e}")
        
        # Use additional heuristics if requested
        if self.use_heuristics:
            # Look for lines with specific title patterns
            lines = content.splitlines()
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                
                # Skip lines that are too short or too long
                if not (self.min_title_length <= len(line) <= self.max_title_length):
                    continue
                
                # Check for common title patterns
                is_title_candidate = (
                    line.isupper() or  # ALL CAPS
                    line.istitle() or  # Title Case
                    re.match(r"^#+ ", line) or  # Markdown title
                    re.match(r"^Title:", line, re.IGNORECASE)  # Explicit title
                )
                
                if is_title_candidate:
                    # Clean up markdown titles
                    cleaned_line = re.sub(r"^#+ ", "", line).strip()
                    if self.min_title_length <= len(cleaned_line) <= self.max_title_length:
                        title_candidates.append(cleaned_line)
        
        # Return the first valid title candidate
        return title_candidates[0] if title_candidates else None


class AuthorExtractor(BaseMetadataExtractor):
    """
    Extract author information from document content.
    """
    
    def __init__(self, use_heuristics: bool = True, use_nlp: bool = True):
        """
        Initialize an author extractor.
        
        Args:
            use_heuristics: Whether to use pattern-based heuristics
            use_nlp: Whether to use NLP-based extraction
        """
        self.use_heuristics = use_heuristics
        self.use_nlp = use_nlp
        
        # Set up NLP pipeline for author extraction if requested
        if use_nlp and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Downloading...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception as e:
                    logger.error(f"Error downloading spaCy model: {e}")
                    self.use_nlp = False
        else:
            self.use_nlp = False and use_nlp  # Only use NLP if requested and available
    
    def extract(self, document: Document) -> Dict[str, Any]:
        """
        Extract author information from a document.
        
        Args:
            document: Document to extract author from
            
        Returns:
            Dictionary containing extracted author information
        """
        metadata = {}
        content = document.content
        
        # Use existing author if available in metadata
        if "author" in document.metadata and document.metadata["author"]:
            return {"author": document.metadata["author"]}
        
        # Try to extract author
        author = self._extract_author(content, document.metadata)
        
        if author:
            metadata["author"] = author
        
        return metadata
    
    def _extract_author(self, content: str, existing_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract author from document content using various methods.
        
        Args:
            content: Document content
            existing_metadata: Existing document metadata
            
        Returns:
            Extracted author or None
        """
        author_candidates = []
        
        # Use document metadata if available
        if "source" in existing_metadata:
            source = existing_metadata["source"]
            
            # Extract from PDF metadata
            if isinstance(source, str) and source.lower().endswith(".pdf") and PYPDF2_AVAILABLE:
                try:
                    with open(source, "rb") as f:
                        pdf_reader = PyPDF2.PdfFileReader(f)
                        info = pdf_reader.getDocumentInfo()
                        if info and info.author:
                            author_candidates.append(info.author.strip())
                except Exception as e:
                    logger.warning(f"Error extracting PDF author: {e}")
            
            # Extract from HTML metadata
            if isinstance(source, str) and source.lower().endswith((".html", ".htm")) and BS4_AVAILABLE:
                try:
                    with open(source, "r", encoding="utf-8") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                        # Check meta tags
                        author_meta = soup.find("meta", attrs={"name": "author"})
                        if author_meta and author_meta.get("content"):
                            author_candidates.append(author_meta.get("content").strip())
                except Exception as e:
                    logger.warning(f"Error extracting HTML author: {e}")
        
        # Use heuristics if requested
        if self.use_heuristics:
            lines = content.splitlines()
            
            # Look for common author patterns in the first few lines
            for line in lines[:20]:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check for common author patterns
                author_match = (
                    re.search(r"(?:Author|By)[:\s]+([^,\n]+)", line, re.IGNORECASE) or
                    re.search(r"^by\s+([^,\n]+)", line, re.IGNORECASE) or
                    re.search(r"written by\s+([^,\n]+)", line, re.IGNORECASE)
                )
                
                if author_match:
                    author_candidates.append(author_match.group(1).strip())
        
        # Use NLP if requested and available
        if self.use_nlp:
            # Process the first part of the document to find person names
            doc = self.nlp(content[:2000])  # Use first 2000 chars for efficiency
            
            # Collect all person entities
            person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            
            # Count occurrences of each person entity
            person_counts = Counter(person_entities)
            
            # Add the most frequent person entities as candidates
            for person, count in person_counts.most_common(3):
                if count >= 2:  # Require at least 2 mentions
                    author_candidates.append(person)
        
        # Return the first valid author candidate
        return author_candidates[0] if author_candidates else None


class DateExtractor(BaseMetadataExtractor):
    """
    Extract date information from document content.
    """
    
    def __init__(self, extract_creation_date: bool = True, extract_content_dates: bool = True):
        """
        Initialize a date extractor.
        
        Args:
            extract_creation_date: Whether to extract document creation date
            extract_content_dates: Whether to extract dates mentioned in content
        """
        self.extract_creation_date = extract_creation_date
        self.extract_content_dates = extract_content_dates
        
        # Compile regular expressions for date patterns
        self.date_patterns = [
            # ISO format: YYYY-MM-DD
            re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
            # US format: MM/DD/YYYY
            re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b"),
            # Written format: Month DD, YYYY
            re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b", re.IGNORECASE),
            # Short month format: MMM DD, YYYY
            re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b", re.IGNORECASE),
        ]
    
    def extract(self, document: Document) -> Dict[str, Any]:
        """
        Extract date information from a document.
        
        Args:
            document: Document to extract dates from
            
        Returns:
            Dictionary containing extracted date information
        """
        metadata = {}
        content = document.content
        
        # Extract creation date if requested
        if self.extract_creation_date:
            creation_date = self._extract_creation_date(document.metadata)
            if creation_date:
                metadata["creation_date"] = creation_date
        
        # Extract dates from content if requested
        if self.extract_content_dates:
            content_dates = self._extract_content_dates(content)
            if content_dates:
                metadata["content_dates"] = content_dates
                # Use the first date as the document date if no creation date was found
                if "document_date" not in metadata and not content_dates:
                    metadata["document_date"] = content_dates[0]
        
        return metadata
    
    def _extract_creation_date(self, existing_metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract document creation date from metadata.
        
        Args:
            existing_metadata: Existing document metadata
            
        Returns:
            Extracted creation date or None
        """
        # Check if creation date is already in metadata
        if "creation_date" in existing_metadata:
            return existing_metadata["creation_date"]
        
        # Try to extract from file metadata
        if "source" in existing_metadata:
            source = existing_metadata["source"]
            
            # Extract from PDF metadata
            if isinstance(source, str) and source.lower().endswith(".pdf") and PYPDF2_AVAILABLE:
                try:
                    with open(source, "rb") as f:
                        pdf_reader = PyPDF2.PdfFileReader(f)
                        info = pdf_reader.getDocumentInfo()
                        if info and "/CreationDate" in info:
                            # PDF date format: D:YYYYMMDDHHmmSS
                            date_str = info["/CreationDate"]
                            if date_str.startswith("D:"):
                                date_str = date_str[2:14]  # Extract YYYYMMDDHHmm
                                try:
                                    date = datetime.datetime.strptime(date_str, "%Y%m%d%H%M")
                                    return date.isoformat()
                                except ValueError:
                                    pass
                except Exception as e:
                    logger.warning(f"Error extracting PDF creation date: {e}")
            
            # Use file creation date as fallback
            if isinstance(source, str) and os.path.exists(source):
                try:
                    file_stats = os.stat(source)
                    return datetime.datetime.fromtimestamp(file_stats.st_ctime).isoformat()
                except Exception as e:
                    logger.warning(f"Error extracting file creation date: {e}")
        
        return None
    
    def _extract_content_dates(self, content: str) -> List[str]:
        """
        Extract dates mentioned in document content.
        
        Args:
            content: Document content
            
        Returns:
            List of extracted dates
        """
        dates = []
        
        # Search for date patterns in content
        for pattern in self.date_patterns:
            matches = pattern.findall(content)
            dates.extend(matches)
        
        # Remove duplicates and sort
        unique_dates = list(set(dates))
        
        return unique_dates


class EntityExtractor(BaseMetadataExtractor):
    """
    Extract named entities from document content.
    """
    
    def __init__(
        self,
        entity_types: Optional[List[str]] = None,
        max_entities: int = 50,
        min_entity_freq: int = 1,
    ):
        """
        Initialize an entity extractor.
        
        Args:
            entity_types: Types of entities to extract (default: all available types)
            max_entities: Maximum number of entities to extract
            min_entity_freq: Minimum frequency for an entity to be included
        """
        self.entity_types = entity_types or [
            "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "DATE", "TIME"
        ]
        self.max_entities = max_entities
        self.min_entity_freq = min_entity_freq
        
        # Set up NLP pipeline for entity extraction
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.nlp_available = True
            except OSError:
                logger.warning("spaCy model not found. Downloading...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                    self.nlp_available = True
                except Exception as e:
                    logger.error(f"Error downloading spaCy model: {e}")
                    self.nlp_available = False
        else:
            logger.warning(
                "spaCy is required for entity extraction. "
                "Install it with `pip install spacy` and "
                "download a model with `python -m spacy download en_core_web_sm`."
            )
            self.nlp_available = False
    
    def extract(self, document: Document) -> Dict[str, Any]:
        """
        Extract named entities from a document.
        
        Args:
            document: Document to extract entities from
            
        Returns:
            Dictionary containing extracted entities
        """
        metadata = {}
        
        # Skip if NLP is not available
        if not self.nlp_available:
            return metadata
        
        content = document.content
        
        # Limit content size for processing efficiency
        max_content_len = 10000  # Limit to first 10K characters for efficiency
        trimmed_content = content[:max_content_len]
        
        # Process the document with spaCy
        doc = self.nlp(trimmed_content)
        
        # Extract entities
        entities_by_type = {}
        entity_counts = {}
        
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                # Normalize entity text
                entity_text = ent.text.strip()
                
                # Skip if entity is too short
                if len(entity_text) < 2:
                    continue
                
                # Count entity occurrences
                if entity_text not in entity_counts:
                    entity_counts[entity_text] = 0
                entity_counts[entity_text] += 1
                
                # Add entity to its type group
                if ent.label_ not in entities_by_type:
                    entities_by_type[ent.label_] = set()
                entities_by_type[ent.label_].add(entity_text)
        
        # Filter entities by frequency and convert to a dictionary
        filtered_entities = {}
        for entity_type, entities in entities_by_type.items():
            # Filter by minimum frequency
            filtered = [
                entity for entity in entities
                if entity_counts.get(entity, 0) >= self.min_entity_freq
            ]
            
            # Sort by frequency (descending) and limit to max_entities
            sorted_entities = sorted(
                filtered,
                key=lambda x: entity_counts.get(x, 0),
                reverse=True
            )[:self.max_entities]
            
            if sorted_entities:
                filtered_entities[entity_type] = sorted_entities
        
        if filtered_entities:
            metadata["entities"] = filtered_entities
        
        return metadata


class KeywordExtractor(BaseMetadataExtractor):
    """
    Extract keywords and key phrases from document content.
    """
    
    def __init__(
        self,
        max_keywords: int = 30,
        min_word_length: int = 3,
        top_n_ngrams: int = 10,
        ngram_range: tuple = (1, 3),
        use_nlp: bool = True,
    ):
        """
        Initialize a keyword extractor.
        
        Args:
            max_keywords: Maximum number of keywords to extract
            min_word_length: Minimum length of words to consider
            top_n_ngrams: Number of top n-grams to extract for each n-gram size
            ngram_range: Range of n-gram sizes to extract (min, max)
            use_nlp: Whether to use NLP techniques for extraction
        """
        self.max_keywords = max_keywords
        self.min_word_length = min_word_length
        self.top_n_ngrams = top_n_ngrams
        self.ngram_range = ngram_range
        self.use_nlp = use_nlp
        
        # Set up NLTK resources if available
        if NLTK_AVAILABLE:
            self.stopwords = set(stopwords.words('english'))
            self.nltk_available = True
        else:
            self.stopwords = set([
                "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
                "which", "this", "that", "these", "those", "then", "just", "so", "than",
                "such", "when", "who", "how", "where", "why", "is", "are", "was", "were",
                "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
                "doing", "would", "should", "could", "ought", "i'm", "you're", "he's",
                "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've",
                "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll",
                "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't",
                "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't",
                "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't",
                "mustn't", "let's", "that's", "who's", "what's", "here's", "there's",
                "when's", "where's", "why's", "how's", "a", "an", "the", "and", "but",
                "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
                "with", "about", "against", "between", "into", "through", "during",
                "before", "after", "above", "below", "to", "from", "up", "down", "in",
                "out", "on", "off", "over", "under", "again", "further", "then", "once",
                "here", "there", "when", "where", "why", "how", "all", "any", "both",
                "each", "few", "more", "most", "other", "some", "such", "no", "nor",
                "not", "only", "own", "same", "so", "than", "too", "very"
            ])
            self.nltk_available = False
            logger.warning(
                "NLTK is not available. Using a basic stopwords list. "
                "Install NLTK for better results with `pip install nltk`."
            )
        
        # Set up spaCy if requested
        if use_nlp and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_available = True
            except OSError:
                logger.warning("spaCy model not found. Downloading...")
                try:
                    spacy.cli.download("en_core_web_sm")
                    self.nlp = spacy.load("en_core_web_sm")
                    self.spacy_available = True
                except Exception as e:
                    logger.error(f"Error downloading spaCy model: {e}")
                    self.spacy_available = False
        else:
            self.spacy_available = False
    
    def extract(self, document: Document) -> Dict[str, Any]:
        """
        Extract keywords from a document.
        
        Args:
            document: Document to extract keywords from
            
        Returns:
            Dictionary containing extracted keywords
        """
        metadata = {}
        content = document.content
        
        # Limit content size for processing efficiency
        max_content_len = 20000  # Limit to first 20K characters for efficiency
        trimmed_content = content[:max_content_len]
        
        # Extract keywords
        keywords = self._extract_keywords(trimmed_content)
        
        if keywords:
            metadata["keywords"] = keywords
        
        return metadata
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text using statistical and NLP methods.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        
        # Clean and normalize text
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        
        # Split into words
        words = text.split()
        
        # Filter out stopwords and short words
        filtered_words = [
            word for word in words
            if word not in self.stopwords and len(word) >= self.min_word_length
        ]
        
        # Count word frequencies
        word_freq = Counter(filtered_words)
        
        # Get the most common words
        top_words = [word for word, _ in word_freq.most_common(self.max_keywords)]
        keywords.extend(top_words)
        
        # Extract n-grams if NLTK is available
        if self.nltk_available:
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                if n == 1:
                    continue  # Skip unigrams as they're already handled
                
                ngrams = self._extract_ngrams(filtered_words, n)
                
                # Count n-gram frequencies
                ngram_freq = Counter(ngrams)
                
                # Get the most common n-grams
                top_ngrams = [
                    " ".join(ngram) for ngram, _ in ngram_freq.most_common(self.top_n_ngrams)
                ]
                keywords.extend(top_ngrams)
        
        # Use spaCy for keyword extraction if available
        if self.use_nlp and self.spacy_available:
            doc = self.nlp(text)
            
            # Extract noun chunks as potential keywords
            noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
            
            # Filter out stopwords and short chunks
            filtered_chunks = []
            for chunk in noun_chunks:
                chunk_words = chunk.split()
                if all(word not in self.stopwords for word in chunk_words) and len(chunk) >= self.min_word_length:
                    filtered_chunks.append(chunk)
            
            # Count chunk frequencies
            chunk_freq = Counter(filtered_chunks)
            
            # Get the most common chunks
            top_chunks = [chunk for chunk, _ in chunk_freq.most_common(self.top_n_ngrams)]
            keywords.extend(top_chunks)
        
        # Remove duplicates and limit to max_keywords
        unique_keywords = []
        seen = set()
        
        for keyword in keywords:
            if keyword not in seen and len(unique_keywords) < self.max_keywords:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords
    
    def _extract_ngrams(self, words: List[str], n: int) -> List[tuple]:
        """
        Extract n-grams from a list of words.
        
        Args:
            words: List of words
            n: Size of n-grams to extract
            
        Returns:
            List of n-grams
        """
        ngrams = []
        
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            
            # Skip n-grams with stopwords
            if any(word in self.stopwords for word in ngram):
                continue
            
            ngrams.append(ngram)
        
        return ngrams


# Factory function to get a metadata extractor
def get_metadata_extractor(
    extractor_type: str = "basic",
    **kwargs
) -> BaseMetadataExtractor:
    """
    Get a metadata extractor based on the specified type.
    
    Args:
        extractor_type: Type of metadata extractor to use
        **kwargs: Additional arguments for the specific extractor
        
    Returns:
        Metadata extractor instance
        
    Raises:
        ValueError: If an unsupported extractor type is specified
    """
    if extractor_type == "basic":
        return BasicMetadataExtractor(**kwargs)
    elif extractor_type == "title":
        return TitleExtractor(**kwargs)
    elif extractor_type == "author":
        return AuthorExtractor(**kwargs)
    elif extractor_type == "date":
        return DateExtractor(**kwargs)
    elif extractor_type == "entity":
        return EntityExtractor(**kwargs)
    elif extractor_type == "keyword":
        return KeywordExtractor(**kwargs)
    else:
        raise ValueError(
            f"Unsupported metadata extractor type: {extractor_type}. "
            "Supported types are: basic, title, author, date, entity, keyword."
        )


# Composite metadata extractor that combines multiple extractors
class CompositeMetadataExtractor(BaseMetadataExtractor):
    """
    Combine multiple metadata extractors into a single extractor.
    """
    
    def __init__(self, extractors: List[BaseMetadataExtractor]):
        """
        Initialize a composite metadata extractor.
        
        Args:
            extractors: List of metadata extractors to use
        """
        self.extractors = extractors
    
    def extract(self, document: Document) -> Dict[str, Any]:
        """
        Extract metadata from a document using multiple extractors.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Dictionary containing all extracted metadata
        """
        metadata = {}
        
        # Apply each extractor in sequence
        for extractor in self.extractors:
            extractor_metadata = extractor.extract(document)
            metadata.update(extractor_metadata)
        
        return metadata


# Create a comprehensive metadata extractor with all available extractors
def create_comprehensive_extractor(**kwargs) -> CompositeMetadataExtractor:
    """
    Create a comprehensive metadata extractor that uses all available extractors.
    
    Args:
        **kwargs: Additional arguments for the extractors
        
    Returns:
        Comprehensive metadata extractor
    """
    extractors = [
        BasicMetadataExtractor(),
        TitleExtractor(),
        AuthorExtractor(),
        DateExtractor(),
        EntityExtractor(),
        KeywordExtractor(),
    ]
    
    return CompositeMetadataExtractor(extractors) 