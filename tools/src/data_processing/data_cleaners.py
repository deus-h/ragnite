"""
Data Cleaners

This module provides tools for cleaning and normalizing text data to improve
the quality of processed documents for retrieval and generation.
"""

import re
import unicodedata
import html
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable, Pattern
import logging

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
except ImportError:
    NLTK_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Import Document class from document_loaders
from .document_loaders import Document


class BaseDataCleaner(ABC):
    """
    Abstract base class for data cleaners.
    """
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """
        Clean text data.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        pass
    
    def clean_document(self, document: Document) -> Document:
        """
        Clean a document.
        
        Args:
            document: Document to clean
            
        Returns:
            Cleaned document
        """
        cleaned_text = self.clean(document.content)
        
        # Add cleaning metadata
        updated_metadata = document.metadata.copy()
        updated_metadata["cleaned"] = True
        updated_metadata["cleaner"] = self.__class__.__name__
        
        return Document(content=cleaned_text, metadata=updated_metadata)
    
    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean a list of documents.
        
        Args:
            documents: List of documents to clean
            
        Returns:
            List of cleaned documents
        """
        return [self.clean_document(doc) for doc in documents]


class WhitespaceNormalizer(BaseDataCleaner):
    """
    Normalize whitespace in text.
    """
    
    def __init__(self, 
                 strip_whitespace: bool = True,
                 normalize_newlines: bool = True,
                 replace_multiple_spaces: bool = True,
                 max_consecutive_newlines: int = 2):
        """
        Initialize a whitespace normalizer.
        
        Args:
            strip_whitespace: Whether to strip leading/trailing whitespace
            normalize_newlines: Whether to normalize newlines to '\n'
            replace_multiple_spaces: Whether to replace multiple spaces with a single space
            max_consecutive_newlines: Maximum number of consecutive newlines to allow
        """
        self.strip_whitespace = strip_whitespace
        self.normalize_newlines = normalize_newlines
        self.replace_multiple_spaces = replace_multiple_spaces
        self.max_consecutive_newlines = max_consecutive_newlines
    
    def clean(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized whitespace
        """
        if not text:
            return ""
        
        # Normalize newlines
        if self.normalize_newlines:
            text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Replace multiple spaces with a single space
        if self.replace_multiple_spaces:
            text = re.sub(r' +', ' ', text)
        
        # Limit consecutive newlines
        if self.max_consecutive_newlines > 0:
            pattern = r'\n{' + str(self.max_consecutive_newlines + 1) + ',}'
            replacement = '\n' * self.max_consecutive_newlines
            text = re.sub(pattern, replacement, text)
        
        # Strip leading/trailing whitespace
        if self.strip_whitespace:
            text = text.strip()
        
        return text


class HTMLCleaner(BaseDataCleaner):
    """
    Clean HTML content from text.
    """
    
    def __init__(self,
                 decode_html_entities: bool = True,
                 remove_html_tags: bool = True,
                 keep_important_tags: bool = True,
                 preserve_line_breaks: bool = True):
        """
        Initialize an HTML cleaner.
        
        Args:
            decode_html_entities: Whether to decode HTML entities
            remove_html_tags: Whether to remove HTML tags
            keep_important_tags: Whether to keep newlines for important tags like <p>, <br>, <div>
            preserve_line_breaks: Whether to preserve line breaks in the cleaned text
        """
        self.decode_html_entities = decode_html_entities
        self.remove_html_tags = remove_html_tags
        self.keep_important_tags = keep_important_tags and remove_html_tags
        self.preserve_line_breaks = preserve_line_breaks
    
    def clean(self, text: str) -> str:
        """
        Clean HTML content from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        result = text
        
        # Replace important tags with newlines before removing all tags
        if self.keep_important_tags and self.remove_html_tags:
            result = re.sub(r'<br\s*/?>|</?p>|</?div>|</?h[1-6]>|</?tr>|</?li>', '\n', result)
        
        # Remove HTML tags
        if self.remove_html_tags:
            result = re.sub(r'<[^>]*>', '', result)
        
        # Decode HTML entities
        if self.decode_html_entities:
            result = html.unescape(result)
        
        # Preserve line breaks
        if self.preserve_line_breaks:
            result = re.sub(r'\n+', '\n', result)
        
        return result


class UnicodeNormalizer(BaseDataCleaner):
    """
    Normalize Unicode characters in text.
    """
    
    def __init__(self,
                 form: str = 'NFKC',
                 remove_control_chars: bool = True,
                 replace_non_ascii: bool = False,
                 ascii_replacement: str = ' '):
        """
        Initialize a Unicode normalizer.
        
        Args:
            form: Unicode normalization form ('NFC', 'NFKC', 'NFD', 'NFKD')
            remove_control_chars: Whether to remove control characters
            replace_non_ascii: Whether to replace non-ASCII characters
            ascii_replacement: Replacement for non-ASCII characters if replace_non_ascii is True
        """
        self.form = form
        self.remove_control_chars = remove_control_chars
        self.replace_non_ascii = replace_non_ascii
        self.ascii_replacement = ascii_replacement
    
    def clean(self, text: str) -> str:
        """
        Normalize Unicode characters in text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Text with normalized Unicode characters
        """
        if not text:
            return ""
        
        # Normalize Unicode
        result = unicodedata.normalize(self.form, text)
        
        # Remove control characters
        if self.remove_control_chars:
            result = ''.join(ch for ch in result if unicodedata.category(ch)[0] != 'C' or ch == '\n' or ch == '\t')
        
        # Replace non-ASCII characters
        if self.replace_non_ascii:
            result = ''.join(ch if ord(ch) < 128 else self.ascii_replacement for ch in result)
            # Collapse multiple replacements
            result = re.sub(f'{re.escape(self.ascii_replacement)}+', self.ascii_replacement, result)
        
        return result


class TextNormalizer(BaseDataCleaner):
    """
    Normalize text content with various text normalization techniques.
    """
    
    def __init__(self,
                 lowercase: bool = True,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False,
                 remove_extra_whitespace: bool = True,
                 collapse_whitespace: bool = True,
                 replace_urls: Optional[str] = None,
                 replace_emails: Optional[str] = None,
                 fix_common_typos: bool = False):
        """
        Initialize a text normalizer.
        
        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            remove_extra_whitespace: Whether to remove extra whitespace
            collapse_whitespace: Whether to collapse consecutive whitespace
            replace_urls: Replacement for URLs (None to leave URLs as is)
            replace_emails: Replacement for email addresses (None to leave emails as is)
            fix_common_typos: Whether to fix common typos
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_extra_whitespace = remove_extra_whitespace
        self.collapse_whitespace = collapse_whitespace
        self.replace_urls = replace_urls
        self.replace_emails = replace_emails
        self.fix_common_typos = fix_common_typos
        
        # Compile regular expressions
        self.url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        self.email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
        
        # Common typos dictionary
        if fix_common_typos:
            self.common_typos = {
                'teh': 'the',
                'adn': 'and',
                'waht': 'what',
                'taht': 'that',
                'wiht': 'with',
                # Add more common typos as needed
            }
        else:
            self.common_typos = {}
    
    def clean(self, text: str) -> str:
        """
        Normalize text content.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        result = text
        
        # Convert to lowercase
        if self.lowercase:
            result = result.lower()
        
        # Replace URLs
        if self.replace_urls is not None:
            result = self.url_pattern.sub(self.replace_urls, result)
        
        # Replace email addresses
        if self.replace_emails is not None:
            result = self.email_pattern.sub(self.replace_emails, result)
        
        # Remove punctuation
        if self.remove_punctuation:
            result = re.sub(r'[^\w\s]', '', result)
        
        # Remove numbers
        if self.remove_numbers:
            result = re.sub(r'\d+', '', result)
        
        # Fix common typos
        if self.fix_common_typos:
            words = result.split()
            corrected_words = [self.common_typos.get(word, word) for word in words]
            result = ' '.join(corrected_words)
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            result = result.strip()
        
        # Collapse consecutive whitespace
        if self.collapse_whitespace:
            result = re.sub(r'\s+', ' ', result)
        
        return result


class RegexCleaner(BaseDataCleaner):
    """
    Clean text using regular expressions.
    """
    
    def __init__(self, 
                 patterns: List[Union[str, Pattern]],
                 replacements: List[str]):
        """
        Initialize a regex cleaner.
        
        Args:
            patterns: List of regex patterns to match
            replacements: List of replacement strings
        """
        if len(patterns) != len(replacements):
            raise ValueError("The number of patterns must match the number of replacements")
        
        self.patterns = [
            re.compile(pattern) if isinstance(pattern, str) else pattern
            for pattern in patterns
        ]
        self.replacements = replacements
    
    def clean(self, text: str) -> str:
        """
        Clean text using regular expressions.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        result = text
        
        for pattern, replacement in zip(self.patterns, self.replacements):
            result = pattern.sub(replacement, result)
        
        return result


class StopwordRemover(BaseDataCleaner):
    """
    Remove stopwords from text.
    """
    
    def __init__(self,
                 language: str = 'english',
                 custom_stopwords: Optional[List[str]] = None,
                 ignore_case: bool = True,
                 tokenize: bool = True,
                 min_word_length: int = 1):
        """
        Initialize a stopword remover.
        
        Args:
            language: Language for stopwords (when using NLTK)
            custom_stopwords: Custom list of stopwords to remove
            ignore_case: Whether to ignore case when matching stopwords
            tokenize: Whether to tokenize the text (if False, only exact matches are removed)
            min_word_length: Minimum length of words to keep after stopword removal
        """
        self.custom_stopwords = set(custom_stopwords or [])
        self.ignore_case = ignore_case
        self.tokenize = tokenize
        self.min_word_length = min_word_length
        
        # Get stopwords from NLTK if available
        if NLTK_AVAILABLE:
            try:
                self.stopwords = set(stopwords.words(language))
                # Add custom stopwords
                self.stopwords.update(self.custom_stopwords)
            except Exception as e:
                logger.warning(f"Error loading NLTK stopwords: {e}")
                self.stopwords = self.custom_stopwords
        else:
            logger.warning(
                "NLTK is not available. Using only custom stopwords. "
                "Install NLTK for better stopword removal with `pip install nltk`."
            )
            self.stopwords = self.custom_stopwords
    
    def clean(self, text: str) -> str:
        """
        Remove stopwords from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with stopwords removed
        """
        if not text:
            return ""
        
        # Normalize case if requested
        working_text = text.lower() if self.ignore_case else text
        
        if self.tokenize:
            # Simple whitespace tokenization
            words = working_text.split()
            
            # Filter out stopwords and short words
            filtered_words = []
            for i, word in enumerate(words):
                # Remove punctuation for comparison
                word_clean = re.sub(r'[^\w\s]', '', word)
                
                if (word_clean.lower() if self.ignore_case else word_clean) not in self.stopwords and len(word_clean) >= self.min_word_length:
                    # Use the original word (with punctuation)
                    filtered_words.append(words[i])
            
            # Reconstruct text
            result = ' '.join(filtered_words)
        else:
            # Replace exact matches only
            result = working_text
            for stopword in self.stopwords:
                # Create a pattern that matches the stopword as a whole word
                pattern = r'\b' + re.escape(stopword) + r'\b'
                result = re.sub(pattern, '', result, flags=re.IGNORECASE if self.ignore_case else 0)
            
            # Collapse multiple spaces
            result = re.sub(r'\s+', ' ', result)
        
        return result


class TextNormalizer(BaseDataCleaner):
    """
    Apply linguistic normalization techniques like stemming and lemmatization.
    """
    
    def __init__(self,
                 normalization_type: str = 'lemmatization',
                 language: str = 'english',
                 preserve_original: bool = False):
        """
        Initialize a text normalizer.
        
        Args:
            normalization_type: Type of normalization ('stemming' or 'lemmatization')
            language: Language for normalization (when using NLTK)
            preserve_original: Whether to preserve original words in the metadata
        """
        self.normalization_type = normalization_type
        self.language = language
        self.preserve_original = preserve_original
        
        if not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for text normalization. "
                "Install it with `pip install nltk`."
            )
        
        # Set up normalizer based on type
        if normalization_type == 'stemming':
            self.normalizer = PorterStemmer()
            self.normalize_word = self.normalizer.stem
        elif normalization_type == 'lemmatization':
            self.normalizer = WordNetLemmatizer()
            self.normalize_word = lambda word: self.normalizer.lemmatize(word)
        else:
            raise ValueError(
                f"Unsupported normalization type: {normalization_type}. "
                "Supported types are: 'stemming', 'lemmatization'."
            )
    
    def clean(self, text: str) -> str:
        """
        Apply normalization to text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Tokenize text
        words = text.split()
        
        # Normalize words
        normalized_words = [self.normalize_word(word) for word in words]
        
        # Join normalized words
        result = ' '.join(normalized_words)
        
        return result
    
    def clean_document(self, document: Document) -> Document:
        """
        Clean a document and preserve original text if requested.
        
        Args:
            document: Document to clean
            
        Returns:
            Cleaned document
        """
        cleaned_text = self.clean(document.content)
        
        # Update metadata
        updated_metadata = document.metadata.copy()
        updated_metadata["cleaned"] = True
        updated_metadata["cleaner"] = self.__class__.__name__
        updated_metadata["normalization_type"] = self.normalization_type
        
        # Preserve original text if requested
        if self.preserve_original:
            updated_metadata["original_text"] = document.content
        
        return Document(content=cleaned_text, metadata=updated_metadata)


class DuplicateRemover(BaseDataCleaner):
    """
    Remove duplicate lines, paragraphs, or sentences from text.
    """
    
    def __init__(self,
                 scope: str = 'line',
                 case_sensitive: bool = False,
                 keep_first: bool = True):
        """
        Initialize a duplicate remover.
        
        Args:
            scope: Scope of deduplication ('line', 'paragraph', or 'sentence')
            case_sensitive: Whether to consider case when comparing for duplicates
            keep_first: Whether to keep the first occurrence (True) or the last (False)
        """
        self.scope = scope
        self.case_sensitive = case_sensitive
        self.keep_first = keep_first
        
        # Set up sentence tokenizer if needed
        if scope == 'sentence' and NLTK_AVAILABLE:
            try:
                from nltk.tokenize import sent_tokenize
                nltk.data.find('tokenizers/punkt')
                self.sent_tokenize = sent_tokenize
            except (ImportError, LookupError):
                nltk.download('punkt')
                from nltk.tokenize import sent_tokenize
                self.sent_tokenize = sent_tokenize
        elif scope == 'sentence':
            logger.warning(
                "NLTK is not available for sentence tokenization. "
                "Using a simple regex-based sentence splitter."
            )
            self.sent_tokenize = lambda text: re.split(r'(?<=[.!?])\s+', text)
    
    def clean(self, text: str) -> str:
        """
        Remove duplicates from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with duplicates removed
        """
        if not text:
            return ""
        
        # Split text based on scope
        if self.scope == 'line':
            segments = text.splitlines()
        elif self.scope == 'paragraph':
            segments = re.split(r'\n\s*\n', text)
        elif self.scope == 'sentence':
            segments = self.sent_tokenize(text)
        else:
            raise ValueError(
                f"Unsupported scope: {self.scope}. "
                "Supported scopes are: 'line', 'paragraph', 'sentence'."
            )
        
        # Remove duplicates
        seen = set()
        unique_segments = []
        
        if self.keep_first:
            # Keep first occurrence
            for segment in segments:
                # Normalize for comparison
                compare_key = segment if self.case_sensitive else segment.lower()
                
                if compare_key not in seen:
                    unique_segments.append(segment)
                    seen.add(compare_key)
        else:
            # Keep last occurrence
            for segment in reversed(segments):
                # Normalize for comparison
                compare_key = segment if self.case_sensitive else segment.lower()
                
                if compare_key not in seen:
                    unique_segments.insert(0, segment)
                    seen.add(compare_key)
        
        # Join segments based on scope
        if self.scope == 'line':
            result = '\n'.join(unique_segments)
        elif self.scope == 'paragraph':
            result = '\n\n'.join(unique_segments)
        elif self.scope == 'sentence':
            result = ' '.join(unique_segments)
        
        return result


class NoiseRemover(BaseDataCleaner):
    """
    Remove noisy elements from text.
    """
    
    def __init__(self,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_phone_numbers: bool = True,
                 remove_social_media_handles: bool = True,
                 remove_html_tags: bool = True,
                 remove_markdown: bool = False,
                 remove_citations: bool = False,
                 remove_special_chars: bool = False,
                 replace_with: str = ' '):
        """
        Initialize a noise remover.
        
        Args:
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_phone_numbers: Whether to remove phone numbers
            remove_social_media_handles: Whether to remove social media handles
            remove_html_tags: Whether to remove HTML tags
            remove_markdown: Whether to remove Markdown formatting
            remove_citations: Whether to remove citations (e.g., [1], [2-4])
            remove_special_chars: Whether to remove special characters
            replace_with: String to replace removed elements with
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phone_numbers = remove_phone_numbers
        self.remove_social_media_handles = remove_social_media_handles
        self.remove_html_tags = remove_html_tags
        self.remove_markdown = remove_markdown
        self.remove_citations = remove_citations
        self.remove_special_chars = remove_special_chars
        self.replace_with = replace_with
        
        # Compile regular expressions
        if remove_urls:
            self.url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        
        if remove_emails:
            self.email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
        
        if remove_phone_numbers:
            self.phone_pattern = re.compile(r'\+?[\d\-\(\)\s]{7,}')
        
        if remove_social_media_handles:
            self.handle_pattern = re.compile(r'@[a-zA-Z0-9_]+')
        
        if remove_html_tags:
            self.html_pattern = re.compile(r'<[^>]*>')
        
        if remove_markdown:
            self.markdown_patterns = [
                re.compile(r'\*\*(.+?)\*\*'),  # Bold
                re.compile(r'\*(.+?)\*'),      # Italic
                re.compile(r'__(.+?)__'),      # Bold
                re.compile(r'_(.+?)_'),        # Italic
                re.compile(r'~~(.+?)~~'),      # Strikethrough
                re.compile(r'`(.+?)`'),        # Inline code
                re.compile(r'```[\s\S]*?```'), # Code block
                re.compile(r'#+\s+'),          # Headers
                re.compile(r'\[([^\]]+)\]\([^\)]+\)'), # Links
                re.compile(r'!\[([^\]]+)\]\([^\)]+\)'), # Images
                re.compile(r'[*+-]\s+'),       # List items
                re.compile(r'\d+\.\s+'),       # Numbered list items
            ]
        
        if remove_citations:
            self.citation_pattern = re.compile(r'\[\d+(?:-\d+)?\]|\(\d+(?:-\d+)?\)')
        
        if remove_special_chars:
            self.special_chars_pattern = re.compile(r'[^\w\s]')
    
    def clean(self, text: str) -> str:
        """
        Remove noisy elements from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        result = text
        
        # Remove URLs
        if self.remove_urls:
            result = self.url_pattern.sub(self.replace_with, result)
        
        # Remove email addresses
        if self.remove_emails:
            result = self.email_pattern.sub(self.replace_with, result)
        
        # Remove phone numbers
        if self.remove_phone_numbers:
            result = self.phone_pattern.sub(self.replace_with, result)
        
        # Remove social media handles
        if self.remove_social_media_handles:
            result = self.handle_pattern.sub(self.replace_with, result)
        
        # Remove HTML tags
        if self.remove_html_tags:
            result = self.html_pattern.sub(self.replace_with, result)
        
        # Remove Markdown formatting
        if self.remove_markdown:
            for pattern in self.markdown_patterns:
                # For patterns with groups, keep the content inside the formatting
                if '(' in pattern.pattern:
                    result = pattern.sub(r'\1', result)
                else:
                    result = pattern.sub(self.replace_with, result)
        
        # Remove citations
        if self.remove_citations:
            result = self.citation_pattern.sub(self.replace_with, result)
        
        # Remove special characters
        if self.remove_special_chars:
            result = self.special_chars_pattern.sub(self.replace_with, result)
        
        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result


class CompositeCleaner(BaseDataCleaner):
    """
    Apply multiple cleaners in sequence.
    """
    
    def __init__(self, cleaners: List[BaseDataCleaner]):
        """
        Initialize a composite cleaner.
        
        Args:
            cleaners: List of cleaners to apply in sequence
        """
        self.cleaners = cleaners
    
    def clean(self, text: str) -> str:
        """
        Apply multiple cleaners in sequence.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        result = text
        
        for cleaner in self.cleaners:
            result = cleaner.clean(result)
        
        return result
    
    def clean_document(self, document: Document) -> Document:
        """
        Clean a document using multiple cleaners.
        
        Args:
            document: Document to clean
            
        Returns:
            Cleaned document
        """
        current_doc = document
        
        for cleaner in self.cleaners:
            current_doc = cleaner.clean_document(current_doc)
        
        return current_doc


# Factory function to get a data cleaner
def get_data_cleaner(
    cleaner_type: str = "basic",
    **kwargs
) -> BaseDataCleaner:
    """
    Get a data cleaner based on the specified type.
    
    Args:
        cleaner_type: Type of data cleaner to use
        **kwargs: Additional arguments for the specific cleaner
        
    Returns:
        Data cleaner instance
        
    Raises:
        ValueError: If an unsupported cleaner type is specified
    """
    if cleaner_type == "whitespace":
        return WhitespaceNormalizer(**kwargs)
    elif cleaner_type == "html":
        return HTMLCleaner(**kwargs)
    elif cleaner_type == "unicode":
        return UnicodeNormalizer(**kwargs)
    elif cleaner_type == "text":
        return TextNormalizer(**kwargs)
    elif cleaner_type == "regex":
        return RegexCleaner(**kwargs)
    elif cleaner_type == "stopword":
        return StopwordRemover(**kwargs)
    elif cleaner_type == "duplicate":
        return DuplicateRemover(**kwargs)
    elif cleaner_type == "noise":
        return NoiseRemover(**kwargs)
    else:
        raise ValueError(
            f"Unsupported data cleaner type: {cleaner_type}. "
            "Supported types are: whitespace, html, unicode, text, regex, "
            "stopword, duplicate, noise."
        )


# Create a standard text cleaner with reasonable defaults
def create_standard_cleaner() -> CompositeCleaner:
    """
    Create a standard text cleaner with reasonable defaults.
    
    Returns:
        Standard composite cleaner
    """
    cleaners = [
        HTMLCleaner(),
        UnicodeNormalizer(),
        WhitespaceNormalizer(),
        NoiseRemover(remove_special_chars=False),
    ]
    
    return CompositeCleaner(cleaners)


# Create a comprehensive text cleaner with all cleaning steps
def create_comprehensive_cleaner() -> CompositeCleaner:
    """
    Create a comprehensive text cleaner with all cleaning steps.
    
    Returns:
        Comprehensive composite cleaner
    """
    cleaners = [
        HTMLCleaner(),
        UnicodeNormalizer(),
        WhitespaceNormalizer(),
        NoiseRemover(),
        DuplicateRemover(),
    ]
    
    # Add stopword remover if NLTK is available
    if NLTK_AVAILABLE:
        cleaners.append(StopwordRemover())
    
    return CompositeCleaner(cleaners) 