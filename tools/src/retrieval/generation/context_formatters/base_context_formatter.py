"""
Base Context Formatter

This module provides the BaseContextFormatter abstract class that defines the interface
for all context formatters.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)


class BaseContextFormatter(ABC):
    """
    Abstract base class for context formatters.
    
    Context formatters transform retrieved documents and their metadata into
    formatted contexts suitable for inclusion in prompts for language models.
    
    Attributes:
        config: Configuration options for the context formatter.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the context formatter.
        
        Args:
            config: Configuration options for the context formatter.
        """
        self.config = config or {}
    
    @abstractmethod
    def format_context(self, documents: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the retrieved documents into a context string.
        
        Args:
            documents: List of document dictionaries, typically including 'content'
                      and metadata fields like 'source', 'title', etc.
            **kwargs: Additional keyword arguments for formatting.
            
        Returns:
            str: The formatted context string.
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration options.
        
        Returns:
            Dict[str, Any]: The configuration options.
        """
        return self.config
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration options.
        
        Args:
            config: The configuration options to set.
        """
        self.config = config
    
    def _truncate_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Truncate text to the specified maximum length.
        
        Args:
            text: The text to truncate.
            max_length: Maximum length of the text. If None, no truncation is performed.
            
        Returns:
            str: The truncated text.
        """
        if max_length is None or len(text) <= max_length:
            return text
        
        return text[:max_length] + "..."
    
    def _get_document_field(self, document: Dict[str, Any], field: str, default: str = "") -> str:
        """
        Get a field from a document with a default value if the field doesn't exist.
        
        Args:
            document: The document dictionary.
            field: The field to get.
            default: Default value if the field doesn't exist.
            
        Returns:
            str: The field value or default.
        """
        return str(document.get(field, default))
    
    def _get_document_content(self, document: Dict[str, Any]) -> str:
        """
        Get the content from a document.
        
        Args:
            document: The document dictionary.
            
        Returns:
            str: The document content.
        
        Raises:
            ValueError: If the document doesn't have a 'content' field.
        """
        if 'content' not in document:
            logger.warning(f"Document missing 'content' field: {document}")
            raise ValueError("Document is missing required 'content' field")
        
        return document['content'] 