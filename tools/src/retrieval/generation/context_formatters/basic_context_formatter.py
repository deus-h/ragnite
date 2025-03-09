"""
Basic Context Formatter

This module provides the BasicContextFormatter class for simple context formatting.
"""

import logging
from typing import Dict, Any, List, Optional, Union

from .base_context_formatter import BaseContextFormatter

# Configure logging
logger = logging.getLogger(__name__)


class BasicContextFormatter(BaseContextFormatter):
    """
    A simple context formatter that combines document contents with minimal formatting.
    
    This formatter concatenates the contents of the provided documents with separators
    and optional numbering, without including metadata or complex formatting.
    
    Attributes:
        config: Configuration options for the context formatter.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the basic context formatter.
        
        Args:
            config: Configuration options for the context formatter.
                   - separator (str): Separator between documents. Default: "\n\n".
                   - numbered (bool): Whether to number the documents. Default: False.
                   - max_length (int): Maximum length for each document. Default: None.
                   - header (str): Optional header text to prepend to the context. Default: "".
                   - footer (str): Optional footer text to append to the context. Default: "".
        """
        super().__init__(config or {})
        
        # Set default configuration
        self.config.setdefault("separator", "\n\n")
        self.config.setdefault("numbered", False)
        self.config.setdefault("max_length", None)
        self.config.setdefault("header", "")
        self.config.setdefault("footer", "")
    
    def format_context(self, documents: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the retrieved documents into a simple context string.
        
        Args:
            documents: List of document dictionaries with 'content' field.
            **kwargs: Additional keyword arguments:
                      - separator: Override the separator between documents.
                      - numbered: Override whether to number the documents.
                      - max_length: Override the maximum length for each document.
                      - header: Override the header text.
                      - footer: Override the footer text.
            
        Returns:
            str: The formatted context string.
        """
        # Apply any overrides from kwargs
        separator = kwargs.get("separator", self.config["separator"])
        numbered = kwargs.get("numbered", self.config["numbered"])
        max_length = kwargs.get("max_length", self.config["max_length"])
        header = kwargs.get("header", self.config["header"])
        footer = kwargs.get("footer", self.config["footer"])
        
        # Format documents
        formatted_docs = []
        for i, doc in enumerate(documents):
            try:
                content = self._get_document_content(doc)
                truncated_content = self._truncate_text(content, max_length)
                
                if numbered:
                    formatted_doc = f"[{i+1}] {truncated_content}"
                else:
                    formatted_doc = truncated_content
                
                formatted_docs.append(formatted_doc)
            except ValueError as e:
                logger.warning(f"Skipping document due to error: {str(e)}")
                continue
        
        # Combine documents
        combined_context = separator.join(formatted_docs)
        
        # Add header and footer
        if header:
            combined_context = f"{header}\n\n{combined_context}"
        
        if footer:
            combined_context = f"{combined_context}\n\n{footer}"
        
        return combined_context
    
    def __repr__(self) -> str:
        """String representation of the BasicContextFormatter."""
        return f"BasicContextFormatter(numbered={self.config['numbered']}, max_length={self.config['max_length']})" 