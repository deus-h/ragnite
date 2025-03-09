"""
Metadata Enriched Formatter

This module provides the MetadataEnrichedFormatter class for formatting context
with document metadata.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Set

from .base_context_formatter import BaseContextFormatter

# Configure logging
logger = logging.getLogger(__name__)


class MetadataEnrichedFormatter(BaseContextFormatter):
    """
    A context formatter that enriches document content with its metadata.
    
    This formatter includes relevant metadata fields with each document,
    such as title, source, author, date, etc., to provide additional
    context about the documents' origin and attributes.
    
    Attributes:
        config: Configuration options for the context formatter.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metadata enriched formatter.
        
        Args:
            config: Configuration options for the context formatter.
                   - separator (str): Separator between documents. Default: "\n\n---\n\n".
                   - metadata_fields (List[str]): Metadata fields to include. Default: ['title', 'source', 'date'].
                   - metadata_format (str): Format for metadata fields. Default: "{field}: {value}".
                   - metadata_separator (str): Separator between metadata fields. Default: "\n".
                   - max_length (int): Maximum length for each document content. Default: None.
                   - header (str): Optional header text to prepend to the context. Default: "".
                   - footer (str): Optional footer text to append to the context. Default: "".
                   - skip_empty_fields (bool): Whether to skip empty metadata fields. Default: True.
        """
        super().__init__(config or {})
        
        # Set default configuration
        self.config.setdefault("separator", "\n\n---\n\n")
        self.config.setdefault("metadata_fields", ["title", "source", "date"])
        self.config.setdefault("metadata_format", "{field}: {value}")
        self.config.setdefault("metadata_separator", "\n")
        self.config.setdefault("max_length", None)
        self.config.setdefault("header", "")
        self.config.setdefault("footer", "")
        self.config.setdefault("skip_empty_fields", True)
    
    def format_context(self, documents: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the retrieved documents into a context string with metadata.
        
        Args:
            documents: List of document dictionaries with 'content' and metadata fields.
            **kwargs: Additional keyword arguments:
                      - separator: Override the separator between documents.
                      - metadata_fields: Override the metadata fields to include.
                      - metadata_format: Override the format for metadata fields.
                      - metadata_separator: Override the separator between metadata fields.
                      - max_length: Override the maximum length for each document content.
                      - header: Override the header text.
                      - footer: Override the footer text.
                      - skip_empty_fields: Override whether to skip empty metadata fields.
            
        Returns:
            str: The formatted context string with metadata.
        """
        # Apply any overrides from kwargs
        separator = kwargs.get("separator", self.config["separator"])
        metadata_fields = kwargs.get("metadata_fields", self.config["metadata_fields"])
        metadata_format = kwargs.get("metadata_format", self.config["metadata_format"])
        metadata_separator = kwargs.get("metadata_separator", self.config["metadata_separator"])
        max_length = kwargs.get("max_length", self.config["max_length"])
        header = kwargs.get("header", self.config["header"])
        footer = kwargs.get("footer", self.config["footer"])
        skip_empty_fields = kwargs.get("skip_empty_fields", self.config["skip_empty_fields"])
        
        # Format documents with metadata
        formatted_docs = []
        for doc in documents:
            try:
                # Get and truncate content
                content = self._get_document_content(doc)
                truncated_content = self._truncate_text(content, max_length)
                
                # Format metadata
                metadata_items = []
                for field in metadata_fields:
                    value = self._get_document_field(doc, field)
                    if skip_empty_fields and not value:
                        continue
                    
                    formatted_field = metadata_format.format(
                        field=field.capitalize(),
                        value=value
                    )
                    metadata_items.append(formatted_field)
                
                # Combine metadata and content
                metadata_block = metadata_separator.join(metadata_items)
                if metadata_block:
                    formatted_doc = f"{metadata_block}\n\n{truncated_content}"
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
    
    def set_metadata_fields(self, fields: List[str]) -> None:
        """
        Set the metadata fields to include.
        
        Args:
            fields: List of metadata field names to include.
        """
        self.config["metadata_fields"] = fields
    
    def get_metadata_fields(self) -> List[str]:
        """
        Get the metadata fields to include.
        
        Returns:
            List[str]: The metadata fields.
        """
        return self.config["metadata_fields"]
    
    def __repr__(self) -> str:
        """String representation of the MetadataEnrichedFormatter."""
        return f"MetadataEnrichedFormatter(fields={self.config['metadata_fields']}, max_length={self.config['max_length']})" 