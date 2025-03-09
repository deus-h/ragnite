"""
Source Attribution Formatter

This module provides the SourceAttributionFormatter class for formatting context
with source attribution.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union, Tuple

from .base_context_formatter import BaseContextFormatter

# Configure logging
logger = logging.getLogger(__name__)


class SourceAttributionFormatter(BaseContextFormatter):
    """
    A context formatter that adds source attribution to document content.
    
    This formatter adds citation markers to document content and includes a
    reference section with full source information, making it easy for language
    models to cite sources in their responses.
    
    Attributes:
        config: Configuration options for the context formatter.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the source attribution formatter.
        
        Args:
            config: Configuration options for the context formatter.
                   - separator (str): Separator between documents. Default: "\n\n".
                   - citation_format (str): Format for citation markers. Default: "[{index}]".
                   - reference_format (str): Format for references. Default: "[{index}] {source}".
                   - reference_header (str): Header for the references section. Default: "\n\nSources:".
                   - source_fields (List[str]): Fields to use for source attribution. Default: ['source', 'title', 'url'].
                   - source_separator (str): Separator between source fields. Default: ", ".
                   - citation_placement (str): Where to place citation markers ('prefix', 'suffix', or 'both'). Default: "suffix".
                   - max_length (int): Maximum length for each document content. Default: None.
                   - header (str): Optional header text to prepend to the context. Default: "".
                   - footer (str): Optional footer text to append to the context. Default: "".
        """
        super().__init__(config or {})
        
        # Set default configuration
        self.config.setdefault("separator", "\n\n")
        self.config.setdefault("citation_format", "[{index}]")
        self.config.setdefault("reference_format", "[{index}] {source}")
        self.config.setdefault("reference_header", "\n\nSources:")
        self.config.setdefault("source_fields", ["source", "title", "url"])
        self.config.setdefault("source_separator", ", ")
        self.config.setdefault("citation_placement", "suffix")
        self.config.setdefault("max_length", None)
        self.config.setdefault("header", "")
        self.config.setdefault("footer", "")
    
    def format_context(self, documents: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the retrieved documents into a context string with source attribution.
        
        Args:
            documents: List of document dictionaries with 'content' and source fields.
            **kwargs: Additional keyword arguments:
                      - separator: Override the separator between documents.
                      - citation_format: Override the format for citation markers.
                      - reference_format: Override the format for references.
                      - reference_header: Override the header for the references section.
                      - source_fields: Override the fields to use for source attribution.
                      - source_separator: Override the separator between source fields.
                      - citation_placement: Override where to place citation markers.
                      - max_length: Override the maximum length for each document content.
                      - header: Override the header text.
                      - footer: Override the footer text.
            
        Returns:
            str: The formatted context string with source attribution.
        """
        # Apply any overrides from kwargs
        separator = kwargs.get("separator", self.config["separator"])
        citation_format = kwargs.get("citation_format", self.config["citation_format"])
        reference_format = kwargs.get("reference_format", self.config["reference_format"])
        reference_header = kwargs.get("reference_header", self.config["reference_header"])
        source_fields = kwargs.get("source_fields", self.config["source_fields"])
        source_separator = kwargs.get("source_separator", self.config["source_separator"])
        citation_placement = kwargs.get("citation_placement", self.config["citation_placement"])
        max_length = kwargs.get("max_length", self.config["max_length"])
        header = kwargs.get("header", self.config["header"])
        footer = kwargs.get("footer", self.config["footer"])
        
        # Format documents with citations
        formatted_docs = []
        references = []
        
        for i, doc in enumerate(documents):
            try:
                # Get and truncate content
                content = self._get_document_content(doc)
                truncated_content = self._truncate_text(content, max_length)
                
                # Create citation marker
                index = i + 1
                citation = citation_format.format(index=index)
                
                # Add citation marker to content
                if citation_placement == "prefix":
                    formatted_doc = f"{citation} {truncated_content}"
                elif citation_placement == "suffix":
                    formatted_doc = f"{truncated_content} {citation}"
                elif citation_placement == "both":
                    formatted_doc = f"{citation} {truncated_content} {citation}"
                else:
                    # Default to suffix if invalid placement
                    formatted_doc = f"{truncated_content} {citation}"
                
                formatted_docs.append(formatted_doc)
                
                # Create reference entry
                source_parts = []
                for field in source_fields:
                    value = self._get_document_field(doc, field)
                    if value:
                        source_parts.append(value)
                
                source_text = source_separator.join(source_parts)
                if source_text:
                    reference = reference_format.format(index=index, source=source_text)
                    references.append(reference)
                
            except ValueError as e:
                logger.warning(f"Skipping document due to error: {str(e)}")
                continue
        
        # Combine documents
        combined_docs = separator.join(formatted_docs)
        
        # Add references section if there are references
        if references:
            references_text = "\n".join(references)
            combined_context = f"{combined_docs}{reference_header}\n{references_text}"
        else:
            combined_context = combined_docs
        
        # Add header and footer
        if header:
            combined_context = f"{header}\n\n{combined_context}"
        
        if footer:
            combined_context = f"{combined_context}\n\n{footer}"
        
        return combined_context
    
    def set_citation_format(self, format_str: str) -> None:
        """
        Set the format for citation markers.
        
        Args:
            format_str: Format string with {index} placeholder.
        """
        self.config["citation_format"] = format_str
    
    def set_reference_format(self, format_str: str) -> None:
        """
        Set the format for references.
        
        Args:
            format_str: Format string with {index} and {source} placeholders.
        """
        self.config["reference_format"] = format_str
    
    def set_source_fields(self, fields: List[str]) -> None:
        """
        Set the fields to use for source attribution.
        
        Args:
            fields: List of field names to use for source attribution.
        """
        self.config["source_fields"] = fields
    
    def set_citation_placement(self, placement: str) -> None:
        """
        Set where to place citation markers.
        
        Args:
            placement: Where to place citation markers ('prefix', 'suffix', or 'both').
        """
        if placement not in ["prefix", "suffix", "both"]:
            logger.warning(f"Invalid citation placement: {placement}. Using 'suffix'.")
            placement = "suffix"
        
        self.config["citation_placement"] = placement
    
    def __repr__(self) -> str:
        """String representation of the SourceAttributionFormatter."""
        return f"SourceAttributionFormatter(citation_placement={self.config['citation_placement']}, max_length={self.config['max_length']})" 