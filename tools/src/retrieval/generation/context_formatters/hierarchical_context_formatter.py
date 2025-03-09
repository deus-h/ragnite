"""
Hierarchical Context Formatter

This module provides the HierarchicalContextFormatter class for organizing context
in a hierarchical structure.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from collections import defaultdict

from .base_context_formatter import BaseContextFormatter

# Configure logging
logger = logging.getLogger(__name__)


class HierarchicalContextFormatter(BaseContextFormatter):
    """
    A context formatter that organizes documents into a hierarchical structure.
    
    This formatter groups documents by categories (e.g., topic, type, source)
    and presents them in a nested structure, making it easier for language models
    to understand the organization of information.
    
    Attributes:
        config: Configuration options for the context formatter.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hierarchical context formatter.
        
        Args:
            config: Configuration options for the context formatter.
                   - hierarchy_field (str): Field to use for primary grouping. Default: "category".
                   - secondary_field (str): Field to use for secondary grouping. Default: None.
                   - group_separator (str): Separator between groups. Default: "\n\n---\n\n".
                   - item_separator (str): Separator between items in a group. Default: "\n\n".
                   - group_format (str): Format for group headers. Default: "## {group}".
                   - subgroup_format (str): Format for subgroup headers. Default: "### {subgroup}".
                   - include_item_headers (bool): Whether to include headers for items. Default: False.
                   - item_header_format (str): Format for item headers. Default: "#### {index}. {title}".
                   - numbered (bool): Whether to number items within groups. Default: True.
                   - max_length (int): Maximum length for each document content. Default: None.
                   - header (str): Optional header text to prepend to the context. Default: "".
                   - footer (str): Optional footer text to append to the context. Default: "".
                   - default_group (str): Group name for documents without the hierarchy field. Default: "Other".
                   - group_sort_key (Callable): Function to sort groups. Default: None (alphabetical).
                   - item_sort_key (Callable): Function to sort items within groups. Default: None (original order).
        """
        super().__init__(config or {})
        
        # Set default configuration
        self.config.setdefault("hierarchy_field", "category")
        self.config.setdefault("secondary_field", None)
        self.config.setdefault("group_separator", "\n\n---\n\n")
        self.config.setdefault("item_separator", "\n\n")
        self.config.setdefault("group_format", "## {group}")
        self.config.setdefault("subgroup_format", "### {subgroup}")
        self.config.setdefault("include_item_headers", False)
        self.config.setdefault("item_header_format", "#### {index}. {title}")
        self.config.setdefault("numbered", True)
        self.config.setdefault("max_length", None)
        self.config.setdefault("header", "")
        self.config.setdefault("footer", "")
        self.config.setdefault("default_group", "Other")
        self.config.setdefault("group_sort_key", None)
        self.config.setdefault("item_sort_key", None)
    
    def format_context(self, documents: List[Dict[str, Any]], **kwargs) -> str:
        """
        Format the retrieved documents into a hierarchical context string.
        
        Args:
            documents: List of document dictionaries with 'content' and metadata fields.
            **kwargs: Additional keyword arguments:
                      - hierarchy_field: Override the field to use for primary grouping.
                      - secondary_field: Override the field to use for secondary grouping.
                      - group_separator: Override the separator between groups.
                      - item_separator: Override the separator between items in a group.
                      - group_format: Override the format for group headers.
                      - subgroup_format: Override the format for subgroup headers.
                      - include_item_headers: Override whether to include headers for items.
                      - item_header_format: Override the format for item headers.
                      - numbered: Override whether to number items within groups.
                      - max_length: Override the maximum length for each document content.
                      - header: Override the header text.
                      - footer: Override the footer text.
                      - default_group: Override the group name for documents without the hierarchy field.
                      - group_sort_key: Override the function to sort groups.
                      - item_sort_key: Override the function to sort items within groups.
            
        Returns:
            str: The formatted hierarchical context string.
        """
        # Apply any overrides from kwargs
        hierarchy_field = kwargs.get("hierarchy_field", self.config["hierarchy_field"])
        secondary_field = kwargs.get("secondary_field", self.config["secondary_field"])
        group_separator = kwargs.get("group_separator", self.config["group_separator"])
        item_separator = kwargs.get("item_separator", self.config["item_separator"])
        group_format = kwargs.get("group_format", self.config["group_format"])
        subgroup_format = kwargs.get("subgroup_format", self.config["subgroup_format"])
        include_item_headers = kwargs.get("include_item_headers", self.config["include_item_headers"])
        item_header_format = kwargs.get("item_header_format", self.config["item_header_format"])
        numbered = kwargs.get("numbered", self.config["numbered"])
        max_length = kwargs.get("max_length", self.config["max_length"])
        header = kwargs.get("header", self.config["header"])
        footer = kwargs.get("footer", self.config["footer"])
        default_group = kwargs.get("default_group", self.config["default_group"])
        group_sort_key = kwargs.get("group_sort_key", self.config["group_sort_key"])
        item_sort_key = kwargs.get("item_sort_key", self.config["item_sort_key"])
        
        # Group documents
        if secondary_field:
            # Two-level hierarchy
            grouped_docs = self._group_documents_two_level(
                documents, 
                hierarchy_field, 
                secondary_field, 
                default_group
            )
            
            formatted_groups = []
            
            # Sort groups if a sort key is provided
            groups = list(grouped_docs.keys())
            if group_sort_key:
                groups.sort(key=group_sort_key)
            
            for group in groups:
                group_header = group_format.format(group=group)
                subgroups = grouped_docs[group]
                
                formatted_subgroups = []
                
                # Sort subgroups if a sort key is provided
                subgroup_names = list(subgroups.keys())
                if group_sort_key:
                    subgroup_names.sort(key=group_sort_key)
                
                for subgroup in subgroup_names:
                    subgroup_header = subgroup_format.format(subgroup=subgroup)
                    docs_in_subgroup = subgroups[subgroup]
                    
                    # Format documents in the subgroup
                    formatted_docs = self._format_documents(
                        docs_in_subgroup,
                        include_item_headers,
                        item_header_format,
                        numbered,
                        max_length,
                        item_sort_key
                    )
                    
                    # Combine documents in the subgroup
                    combined_docs = item_separator.join(formatted_docs)
                    formatted_subgroup = f"{subgroup_header}\n\n{combined_docs}"
                    formatted_subgroups.append(formatted_subgroup)
                
                # Combine subgroups in the group
                combined_subgroups = item_separator.join(formatted_subgroups)
                formatted_group = f"{group_header}\n\n{combined_subgroups}"
                formatted_groups.append(formatted_group)
            
            # Combine all groups
            combined_context = group_separator.join(formatted_groups)
            
        else:
            # Single-level hierarchy
            grouped_docs = self._group_documents(
                documents, 
                hierarchy_field, 
                default_group
            )
            
            formatted_groups = []
            
            # Sort groups if a sort key is provided
            groups = list(grouped_docs.keys())
            if group_sort_key:
                groups.sort(key=group_sort_key)
            
            for group in groups:
                group_header = group_format.format(group=group)
                docs_in_group = grouped_docs[group]
                
                # Format documents in the group
                formatted_docs = self._format_documents(
                    docs_in_group,
                    include_item_headers,
                    item_header_format,
                    numbered,
                    max_length,
                    item_sort_key
                )
                
                # Combine documents in the group
                combined_docs = item_separator.join(formatted_docs)
                formatted_group = f"{group_header}\n\n{combined_docs}"
                formatted_groups.append(formatted_group)
            
            # Combine all groups
            combined_context = group_separator.join(formatted_groups)
        
        # Add header and footer
        if header:
            combined_context = f"{header}\n\n{combined_context}"
        
        if footer:
            combined_context = f"{combined_context}\n\n{footer}"
        
        return combined_context
    
    def _group_documents(self, 
                         documents: List[Dict[str, Any]], 
                         field: str, 
                         default_group: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group documents by a specified field.
        
        Args:
            documents: List of document dictionaries.
            field: Field to group by.
            default_group: Group name for documents without the field.
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Documents grouped by the field.
        """
        grouped = defaultdict(list)
        
        for doc in documents:
            group = self._get_document_field(doc, field) or default_group
            grouped[group].append(doc)
        
        return grouped
    
    def _group_documents_two_level(self, 
                                   documents: List[Dict[str, Any]], 
                                   primary_field: str, 
                                   secondary_field: str,
                                   default_group: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Group documents by primary and secondary fields.
        
        Args:
            documents: List of document dictionaries.
            primary_field: Primary field to group by.
            secondary_field: Secondary field to group by.
            default_group: Group name for documents without the fields.
            
        Returns:
            Dict[str, Dict[str, List[Dict[str, Any]]]]: Two-level grouping of documents.
        """
        grouped = defaultdict(lambda: defaultdict(list))
        
        for doc in documents:
            primary = self._get_document_field(doc, primary_field) or default_group
            secondary = self._get_document_field(doc, secondary_field) or default_group
            grouped[primary][secondary].append(doc)
        
        return grouped
    
    def _format_documents(self, 
                          documents: List[Dict[str, Any]], 
                          include_headers: bool, 
                          header_format: str, 
                          numbered: bool,
                          max_length: Optional[int],
                          sort_key: Optional[Callable] = None) -> List[str]:
        """
        Format a list of documents.
        
        Args:
            documents: List of document dictionaries.
            include_headers: Whether to include headers for items.
            header_format: Format for item headers.
            numbered: Whether to number items.
            max_length: Maximum length for each document content.
            sort_key: Function to sort items.
            
        Returns:
            List[str]: List of formatted documents.
        """
        # Sort documents if a sort key is provided
        docs_to_format = documents.copy()
        if sort_key:
            docs_to_format.sort(key=sort_key)
        
        formatted_docs = []
        
        for i, doc in enumerate(docs_to_format):
            try:
                # Get and truncate content
                content = self._get_document_content(doc)
                truncated_content = self._truncate_text(content, max_length)
                
                # Format the document
                if include_headers:
                    title = self._get_document_field(doc, "title", f"Item {i+1}")
                    index = i + 1 if numbered else ""
                    header = header_format.format(index=index, title=title)
                    formatted_doc = f"{header}\n\n{truncated_content}"
                else:
                    formatted_doc = truncated_content
                
                formatted_docs.append(formatted_doc)
            except ValueError as e:
                logger.warning(f"Skipping document due to error: {str(e)}")
                continue
        
        return formatted_docs
    
    def set_hierarchy_fields(self, primary: str, secondary: Optional[str] = None) -> None:
        """
        Set the fields to use for hierarchical grouping.
        
        Args:
            primary: Field to use for primary grouping.
            secondary: Field to use for secondary grouping.
        """
        self.config["hierarchy_field"] = primary
        self.config["secondary_field"] = secondary
    
    def set_group_formats(self, group_format: str, subgroup_format: Optional[str] = None) -> None:
        """
        Set the formats for group headers.
        
        Args:
            group_format: Format for group headers.
            subgroup_format: Format for subgroup headers.
        """
        self.config["group_format"] = group_format
        if subgroup_format:
            self.config["subgroup_format"] = subgroup_format
    
    def __repr__(self) -> str:
        """String representation of the HierarchicalContextFormatter."""
        secondary = f", secondary={self.config['secondary_field']}" if self.config['secondary_field'] else ""
        return f"HierarchicalContextFormatter(hierarchy={self.config['hierarchy_field']}{secondary})" 