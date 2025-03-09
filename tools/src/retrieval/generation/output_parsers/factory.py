"""
Output Parser Factory

This module provides a factory function for creating output parser instances.
"""

from typing import Any, Dict, List, Optional, Type, Union

from .base_output_parser import BaseOutputParser
from .json_output_parser import JSONOutputParser
from .xml_output_parser import XMLOutputParser
from .markdown_output_parser import MarkdownOutputParser
from .structured_output_parser import StructuredOutputParser

def get_output_parser(
    parser_type: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseOutputParser:
    """
    Factory function to create an output parser instance based on the specified type.
    
    Args:
        parser_type: Type of parser to create. Options:
            - "json": Parser for JSON outputs
            - "xml": Parser for XML outputs
            - "markdown": Parser for Markdown outputs
            - "structured": Parser for custom structured outputs
        config: Optional configuration dictionary for the parser
        **kwargs: Additional keyword arguments to include in the config
        
    Returns:
        An instance of the specified output parser
        
    Raises:
        ValueError: If an unsupported parser type is specified
    """
    # Combine config and kwargs
    if config is None:
        config = {}
    
    combined_config = {**config, **kwargs}
    
    # Create the appropriate parser based on type
    if parser_type.lower() == "json":
        return JSONOutputParser(combined_config)
    elif parser_type.lower() == "xml":
        return XMLOutputParser(combined_config)
    elif parser_type.lower() == "markdown":
        return MarkdownOutputParser(combined_config)
    elif parser_type.lower() == "structured":
        return StructuredOutputParser(combined_config)
    else:
        raise ValueError(f"Unsupported parser type: {parser_type}. "
                        f"Supported types are: json, xml, markdown, structured.") 