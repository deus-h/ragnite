"""
Base Output Parser

This module defines the BaseOutputParser abstract class that all output parsers must implement.
Output parsers transform raw LLM outputs into structured, validated formats for downstream use.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic, Type

T = TypeVar('T')

class BaseOutputParser(Generic[T], ABC):
    """
    Abstract base class for output parsers.
    
    Output parsers transform the raw text output from language models into structured,
    validated formats suitable for downstream use. They can handle formats like JSON, XML,
    Markdown, or custom structured formats.
    
    Attributes:
        config (Dict[str, Any]): Configuration options for the parser.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the output parser.
        
        Args:
            config: Optional configuration dictionary for the parser.
        """
        self.config = config or {}
    
    @abstractmethod
    def parse(self, text: str) -> T:
        """
        Parse the text output from an LLM into a structured format.
        
        Args:
            text: The raw text output from an LLM.
            
        Returns:
            The structured data extracted from the text.
            
        Raises:
            ParsingError: If the text cannot be parsed into the expected format.
        """
        pass
    
    @abstractmethod
    def get_format_instructions(self) -> str:
        """
        Get instructions for the language model on how to format its output.
        
        Returns:
            A string containing instructions on the expected output format.
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the parser.
        
        Returns:
            A dictionary containing the current configuration.
        """
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration of the parser.
        
        Args:
            config: A dictionary containing configuration options to update.
        """
        self.config.update(config)
    
    def validate_output(self, parsed_output: T) -> bool:
        """
        Validate the parsed output against expected schemas or rules.
        
        Args:
            parsed_output: The structured data to validate.
            
        Returns:
            True if the output is valid, False otherwise.
        """
        # Default implementation always returns True
        # Subclasses should override this method to provide proper validation
        return True


class ParsingError(Exception):
    """Exception raised when an output cannot be parsed correctly."""
    
    def __init__(self, message: str, original_text: str, error: Optional[Exception] = None):
        """
        Initialize a ParsingError.
        
        Args:
            message: Description of the error.
            original_text: The original text that failed to parse.
            error: The underlying exception that caused this error, if any.
        """
        self.original_text = original_text
        self.underlying_error = error
        super().__init__(message) 