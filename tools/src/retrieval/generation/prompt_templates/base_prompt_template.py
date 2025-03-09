"""
Base Prompt Template

This module provides the BasePromptTemplate abstract class that defines the interface
for all prompt templates.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)


class BasePromptTemplate(ABC):
    """
    Abstract base class for prompt templates.
    
    All prompt templates should inherit from this class and implement
    the required methods.
    
    Attributes:
        template: The prompt template string.
        config: Configuration options for the prompt template.
    """
    
    def __init__(self, template: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the prompt template.
        
        Args:
            template: The prompt template string with placeholder variables.
            config: Configuration options for the prompt template.
        """
        self.template = template
        self.config = config or {}
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """
        Format the prompt template with the provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template.
            
        Returns:
            str: The formatted prompt.
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
    
    def set_template(self, template: str) -> None:
        """
        Set the prompt template.
        
        Args:
            template: The prompt template string.
        """
        self.template = template
    
    def get_template(self) -> str:
        """
        Get the prompt template.
        
        Returns:
            str: The prompt template string.
        """
        return self.template 