"""
Basic Prompt Template

This module provides the BasicPromptTemplate class for simple prompt templates.
"""

import logging
import string
from typing import Dict, Any, List, Optional, Union

from .base_prompt_template import BasePromptTemplate

# Configure logging
logger = logging.getLogger(__name__)


class BasicPromptTemplate(BasePromptTemplate):
    """
    A simple prompt template that replaces placeholders with provided values.
    
    This template uses Python's string.format() for variable substitution.
    
    Attributes:
        template: The prompt template string.
        config: Configuration options for the prompt template.
    """
    
    def __init__(self, template: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the basic prompt template.
        
        Args:
            template: The prompt template string with {variable} placeholders.
            config: Configuration options for the prompt template.
                   - strip_whitespace: Whether to strip leading/trailing whitespace (default: True)
                   - strip_newlines: Whether to strip excessive newlines (default: False)
        """
        super().__init__(template, config or {})
        self.config.setdefault("strip_whitespace", True)
        self.config.setdefault("strip_newlines", False)
    
    def format(self, **kwargs) -> str:
        """
        Format the prompt template with the provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template.
            
        Returns:
            str: The formatted prompt.
            
        Example:
            template = "Hello, {name}! Today is {day}."
            prompt = template.format(name="Alice", day="Monday")
            # Result: "Hello, Alice! Today is Monday."
        """
        try:
            formatted_prompt = self.template.format(**kwargs)
            
            # Apply post-processing
            if self.config.get("strip_whitespace", True):
                formatted_prompt = formatted_prompt.strip()
            
            if self.config.get("strip_newlines", False):
                # Replace multiple newlines with a single newline
                import re
                formatted_prompt = re.sub(r'\n{3,}', '\n\n', formatted_prompt)
            
            return formatted_prompt
        except KeyError as e:
            missing_key = str(e).strip("'")
            logger.error(f"Missing required variable in prompt template: {missing_key}")
            raise ValueError(f"Missing required variable in prompt template: {missing_key}")
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            raise
    
    def list_variables(self) -> List[str]:
        """
        Extract the variable names from the template.
        
        Returns:
            List[str]: List of variable names used in the template.
        """
        # Extract all variables from the template using string.Formatter
        formatter = string.Formatter()
        variables = []
        
        for _, field_name, _, _ in formatter.parse(self.template):
            if field_name is not None and field_name not in variables:
                variables.append(field_name)
        
        return variables
    
    def validate_variables(self, variables: Dict[str, Any]) -> bool:
        """
        Validate that all required variables are provided.
        
        Args:
            variables: Dictionary of variables to validate.
            
        Returns:
            bool: True if all required variables are provided, False otherwise.
        """
        required_vars = self.list_variables()
        provided_vars = list(variables.keys())
        
        return all(var in provided_vars for var in required_vars)
    
    def __repr__(self) -> str:
        """String representation of the BasicPromptTemplate."""
        return f"BasicPromptTemplate(template='{self.template[:50]}...', variables={self.list_variables()})" 