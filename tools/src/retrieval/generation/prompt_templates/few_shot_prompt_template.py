"""
Few-Shot Prompt Template

This module provides the FewShotPromptTemplate class for creating prompts with few-shot examples.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable

from .basic_prompt_template import BasicPromptTemplate

# Configure logging
logger = logging.getLogger(__name__)


class FewShotPromptTemplate(BasicPromptTemplate):
    """
    A prompt template that includes few-shot examples before the actual prompt.
    
    This template allows for demonstrating the expected behavior through examples
    before presenting the actual task.
    
    Attributes:
        template: The main prompt template string.
        example_template: Template for formatting each example.
        examples: List of example dictionaries.
        example_separator: String to use between examples.
        prefix: Text that comes before the examples.
        suffix: Text that comes after the examples and before the main prompt.
        config: Configuration options for the prompt template.
    """
    
    def __init__(self, 
                 template: str,
                 example_template: str,
                 examples: List[Dict[str, Any]],
                 example_separator: str = "\n\n",
                 prefix: str = "",
                 suffix: str = "\n\n",
                 example_selector: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the few-shot prompt template.
        
        Args:
            template: The main prompt template string with {variable} placeholders.
            example_template: Template string for formatting each example.
            examples: List of example dictionaries, each containing variables for the example_template.
            example_separator: String to use between examples.
            prefix: Text that comes before the examples.
            suffix: Text that comes after the examples and before the main prompt.
            example_selector: Optional function to select/filter examples based on the input.
            config: Configuration options for the prompt template.
                   - max_examples: Maximum number of examples to include (default: all)
                   - randomize_examples: Whether to randomize the order of examples (default: False)
        """
        super().__init__(template, config or {})
        self.example_template = example_template
        self.examples = examples
        self.example_separator = example_separator
        self.prefix = prefix
        self.suffix = suffix
        self.example_selector = example_selector
        
        self.config.setdefault("max_examples", len(examples))
        self.config.setdefault("randomize_examples", False)
    
    def format(self, **kwargs) -> str:
        """
        Format the few-shot prompt template with the provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template.
            
        Returns:
            str: The formatted prompt with few-shot examples.
        """
        # Get examples to use
        examples_to_use = self.examples
        
        # Apply example selector if provided
        if self.example_selector is not None:
            examples_to_use = self.example_selector(examples_to_use, kwargs)
        
        # Apply max_examples limit
        max_examples = self.config.get("max_examples", len(examples_to_use))
        examples_to_use = examples_to_use[:max_examples]
        
        # Randomize examples if configured
        if self.config.get("randomize_examples", False):
            import random
            random.shuffle(examples_to_use)
        
        # Format each example
        formatted_examples = []
        for example in examples_to_use:
            try:
                formatted_example = self.example_template.format(**example)
                formatted_examples.append(formatted_example)
            except KeyError as e:
                logger.warning(f"Skipping example due to missing key: {str(e)}")
                continue
        
        # Combine examples with separator
        examples_text = self.example_separator.join(formatted_examples)
        
        # Format the main template
        main_prompt = super().format(**kwargs)
        
        # Combine everything
        full_prompt = f"{self.prefix}{examples_text}{self.suffix}{main_prompt}"
        
        return full_prompt
    
    def add_example(self, example: Dict[str, Any]) -> None:
        """
        Add a new example to the examples list.
        
        Args:
            example: Dictionary containing variables for the example_template.
        """
        self.examples.append(example)
    
    def set_examples(self, examples: List[Dict[str, Any]]) -> None:
        """
        Replace the entire examples list.
        
        Args:
            examples: New list of example dictionaries.
        """
        self.examples = examples
    
    def set_example_template(self, example_template: str) -> None:
        """
        Set the example template.
        
        Args:
            example_template: New template string for formatting examples.
        """
        self.example_template = example_template
    
    def set_example_selector(self, example_selector: Callable) -> None:
        """
        Set the example selector function.
        
        Args:
            example_selector: Function to select examples based on the input.
        """
        self.example_selector = example_selector
    
    def __repr__(self) -> str:
        """String representation of the FewShotPromptTemplate."""
        return (f"FewShotPromptTemplate(examples={len(self.examples)}, "
                f"template='{self.template[:30]}...', "
                f"example_template='{self.example_template[:30]}...')") 