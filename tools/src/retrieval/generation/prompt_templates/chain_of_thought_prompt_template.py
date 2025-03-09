"""
Chain of Thought Prompt Template

This module provides the ChainOfThoughtPromptTemplate class for prompts that encourage
step-by-step reasoning.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable

from .few_shot_prompt_template import FewShotPromptTemplate

# Configure logging
logger = logging.getLogger(__name__)


class ChainOfThoughtPromptTemplate(FewShotPromptTemplate):
    """
    A prompt template that encourages step-by-step reasoning.
    
    Chain of Thought prompting helps language models break down complex problems
    into smaller steps, often improving performance on reasoning tasks.
    
    This template extends FewShotPromptTemplate with reasoning-specific features,
    making it easy to create prompts that demonstrate and encourage step-by-step thinking.
    
    Attributes:
        template: The main prompt template string.
        example_template: Template for formatting each reasoning example.
        examples: List of example dictionaries, each containing a question, reasoning steps, and answer.
        reasoning_prefix: Text that introduces the reasoning section in examples and the main prompt.
        answer_prefix: Text that introduces the answer section in examples and the main prompt.
        config: Configuration options for the prompt template.
    """
    
    def __init__(self, 
                 template: str,
                 example_template: Optional[str] = None,
                 examples: Optional[List[Dict[str, Any]]] = None,
                 reasoning_prefix: str = "Let's think step by step:\n",
                 answer_prefix: str = "Therefore, the answer is: ",
                 example_separator: str = "\n\n",
                 prefix: str = "I'll solve this by thinking step by step.\n\nHere are some examples:\n\n",
                 suffix: str = "\n\nNow let me solve the new problem:\n\n",
                 example_selector: Optional[Callable] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chain of thought prompt template.
        
        Args:
            template: The main prompt template string with {variable} placeholders.
            example_template: Template string for formatting each example. If None, a default template is used.
            examples: List of example dictionaries, each containing variables for the example_template.
            reasoning_prefix: Text that introduces the reasoning section.
            answer_prefix: Text that introduces the answer section.
            example_separator: String to use between examples.
            prefix: Text that comes before the examples.
            suffix: Text that comes after the examples.
            example_selector: Optional function to select examples based on the input.
            config: Configuration options for the prompt template.
                   - include_answer_in_query: Whether to include the answer prefix in the query (default: True).
                   - include_reasoning_prefix: Whether to include the reasoning prefix (default: True).
                   - additional_reasoning_prompts: List of additional prompts to encourage reasoning (default: []).
        """
        # Default example template if none provided
        if example_template is None:
            example_template = ("Question: {question}\n\n"
                               f"{reasoning_prefix}{{reasoning}}\n\n"
                               f"{answer_prefix}{{answer}}")
        
        super().__init__(
            template=template,
            example_template=example_template,
            examples=examples or [],
            example_separator=example_separator,
            prefix=prefix,
            suffix=suffix,
            example_selector=example_selector,
            config=config or {}
        )
        
        self.reasoning_prefix = reasoning_prefix
        self.answer_prefix = answer_prefix
        
        self.config.setdefault("include_answer_in_query", True)
        self.config.setdefault("include_reasoning_prefix", True)
        self.config.setdefault("additional_reasoning_prompts", [])
    
    def format(self, **kwargs) -> str:
        """
        Format the chain of thought prompt template with the provided variables.
        
        This method first formats examples with questions, reasoning steps, and answers,
        then formats the main prompt with the new question, and optionally adds
        reasoning and answer prefixes.
        
        Args:
            **kwargs: Variables to substitute in the template.
            
        Returns:
            str: The formatted prompt with chain of thought examples and query.
        """
        # Get the base formatted prompt from the parent class
        formatted_prompt = super().format(**kwargs)
        
        # Append reasoning prefix if configured
        if self.config.get("include_reasoning_prefix", True):
            formatted_prompt += f"\n\n{self.reasoning_prefix}"
        
        # Append additional reasoning prompts if any
        additional_prompts = self.config.get("additional_reasoning_prompts", [])
        if additional_prompts:
            formatted_prompt += "\n" + "\n".join(additional_prompts)
        
        # Append answer prefix if configured
        if self.config.get("include_answer_in_query", True):
            formatted_prompt += f"\n\n{self.answer_prefix}"
        
        return formatted_prompt
    
    def set_reasoning_prefix(self, reasoning_prefix: str) -> None:
        """
        Set the reasoning prefix.
        
        Args:
            reasoning_prefix: Text that introduces the reasoning section.
        """
        self.reasoning_prefix = reasoning_prefix
    
    def set_answer_prefix(self, answer_prefix: str) -> None:
        """
        Set the answer prefix.
        
        Args:
            answer_prefix: Text that introduces the answer section.
        """
        self.answer_prefix = answer_prefix
    
    def add_reasoning_prompt(self, prompt: str) -> None:
        """
        Add an additional reasoning prompt.
        
        Args:
            prompt: Additional prompt to encourage reasoning.
        """
        additional_prompts = self.config.get("additional_reasoning_prompts", [])
        additional_prompts.append(prompt)
        self.config["additional_reasoning_prompts"] = additional_prompts
    
    def __repr__(self) -> str:
        """String representation of the ChainOfThoughtPromptTemplate."""
        return (f"ChainOfThoughtPromptTemplate(examples={len(self.examples)}, "
                f"template='{self.template[:30]}...')") 