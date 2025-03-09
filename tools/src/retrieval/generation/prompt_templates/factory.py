"""
Prompt Template Factory

This module provides a factory function for creating prompt templates.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable

from .base_prompt_template import BasePromptTemplate
from .basic_prompt_template import BasicPromptTemplate
from .few_shot_prompt_template import FewShotPromptTemplate
from .chain_of_thought_prompt_template import ChainOfThoughtPromptTemplate
from .structured_prompt_template import StructuredPromptTemplate

# Configure logging
logger = logging.getLogger(__name__)


def get_prompt_template(
    template_type: str,
    template: str,
    examples: Optional[List[Dict[str, Any]]] = None,
    example_template: Optional[str] = None,
    output_format: Optional[str] = None,
    schema: Optional[Union[Dict, str, List[Dict]]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BasePromptTemplate:
    """
    Factory function to create prompt templates.
    
    Args:
        template_type: Type of prompt template ('basic', 'few_shot', 'chain_of_thought', 'structured').
        template: The prompt template string with placeholders.
        examples: List of example dictionaries (for few_shot and chain_of_thought templates).
        example_template: Template for formatting examples (for few_shot and chain_of_thought templates).
        output_format: Output format for structured templates (json, xml, etc.).
        schema: Schema definition for structured templates.
        config: Configuration options for the prompt template.
        **kwargs: Additional arguments passed to the specific template class.
    
    Returns:
        BasePromptTemplate: An instance of the requested prompt template.
    
    Raises:
        ValueError: If the template_type is not supported.
    """
    template_type = template_type.lower()
    config = config or {}
    
    if template_type == "basic":
        return BasicPromptTemplate(template=template, config=config, **kwargs)
    
    elif template_type == "few_shot":
        if examples is None:
            logger.warning("No examples provided for few_shot template. Using empty list.")
            examples = []
        
        if example_template is None:
            logger.warning("No example_template provided for few_shot template. Using default.")
            example_template = "Input: {input}\nOutput: {output}"
        
        return FewShotPromptTemplate(
            template=template,
            example_template=example_template,
            examples=examples,
            config=config,
            **kwargs
        )
    
    elif template_type == "chain_of_thought":
        if examples is None:
            logger.warning("No examples provided for chain_of_thought template. Using empty list.")
            examples = []
        
        return ChainOfThoughtPromptTemplate(
            template=template,
            example_template=example_template,
            examples=examples,
            config=config,
            **kwargs
        )
    
    elif template_type == "structured":
        if output_format is None:
            logger.warning("No output_format provided for structured template. Using 'json'.")
            output_format = "json"
        
        return StructuredPromptTemplate(
            template=template,
            output_format=output_format,
            schema=schema,
            config=config,
            **kwargs
        )
    
    else:
        logger.error(f"Unsupported template type: {template_type}")
        raise ValueError(f"Unsupported template type: {template_type}. "
                        f"Supported types: basic, few_shot, chain_of_thought, structured") 