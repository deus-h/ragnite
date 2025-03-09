"""
Prompt Templates

This module provides various prompt templates for generating prompts
for language models.
"""

from .base_prompt_template import BasePromptTemplate
from .basic_prompt_template import BasicPromptTemplate
from .few_shot_prompt_template import FewShotPromptTemplate
from .chain_of_thought_prompt_template import ChainOfThoughtPromptTemplate
from .structured_prompt_template import StructuredPromptTemplate
from .factory import get_prompt_template

__all__ = [
    'BasePromptTemplate',
    'BasicPromptTemplate',
    'FewShotPromptTemplate',
    'ChainOfThoughtPromptTemplate',
    'StructuredPromptTemplate',
    'get_prompt_template',
] 