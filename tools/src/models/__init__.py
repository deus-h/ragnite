"""
Model Provider Abstraction

This module provides a unified abstraction layer for different LLM providers,
making it easy to switch between different models and providers.
"""

from .base_model import LLMProvider, Message, Role
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .local_model_provider import LocalModelProvider
from .model_factory import get_model_provider

__all__ = [
    'LLMProvider',
    'Message',
    'Role',
    'OpenAIProvider',
    'AnthropicProvider',
    'LocalModelProvider',
    'get_model_provider'
] 