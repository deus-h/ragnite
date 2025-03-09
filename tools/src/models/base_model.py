"""
Base Model Interface

This module defines the abstract base class for all model providers.
"""

import abc
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Union, Generator, Callable
from dataclasses import dataclass


class Role(str, Enum):
    """
    Enum for message roles in LLM conversations.
    
    This matches the common roles used across different providers.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class Message:
    """
    Represents a message in an LLM conversation.
    """
    role: Role
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class LLMProvider(abc.ABC):
    """
    Abstract base class for model providers.
    
    This defines the interface that all model providers must implement.
    """
    
    @abc.abstractmethod
    def generate(self, 
                messages: List[Message], 
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                functions: Optional[List[Dict[str, Any]]] = None,
                function_call: Optional[Union[str, Dict[str, Any]]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate a completion for a list of messages.
        
        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            functions: List of functions that the model can call
            function_call: Controls which function is called, if any
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary with the completion and other metadata
        """
        pass
    
    @abc.abstractmethod
    def generate_stream(self, 
                       messages: List[Message], 
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       functions: Optional[List[Dict[str, Any]]] = None,
                       function_call: Optional[Union[str, Dict[str, Any]]] = None,
                       **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a completion for a list of messages.
        
        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum number of tokens to generate
            functions: List of functions that the model can call
            function_call: Controls which function is called, if any
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generator yielding dictionaries with completion chunks and metadata
        """
        pass
    
    @abc.abstractmethod
    def embed(self,
             texts: List[str],
             **kwargs) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of embedding vectors, one for each input text
        """
        pass
    
    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    @abc.abstractmethod
    def get_max_context_length(self) -> int:
        """
        Get the maximum context length supported by the model.
        
        Returns:
            Maximum number of tokens the model can handle
        """
        pass
    
    @abc.abstractmethod
    def supports_functions(self) -> bool:
        """
        Check if the model supports function calling.
        
        Returns:
            True if the model supports function calling, False otherwise
        """
        pass 