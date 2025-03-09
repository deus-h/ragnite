"""
OpenAI Provider

This module implements the LLMProvider interface for OpenAI models.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Generator, Tuple

from .base_model import LLMProvider, Message, Role

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI implementation of the LLMProvider interface.
    
    This provider supports the following models:
    - GPT-4 family (gpt-4, gpt-4-turbo, gpt-4o, etc.)
    - GPT-3.5 family (gpt-3.5-turbo, etc.)
    - Embeddings (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
    """
    
    # Model context length mapping (in tokens)
    _CONTEXT_LENGTHS = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4o": 128000,
        "text-embedding-ada-002": 8191,
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
    }
    
    # Function calling support
    _FUNCTION_SUPPORT = {
        "gpt-3.5-turbo": True,
        "gpt-3.5-turbo-16k": True,
        "gpt-4": True,
        "gpt-4-32k": True,
        "gpt-4-turbo": True,
        "gpt-4o": True,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
    ):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Default model to use for text generation
            embedding_model: Model to use for embeddings
            organization: OpenAI organization ID
            base_url: Custom base URL for OpenAI API
            timeout: Timeout for API calls
            max_retries: Maximum number of retries for failed API calls
            retry_min_wait: Minimum wait time between retries
            retry_max_wait: Maximum wait time between retries
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "The openai package is required for OpenAIProvider. "
                "Install it with 'pip install openai>=1.0.0'"
            )
        
        self.model = model
        self.embedding_model = embedding_model
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        
        # Initialize the OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
        )
        
        # Initialize tokenizers for different models
        self._tokenizers = {}
        
        # Log initialization
        logger.debug(f"Initialized OpenAIProvider with model {model} and embedding model {embedding_model}")
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal Message objects to OpenAI format.
        
        Args:
            messages: List of internal Message objects
            
        Returns:
            List of messages in OpenAI format
        """
        openai_messages = []
        
        for message in messages:
            openai_message = {
                "role": message.role.value,
                "content": message.content
            }
            
            if message.name:
                openai_message["name"] = message.name
                
            if message.function_call:
                openai_message["function_call"] = message.function_call
                
            openai_messages.append(openai_message)
        
        return openai_messages
    
    def _get_tokenizer(self, model: str):
        """
        Get a tokenizer for the specified model.
        
        Args:
            model: Model name
            
        Returns:
            Tokenizer for the model
            
        Raises:
            ImportError: If tiktoken is not installed
            ValueError: If the model is not supported
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError(
                "The tiktoken package is required for token counting. "
                "Install it with 'pip install tiktoken'"
            )
        
        if model not in self._tokenizers:
            try:
                if model.startswith("gpt-4"):
                    encoding_name = "cl100k_base"
                elif model.startswith("gpt-3.5"):
                    encoding_name = "cl100k_base"
                elif model.startswith("text-embedding-ada"):
                    encoding_name = "cl100k_base"
                elif model.startswith("text-embedding-3"):
                    encoding_name = "cl100k_base"
                else:
                    # Default for newer models
                    encoding_name = "cl100k_base"
                
                self._tokenizers[model] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                raise ValueError(f"Error loading tokenizer for model {model}: {str(e)}")
        
        return self._tokenizers[model]
    
    def _with_retry(self, func, *args, **kwargs):
        """
        Execute a function with exponential backoff retry.
        
        Args:
            func: Function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
            
        Raises:
            Exception: The last exception encountered if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Check if we should retry based on error type
                if hasattr(e, "status_code") and e.status_code in [429, 500, 502, 503, 504]:
                    # Calculate wait time with exponential backoff
                    wait_time = min(
                        self.retry_max_wait,
                        self.retry_min_wait * (2 ** attempt)
                    )
                    
                    logger.warning(
                        f"OpenAI API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    
                    time.sleep(wait_time)
                else:
                    # Non-retriable error
                    raise
        
        # If we reach here, all retries failed
        logger.error(f"OpenAI API request failed after {self.max_retries} attempts: {str(last_exception)}")
        raise last_exception
    
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
            **kwargs: Additional OpenAI-specific parameters
                - model: Override the default model
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Penalty for token frequency
                - presence_penalty: Penalty for token presence
                - stop: Sequences where the API will stop generating
                - seed: Random seed for reproducibility
                
        Returns:
            Dictionary with the completion and other metadata
        """
        # Get the model to use (default or override)
        model = kwargs.pop("model", self.model)
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)
        
        # Set up the parameters
        params = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        
        # Add optional parameters
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if functions is not None and self.supports_functions(model):
            if 'tools' not in kwargs:  # Use 'tools' for newer OpenAI models
                tools = [{"type": "function", "function": function} for function in functions]
                params["tools"] = tools
        if function_call is not None and self.supports_functions(model):
            if 'tool_choice' not in kwargs:  # Use 'tool_choice' for newer OpenAI models
                if isinstance(function_call, str):
                    if function_call == "auto":
                        params["tool_choice"] = "auto"
                    else:
                        params["tool_choice"] = {"type": "function", "function": {"name": function_call}}
                elif isinstance(function_call, dict):
                    params["tool_choice"] = {"type": "function", "function": function_call}
        
        # Add any remaining kwargs
        params.update(kwargs)
        
        # Remove any conflicting parameters
        if 'tools' in params and 'functions' in params:
            del params['functions']
        if 'tool_choice' in params and 'function_call' in params:
            del params['function_call']
        
        # Make the API call with retry
        try:
            response = self._with_retry(
                self.client.chat.completions.create,
                **params
            )
            
            # Convert the response to a standard format
            result = {
                "content": response.choices[0].message.content,
                "role": response.choices[0].message.role,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # Add tool calls / function calls if present
            tool_calls = getattr(response.choices[0].message, 'tool_calls', None)
            if tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tool.id,
                        "type": tool.type,
                        "function": {
                            "name": tool.function.name,
                            "arguments": tool.function.arguments
                        }
                    } for tool in tool_calls
                ]
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            raise
    
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
            **kwargs: Additional OpenAI-specific parameters
                
        Returns:
            Generator yielding dictionaries with completion chunks and metadata
        """
        # Get the model to use (default or override)
        model = kwargs.pop("model", self.model)
        
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages)
        
        # Set up the parameters
        params = {
            "model": model,
            "messages": openai_messages,
            "temperature": temperature,
            "stream": True
        }
        
        # Add optional parameters (same as for generate)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if functions is not None and self.supports_functions(model):
            if 'tools' not in kwargs:
                tools = [{"type": "function", "function": function} for function in functions]
                params["tools"] = tools
        if function_call is not None and self.supports_functions(model):
            if 'tool_choice' not in kwargs:
                if isinstance(function_call, str):
                    if function_call == "auto":
                        params["tool_choice"] = "auto"
                    else:
                        params["tool_choice"] = {"type": "function", "function": {"name": function_call}}
                elif isinstance(function_call, dict):
                    params["tool_choice"] = {"type": "function", "function": function_call}
        
        # Add any remaining kwargs
        params.update(kwargs)
        
        # Remove any conflicting parameters
        if 'tools' in params and 'functions' in params:
            del params['functions']
        if 'tool_choice' in params and 'function_call' in params:
            del params['function_call']
        
        # Make the API call
        try:
            stream = self.client.chat.completions.create(**params)
            
            # Process the stream
            for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Extract the content chunk
                content = delta.content or ""
                
                # Create and yield a standard chunk format
                result = {
                    "content": content,
                    "done": chunk.choices[0].finish_reason is not None,
                }
                
                # Add tool calls if present
                tool_calls = getattr(delta, 'tool_calls', None)
                if tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tool.id,
                            "type": tool.type,
                            "function": {
                                "name": tool.function.name,
                                "arguments": tool.function.arguments
                            }
                        } for tool in tool_calls
                    ]
                
                yield result
                
        except Exception as e:
            logger.error(f"Error streaming completion: {str(e)}")
            raise
    
    def embed(self,
             texts: List[str],
             **kwargs) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            **kwargs: Additional parameters
                - model: Override the default embedding model
                
        Returns:
            List of embedding vectors, one for each input text
        """
        # Get the model to use
        model = kwargs.pop("model", self.embedding_model)
        
        # Handle the case of a single text
        if not isinstance(texts, list):
            texts = [texts]
        
        # Make the API call with retry
        try:
            response = self._with_retry(
                self.client.embeddings.create,
                model=model,
                input=texts,
                **kwargs
            )
            
            # Extract and return the embeddings
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            model: Model to use for tokenization (defaults to the provider's model)
            
        Returns:
            Number of tokens
        """
        if model is None:
            model = self.model
            
        tokenizer = self._get_tokenizer(model)
        return len(tokenizer.encode(text))
    
    def get_max_context_length(self, model: Optional[str] = None) -> int:
        """
        Get the maximum context length supported by the model.
        
        Args:
            model: Model name (defaults to the provider's model)
            
        Returns:
            Maximum number of tokens the model can handle
        """
        if model is None:
            model = self.model
            
        # Get the base model name by removing any version suffix
        base_model = model.split("-")[0] + "-" + model.split("-")[1]
        
        # Check if we have a specific context length for this model
        if model in self._CONTEXT_LENGTHS:
            return self._CONTEXT_LENGTHS[model]
        elif base_model in self._CONTEXT_LENGTHS:
            return self._CONTEXT_LENGTHS[base_model]
        else:
            # Default for unknown models
            logger.warning(f"Unknown model: {model}, using default context length")
            return 4096
    
    def supports_functions(self, model: Optional[str] = None) -> bool:
        """
        Check if the model supports function calling.
        
        Args:
            model: Model name (defaults to the provider's model)
            
        Returns:
            True if the model supports function calling, False otherwise
        """
        if model is None:
            model = self.model
            
        # Get the base model name
        base_model = model.split("-")[0] + "-" + model.split("-")[1]
        
        # Check if we have specific information about function support
        if model in self._FUNCTION_SUPPORT:
            return self._FUNCTION_SUPPORT[model]
        elif base_model in self._FUNCTION_SUPPORT:
            return self._FUNCTION_SUPPORT[base_model]
        else:
            # Default to False for unknown models
            logger.warning(f"Unknown model: {model}, assuming no function support")
            return False 