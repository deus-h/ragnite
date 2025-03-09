"""
Anthropic Provider

This module implements the LLMProvider interface for Anthropic Claude models.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Union, Generator

from .base_model import LLMProvider, Message, Role

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """
    Anthropic implementation of the LLMProvider interface.
    
    This provider supports Claude models, including:
    - claude-3-opus
    - claude-3-sonnet
    - claude-3-haiku
    - claude-2
    - claude-instant
    """
    
    # Model context length mapping (in tokens)
    _CONTEXT_LENGTHS = {
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-2": 100000,
        "claude-2.1": 200000,
        "claude-instant": 100000,
        "claude-instant-1.2": 100000,
    }
    
    # Function calling support
    _FUNCTION_SUPPORT = {
        "claude-3-opus": True,
        "claude-3-sonnet": True,
        "claude-3-haiku": True,
        "claude-2": False,
        "claude-2.1": False,
        "claude-instant": False,
        "claude-instant-1.2": False,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
    ):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)
            model: Default model to use for text generation
            timeout: Timeout for API calls
            max_retries: Maximum number of retries for failed API calls
            retry_min_wait: Minimum wait time between retries
            retry_max_wait: Maximum wait time between retries
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The anthropic package is required for AnthropicProvider. "
                "Install it with 'pip install anthropic'"
            )
        
        self.model = model
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        
        # Initialize the Anthropic client
        self.client = Anthropic(
            api_key=api_key,
            timeout=timeout,
        )
        
        # Log initialization
        logger.debug(f"Initialized AnthropicProvider with model {model}")
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal Message objects to Anthropic format.
        
        Args:
            messages: List of internal Message objects
            
        Returns:
            List of messages in Anthropic format
        """
        # Anthropic requires a specific message format that differs from OpenAI
        # We need to handle system messages differently
        system_message = None
        anthropic_messages = []
        
        for message in messages:
            if message.role == Role.SYSTEM:
                # Claude expects only one system message, save it separately
                system_message = message.content
            elif message.role == Role.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": message.content
                })
            elif message.role == Role.ASSISTANT:
                anthropic_messages.append({
                    "role": "assistant",
                    "content": message.content
                })
            elif message.role == Role.FUNCTION:
                # Anthropic doesn't support function messages directly
                # We need to convert them to user messages with a special format
                anthropic_messages.append({
                    "role": "user",
                    "content": f"Function result from '{message.name}':\n```json\n{message.content}\n```"
                })
        
        return anthropic_messages, system_message
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        # As a rough approximation, divide by 4 for characters per token
        # This is not accurate but provides a reasonable estimate
        return max(1, len(text) // 4)
    
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
                if (
                    # Rate limit or server error status codes
                    (hasattr(e, "status_code") and e.status_code in [429, 500, 502, 503, 504]) or
                    # Check for anthropic specific error types
                    "RateLimitError" in str(type(e)) or 
                    "ServiceUnavailable" in str(type(e)) or
                    "Timeout" in str(type(e))
                ):
                    # Calculate wait time with exponential backoff
                    wait_time = min(
                        self.retry_max_wait,
                        self.retry_min_wait * (2 ** attempt)
                    )
                    
                    logger.warning(
                        f"Anthropic API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    
                    time.sleep(wait_time)
                else:
                    # Non-retriable error
                    raise
        
        # If we reach here, all retries failed
        logger.error(f"Anthropic API request failed after {self.max_retries} attempts: {str(last_exception)}")
        raise last_exception
    
    def _format_tool_use(self, tool_use: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a tool use response from Anthropic.
        
        Args:
            tool_use: Tool use data from Anthropic API
            
        Returns:
            Formatted tool call
        """
        return {
            "name": tool_use["name"],
            "arguments": json.loads(tool_use["input"])
        }
    
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
            **kwargs: Additional Anthropic-specific parameters
                - model: Override the default model
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - stop_sequences: Sequences where the API will stop generating
                
        Returns:
            Dictionary with the completion and other metadata
        """
        # Get the model to use (default or override)
        model = kwargs.pop("model", self.model)
        
        # Convert messages to Anthropic format
        anthropic_messages, system_message = self._convert_messages(messages)
        
        # Set up the parameters
        params = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
        }
        
        # Add system message if present
        if system_message:
            params["system"] = system_message
        
        # Add optional parameters
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Handle tools (functions)
        if functions and self.supports_functions(model):
            # Convert to Anthropic's tool format
            tools = []
            for function in functions:
                tools.append({
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "input_schema": function["parameters"]
                })
            
            params["tools"] = tools
        
        # Add any remaining kwargs (like top_p, top_k, etc.)
        params.update(kwargs)
        
        # Make the API call with retry
        try:
            response = self._with_retry(
                self.client.messages.create,
                **params
            )
            
            # Convert the response to a standard format
            result = {
                "content": response.content[0].text,
                "role": "assistant",
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
            
            # Add tool calls / function calls if present
            tool_calls = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "type": "function",
                        "function": self._format_tool_use(block.tool_use)
                    })
            
            if tool_calls:
                result["tool_calls"] = tool_calls
            
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
            **kwargs: Additional Anthropic-specific parameters
                
        Returns:
            Generator yielding dictionaries with completion chunks and metadata
        """
        # Get the model to use (default or override)
        model = kwargs.pop("model", self.model)
        
        # Convert messages to Anthropic format
        anthropic_messages, system_message = self._convert_messages(messages)
        
        # Set up the parameters
        params = {
            "model": model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "stream": True
        }
        
        # Add system message if present
        if system_message:
            params["system"] = system_message
        
        # Add optional parameters
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Handle tools (functions)
        if functions and self.supports_functions(model):
            # Convert to Anthropic's tool format
            tools = []
            for function in functions:
                tools.append({
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "input_schema": function["parameters"]
                })
            
            params["tools"] = tools
        
        # Add any remaining kwargs
        params.update(kwargs)
        
        # Make the API call
        try:
            stream = self.client.messages.create(**params)
            
            # Process the stream
            for chunk in stream:
                # Handle text content
                if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                    yield {
                        "content": chunk.delta.text,
                        "done": False,
                    }
                # Handle tool use
                elif chunk.type == "content_block_delta" and chunk.delta.type == "tool_use_delta":
                    tool_use_delta = chunk.delta
                    
                    # For tool use, we need to parse the input
                    if hasattr(tool_use_delta, "input") and tool_use_delta.input:
                        try:
                            tool_input = json.loads(tool_use_delta.input)
                            
                            yield {
                                "tool_calls": [{
                                    "type": "function",
                                    "function": {
                                        "name": tool_use_delta.name,
                                        "arguments": tool_input
                                    }
                                }],
                                "content": "",
                                "done": False,
                            }
                        except json.JSONDecodeError:
                            # If the input isn't complete JSON yet, just yield an empty update
                            pass
                
                # Handle message stop
                elif chunk.type == "message_stop":
                    yield {
                        "content": "",
                        "done": True,
                    }
                
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
                
        Returns:
            List of embedding vectors, one for each input text
        """
        # Anthropic doesn't have a dedicated embeddings endpoint (yet)
        # We could use a third-party embedding model, but for now raise an error
        raise NotImplementedError(
            "Embedding is not supported by the AnthropicProvider. "
            "Please use another provider for embeddings."
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            # Use the Anthropic tokenizer if available
            if hasattr(self.client, "count_tokens"):
                response = self.client.count_tokens(text)
                return response.tokens
            else:
                # Fallback to estimate
                return self._estimate_tokens(text)
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}. Using estimate.")
            return self._estimate_tokens(text)
    
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
            return 100000
    
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
            # Only Claude 3 supports tool/function calling
            return "claude-3" in model 