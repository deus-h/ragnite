"""
Google AI Provider (Gemini)

This module implements the LLMProvider interface for Google's Gemini models.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Union, Generator

from .base_model import LLMProvider, Message, Role

# Configure logging
logger = logging.getLogger(__name__)

# Try importing the Google Generative AI client library
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False


class GoogleAIProvider(LLMProvider):
    """
    Google AI implementation of the LLMProvider interface.
    
    This provider supports Gemini models, including:
    - gemini-pro
    - gemini-pro-vision
    - gemini-ultra
    """
    
    # Model context length mapping (in tokens)
    _CONTEXT_LENGTHS = {
        "gemini-pro": 32768,
        "gemini-pro-vision": 16384,
        "gemini-ultra": 32768,
    }
    
    # Function calling support
    _FUNCTION_SUPPORT = {
        "gemini-pro": True,
        "gemini-pro-vision": False,
        "gemini-ultra": True,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro",
        project_id: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
    ):
        """
        Initialize the Google AI provider.
        
        Args:
            api_key: Google AI API key (defaults to GOOGLE_API_KEY environment variable)
            model: Default model to use for text generation
            project_id: Google Cloud project ID (optional)
            timeout: Timeout for API calls
            max_retries: Maximum number of retries for failed API calls
            retry_min_wait: Minimum wait time between retries
            retry_max_wait: Maximum wait time between retries
        """
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError(
                "The google-generativeai package is required for GoogleAIProvider. "
                "Install it with 'pip install google-generativeai'"
            )
        
        self.model = model
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        
        # Initialize the Google Generative AI client
        genai.configure(api_key=api_key)
        
        # Log initialization
        logger.debug(f"Initialized GoogleAIProvider with model {model}")
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        Convert internal Message objects to Google AI format.
        
        Args:
            messages: List of internal Message objects
            
        Returns:
            List of messages in Google AI format
        """
        google_messages = []
        
        for message in messages:
            if message.role == Role.SYSTEM:
                # Google uses a different format for system messages
                google_messages.append({
                    "role": "user",
                    "parts": [{"text": f"System: {message.content}"}]
                })
            elif message.role == Role.USER:
                google_messages.append({
                    "role": "user",
                    "parts": [{"text": message.content}]
                })
            elif message.role == Role.ASSISTANT:
                google_messages.append({
                    "role": "model",
                    "parts": [{"text": message.content}]
                })
            elif message.role == Role.FUNCTION:
                # Google's format for function messages may differ
                google_messages.append({
                    "role": "user",
                    "parts": [{"text": f"Function result from '{message.name}':\n```json\n{message.content}\n```"}]
                })
        
        return google_messages
    
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
                    hasattr(e, "status_code") and 
                    getattr(e, "status_code", 0) in [429, 500, 502, 503, 504]
                ):
                    # Calculate wait time with exponential backoff
                    wait_time = min(
                        self.retry_max_wait,
                        self.retry_min_wait * (2 ** attempt)
                    )
                    
                    logger.warning(
                        f"Google AI API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}. "
                        f"Retrying in {wait_time:.2f} seconds..."
                    )
                    
                    time.sleep(wait_time)
                else:
                    # Non-retriable error
                    raise
        
        # If we reach here, all retries failed
        logger.error(f"Google AI API request failed after {self.max_retries} attempts: {str(last_exception)}")
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
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dictionary with the completion and other metadata
        """
        # Get the model to use (default or override)
        model = kwargs.pop("model", self.model)
        
        # Convert messages to Google AI format
        google_messages = self._convert_messages(messages)
        
        # Set up the parameters
        generation_config = {
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        # Call the API with retry logic
        def _generate():
            # This is a placeholder implementation using the Google Generative AI SDK
            # The actual implementation may differ based on the official documentation
            gemini_model = genai.GenerativeModel(model_name=model)
            response = gemini_model.generate_content(
                google_messages,
                generation_config=generation_config,
                **kwargs
            )
            
            # Convert the response to the expected format
            return {
                "content": response.text,
                "model": model,
                "finish_reason": "stop",  # Replace with actual finish reason
                "usage": {
                    "prompt_tokens": 0,  # Replace with actual count
                    "completion_tokens": 0,  # Replace with actual count
                    "total_tokens": 0,  # Replace with actual count
                },
            }
        
        try:
            return self._with_retry(_generate)
        except Exception as e:
            # Log the error
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
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generator yielding dictionaries with completion chunks and metadata
        """
        # Get the model to use (default or override)
        model = kwargs.pop("model", self.model)
        
        # Convert messages to Google AI format
        google_messages = self._convert_messages(messages)
        
        # Set up the parameters
        generation_config = {
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        
        # Stream the response
        try:
            # This is a placeholder implementation
            gemini_model = genai.GenerativeModel(model_name=model)
            stream = gemini_model.generate_content(
                google_messages,
                generation_config=generation_config,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                yield {
                    "content": chunk.text,
                    "model": model,
                    "finish_reason": None,  # Only the last chunk has a finish reason
                    "usage": None,  # Only the last chunk has usage info
                }
                
        except Exception as e:
            # Log the error
            logger.error(f"Error streaming completion: {str(e)}")
            raise
    
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
        # This is a placeholder implementation
        # The actual implementation will depend on the Google Embeddings API
        try:
            # Example implementation
            embedding_model = genai.GenerativeModel(model_name="embedding-001")
            results = []
            
            for text in texts:
                embedding = embedding_model.embed_content(text)
                results.append(embedding.values)
                
            return results
                
        except Exception as e:
            # Log the error
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        # This is a placeholder implementation
        # Google may provide a tokenizer in their SDK
        try:
            # Example implementation
            # return genai.count_tokens(text)
            
            # Fallback to approximation if not available
            return max(1, len(text) // 4)
                
        except Exception as e:
            # Log the error
            logger.error(f"Error counting tokens: {str(e)}")
            # Fallback to approximation
            return max(1, len(text) // 4)
    
    def get_max_context_length(self) -> int:
        """
        Get the maximum context length supported by the model.
        
        Returns:
            Maximum number of tokens the model can handle
        """
        return self._CONTEXT_LENGTHS.get(self.model, 32768)  # Default to 32k if unknown
    
    def supports_functions(self) -> bool:
        """
        Check if the model supports function calling.
        
        Returns:
            True if the model supports function calling, False otherwise
        """
        return self._FUNCTION_SUPPORT.get(self.model, False) 