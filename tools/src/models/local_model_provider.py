"""
Local Model Provider

This module implements the LLMProvider interface for local models like Llama and Mistral.
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional, Union, Generator, Tuple

from .base_model import LLMProvider, Message, Role

try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TextIteratorStreamer,
        pipeline
    )
    import torch
    from threading import Thread
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import sentence_transformers
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class LocalModelProvider(LLMProvider):
    """
    Local model implementation of the LLMProvider interface.
    
    This provider supports locally hosted models:
    - Llama 2 and 3 family
    - Mistral family
    - Other HuggingFace compatible models
    
    For embeddings, it uses sentence-transformers by default.
    """
    
    # Default model context length mapping (in tokens)
    _CONTEXT_LENGTHS = {
        "llama-2-7b": 4096,
        "llama-2-13b": 4096,
        "llama-2-70b": 4096,
        "llama-3-8b": 8192,
        "llama-3-70b": 8192,
        "mistral-7b": 8192,
        "mistral-7b-instruct": 8192,
        "mistral-8x7b-instruct": 32768,
        "mixtral-8x7b": 32768,
    }
    
    # Default function calling support
    _FUNCTION_SUPPORT = {
        # Most local models don't natively support function calling
        # but can be made to work with the right prompt formatting
        "llama-2-7b": False,
        "llama-2-13b": False,
        "llama-2-70b": False,
        "llama-3-8b": True,  # Llama 3 has better function calling support
        "llama-3-70b": True,
        "mistral-7b": False,
        "mistral-7b-instruct": False,
        "mistral-8x7b-instruct": False,
        "mixtral-8x7b": False,
    }
    
    def __init__(
        self,
        model_name_or_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        max_new_tokens: int = 1024,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flash_attention: bool = False,
        system_prompt_template: Optional[str] = None,
        function_calling_template: Optional[str] = None,
    ):
        """
        Initialize the LocalModelProvider.
        
        Args:
            model_name_or_path: Path to local model or HuggingFace model name
            embedding_model: Model to use for embeddings
            device: Device to use for inference ('cpu', 'cuda', 'cuda:0', etc.)
            torch_dtype: PyTorch data type to use (torch.float16, torch.bfloat16, etc.)
            max_new_tokens: Maximum number of tokens to generate
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            use_flash_attention: Whether to use flash attention for faster inference
            system_prompt_template: Custom template for system prompts
            function_calling_template: Custom template for function calling
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The transformers and torch packages are required for LocalModelProvider. "
                "Install them with 'pip install transformers torch'"
            )
        
        self.model_name_or_path = model_name_or_path
        self.embedding_model_name = embedding_model
        self.max_new_tokens = max_new_tokens
        
        # Initialize system prompt template
        self.system_prompt_template = system_prompt_template or "<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        
        # Initialize function calling template
        self.function_calling_template = function_calling_template or (
            "To call a function, respond with a JSON object with the following structure:\n"
            "```json\n{\"function_call\": {\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\", ...}}}\n```\n"
        )
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set torch dtype
        if torch_dtype is None:
            if "cuda" in self.device:
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Set up model loading parameters
        model_kwargs = {
            "device_map": self.device,
            "torch_dtype": self.torch_dtype,
        }
        
        # Add quantization options if specified
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["quantization_config"] = {"bnb_4bit_compute_dtype": torch.bfloat16}
        
        # Add attention options if specified
        if use_flash_attention and "llama" in model_name_or_path.lower():
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Load the model
        try:
            logger.info(f"Loading model {model_name_or_path} to {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
            logger.info(f"Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Initialize embedding model (lazy loading)
        self._embedding_model = None
        
        # Extract model base name for context length and function support lookup
        self.model_base_name = self._extract_base_model_name(model_name_or_path)
        
        logger.debug(f"Initialized LocalModelProvider with model {model_name_or_path} on {self.device}")
    
    def _extract_base_model_name(self, model_name_or_path: str) -> str:
        """
        Extract the base model name for looking up context length and function support.
        
        Args:
            model_name_or_path: Full model name or path
            
        Returns:
            Base model name
        """
        # Check if it's a local path
        if os.path.exists(model_name_or_path):
            # Try to extract from directory name
            model_dir = os.path.basename(os.path.normpath(model_name_or_path))
            
            # Check for common model names in the directory name
            for name in self._CONTEXT_LENGTHS.keys():
                if name in model_dir.lower():
                    return name
            
            # Default fallback
            return "unknown-local-model"
        
        # It's a HuggingFace model name
        model_name = model_name_or_path.lower()
        
        # Check for common model families
        if "llama-3" in model_name:
            if "70b" in model_name:
                return "llama-3-70b"
            else:
                return "llama-3-8b"
        elif "llama-2" in model_name or "llama2" in model_name:
            if "70b" in model_name:
                return "llama-2-70b"
            elif "13b" in model_name:
                return "llama-2-13b"
            else:
                return "llama-2-7b"
        elif "mixtral" in model_name:
            return "mixtral-8x7b"
        elif "mistral" in model_name:
            if "instruct" in model_name:
                return "mistral-7b-instruct"
            else:
                return "mistral-7b"
        
        # Default fallback
        return "unknown-hf-model"
    
    def _ensure_embedding_model_loaded(self):
        """
        Ensure the embedding model is loaded (lazy loading).
        """
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                import sentence_transformers
                logger.info(f"Loading embedding model {self.embedding_model_name}...")
                self._embedding_model = sentence_transformers.SentenceTransformer(
                    self.embedding_model_name,
                    device=self.device
                )
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading embedding model: {str(e)}")
                raise
        elif not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The sentence-transformers package is required for embeddings. "
                "Install it with 'pip install sentence-transformers'"
            )
    
    def _convert_messages_to_prompt(self, 
                                   messages: List[Message], 
                                   functions: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Convert messages to a formatted prompt string suitable for the model.
        
        Args:
            messages: List of messages in the conversation
            functions: Optional list of functions that can be called
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        system_content = None
        
        # Extract system message if present
        for i, message in enumerate(messages):
            if message.role == Role.SYSTEM:
                system_content = message.content
                break
        
        # Add system prompt if present
        if system_content:
            prompt += self.system_prompt_template.format(system_prompt=system_content)
        
        # Add function definitions if present
        if functions and self.supports_functions():
            prompt += "You can call the following functions:\n\n"
            for function in functions:
                prompt += f"Function: {function['name']}\n"
                prompt += f"Description: {function.get('description', 'No description')}\n"
                prompt += "Parameters:\n"
                
                if 'parameters' in function and 'properties' in function['parameters']:
                    for param_name, param_info in function['parameters']['properties'].items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', 'No description')
                        required = param_name in function['parameters'].get('required', [])
                        prompt += f"- {param_name} ({param_type}{', required' if required else ''}): {param_desc}\n"
                
                prompt += "\n"
            
            prompt += self.function_calling_template + "\n\n"
        
        # Process all messages (except system which was handled separately)
        for message in messages:
            if message.role == Role.SYSTEM:
                continue
                
            if message.role == Role.USER:
                prompt += f"User: {message.content}\n\n"
            elif message.role == Role.ASSISTANT:
                prompt += f"Assistant: {message.content}\n\n"
            elif message.role == Role.FUNCTION:
                prompt += f"Function {message.name or 'unknown'} returned: {message.content}\n\n"
        
        # Add final assistant prefix
        prompt += "Assistant: "
        
        return prompt
    
    def _extract_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract function call from generated text.
        
        Args:
            text: Generated text
            
        Returns:
            Function call data if found, None otherwise
        """
        import json
        import re
        
        # Try to find JSON block
        json_pattern = r'```json\n(.*?)\n```'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        if not json_matches:
            # Try without code block formatting
            json_pattern = r'(\{.*"function_call".*\})'
            json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        for json_str in json_matches:
            try:
                data = json.loads(json_str)
                if 'function_call' in data:
                    return data['function_call']
            except json.JSONDecodeError:
                continue
        
        return None
    
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
            **kwargs: Additional parameters
                - top_p: Nucleus sampling parameter
                - top_k: Top-k sampling parameter
                - repetition_penalty: Penalty for token repetition
                - model: Override the default model
                
        Returns:
            Dictionary with the completion and other metadata
        """
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages, functions)
        
        # Set up generation parameters
        gen_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_tokens or self.max_new_tokens,
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            gen_kwargs["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            gen_kwargs["top_k"] = kwargs["top_k"]
        if "repetition_penalty" in kwargs:
            gen_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]
        
        # Count input tokens
        input_tokens = self.count_tokens(prompt)
        
        # Generate text
        try:
            # Encode the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate output
            with torch.no_grad():
                output = self.model.generate(
                    inputs.input_ids,
                    **gen_kwargs
                )
            
            # Decode the output, skipping the input tokens
            decoded_output = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Count output tokens
            output_tokens = self.count_tokens(decoded_output)
            
            # Create the response
            result = {
                "content": decoded_output,
                "role": "assistant",
                "model": self.model_name_or_path,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }
            
            # Check for function calls if functions were provided
            if functions and self.supports_functions():
                function_call_data = self._extract_function_call(decoded_output)
                if function_call_data:
                    result["tool_calls"] = [{
                        "type": "function",
                        "function": function_call_data
                    }]
                    # Clean up content to remove the JSON
                    import re
                    result["content"] = re.sub(r'```json\n.*?\n```', '', result["content"], flags=re.DOTALL).strip()
            
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
            **kwargs: Additional parameters
                
        Returns:
            Generator yielding dictionaries with completion chunks and metadata
        """
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages, functions)
        
        # Set up generation parameters
        gen_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_tokens or self.max_new_tokens,
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            gen_kwargs["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            gen_kwargs["top_k"] = kwargs["top_k"]
        if "repetition_penalty" in kwargs:
            gen_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]
        
        try:
            # Encode the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Create the streamer
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            
            # Start the generation in a thread
            thread = Thread(target=self.model.generate, kwargs={
                "input_ids": inputs.input_ids,
                **gen_kwargs
            })
            thread.start()
            
            # Collect the complete output for function call detection
            complete_output = ""
            
            # Stream the output
            for text in streamer:
                complete_output += text
                
                # Check for function calls in the complete output so far
                tool_calls = None
                if functions and self.supports_functions():
                    function_call_data = self._extract_function_call(complete_output)
                    if function_call_data:
                        tool_calls = [{
                            "type": "function",
                            "function": function_call_data
                        }]
                
                yield {
                    "content": text,
                    "done": False,
                    "tool_calls": tool_calls
                }
            
            # Signal that generation is done
            yield {
                "content": "",
                "done": True
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
                - model: Override the default embedding model
                - batch_size: Batch size for embedding generation
                
        Returns:
            List of embedding vectors, one for each input text
        """
        self._ensure_embedding_model_loaded()
        
        # Handle single text input
        if not isinstance(texts, list):
            texts = [texts]
        
        # Extract parameters
        batch_size = kwargs.get("batch_size", 32)
        
        try:
            # Generate embeddings
            embeddings = self._embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Convert numpy arrays to lists
            return embeddings.tolist()
            
        except Exception as e:
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
        try:
            # Use the model's tokenizer
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}. Using estimate.")
            # Fallback to a simple estimate
            return len(text.split())
    
    def get_max_context_length(self) -> int:
        """
        Get the maximum context length supported by the model.
        
        Returns:
            Maximum number of tokens the model can handle
        """
        # Check if context length is specified in model config
        if hasattr(self.model.config, "max_position_embeddings"):
            return self.model.config.max_position_embeddings
        
        # Check if we have a predefined context length for this model
        if self.model_base_name in self._CONTEXT_LENGTHS:
            return self._CONTEXT_LENGTHS[self.model_base_name]
        
        # Default fallback
        logger.warning(f"Unknown model context length for {self.model_name_or_path}, using default of 4096")
        return 4096
    
    def supports_functions(self) -> bool:
        """
        Check if the model supports function calling.
        
        Returns:
            True if the model supports function calling, False otherwise
        """
        # Check if we have predefined function support for this model
        if self.model_base_name in self._FUNCTION_SUPPORT:
            return self._FUNCTION_SUPPORT[self.model_base_name]
        
        # Default fallback
        logger.warning(f"Unknown function support for {self.model_name_or_path}, assuming no support")
        return False 