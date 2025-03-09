"""
Streaming RAG Module

This module implements streaming retrieval augmented generation,
enabling token-by-token streaming responses, progressive context retrieval,
and thought streaming for improved user experience and transparency.
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable, AsyncGenerator, Generator, Tuple, Set

# Try to import the model provider and related components
try:
    from tools.src.models import (
        LLMProvider, Message, Role, get_model_provider
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class StreamingRAG:
    """
    Streaming Retrieval Augmented Generation system.
    
    This class implements streaming RAG capabilities that enable:
    1. Token-by-token streaming responses
    2. Progressive context retrieval during generation
    3. Thought streaming for increased transparency
    4. Early stopping for irrelevant retrieval paths
    """
    
    def __init__(
        self,
        model_provider: Optional[Union[str, LLMProvider]] = None,
        retriever: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        progressive_retrieval_prompt: Optional[str] = None,
        thought_stream_prompt: Optional[str] = None,
        max_initial_results: int = 3,
        max_progressive_results: int = 2,
        progressive_threshold: float = 0.7,
        enable_thought_streaming: bool = True,
        progressive_retrieval_interval: int = 50,  # Tokens
        early_stopping_threshold: float = 0.3,
        temperature: float = 0.3,
        streaming_chunk_size: int = 10,  # Tokens
    ):
        """
        Initialize the Streaming RAG system.
        
        Args:
            model_provider: LLM provider for generating responses
            retriever: Function for retrieving passages from a knowledge base
            system_prompt: Custom system prompt for the LLM
            progressive_retrieval_prompt: Custom prompt for progressive retrieval
            thought_stream_prompt: Custom prompt for thought streaming
            max_initial_results: Maximum number of initial retrieval results
            max_progressive_results: Maximum number of progressive retrieval results
            progressive_threshold: Relevance threshold for progressive retrieval
            enable_thought_streaming: Whether to enable thought streaming
            progressive_retrieval_interval: Interval for progressive retrieval checks (in tokens)
            early_stopping_threshold: Threshold for early stopping of retrieval paths
            temperature: Temperature for LLM generation
            streaming_chunk_size: Chunk size for token streaming
        """
        if not MODELS_AVAILABLE:
            raise ImportError(
                "The models module is required. Make sure the 'tools.src.models' module is available."
            )
        
        # Set up the model provider
        if model_provider is None:
            try:
                # Default to OpenAI GPT-4 if not specified
                self.model_provider = get_model_provider("openai", model="gpt-4o")
                logger.info("Using default OpenAI GPT-4o model provider for Streaming RAG")
            except Exception as e:
                logger.warning(f"Error initializing default model provider: {str(e)}")
                raise
        elif isinstance(model_provider, str):
            try:
                self.model_provider = get_model_provider(model_provider)
                logger.info(f"Using {model_provider} model provider for Streaming RAG")
            except Exception as e:
                logger.warning(f"Error initializing model provider {model_provider}: {str(e)}")
                raise
        else:
            self.model_provider = model_provider
        
        # Check if the model provider supports streaming
        self.streaming_supported = hasattr(self.model_provider, "generate_stream")
        if not self.streaming_supported:
            logger.warning("Model provider does not support streaming. Streaming functionality will be limited.")
        
        # Set retriever (can be updated later if None)
        self.retriever = retriever
        
        # Set streaming parameters
        self.max_initial_results = max_initial_results
        self.max_progressive_results = max_progressive_results
        self.progressive_threshold = progressive_threshold
        self.enable_thought_streaming = enable_thought_streaming
        self.progressive_retrieval_interval = progressive_retrieval_interval
        self.early_stopping_threshold = early_stopping_threshold
        self.temperature = temperature
        self.streaming_chunk_size = streaming_chunk_size
        
        # Initialize default prompts
        self._init_default_prompts()
        
        # Override with custom prompts if provided
        if system_prompt:
            self.system_prompt = system_prompt
        if progressive_retrieval_prompt:
            self.progressive_retrieval_prompt = progressive_retrieval_prompt
        if thought_stream_prompt:
            self.thought_stream_prompt = thought_stream_prompt
        
        logger.debug("Initialized Streaming RAG with streaming and progressive retrieval capabilities")
    
    def _init_default_prompts(self):
        """Initialize default prompts for the Streaming RAG system."""
        
        # Main system prompt
        self.system_prompt = """
        You are an expert at answering questions using the most relevant information.
        Your goal is to provide accurate, helpful responses based on the context provided.
        
        As you generate your response:
        1. Begin answering immediately using the initial context
        2. Use any additional context that becomes available during generation
        3. Maintain a coherent, fluent response despite receiving additional information
        4. Clearly cite sources when using specific information from the context
        5. If the available information doesn't fully answer the question, acknowledge limitations
        
        Prioritize accuracy and relevance in your response. If new information contradicts what
        you've already said, find a smooth way to correct yourself while maintaining coherence.
        """
        
        # Prompt for progressive retrieval
        self.progressive_retrieval_prompt = """
        Based on your response so far, identify the most important missing information or topic
        that would help answer the user's question more completely.
        
        Please follow these guidelines:
        1. Be specific about what information would be most valuable to retrieve next
        2. Focus on facts, data, or clarifications that are directly relevant to the query
        3. Don't request information that's already covered in the existing context
        4. If you believe you already have all necessary information, indicate that
        
        Original question: {query}
        
        Response so far: {response_so_far}
        
        Existing context: {existing_context}
        
        What specific information would be most valuable to retrieve next? Answer in 10 words or less.
        """
        
        # Prompt for thought streaming
        self.thought_stream_prompt = """
        For the following question, share your step-by-step reasoning as you work through
        the answer. Make your thought process explicit and clear.
        
        Question: {query}
        
        Context:
        {context}
        
        Begin with "Thinking: " for your internal reasoning, followed by a detailed response
        that addresses the question directly.
        """
    
    def set_retriever(self, retriever: Callable):
        """
        Set or update the retriever function.
        
        Args:
            retriever: Function for retrieving passages from a knowledge base
        """
        self.retriever = retriever
        logger.info("Retriever function updated")
    
    def _retrieve_initial_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve initial context for the query.
        
        Args:
            query: The user's query
            
        Returns:
            List of retrieved passages
        """
        if not self.retriever:
            logger.warning("Retriever not configured")
            return []
        
        try:
            # Retrieve initial passages
            logger.info(f"Retrieving initial context for query: {query}")
            passages = self.retriever(
                query=query,
                limit=self.max_initial_results
            )
            
            logger.info(f"Retrieved {len(passages)} initial passages")
            return passages
            
        except Exception as e:
            logger.error(f"Error retrieving initial context: {str(e)}")
            return []
    
    def _format_context(self, passages: List[Dict[str, Any]]) -> str:
        """
        Format passages for inclusion in LLM context.
        
        Args:
            passages: List of retrieved passages
            
        Returns:
            Formatted context string
        """
        if not passages:
            return "No relevant context found."
        
        formatted = ""
        
        for i, passage in enumerate(passages):
            content = passage.get("content", "No content")
            source = passage.get("source", "Unknown")
            title = passage.get("title", "Untitled")
            
            formatted += f"\nSOURCE {i+1} [{source}: {title}]:\n{content}\n"
        
        return formatted
    
    def _extract_next_retrieval_query(self, 
                                     query: str, 
                                     response_so_far: str, 
                                     existing_context: str) -> str:
        """
        Extract the next retrieval query based on the response generated so far.
        
        Args:
            query: The original user query
            response_so_far: The response generated so far
            existing_context: The context used so far
            
        Returns:
            Next retrieval query or empty string if no additional retrieval needed
        """
        if not response_so_far:
            return ""
        
        # Format the progressive retrieval prompt
        formatted_prompt = self.progressive_retrieval_prompt.format(
            query=query,
            response_so_far=response_so_far,
            existing_context=existing_context
        )
        
        # Create messages for query extraction
        messages = [
            Message(role=Role.SYSTEM, content="You help identify information gaps that need to be filled."),
            Message(role=Role.USER, content=formatted_prompt)
        ]
        
        # Generate next retrieval query
        try:
            response = self.model_provider.generate(
                messages=messages,
                temperature=0.2,  # Low temperature for more focused extraction
                max_tokens=20  # Short response
            )
            
            next_query = response.get("content", "").strip()
            
            # Check if the response indicates no more retrieval is needed
            if any(phrase in next_query.lower() for phrase in ["no more", "all necessary", "sufficient", "complete", "nothing"]):
                logger.info("Progressive retrieval: No additional information needed")
                return ""
            
            logger.info(f"Progressive retrieval query: {next_query}")
            return next_query
            
        except Exception as e:
            logger.error(f"Error extracting next retrieval query: {str(e)}")
            return ""
    
    def _retrieve_progressive_context(self, 
                                   next_query: str,
                                   original_query: str,
                                   existing_passages_ids: Set[str]) -> List[Dict[str, Any]]:
        """
        Retrieve additional context based on an extracted query.
        
        Args:
            next_query: The extracted query for additional information
            original_query: The original user query
            existing_passages_ids: IDs of passages already retrieved
            
        Returns:
            List of newly retrieved passages
        """
        if not self.retriever or not next_query:
            return []
        
        try:
            # Combine the original query with the extracted query for better retrieval
            combined_query = f"{original_query} {next_query}"
            
            # Retrieve additional passages
            logger.info(f"Retrieving progressive context for query: {combined_query}")
            passages = self.retriever(
                query=combined_query,
                limit=self.max_progressive_results
            )
            
            # Filter out passages that we've already retrieved
            new_passages = [p for p in passages if p.get("id", "") not in existing_passages_ids]
            
            # Filter by relevance threshold
            relevant_passages = [p for p in new_passages if p.get("score", 0) >= self.progressive_threshold]
            
            logger.info(f"Retrieved {len(relevant_passages)} new relevant passages")
            return relevant_passages
            
        except Exception as e:
            logger.error(f"Error retrieving progressive context: {str(e)}")
            return []
    
    def _prepare_streaming_messages(self, 
                                   query: str, 
                                   context: str,
                                   enable_thought_streaming: bool = None) -> List[Dict[str, Any]]:
        """
        Prepare messages for streaming generation.
        
        Args:
            query: The user's query
            context: The context for generation
            enable_thought_streaming: Whether to enable thought streaming
            
        Returns:
            List of messages for the model
        """
        if enable_thought_streaming is None:
            enable_thought_streaming = self.enable_thought_streaming
        
        if enable_thought_streaming:
            # Create a thought streaming prompt
            user_prompt = self.thought_stream_prompt.format(
                query=query,
                context=context
            )
        else:
            # Standard prompt
            user_prompt = f"""
            Please answer the following question using the provided context:
            
            Question: {query}
            
            Context:
            {context}
            
            Answer:
            """
        
        # Create messages
        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            Message(role=Role.USER, content=user_prompt)
        ]
        
        return messages
    
    async def _process_stream_with_progressive_retrieval(self, 
                                                     stream_generator, 
                                                     query: str,
                                                     initial_context: List[Dict[str, Any]]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a token stream with progressive retrieval.
        
        Args:
            stream_generator: Generator yielding tokens
            query: The original user query
            initial_context: Initial retrieved passages
            
        Yields:
            Tokens with progressive retrieval updates
        """
        response_so_far = ""
        token_count = 0
        retrieved_passage_ids = {p.get("id", f"p{i}") for i, p in enumerate(initial_context)}
        formatted_context = self._format_context(initial_context)
        
        # Dictionary to store information about progressive retrievals
        retrieval_info = {
            "progressive_retrievals": [],
            "tokens_at_retrievals": [],
            "new_passages_count": 0,
            "total_tokens": 0
        }
        
        async for chunk in stream_generator:
            # Extract token from chunk
            token = chunk.get("content", "")
            if not token:
                continue
            
            # Update response built so far
            response_so_far += token
            token_count += 1
            
            # Check if we should perform progressive retrieval
            if token_count % self.progressive_retrieval_interval == 0:
                # Extract next retrieval query
                next_query = self._extract_next_retrieval_query(
                    query=query,
                    response_so_far=response_so_far,
                    existing_context=formatted_context
                )
                
                if next_query:
                    # Retrieve additional passages
                    new_passages = self._retrieve_progressive_context(
                        next_query=next_query,
                        original_query=query,
                        existing_passages_ids=retrieved_passage_ids
                    )
                    
                    if new_passages:
                        # Update tracking
                        retrieval_info["progressive_retrievals"].append(next_query)
                        retrieval_info["tokens_at_retrievals"].append(token_count)
                        retrieval_info["new_passages_count"] += len(new_passages)
                        
                        # Add to context
                        formatted_new_context = self._format_context(new_passages)
                        formatted_context += "\n" + formatted_new_context
                        
                        # Update retrieved IDs
                        retrieved_passage_ids.update({p.get("id", f"p{i}") for i, p in enumerate(new_passages)})
                        
                        # Include retrieval info in the yielded chunk
                        chunk["progressive_retrieval"] = {
                            "query": next_query,
                            "passages": new_passages
                        }
            
            # Update total tokens
            retrieval_info["total_tokens"] = token_count
            
            # Include retrieval summary info in every chunk
            chunk["retrieval_info"] = retrieval_info
            
            yield chunk
    
    async def _stream_with_progressive_retrieval(self, 
                                              query: str,
                                              context: List[Dict[str, Any]] = None,
                                              enable_thought_streaming: bool = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a response with progressive retrieval during generation.
        
        Args:
            query: The user's query
            context: Initial context (optional, will be retrieved if not provided)
            enable_thought_streaming: Whether to enable thought streaming
            
        Yields:
            Tokens from the streaming generation with progressive retrieval
        """
        # Get initial context if not provided
        if context is None:
            context = self._retrieve_initial_context(query)
        
        # Format context
        formatted_context = self._format_context(context)
        
        # Prepare messages
        messages = self._prepare_streaming_messages(
            query=query,
            context=formatted_context,
            enable_thought_streaming=enable_thought_streaming
        )
        
        try:
            # Start the streaming generation
            stream_generator = self.model_provider.generate_stream(
                messages=messages,
                temperature=self.temperature
            )
            
            # Process the stream with progressive retrieval
            async for chunk in self._process_stream_with_progressive_retrieval(
                stream_generator=stream_generator,
                query=query,
                initial_context=context
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming generation: {str(e)}")
            yield {"content": f"Error: {str(e)}", "error": True}
    
    def _post_process_thought_stream(self, response: str) -> str:
        """
        Post-process a thought-streamed response to clean it up.
        
        Args:
            response: The raw response with thought streaming
            
        Returns:
            Cleaned response with thoughts removed
        """
        # If the response has thoughts, extract just the answer
        if "Thinking:" in response:
            # Extract all the thinking parts
            thinking_parts = []
            answer_parts = []
            
            in_thinking = False
            lines = response.split("\n")
            
            for line in lines:
                if line.strip().startswith("Thinking:"):
                    in_thinking = True
                    thinking_parts.append(line.strip())
                elif in_thinking and line.strip() and not any(line.strip().startswith(prefix) for prefix in ["Answer:", "Response:", "I think", "Based on"]):
                    thinking_parts.append(line.strip())
                else:
                    in_thinking = False
                    answer_parts.append(line)
            
            # Join the answer parts
            answer = "\n".join(answer_parts).strip()
            return answer
        
        return response
    
    async def generate_stream(self, 
                            query: str,
                            context: Optional[List[Dict[str, Any]]] = None,
                            enable_progressive_retrieval: bool = True,
                            enable_thought_streaming: bool = None,
                            post_process_thoughts: bool = True,
                            **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a streaming response to a query.
        
        Args:
            query: The user's query
            context: Initial context (optional, will be retrieved if not provided)
            enable_progressive_retrieval: Whether to enable progressive retrieval
            enable_thought_streaming: Whether to enable thought streaming
            post_process_thoughts: Whether to post-process thoughts from the final response
            **kwargs: Additional parameters
            
        Yields:
            Tokens from the streaming generation
        """
        if not self.streaming_supported:
            logger.warning("Model provider does not support streaming. Using non-streaming implementation.")
            response = await self.generate(
                query=query,
                context=context,
                enable_progressive_retrieval=enable_progressive_retrieval,
                enable_thought_streaming=enable_thought_streaming,
                **kwargs
            )
            
            # Return the full response as a single chunk
            yield {
                "content": response.get("answer", ""),
                "retrieval_info": response.get("retrieval_info", {}),
                "is_complete": True
            }
            return
        
        if enable_thought_streaming is None:
            enable_thought_streaming = self.enable_thought_streaming
        
        # Get initial context if not provided
        initial_context = context or self._retrieve_initial_context(query)
        
        if not enable_progressive_retrieval:
            # Simple streaming without progressive retrieval
            formatted_context = self._format_context(initial_context)
            messages = self._prepare_streaming_messages(
                query=query,
                context=formatted_context,
                enable_thought_streaming=enable_thought_streaming
            )
            
            full_response = ""
            
            try:
                # Start the streaming generation
                async for chunk in self.model_provider.generate_stream(
                    messages=messages,
                    temperature=self.temperature
                ):
                    token = chunk.get("content", "")
                    full_response += token
                    
                    # Add a flag for the final token
                    if chunk.get("is_complete", False):
                        if post_process_thoughts and enable_thought_streaming:
                            # Clean up the response
                            processed_response = self._post_process_thought_stream(full_response)
                            # Only yield the clean response for the final chunk
                            yield {
                                "content": processed_response,
                                "is_complete": True,
                                "raw_response": full_response
                            }
                        else:
                            yield chunk
                    else:
                        yield chunk
                    
            except Exception as e:
                logger.error(f"Error in streaming generation: {str(e)}")
                yield {"content": f"Error: {str(e)}", "error": True, "is_complete": True}
            
        else:
            # Streaming with progressive retrieval
            full_response = ""
            
            async for chunk in self._stream_with_progressive_retrieval(
                query=query,
                context=initial_context,
                enable_thought_streaming=enable_thought_streaming
            ):
                token = chunk.get("content", "")
                full_response += token
                
                # Add a flag for the final token
                if chunk.get("is_complete", False):
                    if post_process_thoughts and enable_thought_streaming:
                        # Clean up the response
                        processed_response = self._post_process_thought_stream(full_response)
                        # Include all the original chunk data
                        final_chunk = {**chunk, "content": processed_response, "raw_response": full_response}
                        yield final_chunk
                    else:
                        yield chunk
                else:
                    yield chunk
    
    async def generate(self, 
                    query: str,
                    context: Optional[List[Dict[str, Any]]] = None,
                    enable_progressive_retrieval: bool = True,
                    enable_thought_streaming: bool = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Generate a complete response to a query (non-streaming).
        
        Args:
            query: The user's query
            context: Initial context (optional, will be retrieved if not provided)
            enable_progressive_retrieval: Whether to enable progressive retrieval
            enable_thought_streaming: Whether to enable thought streaming
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with the generated response and metadata
        """
        if enable_thought_streaming is None:
            enable_thought_streaming = self.enable_thought_streaming
        
        # Get initial context if not provided
        initial_context = context or self._retrieve_initial_context(query)
        
        # Initialize result
        result = {
            "query": query,
            "answer": "",
            "initial_context": initial_context,
            "progressive_contexts": [],
            "retrieval_info": {
                "progressive_retrievals": [],
                "new_passages_count": 0,
                "total_passages": len(initial_context)
            }
        }
        
        if not enable_progressive_retrieval:
            # Simple generation without progressive retrieval
            formatted_context = self._format_context(initial_context)
            messages = self._prepare_streaming_messages(
                query=query,
                context=formatted_context,
                enable_thought_streaming=enable_thought_streaming
            )
            
            # Generate response
            try:
                response = self.model_provider.generate(
                    messages=messages,
                    temperature=self.temperature
                )
                
                answer = response.get("content", "")
                
                # Post-process to remove thinking if needed
                if enable_thought_streaming:
                    answer = self._post_process_thought_stream(answer)
                
                result["answer"] = answer
                
            except Exception as e:
                logger.error(f"Error in generation: {str(e)}")
                result["error"] = str(e)
                result["answer"] = f"Error generating answer: {str(e)}"
            
        else:
            # Generation with progressive retrieval
            # Use the streaming implementation and collect all chunks
            full_response = ""
            
            # Handle progressive retrieval information
            progressive_retrievals = []
            progressive_contexts = []
            
            async for chunk in self._stream_with_progressive_retrieval(
                query=query,
                context=initial_context,
                enable_thought_streaming=enable_thought_streaming
            ):
                token = chunk.get("content", "")
                full_response += token
                
                # Track progressive retrieval
                if "progressive_retrieval" in chunk:
                    retrieval_data = chunk["progressive_retrieval"]
                    progressive_retrievals.append(retrieval_data["query"])
                    progressive_contexts.append(retrieval_data["passages"])
            
            # Post-process to remove thinking if needed
            if enable_thought_streaming:
                answer = self._post_process_thought_stream(full_response)
            else:
                answer = full_response
            
            # Update the result
            result["answer"] = answer
            result["progressive_contexts"] = progressive_contexts
            result["retrieval_info"]["progressive_retrievals"] = progressive_retrievals
            result["retrieval_info"]["new_passages_count"] = sum(len(ctx) for ctx in progressive_contexts)
            result["retrieval_info"]["total_passages"] = len(initial_context) + result["retrieval_info"]["new_passages_count"]
        
        return result
    
    def stop_generation(self):
        """Stop any ongoing generation (if supported by the model provider)."""
        if hasattr(self.model_provider, "stop_generation"):
            self.model_provider.stop_generation()
            logger.info("Generation stopped")
        else:
            logger.warning("Model provider does not support stopping generation")
    
    # Synchronous versions of the async methods (for simpler usage)
    
    def generate_stream_sync(self, 
                          query: str,
                          context: Optional[List[Dict[str, Any]]] = None,
                          enable_progressive_retrieval: bool = True,
                          enable_thought_streaming: bool = None,
                          post_process_thoughts: bool = True,
                          **kwargs) -> Generator[Dict[str, Any], None, None]:
        """
        Synchronous version of generate_stream.
        
        Args:
            query: The user's query
            context: Initial context (optional, will be retrieved if not provided)
            enable_progressive_retrieval: Whether to enable progressive retrieval
            enable_thought_streaming: Whether to enable thought streaming
            post_process_thoughts: Whether to post-process thoughts from the final response
            **kwargs: Additional parameters
            
        Yields:
            Tokens from the streaming generation
        """
        import asyncio
        
        # Create a new event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def run_async_generator():
            async for chunk in self.generate_stream(
                query=query,
                context=context,
                enable_progressive_retrieval=enable_progressive_retrieval,
                enable_thought_streaming=enable_thought_streaming,
                post_process_thoughts=post_process_thoughts,
                **kwargs
            ):
                yield chunk
        
        # Convert async generator to sync generator
        gen = run_async_generator()
        
        while True:
            try:
                chunk = loop.run_until_complete(gen.__anext__())
                yield chunk
                
                # If this is the last chunk, break
                if chunk.get("is_complete", False):
                    break
            except StopAsyncIteration:
                break
    
    def generate_sync(self, 
                    query: str,
                    context: Optional[List[Dict[str, Any]]] = None,
                    enable_progressive_retrieval: bool = True,
                    enable_thought_streaming: bool = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Synchronous version of generate.
        
        Args:
            query: The user's query
            context: Initial context (optional, will be retrieved if not provided)
            enable_progressive_retrieval: Whether to enable progressive retrieval
            enable_thought_streaming: Whether to enable thought streaming
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with the generated response and metadata
        """
        import asyncio
        
        # Create a new event loop if there isn't one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.generate(
                query=query,
                context=context,
                enable_progressive_retrieval=enable_progressive_retrieval,
                enable_thought_streaming=enable_thought_streaming,
                **kwargs
            )
        ) 