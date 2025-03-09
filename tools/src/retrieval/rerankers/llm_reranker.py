"""
LLM Reranker

This module provides the LLMReranker class that uses Large Language Models (LLMs)
to rerank documents based on their relevance to a query.
"""

import logging
import json
import time
import os
from typing import Dict, Any, List, Optional, Union, Callable, Tuple

from .base_reranker import BaseReranker

# Configure logging
logger = logging.getLogger(__name__)


class LLMReranker(BaseReranker):
    """
    Reranker that uses Large Language Models (LLMs) to rerank documents.
    
    This reranker uses LLMs to evaluate the relevance of documents to a query.
    It can work with any LLM API that follows a standard interface, including
    OpenAI, Anthropic, local models, etc.
    
    Attributes:
        llm_provider: Function to call the LLM API.
        scoring_method: Method to extract scores from LLM outputs.
        prompt_template: Template for prompts sent to the LLM.
        batch_size: Number of documents to process in a single batch.
        config: Configuration options for the reranker.
    """
    
    def __init__(self, 
                 llm_provider: Callable,
                 scoring_method: str = "direct",
                 prompt_template: Optional[str] = None,
                 batch_size: int = 4,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLMReranker.
        
        Args:
            llm_provider: Function to call the LLM API. Should accept at least
                a 'prompt' parameter and return text.
            scoring_method: Method to extract scores from LLM outputs.
                Options: 'direct', 'json', 'scale_1_10', 'scale_1_5'.
            prompt_template: Template for prompts sent to the LLM.
                Uses {query} and {document} placeholders.
            batch_size: Number of documents to process in a single batch.
            config: Additional configuration options.
        """
        super().__init__(config or {})
        
        self.llm_provider = llm_provider
        self.scoring_method = scoring_method
        self.batch_size = batch_size
        
        # Set prompt template
        if prompt_template is None:
            self.prompt_template = (
                "On a scale of 0-10, rate how relevant the following document "
                "is to the query. Return just the numerical score.\n\n"
                "Query: {query}\n\n"
                "Document: {document}\n\n"
                "Relevance score (0-10):"
            )
        else:
            self.prompt_template = prompt_template
        
        # Update configuration
        self.config.update({
            "scoring_method": scoring_method,
            "batch_size": batch_size,
            "prompt_template": self.prompt_template
        })
        
        # Define mapping of scoring methods to score extraction functions
        self._scoring_methods = {
            "direct": self._extract_score_direct,
            "json": self._extract_score_json,
            "scale_1_10": self._extract_score_scale_1_10,
            "scale_1_5": self._extract_score_scale_1_5
        }
        
        if self.scoring_method not in self._scoring_methods:
            valid_methods = list(self._scoring_methods.keys())
            raise ValueError(f"Invalid scoring method: {scoring_method}. "
                             f"Valid options are: {valid_methods}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query using an LLM.
        
        Args:
            query: The query string.
            documents: List of document dictionaries, each with at least 'id' and 'content' or 'text' keys.
            top_k: The number of top documents to return. If None, all documents are returned.
            **kwargs: Additional keyword arguments passed to the LLM provider.
        
        Returns:
            List of reranked document dictionaries with updated scores.
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        logger.info(f"Reranking {len(documents)} documents with LLM")
        
        # Extract text from documents
        doc_texts = []
        for doc in documents:
            if "content" in doc:
                doc_texts.append(doc["content"])
            elif "text" in doc:
                doc_texts.append(doc["text"])
            else:
                raise ValueError("Documents must have either 'content' or 'text' keys")
        
        # Compute relevance scores
        scores = self._compute_scores(query, doc_texts, **kwargs)
        
        # Update document scores
        reranked_docs = []
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            # Store the original score if available
            if "score" in doc_copy:
                doc_copy["original_score"] = doc_copy["score"]
            doc_copy["score"] = scores[i]
            reranked_docs.append(doc_copy)
        
        # Sort by score in descending order
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]
        
        logger.info(f"Reranking complete, returning {len(reranked_docs)} documents")
        return reranked_docs
    
    def _compute_scores(self, query: str, doc_texts: List[str], **kwargs) -> List[float]:
        """
        Compute relevance scores for a list of documents using an LLM.
        
        Args:
            query: The query string.
            doc_texts: List of document texts.
            **kwargs: Additional keyword arguments passed to the LLM provider.
        
        Returns:
            List of relevance scores.
        """
        scores = []
        
        # Process in batches
        for i in range(0, len(doc_texts), self.batch_size):
            batch_docs = doc_texts[i:i+self.batch_size]
            batch_scores = []
            
            for doc_text in batch_docs:
                # Create the prompt
                prompt = self.prompt_template.format(
                    query=query,
                    document=doc_text
                )
                
                # Call the LLM provider
                try:
                    llm_response = self.llm_provider(prompt=prompt, **kwargs)
                    
                    # Extract the score from the response
                    score = self._scoring_methods[self.scoring_method](llm_response)
                    batch_scores.append(score)
                    
                    # Optional delay to avoid rate limits
                    if self.config.get("rate_limit_delay", 0) > 0:
                        time.sleep(self.config["rate_limit_delay"])
                        
                except Exception as e:
                    logger.error(f"Error calling LLM provider: {e}")
                    # Assign a default low score in case of failure
                    batch_scores.append(0.0)
            
            scores.extend(batch_scores)
        
        return scores
    
    def _extract_score_direct(self, llm_response: str) -> float:
        """
        Extract a score from the LLM response, expecting a direct numerical value.
        
        Args:
            llm_response: The raw response from the LLM.
        
        Returns:
            Extracted float score.
        """
        try:
            # Try to extract a number from the response
            # First, extract the first line
            first_line = llm_response.strip().split('\n')[0]
            
            # Remove any non-numeric characters except for decimal points
            numeric_text = ''.join(c for c in first_line if c.isdigit() or c == '.')
            
            # Try to convert to float
            score = float(numeric_text)
            
            # Normalize to 0-1 range if the score is on a 0-10 scale
            if score > 1.0 and score <= 10.0:
                score = score / 10.0
            
            return score
        except (ValueError, IndexError):
            logger.warning(f"Could not extract direct score from: '{llm_response}'")
            return 0.0
    
    def _extract_score_json(self, llm_response: str) -> float:
        """
        Extract a score from the LLM response, expecting a JSON structure.
        
        Args:
            llm_response: The raw response from the LLM, expected to contain JSON.
        
        Returns:
            Extracted float score.
        """
        try:
            # Try to find and parse JSON in the response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = llm_response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Look for a score field in the JSON
                if 'score' in parsed:
                    score = float(parsed['score'])
                elif 'relevance' in parsed:
                    score = float(parsed['relevance'])
                elif 'relevance_score' in parsed:
                    score = float(parsed['relevance_score'])
                else:
                    # No recognizable score field
                    logger.warning(f"No score field in JSON: {parsed}")
                    return 0.0
                
                # Normalize to 0-1 range if needed
                if score > 1.0 and score <= 10.0:
                    score = score / 10.0
                
                return score
            else:
                logger.warning(f"No valid JSON found in: '{llm_response}'")
                return 0.0
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error parsing JSON score: {e}")
            return 0.0
    
    def _extract_score_scale_1_10(self, llm_response: str) -> float:
        """
        Extract a score from the LLM response, expecting a 1-10 scale rating.
        
        Args:
            llm_response: The raw response from the LLM.
        
        Returns:
            Normalized float score (0-1 range).
        """
        try:
            # Try to find a number between 1 and 10
            # First clean the response to just digits and decimals
            clean_text = ''.join(c for c in llm_response if c.isdigit() or c == '.')
            
            # Extract the first valid number
            number_strings = []
            current = ""
            for c in clean_text:
                if c.isdigit() or c == '.':
                    current += c
                elif current:
                    number_strings.append(current)
                    current = ""
            
            if current:
                number_strings.append(current)
            
            # Find the first valid number between 1 and 10
            for num_str in number_strings:
                try:
                    num = float(num_str)
                    if 1 <= num <= 10:
                        # Normalize to 0-1 scale
                        return num / 10.0
                except ValueError:
                    continue
            
            logger.warning(f"No valid score on 1-10 scale found in: '{llm_response}'")
            return 0.0
        except Exception as e:
            logger.warning(f"Error extracting 1-10 scale score: {e}")
            return 0.0
    
    def _extract_score_scale_1_5(self, llm_response: str) -> float:
        """
        Extract a score from the LLM response, expecting a 1-5 scale rating.
        
        Args:
            llm_response: The raw response from the LLM.
        
        Returns:
            Normalized float score (0-1 range).
        """
        try:
            # Try to find a number between 1 and 5
            # First clean the response to just digits and decimals
            clean_text = ''.join(c for c in llm_response if c.isdigit() or c == '.')
            
            # Extract the first valid number
            number_strings = []
            current = ""
            for c in clean_text:
                if c.isdigit() or c == '.':
                    current += c
                elif current:
                    number_strings.append(current)
                    current = ""
            
            if current:
                number_strings.append(current)
            
            # Find the first valid number between 1 and 5
            for num_str in number_strings:
                try:
                    num = float(num_str)
                    if 1 <= num <= 5:
                        # Normalize to 0-1 scale
                        return num / 5.0
                except ValueError:
                    continue
            
            logger.warning(f"No valid score on 1-5 scale found in: '{llm_response}'")
            return 0.0
        except Exception as e:
            logger.warning(f"Error extracting 1-5 scale score: {e}")
            return 0.0
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the reranker configuration.
        
        Args:
            config: New configuration parameters.
        """
        super().set_config(config)
        
        # Update reranker-specific configuration
        if "batch_size" in config:
            self.batch_size = config["batch_size"]
        if "prompt_template" in config:
            self.prompt_template = config["prompt_template"]
        if "scoring_method" in config:
            if config["scoring_method"] in self._scoring_methods:
                self.scoring_method = config["scoring_method"]
            else:
                valid_methods = list(self._scoring_methods.keys())
                logger.warning(f"Invalid scoring method: {config['scoring_method']}. "
                              f"Valid options are: {valid_methods}")
    
    @property
    def supported_backends(self) -> List[str]:
        """
        Get the list of supported backend types.
        
        Returns:
            List of supported backend types.
        """
        return ["openai", "anthropic", "llama", "huggingface", "local_llm"]
    
    @classmethod
    def from_openai(cls, 
                    api_key: Optional[str] = None, 
                    model: str = "gpt-3.5-turbo", 
                    temperature: float = 0.0,
                    **kwargs) -> 'LLMReranker':
        """
        Create an LLMReranker using the OpenAI API.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
            model: OpenAI model to use.
            temperature: Temperature parameter for the model.
            **kwargs: Additional parameters for LLMReranker.
        
        Returns:
            Initialized LLMReranker instance.
        
        Raises:
            ImportError: If the openai package is not installed.
            ValueError: If the API key is not provided or found in environment.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The openai package is required to use OpenAI models. "
                "Please install it with `pip install openai`."
            )
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Provide it as a parameter or "
                "set the OPENAI_API_KEY environment variable."
            )
        
        # Configure the OpenAI client
        openai.api_key = api_key
        
        def openai_provider(prompt: str, **provider_kwargs) -> str:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **provider_kwargs
            )
            return response.choices[0].message.content
        
        # Create scoring prompt template if not provided
        prompt_template = kwargs.pop('prompt_template', None)
        if prompt_template is None:
            prompt_template = (
                "Please rate how relevant the following document is to the query "
                "on a scale of 0 to 10, where 0 means completely irrelevant and "
                "10 means perfectly relevant. Return only the numerical score.\n\n"
                "Query: {query}\n\n"
                "Document: {document}\n\n"
                "Relevance score (0-10):"
            )
        
        # Configure rate limiting
        config = kwargs.pop('config', {})
        if 'rate_limit_delay' not in config:
            config['rate_limit_delay'] = 0.5  # Default delay to avoid OpenAI rate limits
        
        return cls(
            llm_provider=openai_provider,
            prompt_template=prompt_template,
            scoring_method=kwargs.pop('scoring_method', 'direct'),
            batch_size=kwargs.pop('batch_size', 4),
            config={
                **config,
                "model": model,
                "temperature": temperature,
                "provider": "openai"
            }
        )
    
    @classmethod
    def from_anthropic(cls, 
                      api_key: Optional[str] = None, 
                      model: str = "claude-3-haiku-20240307", 
                      temperature: float = 0.0,
                      **kwargs) -> 'LLMReranker':
        """
        Create an LLMReranker using the Anthropic API.
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY environment variable.
            model: Anthropic model to use.
            temperature: Temperature parameter for the model.
            **kwargs: Additional parameters for LLMReranker.
        
        Returns:
            Initialized LLMReranker instance.
        
        Raises:
            ImportError: If the anthropic package is not installed.
            ValueError: If the API key is not provided or found in environment.
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required to use Anthropic models. "
                "Please install it with `pip install anthropic`."
            )
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Provide it as a parameter or "
                "set the ANTHROPIC_API_KEY environment variable."
            )
        
        def anthropic_provider(prompt: str, **provider_kwargs) -> str:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **provider_kwargs
            )
            return response.content[0].text
        
        # Create scoring prompt template if not provided
        prompt_template = kwargs.pop('prompt_template', None)
        if prompt_template is None:
            prompt_template = (
                "Please rate how relevant the following document is to the query "
                "on a scale of 0 to 10, where 0 means completely irrelevant and "
                "10 means perfectly relevant. Return only the numerical score.\n\n"
                "Query: {query}\n\n"
                "Document: {document}\n\n"
                "Relevance score (0-10):"
            )
        
        # Configure rate limiting
        config = kwargs.pop('config', {})
        if 'rate_limit_delay' not in config:
            config['rate_limit_delay'] = 0.5  # Default delay to avoid rate limits
        
        return cls(
            llm_provider=anthropic_provider,
            prompt_template=prompt_template,
            scoring_method=kwargs.pop('scoring_method', 'direct'),
            batch_size=kwargs.pop('batch_size', 4),
            config={
                **config,
                "model": model,
                "temperature": temperature,
                "provider": "anthropic"
            }
        ) 