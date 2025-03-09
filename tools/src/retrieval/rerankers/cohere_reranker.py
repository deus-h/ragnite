"""
Cohere Reranker

This module provides a CohereReranker that uses Cohere's Rerank API
to rerank retrieval results for improved relevance.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable

from .base_reranker import BaseReranker

# Configure logging
logger = logging.getLogger(__name__)

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logger.warning("Cohere package not available. Install with 'pip install cohere'")


class CohereReranker(BaseReranker):
    """
    A reranker that uses Cohere's Rerank API to improve the relevance of retrieval results.
    
    Cohere's Rerank API provides a sophisticated cross-encoder based reranking solution
    that can significantly improve the relevance of retrieval results compared to
    initial vector similarity or keyword matching.
    
    Attributes:
        api_key (str): Cohere API key
        model (str): Cohere reranking model to use
        client: Cohere client instance
        config (Dict[str, Any]): Configuration for the reranker
    """
    
    def __init__(self, 
                 api_key: str,
                 model: str = "rerank-english-v2.0",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CohereReranker.
        
        Args:
            api_key: Cohere API key
            model: Cohere reranking model to use (default: "rerank-english-v2.0")
                Available models: 
                - "rerank-english-v2.0" (English)
                - "rerank-multilingual-v2.0" (Multilingual)
            config: Optional configuration dictionary with the following keys:
                - batch_size: Number of documents to rerank in a single API call (default: 32)
                - max_rerank: Maximum number of documents to rerank (default: 100)
                - top_n: Number of results to return after reranking (default: None, returns all)
                - min_score_threshold: Minimum relevance score threshold (default: 0.0)
                - retry_attempts: Number of retry attempts for API calls (default: 3)
                - retry_delay: Delay between retry attempts in seconds (default: 1.0)
                - return_original_if_empty: Return original results if reranking returns no results (default: True)
                - include_original_scores: Include original scores in results (default: True)
                
        Raises:
            ImportError: If the cohere package is not installed
            ValueError: If api_key is not provided
        """
        super().__init__(config)
        
        if not COHERE_AVAILABLE:
            raise ImportError("The cohere package is required. Install with 'pip install cohere'")
        
        if not api_key:
            raise ValueError("Cohere API key is required")
        
        self.api_key = api_key
        self.model = model
        self.client = cohere.Client(api_key)
        
        # Set default configuration
        default_config = {
            'batch_size': 32,
            'max_rerank': 100,
            'top_n': None,
            'min_score_threshold': 0.0,
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'return_original_if_empty': True,
            'include_original_scores': True
        }
        
        # Update config with default values for missing keys
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        logger.debug(f"Initialized CohereReranker with model '{model}' and config: {self.config}")
    
    def rerank(self, 
              query: str, 
              results: List[Dict[str, Any]], 
              **kwargs) -> List[Dict[str, Any]]:
        """
        Rerank search results using Cohere's Rerank API.
        
        Args:
            query: The search query
            results: The original search results to rerank
            **kwargs: Additional parameters to override config
                
        Returns:
            Reranked results
            
        Raises:
            ValueError: If results is empty
            RuntimeError: If Cohere API call fails after retry attempts
        """
        if not results:
            logger.warning("No results to rerank")
            return []
        
        # Override config with kwargs
        config = self.config.copy()
        config.update(kwargs)
        
        # Limit number of documents to rerank
        results_to_rerank = results[:min(len(results), config['max_rerank'])]
        logger.debug(f"Reranking {len(results_to_rerank)} out of {len(results)} results")
        
        # Extract documents for reranking
        docs = []
        for result in results_to_rerank:
            # Prefer 'content' field, fall back to 'text' or 'snippet'
            doc_text = result.get('content', result.get('text', result.get('snippet', '')))
            docs.append(doc_text)
        
        # Check if we have valid documents to rerank
        if not docs or all(not doc for doc in docs):
            logger.warning("No valid document content found for reranking")
            return results
        
        # Process results in batches to avoid API limits
        all_reranked_results = []
        batch_size = config['batch_size']
        
        for batch_start in range(0, len(results_to_rerank), batch_size):
            batch_end = min(batch_start + batch_size, len(results_to_rerank))
            batch_docs = docs[batch_start:batch_end]
            batch_results = results_to_rerank[batch_start:batch_end]
            
            # Rerank the batch
            reranked_batch = self._rerank_batch(query, batch_docs, batch_results, config)
            all_reranked_results.extend(reranked_batch)
        
        # Sort by relevance score
        all_reranked_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Apply minimum score threshold
        filtered_results = [
            result for result in all_reranked_results 
            if result['relevance_score'] >= config['min_score_threshold']
        ]
        
        # If filtered results are empty and configured to return originals, do so
        if not filtered_results and config['return_original_if_empty']:
            logger.warning("No results above threshold, returning original results")
            return results
        
        # Apply top_n limit if specified
        if config['top_n'] is not None:
            filtered_results = filtered_results[:config['top_n']]
        
        logger.info(f"Reranking complete. Returned {len(filtered_results)} results")
        return filtered_results
    
    def _rerank_batch(self, 
                     query: str, 
                     docs: List[str], 
                     original_results: List[Dict[str, Any]],
                     config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rerank a batch of documents using Cohere's Rerank API.
        
        Args:
            query: The search query
            docs: List of document texts to rerank
            original_results: The original search results corresponding to docs
            config: Reranking configuration
            
        Returns:
            Reranked results for the batch
            
        Raises:
            RuntimeError: If Cohere API call fails after retry attempts
        """
        retry_attempts = config['retry_attempts']
        retry_delay = config['retry_delay']
        
        # Try the API call with retries
        for attempt in range(retry_attempts):
            try:
                logger.debug(f"Calling Cohere Rerank API (attempt {attempt+1}/{retry_attempts})")
                
                # Call the Cohere Rerank API
                response = self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=docs,
                    top_n=len(docs)  # Rerank all docs in the batch
                )
                
                # Process the response
                reranked_results = []
                
                for i, result in enumerate(response.results):
                    # Get the original result based on the index in the response
                    original_result = original_results[result.index]
                    
                    # Create a new result with reranking data
                    reranked_result = original_result.copy()
                    reranked_result['relevance_score'] = result.relevance_score
                    
                    # Optionally include original score
                    if config['include_original_scores']:
                        reranked_result['original_score'] = original_result.get('score')
                    
                    # Update the main score to the relevance score for consistent interface
                    reranked_result['score'] = result.relevance_score
                    
                    reranked_results.append(reranked_result)
                
                return reranked_results
                
            except Exception as e:
                logger.warning(f"Cohere Rerank API error (attempt {attempt+1}/{retry_attempts}): {str(e)}")
                
                if attempt < retry_attempts - 1:
                    # Wait before retrying
                    time.sleep(retry_delay)
                    # Increase delay for next retry (exponential backoff)
                    retry_delay *= 2
                else:
                    # Last attempt failed
                    logger.error(f"Cohere Rerank API failed after {retry_attempts} attempts")
                    raise RuntimeError(f"Cohere Rerank API failed: {str(e)}")
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the reranker configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config.update(config)
        logger.info(f"Updated configuration: {config}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if the configuration is valid, False otherwise
        """
        # Check if config is a dict
        if not isinstance(config, dict):
            return False
        
        # Check specific config values
        if 'batch_size' in config and not isinstance(config['batch_size'], int):
            return False
        if 'max_rerank' in config and not isinstance(config['max_rerank'], int):
            return False
        if 'top_n' in config and config['top_n'] is not None and not isinstance(config['top_n'], int):
            return False
        if 'min_score_threshold' in config and not isinstance(config['min_score_threshold'], (int, float)):
            return False
        
        return True 