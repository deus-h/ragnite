"""
MonoT5 Reranker

This module provides the MonoT5Reranker class that uses T5 models fine-tuned for ranking
to rerank documents based on their relevance to a query.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
import torch
import numpy as np

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .base_reranker import BaseReranker

# Configure logging
logger = logging.getLogger(__name__)


class MonoT5Reranker(BaseReranker):
    """
    Reranker that uses T5 models fine-tuned for ranking to rerank documents.
    
    This reranker uses the MonoT5 approach, where a T5 model is fine-tuned to generate
    'true' or 'false' based on query-document relevance. The probability of generating
    'true' is used as the relevance score.
    
    Attributes:
        model: The T5 model for ranking.
        tokenizer: The tokenizer for the T5 model.
        device: The device to run the model on ('cuda' or 'cpu').
        batch_size: The batch size for inference.
        max_length: The maximum sequence length for the model.
        config: Configuration options for the reranker.
    """
    
    def __init__(self, 
                 model_name: str = "castorini/monot5-base-msmarco",
                 device: Optional[str] = None,
                 batch_size: int = 8,
                 max_length: int = 512,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MonoT5Reranker.
        
        Args:
            model_name: The name or path of the MonoT5 model to use.
            device: The device to run the model on ('cuda' or 'cpu').
            batch_size: The batch size for inference.
            max_length: The maximum sequence length for the model.
            config: Additional configuration options.
        
        Raises:
            ImportError: If the transformers library is not installed.
        """
        super().__init__(config or {})
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The transformers library is required to use the MonoT5Reranker. "
                "Please install it with `pip install transformers`."
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Set the device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading MonoT5 model: {model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"MonoT5 model loaded, using device: {self.device}")
        
        # Update configuration
        self.config.update({
            "model_name": model_name,
            "device": self.device,
            "batch_size": batch_size,
            "max_length": max_length
        })
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], 
               top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Rerank documents based on their relevance to the query using MonoT5.
        
        Args:
            query: The query string.
            documents: List of document dictionaries, each with at least 'id' and 'content' or 'text' keys.
            top_k: The number of top documents to return. If None, all documents are returned.
            **kwargs: Additional keyword arguments.
        
        Returns:
            List of reranked document dictionaries with updated scores.
        """
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        logger.info(f"Reranking {len(documents)} documents with MonoT5")
        
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
        scores = self._compute_scores(query, doc_texts)
        
        # Update document scores
        reranked_docs = []
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            # Store the original score if available
            if "score" in doc_copy:
                doc_copy["original_score"] = doc_copy["score"]
            doc_copy["score"] = float(scores[i])
            reranked_docs.append(doc_copy)
        
        # Sort by score in descending order
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]
        
        logger.info(f"Reranking complete, returning {len(reranked_docs)} documents")
        return reranked_docs
    
    def _compute_scores(self, query: str, doc_texts: List[str]) -> List[float]:
        """
        Compute relevance scores for a list of documents using MonoT5.
        
        Args:
            query: The query string.
            doc_texts: List of document texts.
        
        Returns:
            List of relevance scores.
        """
        # Prepare input data
        inputs = [f"Query: {query} Document: {doc}" for doc in doc_texts]
        
        scores = []
        
        # Process in batches
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i+self.batch_size]
            
            # Tokenize inputs
            with torch.no_grad():
                encoded_inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate predictions
                outputs = self.model.generate(
                    **encoded_inputs,
                    max_length=2,  # Only need to generate "true" or "false"
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Extract the probability of generating "true"
                # The score for "true" is typically at index 1846 for t5-base tokenizer
                true_id = self.tokenizer.encode("true")[0]
                relevance_scores = []
                
                for beam_scores in outputs.scores:
                    # Get the scores for the first token
                    first_token_scores = beam_scores[0]
                    # Get the score for "true" token
                    true_score = first_token_scores[true_id].cpu().numpy()
                    relevance_scores.append(float(true_score))
                
                scores.extend(relevance_scores)
        
        return scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the MonoT5 model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the reranker configuration.
        
        Args:
            config: New configuration parameters.
        """
        super().set_config(config)
        
        # Update model-specific configuration
        if "batch_size" in config:
            self.batch_size = config["batch_size"]
        if "max_length" in config:
            self.max_length = config["max_length"]
    
    @property
    def supported_backends(self) -> List[str]:
        """
        Get the list of supported backend types.
        
        Returns:
            List of supported backend types.
        """
        return ["transformers", "huggingface", "pytorch"] 