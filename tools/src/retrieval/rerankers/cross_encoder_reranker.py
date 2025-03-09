"""
Cross-Encoder Reranker

This module provides the CrossEncoderReranker class that uses cross-encoder models
to rerank retrieval results based on query-document relevance scoring.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Callable
from sentence_transformers import CrossEncoder

from .base_reranker import BaseReranker

# Configure logging
logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Reranker that uses cross-encoder models to rerank retrieval results.
    
    Cross-encoders directly score the relevance of query-document pairs by processing
    them together through a transformer model. This typically provides higher-quality
    relevance scoring than bi-encoders (e.g., BERT, RoBERTa) but is more computationally
    intensive because each query-document pair must be processed separately.
    
    Attributes:
        model: Cross-encoder model from sentence-transformers
        model_name: Name of the model being used
        config: Configuration dictionary
        batch_size: Batch size for model inference
        scale_scores: Whether to scale scores to [0, 1] range
        content_field: Field name containing document content
    """
    
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, 
                 model_name_or_path: Optional[str] = None,
                 model: Optional[CrossEncoder] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CrossEncoderReranker.
        
        Args:
            model_name_or_path: Hugging Face model name or path (default: ms-marco-MiniLM-L-6-v2)
            model: Pre-instantiated cross-encoder model (alternative to model_name_or_path)
            config: Optional configuration dictionary with the following possible keys:
                - batch_size: Batch size for model inference (default: 32)
                - scale_scores: Whether to scale scores to [0, 1] range (default: True)
                - content_field: Field name containing document content (default: 'content')
                - max_length: Maximum sequence length (default: None - model default)
                - normalize_scores: Normalize scores with softmax (default: False)
                - use_gpu: Whether to use GPU if available (default: True)
        
        Raises:
            ValueError: If neither model_name_or_path nor model is provided
            ImportError: If sentence-transformers is not installed
        """
        super().__init__(config)
        
        # Set default configuration
        default_config = {
            'batch_size': 32,
            'scale_scores': True,
            'content_field': 'content',
            'max_length': None,
            'normalize_scores': False,
            'use_gpu': True,
        }
        
        # Update with user-provided config
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.model = None
        self.model_name = model_name_or_path or self.DEFAULT_MODEL
        
        # Initialize model
        try:
            if model is not None:
                self.model = model
                self.model_name = getattr(model, 'model_name', 'custom_cross_encoder')
            else:
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=self.config['max_length'],
                    device='cuda' if self.config['use_gpu'] else 'cpu'
                )
                
            logger.info(f"Initialized CrossEncoderReranker with model: {self.model_name}")
        except ImportError:
            logger.error("Failed to import CrossEncoder from sentence-transformers. "
                         "Please install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers is required for CrossEncoderReranker")
    
    def rerank(self, query: str, results: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results using the cross-encoder model.
        
        Args:
            query: The search query
            results: Initial retrieval results
            **kwargs: Additional arguments:
                - batch_size: Override config batch_size
                - scale_scores: Override config scale_scores
                - content_field: Override config content_field
        
        Returns:
            Reranked results with updated scores
        """
        if not results:
            logger.warning("Empty results list provided to rerank")
            return results
        
        # Validate results
        if not self.validate_results(results):
            logger.error("Invalid results format provided to rerank")
            return results
        
        # Get configuration overrides from kwargs
        batch_size = kwargs.get('batch_size', self.config['batch_size'])
        scale_scores = kwargs.get('scale_scores', self.config['scale_scores'])
        content_field = kwargs.get('content_field', self.config['content_field'])
        normalize_scores = kwargs.get('normalize_scores', self.config['normalize_scores'])
        
        # Prepare query-document pairs for scoring
        pairs = []
        for result in results:
            # Try to get document content from the specified field, falling back to 'text' if needed
            document_content = result.get(content_field, result.get('text', result.get('content', '')))
            pairs.append([query, document_content])
        
        # Score all pairs
        try:
            scores = self.model.predict(pairs, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {e}")
            # Return original results if scoring fails
            return results
        
        # Scale scores to [0, 1] if configured to do so
        if scale_scores:
            scores = self._scale_scores(scores)
        
        # Apply softmax normalization if configured
        if normalize_scores:
            scores = self._softmax(scores)
        
        # Update results with new scores
        for i, result in enumerate(results):
            result['score'] = float(scores[i])
        
        # Sort by new scores in descending order
        reranked_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return reranked_results
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration of the reranker.
        
        Args:
            config: New configuration dictionary
        """
        self.config.update(config)
        
        # Reload model if max_length or device changed
        if 'max_length' in config or 'use_gpu' in config:
            device = 'cuda' if self.config['use_gpu'] else 'cpu'
            
            # Only reload if we instantiated the model ourselves (not a custom provided model)
            if hasattr(self.model, 'model_name') and self.model.model_name == self.model_name:
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=self.config['max_length'],
                    device=device
                )
                logger.info(f"Reloaded CrossEncoderReranker with new config: "
                           f"max_length={self.config['max_length']}, device={device}")
    
    def _scale_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Scale scores to [0, 1] range.
        
        Args:
            scores: Numpy array of scores
        
        Returns:
            Scaled scores
        """
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            return np.ones_like(scores)
        
        return (scores - min_score) / (max_score - min_score)
    
    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply softmax normalization to scores.
        
        Args:
            scores: Numpy array of scores
        
        Returns:
            Softmax-normalized scores
        """
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()
    
    def explain_reranking(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Provide a detailed explanation of the reranking process.
        
        Args:
            query: The search query
            results: Initial retrieval results
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with explanation of the reranking process and results
        """
        # Create a copy of results to avoid modifying the originals
        results_copy = [{k: v for k, v in result.items()} for result in results]
        
        # Store original scores and ranks
        for i, result in enumerate(results_copy):
            result['original_score'] = result.get('score', 0.0)
            result['original_rank'] = i
        
        # Get configuration overrides from kwargs
        content_field = kwargs.get('content_field', self.config['content_field'])
        
        # Get reranked results
        reranked_results = self.rerank(query, results_copy, **kwargs)
        
        # Create explanation
        explanation = {
            'query': query,
            'reranking_method': 'CrossEncoder',
            'model_name': self.model_name,
            'config': self.config,
            'results': reranked_results,
            'detailed_changes': []
        }
        
        # Add detailed changes for each result
        for i, result in enumerate(reranked_results):
            original_rank = result.get('original_rank', -1)
            original_score = result.get('original_score', 0.0)
            new_score = result.get('score', 0.0)
            
            # Calculate rank and score changes
            rank_change = original_rank - i if original_rank >= 0 else None
            score_change = new_score - original_score
            
            # Get document content for the explanation
            document_content = result.get(content_field, result.get('text', result.get('content', '')))
            
            # Truncate document content for readability
            if document_content and len(document_content) > 200:
                document_content = document_content[:200] + "..."
            
            explanation['detailed_changes'].append({
                'id': result.get('id', ''),
                'original_rank': original_rank,
                'new_rank': i,
                'rank_change': rank_change,
                'original_score': original_score,
                'new_score': new_score,
                'score_change': score_change,
                'document_preview': document_content
            })
        
        return explanation 