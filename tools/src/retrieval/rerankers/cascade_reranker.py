"""
Cascade Reranker

This module provides a CascadeReranker that implements a multi-stage reranking pipeline,
progressively refining results using increasingly sophisticated (and expensive) rerankers.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from .base_reranker import BaseReranker

# Configure logging
logger = logging.getLogger(__name__)


class CascadeReranker(BaseReranker):
    """
    Implements a cascade reranking pipeline that applies multiple rerankers in stages.
    
    The cascade approach progressively filters and reranks results, starting with faster,
    less precise rerankers and moving to slower, more precise ones. This balances
    performance and cost by only applying expensive rerankers to the most promising candidates.
    
    Attributes:
        rerankers (List[Tuple[BaseReranker, Dict[str, Any]]]): List of (reranker, config) pairs
        config (Dict[str, Any]): Configuration for the cascade reranker
    """
    
    def __init__(self, 
                 rerankers: List[Union[BaseReranker, Tuple[BaseReranker, Dict[str, Any]]]],
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CascadeReranker.
        
        Args:
            rerankers: List of rerankers to use in the cascade. Each element can be either:
                - A BaseReranker instance
                - A tuple of (BaseReranker, Dict[str, Any]) where the dict contains stage-specific config
            config: Optional configuration dictionary with the following keys:
                - stage_names: List of names for each stage (for logging/debugging)
                - min_results: Minimum number of results to maintain after each stage
                - early_stopping: Whether to stop if a stage produces high-confidence results
                - early_stopping_threshold: Score threshold for early stopping
                - early_stopping_min_results: Minimum results needed for early stopping
                - always_use_final_reranker: Whether to always apply the final reranker
                - return_stage_results: Include intermediate results in final output
                
        Raises:
            ValueError: If rerankers is empty
            TypeError: If any element in rerankers is not a BaseReranker or (BaseReranker, dict) tuple
        """
        super().__init__(config)
        
        if not rerankers:
            raise ValueError("At least one reranker is required for cascade reranking")
        
        # Process and validate rerankers
        self.rerankers = []
        for i, item in enumerate(rerankers):
            if isinstance(item, BaseReranker):
                # Just a reranker without stage-specific config
                self.rerankers.append((item, {}))
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], BaseReranker) and isinstance(item[1], dict):
                # A (reranker, config) tuple
                self.rerankers.append(item)
            else:
                raise TypeError(f"Item {i} in rerankers must be a BaseReranker or a (BaseReranker, dict) tuple")
        
        # Set default configuration
        default_config = {
            'stage_names': [f"Stage {i+1}" for i in range(len(self.rerankers))],
            'min_results': 5,
            'early_stopping': True,
            'early_stopping_threshold': 0.85,
            'early_stopping_min_results': 3,
            'always_use_final_reranker': True,
            'return_stage_results': False
        }
        
        # Update config with default values for missing keys
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Ensure stage_names has the correct length
        if len(self.config['stage_names']) != len(self.rerankers):
            logger.warning(f"Length of stage_names ({len(self.config['stage_names'])}) "
                          f"doesn't match number of rerankers ({len(self.rerankers)}). "
                          f"Using default stage names.")
            self.config['stage_names'] = [f"Stage {i+1}" for i in range(len(self.rerankers))]
        
        logger.debug(f"Initialized CascadeReranker with {len(self.rerankers)} stages and config: {self.config}")
    
    def rerank(self, 
              query: str, 
              results: List[Dict[str, Any]], 
              **kwargs) -> List[Dict[str, Any]]:
        """
        Rerank results using the cascade of rerankers.
        
        Args:
            query: The search query
            results: The original search results to rerank
            **kwargs: Additional parameters to override config
                
        Returns:
            Reranked results, potentially with stage information if return_stage_results is True
            
        Raises:
            ValueError: If results is empty
        """
        if not results:
            logger.warning("No results to rerank")
            return []
        
        # Override config with kwargs
        config = self.config.copy()
        config.update(kwargs)
        
        # Track results from each stage for potential return and debugging
        all_stage_results = []
        
        # Initialize current results with input results
        current_results = results
        
        # Process each stage in the cascade
        for i, (reranker, stage_config) in enumerate(self.rerankers):
            stage_name = config['stage_names'][i]
            is_final_stage = (i == len(self.rerankers) - 1)
            
            # Skip stages with no results
            if not current_results:
                logger.warning(f"{stage_name}: No results to rerank, skipping")
                continue
            
            # Skip non-final stages if results are already high quality
            if (config['early_stopping'] and not is_final_stage and 
                self._check_early_stopping(current_results, config)):
                logger.info(f"{stage_name}: Early stopping triggered - results have high confidence")
                
                # Still use final reranker if configured
                if config['always_use_final_reranker'] and i < len(self.rerankers) - 1:
                    logger.info(f"Skipping to final reranker as configured")
                    final_reranker, final_config = self.rerankers[-1]
                    
                    # Combine stage config with global config
                    combined_config = config.copy()
                    combined_config.update(final_config)
                    
                    # Apply final reranker
                    current_results = final_reranker.rerank(query, current_results, **combined_config)
                    all_stage_results.append({
                        'stage': config['stage_names'][-1],
                        'results': current_results.copy()
                    })
                    break
                else:
                    # Skip remaining stages
                    break
            
            # Log the current stage
            logger.info(f"{stage_name}: Reranking {len(current_results)} results")
            
            # Combine stage config with global config
            combined_config = config.copy()
            combined_config.update(stage_config)
            
            # Apply the current reranker
            current_results = reranker.rerank(query, current_results, **combined_config)
            
            # Store results from this stage
            all_stage_results.append({
                'stage': stage_name,
                'results': current_results.copy()
            })
            
            # Apply minimum results constraint if needed
            min_results = combined_config.get('min_results', config['min_results'])
            if len(current_results) < min_results and len(results) >= min_results:
                logger.warning(f"{stage_name}: Reranker returned fewer than {min_results} results, "
                              f"adding more from original results")
                
                # Add more results from original set
                original_ids = {r['id'] for r in current_results}
                for r in results:
                    if r['id'] not in original_ids and len(current_results) < min_results:
                        current_results.append(r)
                        original_ids.add(r['id'])
        
        # If configured to return stage results, include them in the output
        if config['return_stage_results']:
            # Return a dictionary with final results and all stage results
            return {
                'results': current_results,
                'stages': all_stage_results
            }
        else:
            # Just return the final reranked results
            return current_results
    
    def _check_early_stopping(self, results: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
        """
        Check if early stopping should be triggered based on result quality.
        
        Args:
            results: Current set of reranked results
            config: Configuration settings
            
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        threshold = config['early_stopping_threshold']
        min_results = config['early_stopping_min_results']
        
        # Count results above threshold
        high_confidence_count = sum(1 for r in results if r.get('score', 0) >= threshold)
        
        # Trigger early stopping if we have enough high-confidence results
        return high_confidence_count >= min_results
    
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
        
        # Check specific config values if present
        if 'min_results' in config and not isinstance(config['min_results'], int):
            return False
        if 'early_stopping' in config and not isinstance(config['early_stopping'], bool):
            return False
        if 'early_stopping_threshold' in config and not isinstance(config['early_stopping_threshold'], (int, float)):
            return False
        if 'early_stopping_min_results' in config and not isinstance(config['early_stopping_min_results'], int):
            return False
        if 'stage_names' in config and not isinstance(config['stage_names'], list):
            return False
        
        return True 