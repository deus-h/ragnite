"""
Multi-Index Hybrid Searcher

This module provides the MultiIndexHybridSearcher class that allows searching
across multiple indices or collections.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Sequence

from .base_hybrid_searcher import BaseHybridSearcher

# Configure logging
logger = logging.getLogger(__name__)


class MultiIndexHybridSearcher(BaseHybridSearcher):
    """
    Hybrid searcher that enables searching across multiple indices or collections.
    
    This searcher allows you to search across multiple sources of information (indices,
    collections, databases, etc.) and combine the results using various strategies.
    Each source can have its own search function and weight in the final results.
    
    Attributes:
        search_funcs (List[Dict]): List of search function configurations
        config (Dict[str, Any]): Configuration dictionary
    """
    
    def __init__(self,
                search_funcs: List[Dict[str, Any]],
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MultiIndexHybridSearcher.
        
        Args:
            search_funcs: List of dictionaries with search function configurations, each containing:
                - 'func': The search function (required)
                - 'weight': Weight for this search function's results (default: 1.0)
                - 'name': Name of the index/collection/source (default: "index_{i}")
                - 'params': Additional parameters to pass to the search function (optional)
            config: Optional configuration dictionary with the following possible keys:
                - 'combination_method': Method to combine results (default: 'linear_combination')
                - 'min_score_threshold': Minimum score for results to be included (default: 0.0)
                - 'normalize_scores': Whether to normalize scores before combining (default: True)
                - 'expand_results': Whether to include more results in initial search (default: True)
                - 'include_source': Whether to include source information in results (default: True)
        
        Raises:
            ValueError: If no search functions are provided or if any search function is invalid
        """
        super().__init__(config)
        
        if not search_funcs:
            raise ValueError("At least one search function must be provided")
        
        # Validate and prepare search functions
        self.search_funcs = []
        for i, func_config in enumerate(search_funcs):
            if 'func' not in func_config or not callable(func_config['func']):
                raise ValueError(f"Search function at index {i} is missing or not callable")
            
            # Create a standardized config for each search function
            self.search_funcs.append({
                'func': func_config['func'],
                'weight': func_config.get('weight', 1.0),
                'name': func_config.get('name', f"index_{i}"),
                'params': func_config.get('params', {})
            })
        
        # Set default configuration
        default_config = {
            'combination_method': 'linear_combination',
            'min_score_threshold': 0.0,
            'normalize_scores': True,
            'expand_results': True,
            'include_source': True
        }
        
        # Update with user-provided config
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        # Normalize weights to sum to 1.0 if they don't already
        self._normalize_weights()
        
        logger.info(f"Initialized MultiIndexHybridSearcher with {len(self.search_funcs)} search functions")
    
    def _normalize_weights(self) -> None:
        """
        Normalize the search function weights to sum to 1.0.
        """
        total_weight = sum(func_config['weight'] for func_config in self.search_funcs)
        
        # Only normalize if total_weight is not 1.0 (allowing for floating-point imprecision)
        if abs(total_weight - 1.0) > 1e-6 and total_weight > 0:
            for func_config in self.search_funcs:
                func_config['weight'] = func_config['weight'] / total_weight
                
            logger.info(f"Normalized search function weights to sum to 1.0")
    
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search across all indices and combine the results.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional arguments to pass to all search functions
        
        Returns:
            List of dictionaries with search results
        """
        # Calculate the number of results to request from each function
        # If expand_results is True, we request more results to improve recall
        per_func_limit = limit
        if self.config['expand_results']:
            per_func_limit = max(limit * 2, 20)  # Request at least 20 results
        
        # Execute all search functions in parallel and collect results
        all_results = []
        for func_config in self.search_funcs:
            try:
                # Combine global kwargs with function-specific params
                combined_kwargs = {**kwargs}
                if func_config['params']:
                    combined_kwargs.update(func_config['params'])
                
                # Call the search function
                results = func_config['func'](query, limit=per_func_limit, **combined_kwargs)
                
                # Add source information if configured
                if self.config['include_source']:
                    for result in results:
                        result['source'] = func_config['name']
                        result['source_weight'] = func_config['weight']
                
                # Store the results with their function index and weight
                all_results.append({
                    'results': results,
                    'weight': func_config['weight'],
                    'name': func_config['name']
                })
                
            except Exception as e:
                logger.error(f"Error executing search function '{func_config['name']}': {str(e)}")
                # Continue with other search functions even if one fails
        
        # If all search functions failed, return an empty list
        if not all_results:
            logger.warning("All search functions failed, returning empty results")
            return []
        
        # Normalize scores within each result set if configured
        if self.config['normalize_scores']:
            for result_set in all_results:
                result_set['results'] = self._normalize_scores(result_set['results'])
        
        # Combine results based on the specified method
        if self.config['combination_method'] == 'reciprocal_rank_fusion':
            combined_results = self._combine_by_reciprocal_rank_fusion(
                all_results, limit
            )
        else:  # default to linear_combination
            combined_results = self._combine_by_linear_weights(
                all_results, limit
            )
        
        # Filter results below the minimum score threshold
        if self.config['min_score_threshold'] > 0:
            combined_results = [
                result for result in combined_results 
                if result.get('score', 0) >= self.config['min_score_threshold']
            ]
        
        return combined_results
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Update the configuration of the hybrid searcher.
        
        Args:
            config: New configuration dictionary
        """
        self.config.update(config)
    
    def _combine_by_linear_weights(self, 
                                  all_results: List[Dict[str, Any]], 
                                  limit: int) -> List[Dict[str, Any]]:
        """
        Combine results from multiple sources using linear weights.
        
        Args:
            all_results: List of result sets with their weights
            limit: Maximum number of results to return
        
        Returns:
            Combined results
        """
        # Create a dictionary to track combined scores
        combined_scores = {}
        
        # Process each result set
        for result_set in all_results:
            # Skip empty result sets
            if not result_set['results']:
                continue
                
            weight = result_set['weight']
            source_name = result_set['name']
            
            # Process each result in the set
            for result in result_set['results']:
                doc_id = result['id']
                score = result.get('score', 0.0) * weight
                
                # If the document is already in the combined scores, update it
                if doc_id in combined_scores:
                    # Add the weighted score
                    combined_scores[doc_id]['score'] += score
                    
                    # If source tracking is enabled, update the sources
                    if self.config['include_source']:
                        sources = combined_scores[doc_id].get('sources', [])
                        sources.append({
                            'name': source_name,
                            'score': result.get('score', 0.0),
                            'weight': weight,
                            'weighted_score': score
                        })
                        combined_scores[doc_id]['sources'] = sources
                else:
                    # Create a new entry with the basic fields
                    combined_scores[doc_id] = {
                        'id': doc_id,
                        'score': score,
                        'content': result.get('content', result.get('text', '')),
                        'metadata': result.get('metadata', {})
                    }
                    
                    # If source tracking is enabled, initialize sources
                    if self.config['include_source']:
                        combined_scores[doc_id]['sources'] = [{
                            'name': source_name,
                            'score': result.get('score', 0.0),
                            'weight': weight,
                            'weighted_score': score
                        }]
        
        # Convert to list and sort by score
        combined_results = list(combined_scores.values())
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top 'limit' results
        return combined_results[:limit]
    
    def _combine_by_reciprocal_rank_fusion(self, 
                                          all_results: List[Dict[str, Any]], 
                                          limit: int) -> List[Dict[str, Any]]:
        """
        Combine results from multiple sources using reciprocal rank fusion.
        
        Args:
            all_results: List of result sets with their weights
            limit: Maximum number of results to return
        
        Returns:
            Combined results
        """
        # Constant k for RRF formula
        k = 60  # Standard value used in RRF
        
        # Create dictionaries to store ranks for each document from each source
        ranks = {}
        
        # Process each result set to get ranks
        for result_set in all_results:
            # Skip empty result sets
            if not result_set['results']:
                continue
                
            weight = result_set['weight']
            source_name = result_set['name']
            
            # Sort results by score and get ranks
            sorted_results = sorted(
                result_set['results'], 
                key=lambda x: x.get('score', 0.0), 
                reverse=True
            )
            
            # Store ranks for each document
            for i, result in enumerate(sorted_results):
                doc_id = result['id']
                
                # Initialize document entry if it doesn't exist
                if doc_id not in ranks:
                    ranks[doc_id] = {
                        'id': doc_id,
                        'ranks': {},
                        'content': result.get('content', result.get('text', '')),
                        'metadata': result.get('metadata', {}),
                        'sources': [] if self.config['include_source'] else None
                    }
                
                # Store rank and other info
                ranks[doc_id]['ranks'][source_name] = {
                    'rank': i + 1,  # 1-based ranking
                    'weight': weight,
                    'original_score': result.get('score', 0.0)
                }
                
                # Store source information if needed
                if self.config['include_source']:
                    ranks[doc_id]['sources'].append({
                        'name': source_name,
                        'rank': i + 1,
                        'score': result.get('score', 0.0),
                        'weight': weight
                    })
        
        # Calculate RRF score for each document
        for doc_id, doc_info in ranks.items():
            rrf_score = 0.0
            
            for source, rank_info in doc_info['ranks'].items():
                # Apply weighted RRF formula
                source_rrf = 1.0 / (k + rank_info['rank'])
                rrf_score += rank_info['weight'] * source_rrf
            
            # Store the final RRF score
            doc_info['score'] = rrf_score
        
        # Convert to list and sort by RRF score
        combined_results = list(ranks.values())
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Clean up internal structure before returning
        for result in combined_results:
            if 'ranks' in result:
                del result['ranks']
        
        # Take top 'limit' results
        return combined_results[:limit]
    
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize scores in the results to be between 0 and 1.
        
        Args:
            results: List of search results
        
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        # Extract scores
        scores = [result.get('score', 0.0) for result in results]
        
        # Find min and max scores
        min_score = min(scores)
        max_score = max(scores)
        
        # If all scores are the same, return as is
        if max_score == min_score:
            return results
        
        # Create a deep copy of results to avoid modifying the original
        normalized_results = [{k: v for k, v in result.items()} for result in results]
        
        # Normalize scores
        for i, result in enumerate(normalized_results):
            normalized_score = (result.get('score', 0.0) - min_score) / (max_score - min_score)
            normalized_results[i]['score'] = normalized_score
        
        return normalized_results
    
    def explain_search(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Explain the hybrid search process and results.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional arguments to pass to all search functions
        
        Returns:
            Dictionary with explanation of the search process and results
        """
        # Calculate the number of results to request from each function
        per_func_limit = limit
        if self.config['expand_results']:
            per_func_limit = max(limit * 2, 20)
        
        # Execute all search functions and collect results with explanations
        component_results = []
        for func_config in self.search_funcs:
            try:
                # Combine global kwargs with function-specific params
                combined_kwargs = {**kwargs}
                if func_config['params']:
                    combined_kwargs.update(func_config['params'])
                
                # Call the search function
                results = func_config['func'](query, limit=per_func_limit, **combined_kwargs)
                
                # Add source information
                if self.config['include_source']:
                    for result in results:
                        result['source'] = func_config['name']
                
                # Normalize scores if configured
                if self.config['normalize_scores']:
                    results = self._normalize_scores(results)
                
                # Store the component results
                component_results.append({
                    'name': func_config['name'],
                    'weight': func_config['weight'],
                    'results': results
                })
                
            except Exception as e:
                logger.error(f"Error executing search function '{func_config['name']}': {str(e)}")
                component_results.append({
                    'name': func_config['name'],
                    'weight': func_config['weight'],
                    'error': str(e),
                    'results': []
                })
        
        # Combine results based on the specified method
        all_results = [cr for cr in component_results if cr.get('results')]
        
        if not all_results:
            combined_results = []
        elif self.config['combination_method'] == 'reciprocal_rank_fusion':
            combined_results = self._combine_by_reciprocal_rank_fusion(
                all_results, limit
            )
        else:  # default to linear_combination
            combined_results = self._combine_by_linear_weights(
                all_results, limit
            )
        
        # Filter results below the minimum score threshold
        if self.config['min_score_threshold'] > 0:
            combined_results = [
                result for result in combined_results 
                if result.get('score', 0) >= self.config['min_score_threshold']
            ]
        
        # Create explanation
        explanation = {
            'query': query,
            'search_strategy': 'Multi-Index Hybrid Search',
            'description': 'Combined search results from multiple indices/collections',
            'config': self.config,
            'results': combined_results,
            'components': component_results,
            'indices': [
                {
                    'name': func_config['name'],
                    'weight': func_config['weight']
                }
                for func_config in self.search_funcs
            ]
        }
        
        return explanation
    
    @property
    def supported_backends(self) -> List[str]:
        """
        Get the list of supported backend types.
        
        Returns:
            List of supported backend types (always ['generic'])
        """
        return ['generic']  # This searcher is not specific to any backend
    
    def get_component_weights(self) -> Dict[str, float]:
        """
        Get the current weights for each search component.
        
        Returns:
            Dictionary with component names as keys and weights as values
        """
        return {
            func_config['name']: func_config['weight']
            for func_config in self.search_funcs
        }
    
    def set_component_weights(self, weights: Dict[str, float]) -> None:
        """
        Set the weights for each search component.
        
        Args:
            weights: Dictionary with component names as keys and weights as values
        
        Raises:
            ValueError: If a component name is not found
        """
        # Validate that all component names exist
        for name in weights.keys():
            if not any(func_config['name'] == name for func_config in self.search_funcs):
                raise ValueError(f"Component name not found: {name}")
        
        # Update weights
        for func_config in self.search_funcs:
            if func_config['name'] in weights:
                func_config['weight'] = weights[func_config['name']]
        
        # Normalize weights
        self._normalize_weights()
        
        logger.info(f"Updated component weights: {self.get_component_weights()}") 