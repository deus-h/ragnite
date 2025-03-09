"""
Unified Retriever Interface

This module provides a unified interface for all retrieval types, including vector search,
keyword search, hybrid search, and contextual retrieval.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum, auto

from .hybrid_searchers import (
    get_hybrid_searcher,
    BaseHybridSearcher,
    VectorKeywordHybridSearcher,
    BM25VectorHybridSearcher,
    MultiIndexHybridSearcher,
    WeightedHybridSearcher,
    ContextualHybridSearcher
)

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Enum representing different retrieval strategies."""
    VECTOR = auto()
    KEYWORD = auto()
    BM25 = auto()
    VECTOR_KEYWORD_HYBRID = auto()
    BM25_VECTOR_HYBRID = auto()
    MULTI_INDEX_HYBRID = auto()
    WEIGHTED_HYBRID = auto()
    CONTEXTUAL_HYBRID = auto()


class UnifiedRetriever:
    """
    Unified interface for all retrieval types.
    
    This class provides a consistent API for retrieving documents using different
    search strategies, including vector search, keyword search, hybrid search, and
    contextual retrieval.
    
    Attributes:
        strategy (RetrievalStrategy): The retrieval strategy to use
        retriever: The underlying retriever object
        config (Dict[str, Any]): Configuration settings
    """
    
    def __init__(self, 
                 strategy: Union[RetrievalStrategy, str],
                 **kwargs):
        """
        Initialize the UnifiedRetriever.
        
        Args:
            strategy: The retrieval strategy to use. Can be a RetrievalStrategy enum
                or a string (case-insensitive):
                - 'vector': Vector search
                - 'keyword': Keyword search
                - 'bm25': BM25 search
                - 'vector_keyword_hybrid': Hybrid vector and keyword search
                - 'bm25_vector_hybrid': Hybrid BM25 and vector search
                - 'multi_index_hybrid': Multi-index hybrid search
                - 'weighted_hybrid': Weighted hybrid search
                - 'contextual_hybrid': Contextual hybrid search using Anthropic's technique
            **kwargs: Additional arguments for the specific retriever
                
        Common kwargs for all retrievers:
            config (Dict[str, Any]): Configuration dictionary
        
        Specific kwargs for different retrievers:
            - For VECTOR: vector_search_func (required)
            - For KEYWORD: keyword_search_func (required)
            - For BM25: bm25_search_func (required) or corpus and doc_ids (required)
            - For VECTOR_KEYWORD_HYBRID: vector_search_func (required), keyword_search_func (required)
            - For BM25_VECTOR_HYBRID: vector_search_func (required), bm25_search_func or (corpus and doc_ids)
            - For MULTI_INDEX_HYBRID: search_funcs (required)
            - For WEIGHTED_HYBRID: search_funcs (required)
            - For CONTEXTUAL_HYBRID: vector_search_func (required), embedding_model (required)
            
        Raises:
            ValueError: If an unsupported strategy is provided
            ValueError: If required arguments for a specific strategy are missing
        """
        # Convert string to enum if necessary
        if isinstance(strategy, str):
            strategy = self._parse_strategy(strategy)
        
        self.strategy = strategy
        self.config = kwargs.get('config', {})
        
        # Initialize the appropriate retriever based on the strategy
        if strategy == RetrievalStrategy.VECTOR:
            self._init_vector_retriever(**kwargs)
        elif strategy == RetrievalStrategy.KEYWORD:
            self._init_keyword_retriever(**kwargs)
        elif strategy == RetrievalStrategy.BM25:
            self._init_bm25_retriever(**kwargs)
        elif strategy in [
            RetrievalStrategy.VECTOR_KEYWORD_HYBRID,
            RetrievalStrategy.BM25_VECTOR_HYBRID,
            RetrievalStrategy.MULTI_INDEX_HYBRID,
            RetrievalStrategy.WEIGHTED_HYBRID,
            RetrievalStrategy.CONTEXTUAL_HYBRID
        ]:
            self._init_hybrid_retriever(strategy, **kwargs)
        else:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")
        
        logger.info(f"Initialized UnifiedRetriever with strategy: {strategy}")
    
    def _parse_strategy(self, strategy_str: str) -> RetrievalStrategy:
        """
        Parse a string into a RetrievalStrategy enum.
        
        Args:
            strategy_str: Strategy name as a string
            
        Returns:
            RetrievalStrategy enum
            
        Raises:
            ValueError: If the strategy string is not recognized
        """
        strategy_str = strategy_str.upper().replace('-', '_')
        
        if strategy_str == 'VECTOR':
            return RetrievalStrategy.VECTOR
        elif strategy_str == 'KEYWORD':
            return RetrievalStrategy.KEYWORD
        elif strategy_str == 'BM25':
            return RetrievalStrategy.BM25
        elif strategy_str in ['VECTOR_KEYWORD', 'VECTOR_KEYWORD_HYBRID']:
            return RetrievalStrategy.VECTOR_KEYWORD_HYBRID
        elif strategy_str in ['BM25_VECTOR', 'BM25_VECTOR_HYBRID']:
            return RetrievalStrategy.BM25_VECTOR_HYBRID
        elif strategy_str in ['MULTI_INDEX', 'MULTI_INDEX_HYBRID']:
            return RetrievalStrategy.MULTI_INDEX_HYBRID
        elif strategy_str in ['WEIGHTED', 'WEIGHTED_HYBRID']:
            return RetrievalStrategy.WEIGHTED_HYBRID
        elif strategy_str in ['CONTEXTUAL', 'CONTEXTUAL_HYBRID']:
            return RetrievalStrategy.CONTEXTUAL_HYBRID
        else:
            valid_strategies = [e.name for e in RetrievalStrategy]
            raise ValueError(f"Unknown retrieval strategy: {strategy_str}. "
                            f"Valid strategies: {valid_strategies}")
    
    def _init_vector_retriever(self, **kwargs):
        """Initialize a vector retriever."""
        vector_search_func = kwargs.get('vector_search_func')
        if not vector_search_func:
            raise ValueError("vector_search_func is required for VECTOR strategy")
        
        self.retriever = vector_search_func
    
    def _init_keyword_retriever(self, **kwargs):
        """Initialize a keyword retriever."""
        keyword_search_func = kwargs.get('keyword_search_func')
        if not keyword_search_func:
            raise ValueError("keyword_search_func is required for KEYWORD strategy")
        
        self.retriever = keyword_search_func
    
    def _init_bm25_retriever(self, **kwargs):
        """Initialize a BM25 retriever."""
        bm25_search_func = kwargs.get('bm25_search_func')
        corpus = kwargs.get('corpus')
        doc_ids = kwargs.get('doc_ids')
        
        if bm25_search_func:
            self.retriever = bm25_search_func
        elif corpus and doc_ids:
            from rank_bm25 import BM25Plus
            
            # Tokenize corpus for BM25
            tokenized_corpus = [doc.split() for doc in corpus]
            
            # Create BM25 index
            bm25 = BM25Plus(tokenized_corpus)
            
            # Create search function
            def bm25_search(query, limit=10, **search_kwargs):
                tokenized_query = query.split()
                scores = bm25.get_scores(tokenized_query)
                
                # Create a list of (index, score) tuples and sort by score
                results_with_scores = [(i, scores[i]) for i in range(len(scores))]
                results_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 'limit' results
                top_results = results_with_scores[:limit]
                
                # Create result objects
                results = []
                for idx, score in top_results:
                    results.append({
                        'id': doc_ids[idx],
                        'score': float(score),
                        'content': corpus[idx]
                    })
                
                return results
            
            self.retriever = bm25_search
        else:
            raise ValueError("Either bm25_search_func or both corpus and doc_ids must be provided for BM25 strategy")
    
    def _init_hybrid_retriever(self, strategy: RetrievalStrategy, **kwargs):
        """Initialize a hybrid retriever based on the strategy."""
        strategy_map = {
            RetrievalStrategy.VECTOR_KEYWORD_HYBRID: "vector_keyword",
            RetrievalStrategy.BM25_VECTOR_HYBRID: "bm25_vector",
            RetrievalStrategy.MULTI_INDEX_HYBRID: "multi_index",
            RetrievalStrategy.WEIGHTED_HYBRID: "weighted",
            RetrievalStrategy.CONTEXTUAL_HYBRID: "contextual"
        }
        
        # Create a hybrid searcher using the factory
        self.retriever = get_hybrid_searcher(
            searcher_type=strategy_map[strategy],
            **kwargs
        )
    
    def retrieve(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents matching the query.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results, each containing at least:
                - 'id': Document ID
                - 'score': Relevance score
                - Additional metadata depending on the retriever
        """
        logger.info(f"Retrieving documents for query: '{query}' with strategy: {self.strategy.name}")
        
        try:
            # Call the appropriate search method based on the retriever type
            if isinstance(self.retriever, BaseHybridSearcher):
                results = self.retriever.search(query, limit, **kwargs)
            else:
                # For function-based retrievers
                results = self.retriever(query, limit, **kwargs)
            
            logger.info(f"Retrieved {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def explain(self, query: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """
        Explain how the search results were generated.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            Explanation of the search process
            
        Raises:
            NotImplementedError: For retriever types that don't support explanation
        """
        logger.info(f"Explaining retrieval for query: '{query}' with strategy: {self.strategy.name}")
        
        try:
            # For hybrid searchers that implement explain_search
            if isinstance(self.retriever, BaseHybridSearcher):
                return self.retriever.explain_search(query, limit, **kwargs)
            
            # For function-based retrievers, create a basic explanation
            results = self.retrieve(query, limit, **kwargs)
            
            return {
                "query": query,
                "results": results,
                "strategy": self.strategy.name,
                "explanation": f"Used {self.strategy.name} retrieval strategy"
            }
            
        except Exception as e:
            logger.error(f"Error explaining retrieval: {str(e)}")
            raise
    
    def get_strategy_name(self) -> str:
        """
        Get the name of the current retrieval strategy.
        
        Returns:
            Strategy name as a string
        """
        return self.strategy.name
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the retriever configuration.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            NotImplementedError: For retriever types that don't support configuration
        """
        self.config.update(config)
        
        # For hybrid searchers that implement set_config
        if isinstance(self.retriever, BaseHybridSearcher):
            self.retriever.set_config(config)
            logger.info(f"Updated configuration for {self.strategy.name}")
        else:
            logger.warning(f"Configuration update not supported for {self.strategy.name}")
    
    def get_component_weights(self) -> Dict[str, float]:
        """
        Get the weights assigned to each search component.
        
        Returns:
            Dictionary mapping component names to their weights
            
        Raises:
            NotImplementedError: For retriever types that don't support component weights
        """
        if isinstance(self.retriever, BaseHybridSearcher):
            return self.retriever.get_component_weights()
        else:
            logger.warning(f"Component weights not supported for {self.strategy.name}")
            return {}
    
    def set_component_weights(self, weights: Dict[str, float]) -> None:
        """
        Set the weights for search components.
        
        Args:
            weights: Dictionary mapping component names to their weights
            
        Raises:
            NotImplementedError: For retriever types that don't support component weights
        """
        if isinstance(self.retriever, BaseHybridSearcher):
            self.retriever.set_component_weights(weights)
            logger.info(f"Updated component weights for {self.strategy.name}")
        else:
            logger.warning(f"Setting component weights not supported for {self.strategy.name}") 