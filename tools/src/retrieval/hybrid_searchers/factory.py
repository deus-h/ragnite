"""
Hybrid Searcher Factory

This module provides a factory function for creating hybrid searchers.
"""

import logging
from typing import Callable, Dict, Any, Optional, List

from .base_hybrid_searcher import BaseHybridSearcher
from .vector_keyword_hybrid_searcher import VectorKeywordHybridSearcher
from .bm25_vector_hybrid_searcher import BM25VectorHybridSearcher
from .multi_index_hybrid_searcher import MultiIndexHybridSearcher
from .weighted_hybrid_searcher import WeightedHybridSearcher

# Configure logging
logger = logging.getLogger(__name__)


def get_hybrid_searcher(
    searcher_type: str,
    vector_search_func: Optional[Callable] = None,
    keyword_search_func: Optional[Callable] = None,
    bm25_search_func: Optional[Callable] = None,
    corpus: Optional[List[str]] = None,
    doc_ids: Optional[List[str]] = None,
    search_funcs: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> BaseHybridSearcher:
    """
    Factory function to create a hybrid searcher of the specified type.
    
    Args:
        searcher_type: Type of hybrid searcher to create
            - 'vector_keyword': VectorKeywordHybridSearcher
            - 'bm25_vector': BM25VectorHybridSearcher
            - 'multi_index': MultiIndexHybridSearcher
            - 'weighted': WeightedHybridSearcher
        vector_search_func: Function for vector similarity search
        keyword_search_func: Function for keyword search (for VectorKeywordHybridSearcher)
        bm25_search_func: Function for BM25 search (for BM25VectorHybridSearcher)
        corpus: Document corpus for internal BM25 index (for BM25VectorHybridSearcher)
        doc_ids: Document IDs corresponding to corpus (for BM25VectorHybridSearcher)
        search_funcs: List of search function configs (for MultiIndexHybridSearcher and WeightedHybridSearcher)
        **kwargs: Additional arguments to pass to the searcher constructor
    
    Returns:
        A hybrid searcher instance
    
    Raises:
        ValueError: If the searcher type is not supported
        ValueError: If required arguments for a specific searcher type are missing
    """
    searcher_type = searcher_type.lower().replace('_', '').replace('-', '')
    
    # Create VectorKeywordHybridSearcher
    if searcher_type in ['vectorkeyword', 'keywordvector', 'vectorkeywordhybrid', 'keywordvectorhybrid']:
        if vector_search_func is None:
            raise ValueError("vector_search_func is required for VectorKeywordHybridSearcher")
        if keyword_search_func is None:
            raise ValueError("keyword_search_func is required for VectorKeywordHybridSearcher")
        
        return VectorKeywordHybridSearcher(
            vector_search_func=vector_search_func,
            keyword_search_func=keyword_search_func,
            config=kwargs.get('config', None)
        )
    
    # Create BM25VectorHybridSearcher
    elif searcher_type in ['bm25vector', 'vectorbm25', 'bm25vectorhybrid', 'vectorbm25hybrid']:
        if vector_search_func is None:
            raise ValueError("vector_search_func is required for BM25VectorHybridSearcher")
        
        # Either bm25_search_func or (corpus and doc_ids) must be provided
        if bm25_search_func is None and (corpus is None or doc_ids is None):
            raise ValueError("Either bm25_search_func or both corpus and doc_ids must be provided for BM25VectorHybridSearcher")
        
        return BM25VectorHybridSearcher(
            vector_search_func=vector_search_func,
            bm25_search_func=bm25_search_func,
            corpus=corpus,
            doc_ids=doc_ids,
            config=kwargs.get('config', None)
        )
    
    # Create MultiIndexHybridSearcher
    elif searcher_type in ['multiindex', 'multindex', 'multiindexhybrid', 'multiindexhybridsearcher']:
        if search_funcs is None or not search_funcs:
            raise ValueError("search_funcs is required for MultiIndexHybridSearcher and must contain at least one search function")
        
        return MultiIndexHybridSearcher(
            search_funcs=search_funcs,
            config=kwargs.get('config', None)
        )
    
    # Create WeightedHybridSearcher
    elif searcher_type in ['weighted', 'weightedhybrid', 'weightedhybridsearcher']:
        if search_funcs is None or not search_funcs:
            raise ValueError("search_funcs is required for WeightedHybridSearcher and must contain at least one search function")
        
        return WeightedHybridSearcher(
            search_funcs=search_funcs,
            config=kwargs.get('config', None)
        )
    
    else:
        supported_searchers = ["vector_keyword", "bm25_vector", "multi_index", "weighted"]
        raise ValueError(f"Unsupported hybrid searcher type: {searcher_type}. "
                         f"Supported types: {supported_searchers}") 