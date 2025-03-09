"""
Retrieval Tools

This module provides tools for retrieval in RAG systems, including query processors,
retrieval debuggers, filter builders, hybrid searchers, rerankers, and generation tools.
"""

from .query_processors import BaseQueryProcessor, QueryExpander, QueryRewriter, QueryDecomposer, QueryTranslator
from .query_processors.factory import get_query_processor

from .retrieval_debuggers import BaseRetrievalDebugger, RetrievalInspector, QueryAnalyzer, ContextAnalyzer, RetrievalVisualizer
from .retrieval_debuggers.factory import get_retrieval_debugger

from .filter_builders import BaseFilterBuilder, MetadataFilterBuilder, DateFilterBuilder, NumericFilterBuilder, CompositeFilterBuilder
from .filter_builders.factory import get_filter_builder

from .hybrid_searchers import BaseHybridSearcher, VectorKeywordHybridSearcher, BM25VectorHybridSearcher, MultiIndexHybridSearcher, WeightedHybridSearcher
from .hybrid_searchers.factory import get_hybrid_searcher

from .rerankers import BaseReranker, CrossEncoderReranker, MonoT5Reranker, LLMReranker, EnsembleReranker
from .rerankers.factory import get_reranker

from .generation import (
    BasePromptTemplate, 
    BasicPromptTemplate,
    FewShotPromptTemplate,
    ChainOfThoughtPromptTemplate,
    StructuredPromptTemplate,
    get_prompt_template,
    
    BaseContextFormatter,
    BasicContextFormatter,
    MetadataEnrichedFormatter,
    SourceAttributionFormatter,
    HierarchicalContextFormatter,
    get_context_formatter,
    
    BaseOutputParser,
    JSONOutputParser,
    XMLOutputParser,
    MarkdownOutputParser,
    StructuredOutputParser,
    get_output_parser,
    ParsingError,
    
    BaseHallucinationDetector,
    HallucinationDetectionResult,
    FactualConsistencyDetector,
    SourceVerificationDetector,
    ContradictionDetector,
    UncertaintyDetector,
    get_hallucination_detector,
    
    BaseCitationGenerator,
    AcademicCitationGenerator,
    LegalCitationGenerator,
    WebCitationGenerator,
    CustomCitationGenerator,
    get_citation_generator
)

# Import statements for other components will be added as they are implemented

__all__ = [
    # Query Processors
    'BaseQueryProcessor',
    'QueryExpander',
    'QueryRewriter',
    'QueryDecomposer',
    'QueryTranslator',
    'get_query_processor',
    
    # Retrieval Debuggers
    'BaseRetrievalDebugger',
    'RetrievalInspector',
    'QueryAnalyzer',
    'ContextAnalyzer',
    'RetrievalVisualizer',
    'get_retrieval_debugger',
    
    # Filter Builders
    'BaseFilterBuilder',
    'MetadataFilterBuilder',
    'DateFilterBuilder',
    'NumericFilterBuilder',
    'CompositeFilterBuilder',
    'get_filter_builder',
    
    # Hybrid Searchers
    'BaseHybridSearcher',
    'VectorKeywordHybridSearcher',
    'BM25VectorHybridSearcher',
    'MultiIndexHybridSearcher',
    'WeightedHybridSearcher',
    'get_hybrid_searcher',
    
    # Rerankers
    'BaseReranker',
    'CrossEncoderReranker',
    'MonoT5Reranker',
    'LLMReranker',
    'EnsembleReranker',
    'get_reranker',
    
    # Generation - Prompt Templates
    'BasePromptTemplate',
    'BasicPromptTemplate',
    'FewShotPromptTemplate',
    'ChainOfThoughtPromptTemplate',
    'StructuredPromptTemplate',
    'get_prompt_template',
    
    # Generation - Context Formatters
    'BaseContextFormatter',
    'BasicContextFormatter',
    'MetadataEnrichedFormatter',
    'SourceAttributionFormatter',
    'HierarchicalContextFormatter',
    'get_context_formatter',
    
    # Generation - Output Parsers
    'BaseOutputParser',
    'JSONOutputParser',
    'XMLOutputParser',
    'MarkdownOutputParser',
    'StructuredOutputParser',
    'get_output_parser',
    'ParsingError',
    
    # Generation - Hallucination Detectors
    'BaseHallucinationDetector',
    'HallucinationDetectionResult',
    'FactualConsistencyDetector',
    'SourceVerificationDetector',
    'ContradictionDetector',
    'UncertaintyDetector',
    'get_hallucination_detector',
    
    # Generation - Citation Generators
    'BaseCitationGenerator',
    'AcademicCitationGenerator',
    'LegalCitationGenerator',
    'WebCitationGenerator',
    'CustomCitationGenerator',
    'get_citation_generator',
] 