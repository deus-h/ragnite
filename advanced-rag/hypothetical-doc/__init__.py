"""
HyDE (Hypothetical Document Embeddings) Module for RAGNITE

This module implements enhanced retrieval using hypothetical document embeddings
to improve semantic search by bridging the lexical gap between queries and documents.

Key components:
- HyDEEngine: Core engine for generating and using hypothetical documents
- HyDETemplates: Domain-specific templates for different applications
- HyDEEvaluator: Tools for evaluating HyDE quality and effectiveness
"""

from .hyde_engine import HyDEEngine
from .templates import HyDETemplates
from .evaluation import HyDEEvaluator

__all__ = ['HyDEEngine', 'HyDETemplates', 'HyDEEvaluator'] 