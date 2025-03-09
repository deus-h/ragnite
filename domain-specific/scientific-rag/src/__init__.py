"""
Scientific RAG: A domain-specific Retrieval-Augmented Generation system for scientific research.

This module provides specialized components for processing, retrieving, and generating
content based on scientific literature and research papers.
"""

from . import chunking
from . import embedding
from . import retrieval
from . import generation
from . import evaluation

__version__ = "0.1.0" 