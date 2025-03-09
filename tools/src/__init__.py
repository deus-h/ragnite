"""
RAG Utility Tools

This package provides a collection of utility tools for building, debugging,
and optimizing Retrieval-Augmented Generation (RAG) systems.
"""

__version__ = "0.1.0"

# Import key components for easier access
from .data_processing.document_loaders import Document, DirectoryLoader
from .data_processing.text_chunkers import get_chunker
from .data_processing.metadata_extractors import create_comprehensive_extractor
from .data_processing.data_cleaners import create_standard_cleaner
from .data_processing.data_augmentation import create_standard_augmentation_pipeline

# Define what gets imported with "from tools.src import *"
__all__ = [
    'Document',
    'DirectoryLoader',
    'get_chunker',
    'create_comprehensive_extractor',
    'create_standard_cleaner',
    'create_standard_augmentation_pipeline',
] 