"""
Multi-Modal RAG Module for RAGNITE

This module implements advanced multi-modal retrieval augmented generation,
supporting both text and image inputs/outputs. It integrates vision-language models
for understanding visual content and specialized retrievers for different content types.
"""

from .multimodal_rag import MultiModalRAG
from .image_retriever import ImageRetriever

__all__ = ['MultiModalRAG', 'ImageRetriever'] 