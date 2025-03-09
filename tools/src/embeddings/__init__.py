"""
Embedding Tools

This module provides tools for working with embeddings, including embedding generators,
visualizers, analyzers, model adapters, and dimensionality reduction techniques.
"""

# Import embedding generators
from .embedding_generators import (
    BaseEmbeddingGenerator,
    SentenceTransformerGenerator,
    HuggingFaceGenerator,
    OpenAIGenerator,
    TensorFlowGenerator,
    CustomGenerator,
    get_embedding_generator
)

# Import embedding visualizers
from .embedding_visualizers import (
    BaseEmbeddingVisualizer,
    MatplotlibVisualizer,
    PlotlyVisualizer,
    HeatmapVisualizer,
    get_embedding_visualizer
)

# Import embedding analyzers
from .embedding_analyzers import (
    BaseEmbeddingAnalyzer,
    SimilarityAnalyzer,
    ClusteringAnalyzer,
    OutlierAnalyzer,
    DimensionalityAnalyzer,
    get_embedding_analyzer
)

# Import model adapters
from .model_adapters import (
    BaseModelAdapter,
    DimensionalityAdapter,
    NormalizationAdapter,
    LinearAdapter,
    CompositeAdapter,
    get_model_adapter
)

# Import dimensionality reduction
from .dimensionality_reduction import (
    BaseDimensionalityReducer,
    PCAReducer,
    SVDReducer,
    TSNEReducer,
    UMAPReducer,
    get_dimensionality_reducer
)

__all__ = [
    # Embedding generators
    "BaseEmbeddingGenerator",
    "SentenceTransformerGenerator",
    "HuggingFaceGenerator",
    "OpenAIGenerator",
    "TensorFlowGenerator",
    "CustomGenerator",
    "get_embedding_generator",
    
    # Embedding visualizers
    "BaseEmbeddingVisualizer",
    "MatplotlibVisualizer",
    "PlotlyVisualizer",
    "HeatmapVisualizer",
    "get_embedding_visualizer",
    
    # Embedding analyzers
    "BaseEmbeddingAnalyzer",
    "SimilarityAnalyzer",
    "ClusteringAnalyzer",
    "OutlierAnalyzer",
    "DimensionalityAnalyzer",
    "get_embedding_analyzer",
    
    # Model adapters
    "BaseModelAdapter",
    "DimensionalityAdapter",
    "NormalizationAdapter",
    "LinearAdapter",
    "CompositeAdapter",
    "get_model_adapter",
    
    # Dimensionality reduction
    "BaseDimensionalityReducer",
    "PCAReducer",
    "SVDReducer",
    "TSNEReducer",
    "UMAPReducer",
    "get_dimensionality_reducer"
] 