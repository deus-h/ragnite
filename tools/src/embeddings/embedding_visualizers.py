"""
Embedding Visualizers

This module provides tools for visualizing embeddings in 2D and 3D space,
including scatter plots, heatmaps, and interactive visualizations.
"""

import os
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class BaseEmbeddingVisualizer(ABC):
    """
    Abstract base class for embedding visualizers.
    """
    
    @abstractmethod
    def visualize(
        self,
        embeddings: List[List[float]],
        labels: Optional[List[Any]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """
        Visualize embeddings.
        
        Args:
            embeddings: List of embeddings to visualize
            labels: Optional list of labels for each embedding
            metadata: Optional list of metadata for each embedding
            **kwargs: Additional arguments for visualization
            
        Returns:
            Visualization object or path to saved visualization
        """
        pass
    
    def save(self, path: str, **kwargs) -> None:
        """
        Save the visualization to a file.
        
        Args:
            path: Path to save the visualization to
            **kwargs: Additional arguments for saving
        """
        raise NotImplementedError("This visualizer does not support saving.")


class MatplotlibVisualizer(BaseEmbeddingVisualizer):
    """
    Visualize embeddings using Matplotlib.
    """
    
    def __init__(
        self,
        reducer_type: str = "pca",
        n_components: int = 2,
        figsize: Tuple[int, int] = (10, 8),
        title: str = "Embedding Visualization",
        cmap: str = "viridis",
        alpha: float = 0.7,
        s: int = 50,
        random_state: int = 42
    ):
        """
        Initialize a Matplotlib visualizer.
        
        Args:
            reducer_type: Type of dimensionality reducer to use ('pca', 'tsne')
            n_components: Number of components to reduce to (2 or 3)
            figsize: Figure size (width, height)
            title: Plot title
            cmap: Colormap for points
            alpha: Alpha value for points
            s: Size of points
            random_state: Random seed for reproducibility
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for MatplotlibVisualizer. "
                "Install it with `pip install matplotlib`."
            )
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for dimensionality reduction. "
                "Install it with `pip install scikit-learn`."
            )
        
        if n_components not in (2, 3):
            raise ValueError("n_components must be 2 or 3.")
        
        self.reducer_type = reducer_type
        self.n_components = n_components
        self.figsize = figsize
        self.title = title
        self.cmap = cmap
        self.alpha = alpha
        self.s = s
        self.random_state = random_state
        self.fig = None
        self.ax = None
    
    def _reduce_dimensions(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Reduce the dimensionality of embeddings.
        
        Args:
            embeddings: List of embeddings to reduce
            
        Returns:
            Reduced embeddings as a numpy array
        """
        if not embeddings:
            raise ValueError("No embeddings provided for visualization.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Check if embeddings already have the right dimensionality
        if embeddings_array.shape[1] == self.n_components:
            return embeddings_array
        
        # Reduce dimensionality
        if self.reducer_type == "pca":
            reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.reducer_type == "tsne":
            reducer = TSNE(
                n_components=self.n_components,
                random_state=self.random_state
            )
        else:
            raise ValueError(
                f"Unsupported reducer type: {self.reducer_type}. "
                "Supported types are: 'pca', 'tsne'."
            )
        
        reduced = reducer.fit_transform(embeddings_array)
        
        return reduced
    
    def visualize(
        self,
        embeddings: List[List[float]],
        labels: Optional[List[Any]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Visualize embeddings using Matplotlib.
        
        Args:
            embeddings: List of embeddings to visualize
            labels: Optional list of labels for each embedding
            metadata: Optional list of metadata for each embedding
            **kwargs: Additional arguments for visualization
                - figsize: Figure size (width, height)
                - title: Plot title
                - cmap: Colormap for points
                - alpha: Alpha value for points
                - s: Size of points
                
        Returns:
            Matplotlib figure
        """
        # Update parameters from kwargs
        figsize = kwargs.get("figsize", self.figsize)
        title = kwargs.get("title", self.title)
        cmap = kwargs.get("cmap", self.cmap)
        alpha = kwargs.get("alpha", self.alpha)
        s = kwargs.get("s", self.s)
        
        # Reduce dimensionality
        reduced = self._reduce_dimensions(embeddings)
        
        # Create figure
        self.fig = plt.figure(figsize=figsize)
        
        # Create 2D or 3D plot
        if self.n_components == 2:
            self.ax = self.fig.add_subplot(111)
            
            # Plot points
            if labels is not None:
                # Convert labels to numeric values if they are not already
                unique_labels = list(set(labels))
                label_to_int = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = [label_to_int[label] for label in labels]
                
                scatter = self.ax.scatter(
                    reduced[:, 0],
                    reduced[:, 1],
                    c=numeric_labels,
                    cmap=cmap,
                    alpha=alpha,
                    s=s
                )
                
                # Add legend
                if len(unique_labels) <= 20:  # Only show legend if not too many labels
                    legend1 = self.ax.legend(
                        scatter.legend_elements()[0],
                        unique_labels,
                        loc="upper right",
                        title="Labels"
                    )
                    self.ax.add_artist(legend1)
            else:
                self.ax.scatter(
                    reduced[:, 0],
                    reduced[:, 1],
                    alpha=alpha,
                    s=s
                )
            
            # Set labels and title
            self.ax.set_xlabel("Component 1")
            self.ax.set_ylabel("Component 2")
            
        else:  # 3D plot
            self.ax = self.fig.add_subplot(111, projection="3d")
            
            # Plot points
            if labels is not None:
                # Convert labels to numeric values if they are not already
                unique_labels = list(set(labels))
                label_to_int = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = [label_to_int[label] for label in labels]
                
                scatter = self.ax.scatter(
                    reduced[:, 0],
                    reduced[:, 1],
                    reduced[:, 2],
                    c=numeric_labels,
                    cmap=cmap,
                    alpha=alpha,
                    s=s
                )
                
                # Add legend
                if len(unique_labels) <= 20:  # Only show legend if not too many labels
                    legend1 = self.ax.legend(
                        scatter.legend_elements()[0],
                        unique_labels,
                        loc="upper right",
                        title="Labels"
                    )
                    self.ax.add_artist(legend1)
            else:
                self.ax.scatter(
                    reduced[:, 0],
                    reduced[:, 1],
                    reduced[:, 2],
                    alpha=alpha,
                    s=s
                )
            
            # Set labels
            self.ax.set_xlabel("Component 1")
            self.ax.set_ylabel("Component 2")
            self.ax.set_zlabel("Component 3")
        
        # Set title
        self.ax.set_title(title)
        
        # Add grid
        self.ax.grid(True, linestyle="--", alpha=0.7)
        
        # Tight layout
        plt.tight_layout()
        
        return self.fig
    
    def save(self, path: str, dpi: int = 300, **kwargs) -> None:
        """
        Save the visualization to a file.
        
        Args:
            path: Path to save the visualization to
            dpi: DPI for the saved image
            **kwargs: Additional arguments for saving
        """
        if self.fig is None:
            raise ValueError("No visualization to save. Call visualize() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save figure
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved visualization to {path}")


class PlotlyVisualizer(BaseEmbeddingVisualizer):
    """
    Visualize embeddings using Plotly for interactive visualizations.
    """
    
    def __init__(
        self,
        reducer_type: str = "pca",
        n_components: int = 2,
        title: str = "Embedding Visualization",
        colorscale: str = "Viridis",
        opacity: float = 0.7,
        size: int = 5,
        random_state: int = 42
    ):
        """
        Initialize a Plotly visualizer.
        
        Args:
            reducer_type: Type of dimensionality reducer to use ('pca', 'tsne')
            n_components: Number of components to reduce to (2 or 3)
            title: Plot title
            colorscale: Colorscale for points
            opacity: Opacity value for points
            size: Size of points
            random_state: Random seed for reproducibility
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "plotly is required for PlotlyVisualizer. "
                "Install it with `pip install plotly`."
            )
        
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for dimensionality reduction. "
                "Install it with `pip install scikit-learn`."
            )
        
        if n_components not in (2, 3):
            raise ValueError("n_components must be 2 or 3.")
        
        self.reducer_type = reducer_type
        self.n_components = n_components
        self.title = title
        self.colorscale = colorscale
        self.opacity = opacity
        self.size = size
        self.random_state = random_state
        self.fig = None
    
    def _reduce_dimensions(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Reduce the dimensionality of embeddings.
        
        Args:
            embeddings: List of embeddings to reduce
            
        Returns:
            Reduced embeddings as a numpy array
        """
        if not embeddings:
            raise ValueError("No embeddings provided for visualization.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Check if embeddings already have the right dimensionality
        if embeddings_array.shape[1] == self.n_components:
            return embeddings_array
        
        # Reduce dimensionality
        if self.reducer_type == "pca":
            reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.reducer_type == "tsne":
            reducer = TSNE(
                n_components=self.n_components,
                random_state=self.random_state
            )
        else:
            raise ValueError(
                f"Unsupported reducer type: {self.reducer_type}. "
                "Supported types are: 'pca', 'tsne'."
            )
        
        reduced = reducer.fit_transform(embeddings_array)
        
        return reduced
    
    def visualize(
        self,
        embeddings: List[List[float]],
        labels: Optional[List[Any]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> go.Figure:
        """
        Visualize embeddings using Plotly.
        
        Args:
            embeddings: List of embeddings to visualize
            labels: Optional list of labels for each embedding
            metadata: Optional list of metadata for each embedding
            **kwargs: Additional arguments for visualization
                - title: Plot title
                - colorscale: Colorscale for points
                - opacity: Opacity value for points
                - size: Size of points
                
        Returns:
            Plotly figure
        """
        # Update parameters from kwargs
        title = kwargs.get("title", self.title)
        colorscale = kwargs.get("colorscale", self.colorscale)
        opacity = kwargs.get("opacity", self.opacity)
        size = kwargs.get("size", self.size)
        
        # Reduce dimensionality
        reduced = self._reduce_dimensions(embeddings)
        
        # Prepare hover text if metadata is provided
        hover_text = None
        if metadata is not None:
            hover_text = []
            for i, meta in enumerate(metadata):
                text = "<br>".join([f"{k}: {v}" for k, v in meta.items()])
                if labels is not None:
                    text = f"Label: {labels[i]}<br>" + text
                hover_text.append(text)
        
        # Create figure
        if self.n_components == 2:
            # Create 2D scatter plot
            if labels is not None:
                # Convert labels to strings
                str_labels = [str(label) for label in labels]
                
                # Create figure with px for better legend handling
                self.fig = px.scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    color=str_labels,
                    labels={"color": "Label"},
                    title=title,
                    opacity=opacity,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    hover_name=str_labels if hover_text is None else None,
                    hover_data=metadata if metadata is not None else None,
                    custom_data=[range(len(reduced))]
                )
                
                # Update marker size
                self.fig.update_traces(marker=dict(size=size))
                
            else:
                # Create basic scatter plot
                self.fig = go.Figure(
                    data=[go.Scatter(
                        x=reduced[:, 0],
                        y=reduced[:, 1],
                        mode="markers",
                        marker=dict(
                            size=size,
                            opacity=opacity,
                            colorscale=colorscale
                        ),
                        text=hover_text,
                        hoverinfo="text" if hover_text is not None else "none"
                    )]
                )
                
                # Set title
                self.fig.update_layout(title=title)
            
            # Update layout
            self.fig.update_layout(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                legend_title="Labels",
                hovermode="closest"
            )
            
        else:  # 3D plot
            # Create 3D scatter plot
            if labels is not None:
                # Convert labels to strings
                str_labels = [str(label) for label in labels]
                
                # Create figure with px for better legend handling
                self.fig = px.scatter_3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    color=str_labels,
                    labels={"color": "Label"},
                    title=title,
                    opacity=opacity,
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    hover_name=str_labels if hover_text is None else None,
                    hover_data=metadata if metadata is not None else None,
                    custom_data=[range(len(reduced))]
                )
                
                # Update marker size
                self.fig.update_traces(marker=dict(size=size))
                
            else:
                # Create basic 3D scatter plot
                self.fig = go.Figure(
                    data=[go.Scatter3d(
                        x=reduced[:, 0],
                        y=reduced[:, 1],
                        z=reduced[:, 2],
                        mode="markers",
                        marker=dict(
                            size=size,
                            opacity=opacity,
                            colorscale=colorscale
                        ),
                        text=hover_text,
                        hoverinfo="text" if hover_text is not None else "none"
                    )]
                )
                
                # Set title
                self.fig.update_layout(title=title)
            
            # Update layout
            self.fig.update_layout(
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3"
                ),
                legend_title="Labels",
                hovermode="closest"
            )
        
        return self.fig
    
    def save(
        self,
        path: str,
        format: str = "html",
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: float = 1.0,
        **kwargs
    ) -> None:
        """
        Save the visualization to a file.
        
        Args:
            path: Path to save the visualization to
            format: Format to save as ('html', 'png', 'jpg', 'svg', 'pdf')
            width: Width of the saved image (for non-HTML formats)
            height: Height of the saved image (for non-HTML formats)
            scale: Scale factor for the saved image (for non-HTML formats)
            **kwargs: Additional arguments for saving
        """
        if self.fig is None:
            raise ValueError("No visualization to save. Call visualize() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save figure
        if format == "html":
            self.fig.write_html(path)
        else:
            self.fig.write_image(
                path,
                width=width,
                height=height,
                scale=scale
            )
        
        logger.info(f"Saved visualization to {path}")


class HeatmapVisualizer(BaseEmbeddingVisualizer):
    """
    Visualize embedding similarity as a heatmap.
    """
    
    def __init__(
        self,
        metric: str = "cosine",
        figsize: Tuple[int, int] = (10, 8),
        title: str = "Embedding Similarity Heatmap",
        cmap: str = "viridis",
        annotate: bool = False,
        fmt: str = ".2f"
    ):
        """
        Initialize a heatmap visualizer.
        
        Args:
            metric: Similarity metric to use ('cosine', 'euclidean', 'dot')
            figsize: Figure size (width, height)
            title: Plot title
            cmap: Colormap for heatmap
            annotate: Whether to annotate the heatmap with values
            fmt: Format string for annotations
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for HeatmapVisualizer. "
                "Install it with `pip install matplotlib`."
            )
        
        self.metric = metric
        self.figsize = figsize
        self.title = title
        self.cmap = cmap
        self.annotate = annotate
        self.fmt = fmt
        self.fig = None
        self.ax = None
    
    def _compute_similarity(self, embeddings: List[List[float]]) -> np.ndarray:
        """
        Compute similarity matrix between embeddings.
        
        Args:
            embeddings: List of embeddings to compute similarity for
            
        Returns:
            Similarity matrix as a numpy array
        """
        if not embeddings:
            raise ValueError("No embeddings provided for visualization.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Compute similarity matrix
        if self.metric == "cosine":
            # Normalize embeddings
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            normalized = embeddings_array / norms
            
            # Compute cosine similarity
            similarity = np.dot(normalized, normalized.T)
            
        elif self.metric == "euclidean":
            # Compute pairwise distances
            n = embeddings_array.shape[0]
            similarity = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    similarity[i, j] = np.linalg.norm(
                        embeddings_array[i] - embeddings_array[j]
                    )
            
            # Convert distance to similarity
            similarity = 1 / (1 + similarity)
            
        elif self.metric == "dot":
            # Compute dot product
            similarity = np.dot(embeddings_array, embeddings_array.T)
            
        else:
            raise ValueError(
                f"Unsupported similarity metric: {self.metric}. "
                "Supported metrics are: 'cosine', 'euclidean', 'dot'."
            )
        
        return similarity
    
    def visualize(
        self,
        embeddings: List[List[float]],
        labels: Optional[List[Any]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Visualize embedding similarity as a heatmap.
        
        Args:
            embeddings: List of embeddings to visualize
            labels: Optional list of labels for each embedding
            metadata: Optional list of metadata for each embedding
            **kwargs: Additional arguments for visualization
                - figsize: Figure size (width, height)
                - title: Plot title
                - cmap: Colormap for heatmap
                - annotate: Whether to annotate the heatmap with values
                - fmt: Format string for annotations
                
        Returns:
            Matplotlib figure
        """
        # Update parameters from kwargs
        figsize = kwargs.get("figsize", self.figsize)
        title = kwargs.get("title", self.title)
        cmap = kwargs.get("cmap", self.cmap)
        annotate = kwargs.get("annotate", self.annotate)
        fmt = kwargs.get("fmt", self.fmt)
        
        # Compute similarity matrix
        similarity = self._compute_similarity(embeddings)
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = self.ax.imshow(similarity, cmap=cmap)
        
        # Add colorbar
        cbar = self.fig.colorbar(im, ax=self.ax)
        cbar.set_label(f"{self.metric.capitalize()} Similarity")
        
        # Set title
        self.ax.set_title(title)
        
        # Set labels
        if labels is not None:
            # Limit the number of labels to avoid overcrowding
            max_labels = 30
            if len(labels) <= max_labels:
                # Set tick labels
                self.ax.set_xticks(np.arange(len(labels)))
                self.ax.set_yticks(np.arange(len(labels)))
                self.ax.set_xticklabels(labels)
                self.ax.set_yticklabels(labels)
                
                # Rotate x tick labels
                plt.setp(
                    self.ax.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor"
                )
            else:
                # Too many labels, show only indices
                self.ax.set_xticks(np.arange(0, len(labels), len(labels) // max_labels))
                self.ax.set_yticks(np.arange(0, len(labels), len(labels) // max_labels))
        
        # Annotate heatmap
        if annotate:
            # Only annotate if not too many embeddings
            if len(embeddings) <= 20:
                for i in range(len(embeddings)):
                    for j in range(len(embeddings)):
                        self.ax.text(
                            j, i, format(similarity[i, j], fmt),
                            ha="center", va="center",
                            color="white" if similarity[i, j] < 0.7 else "black"
                        )
        
        # Tight layout
        plt.tight_layout()
        
        return self.fig
    
    def save(self, path: str, dpi: int = 300, **kwargs) -> None:
        """
        Save the visualization to a file.
        
        Args:
            path: Path to save the visualization to
            dpi: DPI for the saved image
            **kwargs: Additional arguments for saving
        """
        if self.fig is None:
            raise ValueError("No visualization to save. Call visualize() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save figure
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved visualization to {path}")


# Factory function to get an embedding visualizer
def get_embedding_visualizer(
    visualizer_type: str = "matplotlib",
    n_components: int = 2,
    **kwargs
) -> BaseEmbeddingVisualizer:
    """
    Get an embedding visualizer based on the specified type.
    
    Args:
        visualizer_type: Type of visualizer to use ('matplotlib', 'plotly', 'heatmap')
        n_components: Number of components to visualize (2 or 3)
        **kwargs: Additional arguments for the specific visualizer
        
    Returns:
        Embedding visualizer instance
        
    Raises:
        ValueError: If an unsupported visualizer type is specified
    """
    if visualizer_type == "matplotlib":
        return MatplotlibVisualizer(
            n_components=n_components,
            **kwargs
        )
    elif visualizer_type == "plotly":
        return PlotlyVisualizer(
            n_components=n_components,
            **kwargs
        )
    elif visualizer_type == "heatmap":
        return HeatmapVisualizer(**kwargs)
    else:
        raise ValueError(
            f"Unsupported visualizer type: {visualizer_type}. "
            "Supported types are: 'matplotlib', 'plotly', 'heatmap'."
        ) 