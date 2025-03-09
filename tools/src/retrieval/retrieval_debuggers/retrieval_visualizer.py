"""
Retrieval Visualizer

This module provides the RetrievalVisualizer class for visualizing retrieval results
and metrics to aid in debugging and optimizing retrieval systems.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_debugger import BaseRetrievalDebugger

# Configure logging
logger = logging.getLogger(__name__)

# Default colors for visualizations
DEFAULT_COLORS = list(mcolors.TABLEAU_COLORS.values())
DEFAULT_CMAP = 'viridis'

class RetrievalVisualizer(BaseRetrievalDebugger):
    """
    Visualization tool for retrieval results and metrics.
    
    This class provides methods to visualize various aspects of retrieval results, including:
    - Score distributions
    - Relevance visualizations
    - Content similarity maps
    - Performance comparisons
    - Query-document similarity heatmaps
    
    The visualizer can generate static plots or interactive visualizations based on configuration.
    """
    
    def __init__(
        self,
        content_field: str = "content",
        score_field: str = "score",
        metadata_fields: Optional[List[str]] = None,
        interactive: bool = False,
        plot_style: str = "whitegrid",
        default_figsize: Tuple[int, int] = (10, 6),
        color_palette: str = "viridis",
        embedding_dim_reduction: str = "tsne",
        custom_plotting_fn: Optional[Callable] = None,
        output_dir: Optional[str] = None,
        save_plots: bool = False,
        plot_format: str = "png",
        dpi: int = 150
    ):
        """
        Initialize the RetrievalVisualizer.
        
        Args:
            content_field: Field containing document text
            score_field: Field containing retrieval scores
            metadata_fields: List of metadata fields to include in visualizations
            interactive: Whether to create interactive visualizations (requires plotly)
            plot_style: Seaborn style for plots ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
            default_figsize: Default figure size (width, height) in inches
            color_palette: Color palette for plots (any valid seaborn or matplotlib palette)
            embedding_dim_reduction: Method for dimensionality reduction ('tsne', 'pca', 'umap')
            custom_plotting_fn: Optional custom function for specialized plotting
            output_dir: Directory to save plots (if save_plots is True)
            save_plots: Whether to save generated plots to disk
            plot_format: Format to save plots ('png', 'jpg', 'svg', 'pdf')
            dpi: Resolution for saved plots
        """
        super().__init__()
        
        self.content_field = content_field
        self.score_field = score_field
        self.metadata_fields = metadata_fields or []
        self.interactive = interactive
        self.default_figsize = default_figsize
        self.color_palette = color_palette
        self.embedding_dim_reduction = embedding_dim_reduction
        self.custom_plotting_fn = custom_plotting_fn
        self.output_dir = output_dir
        self.save_plots = save_plots
        self.plot_format = plot_format
        self.dpi = dpi
        
        # Set plot style
        sns.set_style(plot_style)
        sns.set_palette(color_palette)
        
        # Check for interactive visualization dependencies if needed
        if self.interactive:
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                self.px = px
                self.go = go
            except ImportError:
                logger.warning("Plotly not installed. Interactive visualizations disabled.")
                self.interactive = False
        
        # Initialize dimensionality reduction models lazily
        self._tsne_model = None
        self._pca_model = None
        self._umap_model = None
        
        # Initialize vectorizer for text similarity
        self._vectorizer = None
    
    def _extract_content(self, result: Dict[str, Any]) -> str:
        """Extract content from a retrieval result."""
        # Handle nested content fields with dot notation
        if "." in self.content_field:
            parts = self.content_field.split(".")
            temp = result
            for part in parts:
                if not isinstance(temp, dict) or part not in temp:
                    return ""
                temp = temp[part]
            return str(temp) if temp is not None else ""
        
        # Direct field access
        content = result.get(self.content_field, "")
        return str(content) if content is not None else ""
    
    def _extract_score(self, result: Dict[str, Any]) -> float:
        """Extract score from a retrieval result."""
        # Handle nested score fields with dot notation
        if "." in self.score_field:
            parts = self.score_field.split(".")
            temp = result
            for part in parts:
                if not isinstance(temp, dict) or part not in temp:
                    return 0.0
                temp = temp[part]
            return float(temp) if temp is not None else 0.0
        
        # Direct field access
        score = result.get(self.score_field, 0.0)
        return float(score) if score is not None else 0.0
    
    def _extract_metadata(self, result: Dict[str, Any], field: str) -> Any:
        """Extract metadata field from a result."""
        if not field:
            return None
            
        # Handle nested fields with dot notation
        if "." in field:
            parts = field.split(".")
            temp = result
            for part in parts:
                if not isinstance(temp, dict) or part not in temp:
                    return None
                temp = temp[part]
            return temp
            
        # Handle top-level metadata field
        if "metadata" in result and isinstance(result["metadata"], dict):
            return result["metadata"].get(field)
            
        # Direct field access
        return result.get(field)
    
    def _save_figure(self, fig: Figure, filename: str) -> None:
        """Save the figure to disk if save_plots is enabled."""
        if not self.save_plots or not self.output_dir:
            return
            
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Build full path
        filepath = os.path.join(self.output_dir, f"{filename}.{self.plot_format}")
        
        # Save figure
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved figure to {filepath}")
    
    def _preprocess_texts(self, texts: List[str]) -> np.ndarray:
        """Convert a list of texts to a TF-IDF matrix."""
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(max_features=100)
            
        return self._vectorizer.fit_transform(texts).toarray()
    
    def _reduce_dimensions(self, high_dim_data: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce dimensions of the data for visualization."""
        if high_dim_data.shape[0] <= 1:
            return np.zeros((high_dim_data.shape[0], n_components))
            
        if self.embedding_dim_reduction == 'tsne':
            if self._tsne_model is None:
                self._tsne_model = TSNE(n_components=n_components, random_state=42)
            return self._tsne_model.fit_transform(high_dim_data)
            
        elif self.embedding_dim_reduction == 'pca':
            if self._pca_model is None:
                self._pca_model = PCA(n_components=n_components)
            return self._pca_model.fit_transform(high_dim_data)
            
        elif self.embedding_dim_reduction == 'umap':
            try:
                import umap
                if self._umap_model is None:
                    self._umap_model = umap.UMAP(n_components=n_components, random_state=42)
                return self._umap_model.fit_transform(high_dim_data)
            except ImportError:
                logger.warning("UMAP not installed. Falling back to PCA.")
                if self._pca_model is None:
                    self._pca_model = PCA(n_components=n_components)
                return self._pca_model.fit_transform(high_dim_data)
                
        else:
            # Default to PCA
            if self._pca_model is None:
                self._pca_model = PCA(n_components=n_components)
            return self._pca_model.fit_transform(high_dim_data)
    
    def plot_score_distribution(self, results: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Plot the distribution of retrieval scores.
        
        Args:
            results: List of retrieval results
            **kwargs: Additional plot configuration options
            
        Returns:
            Matplotlib figure or Plotly figure depending on interactive setting
        """
        scores = [self._extract_score(result) for result in results]
        
        if len(scores) == 0:
            logger.warning("No scores to plot")
            return None
            
        title = kwargs.get('title', 'Retrieval Score Distribution')
        xlabel = kwargs.get('xlabel', 'Score')
        ylabel = kwargs.get('ylabel', 'Frequency')
        bins = kwargs.get('bins', min(20, len(scores)))
        figsize = kwargs.get('figsize', self.default_figsize)
        
        if self.interactive:
            import plotly.express as px
            fig = px.histogram(
                x=scores,
                nbins=bins,
                title=title,
                labels={'x': xlabel, 'y': ylabel},
                opacity=0.7,
                template='plotly_white'
            )
            
            # Add vertical line for average score
            avg_score = sum(scores) / len(scores)
            fig.add_vline(x=avg_score, line_dash="dash", line_color="red",
                         annotation_text=f"Average: {avg_score:.2f}",
                         annotation_position="top right")
                         
            return fig
        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(scores, bins=bins, alpha=0.7, color='steelblue')
            
            # Add vertical line for average score
            avg_score = sum(scores) / len(scores)
            ax.axvline(avg_score, color='red', linestyle='dashed', linewidth=1)
            ax.text(avg_score, ax.get_ylim()[1]*0.9, f' Avg: {avg_score:.2f}',
                   color='red', verticalalignment='top')
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure if enabled
            if self.save_plots:
                self._save_figure(fig, "score_distribution")
                
            return fig
            
    def plot_rank_vs_score(self, results: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Plot retrieval rank vs score to visualize score decay.
        
        Args:
            results: List of retrieval results (assumed to be in rank order)
            **kwargs: Additional plot configuration options
            
        Returns:
            Matplotlib figure or Plotly figure depending on interactive setting
        """
        if not results:
            logger.warning("No results to plot")
            return None
            
        scores = [self._extract_score(result) for result in results]
        ranks = list(range(1, len(scores) + 1))
        
        title = kwargs.get('title', 'Retrieval Rank vs Score')
        xlabel = kwargs.get('xlabel', 'Rank')
        ylabel = kwargs.get('ylabel', 'Score')
        figsize = kwargs.get('figsize', self.default_figsize)
        marker_size = kwargs.get('marker_size', 80)
        
        if self.interactive:
            import plotly.express as px
            
            fig = px.scatter(
                x=ranks, 
                y=scores,
                title=title,
                labels={'x': xlabel, 'y': ylabel},
                size=[marker_size] * len(ranks),
                template='plotly_white'
            )
            
            # Add trendline
            fig.add_traces(
                px.scatter(x=ranks, y=scores, trendline="lowess").data[1]
            )
            
            # Update layout
            fig.update_traces(marker=dict(color='steelblue'))
            fig.update_layout(showlegend=False)
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot the points
            ax.scatter(ranks, scores, s=marker_size, alpha=0.7, color='steelblue')
            
            # Add trendline
            z = np.polyfit(ranks, scores, 1)
            p = np.poly1d(z)
            ax.plot(ranks, p(ranks), "r--", alpha=0.7, linewidth=2)
            
            # Add smoothed trendline
            from scipy.signal import savgol_filter
            if len(ranks) > 3:
                window_length = min(9, len(ranks) - (len(ranks) % 2 == 0))
                if window_length > 2:
                    smoothed = savgol_filter(scores, window_length, 2)
                    ax.plot(ranks, smoothed, color='green', alpha=0.7, linewidth=2)
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure if enabled
            if self.save_plots:
                self._save_figure(fig, "rank_vs_score")
                
            return fig
    
    def plot_content_similarity_matrix(self, results: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Plot a similarity matrix for retrieved document content.
        
        Args:
            results: List of retrieval results
            **kwargs: Additional plot configuration options
            
        Returns:
            Matplotlib figure or Plotly figure depending on interactive setting
        """
        if not results:
            logger.warning("No results to plot")
            return None
            
        contents = [self._extract_content(result) for result in results]
        
        # Filter out empty content
        valid_indices = [i for i, content in enumerate(contents) if content.strip()]
        if not valid_indices:
            logger.warning("No valid content to plot")
            return None
            
        valid_contents = [contents[i] for i in valid_indices]
        
        # Transform text to TF-IDF vectors
        try:
            tfidf_matrix = self._preprocess_texts(valid_contents)
        except Exception as e:
            logger.error(f"Error preprocessing texts: {e}")
            return None
        
        # Compute similarity matrix
        similarity_matrix = np.zeros((len(valid_contents), len(valid_contents)))
        for i in range(len(valid_contents)):
            for j in range(len(valid_contents)):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Cosine similarity between document vectors
                    dot_product = np.dot(tfidf_matrix[i], tfidf_matrix[j])
                    norm_i = np.linalg.norm(tfidf_matrix[i])
                    norm_j = np.linalg.norm(tfidf_matrix[j])
                    if norm_i > 0 and norm_j > 0:
                        similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
                    else:
                        similarity_matrix[i, j] = 0.0
        
        # Plot options
        title = kwargs.get('title', 'Document Content Similarity Matrix')
        figsize = kwargs.get('figsize', self.default_figsize)
        cmap = kwargs.get('cmap', 'viridis')
        include_values = kwargs.get('include_values', True)
        
        # Create labels (document numbers)
        doc_ranks = [valid_indices[i] + 1 for i in range(len(valid_contents))]
        
        if self.interactive:
            import plotly.figure_factory as ff
            
            fig = ff.create_annotated_heatmap(
                z=similarity_matrix,
                x=doc_ranks,
                y=doc_ranks,
                annotation_text=np.around(similarity_matrix, decimals=2) if include_values else None,
                colorscale=cmap,
                showscale=True
            )
            
            fig.update_layout(
                title=title,
                xaxis_title="Document Rank",
                yaxis_title="Document Rank",
                xaxis_tickvals=doc_ranks,
                yaxis_tickvals=doc_ranks
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            heatmap = ax.imshow(similarity_matrix, cmap=cmap)
            
            # Add colorbar
            cbar = plt.colorbar(heatmap, ax=ax)
            cbar.set_label('Similarity')
            
            # Add annotations
            if include_values:
                for i in range(len(valid_contents)):
                    for j in range(len(valid_contents)):
                        ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                                ha="center", va="center", color="black" if similarity_matrix[i, j] < 0.7 else "white")
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(valid_contents)))
            ax.set_yticks(np.arange(len(valid_contents)))
            ax.set_xticklabels(doc_ranks)
            ax.set_yticklabels(doc_ranks)
            
            ax.set_title(title)
            ax.set_xlabel("Document Rank")
            ax.set_ylabel("Document Rank")
            
            plt.tight_layout()
            
            # Save figure if enabled
            if self.save_plots:
                self._save_figure(fig, "content_similarity_matrix")
                
            return fig
        
    def plot_content_similarity_map(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Any:
        """
        Plot a 2D similarity map of retrieved document content and the query.
        
        Args:
            query: The search query
            results: List of retrieval results
            **kwargs: Additional plot configuration options
            
        Returns:
            Matplotlib figure or Plotly figure depending on interactive setting
        """
        if not results:
            logger.warning("No results to plot")
            return None
            
        contents = [self._extract_content(result) for result in results]
        
        # Filter out empty content
        valid_indices = [i for i, content in enumerate(contents) if content.strip()]
        if not valid_indices:
            logger.warning("No valid content to plot")
            return None
            
        valid_contents = [contents[i] for i in valid_indices]
        
        # Add query to the contents for mapping
        all_texts = [query] + valid_contents
        
        # Transform text to vectors
        try:
            tfidf_matrix = self._preprocess_texts(all_texts)
        except Exception as e:
            logger.error(f"Error preprocessing texts: {e}")
            return None
        
        # Reduce dimensions for visualization
        try:
            reduced_data = self._reduce_dimensions(tfidf_matrix, n_components=2)
        except Exception as e:
            logger.error(f"Error reducing dimensions: {e}")
            return None
        
        # Extract query and document positions
        query_pos = reduced_data[0]
        doc_positions = reduced_data[1:]
        
        # Get document scores for coloring
        scores = [self._extract_score(results[i]) for i in valid_indices]
        
        # Create document labels
        doc_labels = [f"Doc {valid_indices[i]+1}" for i in range(len(valid_indices))]
        
        # Plot options
        title = kwargs.get('title', 'Document Content Similarity Map')
        figsize = kwargs.get('figsize', self.default_figsize)
        query_marker_size = kwargs.get('query_marker_size', 150)
        doc_marker_size = kwargs.get('doc_marker_size', 100)
        
        if self.interactive:
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Create a dataframe for plotting
            import pandas as pd
            df = pd.DataFrame({
                'x': [query_pos[0]] + list(doc_positions[:, 0]),
                'y': [query_pos[1]] + list(doc_positions[:, 1]),
                'label': ['Query'] + doc_labels,
                'type': ['Query'] + ['Document'] * len(doc_labels),
                'score': [1.0] + scores
            })
            
            # Create figure
            fig = px.scatter(
                df, x='x', y='y', color='score', text='label', 
                color_continuous_scale=self.color_palette,
                title=title,
                labels={'score': 'Similarity Score'},
                template='plotly_white',
                size='type',
                size_map={'Query': query_marker_size, 'Document': doc_marker_size}
            )
            
            # Update marker appearance
            fig.update_traces(
                textposition='top center',
                marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                selector=dict(mode='markers+text')
            )
            
            # Add rings around the query
            for radius in [0.5, 1.0, 1.5]:
                theta = np.linspace(0, 2*np.pi, 100)
                x = query_pos[0] + radius * np.cos(theta)
                y = query_pos[1] + radius * np.sin(theta)
                
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='lines', 
                    line=dict(color='rgba(0,0,0,0.2)', dash='dash'),
                    showlegend=False, hoverinfo='none'
                ))
            
            # Update layout
            fig.update_layout(
                legend_title_text='Document Type'
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot query point
            ax.scatter(query_pos[0], query_pos[1], s=query_marker_size, c='red', marker='*', label='Query')
            
            # Plot document points with score-based coloring
            scatter = ax.scatter(
                doc_positions[:, 0], 
                doc_positions[:, 1], 
                s=doc_marker_size, 
                c=scores, 
                cmap=self.color_palette,
                alpha=0.8,
                edgecolors='black', 
                linewidths=1
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Similarity Score')
            
            # Add document labels
            for i, (x, y) in enumerate(doc_positions):
                ax.annotate(
                    doc_labels[i],
                    (x, y),
                    xytext=(0, 5),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8
                )
            
            # Add rings around the query
            for radius in [0.5, 1.0, 1.5]:
                circle = plt.Circle(
                    (query_pos[0], query_pos[1]), 
                    radius, 
                    fill=False, 
                    linestyle='--', 
                    alpha=0.3, 
                    color='gray'
                )
                ax.add_patch(circle)
            
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Make axes equal to ensure circles appear as circles
            ax.set_aspect('equal')
            
            plt.tight_layout()
            
            # Save figure if enabled
            if self.save_plots:
                self._save_figure(fig, "content_similarity_map")
                
            return fig
    
    def plot_metadata_distribution(self, results: List[Dict[str, Any]], metadata_field: str, **kwargs) -> Any:
        """
        Plot the distribution of a specific metadata field across retrieved documents.
        
        Args:
            results: List of retrieval results
            metadata_field: Field to visualize
            **kwargs: Additional plot configuration options
            
        Returns:
            Matplotlib figure or Plotly figure depending on interactive setting
        """
        if not results:
            logger.warning("No results to plot")
            return None
            
        if metadata_field not in self.metadata_fields and metadata_field not in results[0].get("metadata", {}):
            logger.warning(f"Metadata field '{metadata_field}' not found")
            return None
            
        # Extract metadata values
        values = []
        for result in results:
            value = self._extract_metadata(result, metadata_field)
            if value is not None:
                values.append(value)
                
        if not values:
            logger.warning(f"No valid values for metadata field '{metadata_field}'")
            return None
            
        # Plot options
        title = kwargs.get('title', f'Distribution of {metadata_field}')
        xlabel = kwargs.get('xlabel', metadata_field)
        ylabel = kwargs.get('ylabel', 'Count')
        figsize = kwargs.get('figsize', self.default_figsize)
        
        # Different plot types based on the data type
        if all(isinstance(v, (int, float)) for v in values):
            # Numeric data - plot histogram
            bins = kwargs.get('bins', min(20, len(set(values))))
            
            if self.interactive:
                import plotly.express as px
                fig = px.histogram(
                    x=values,
                    nbins=bins,
                    title=title,
                    labels={'x': xlabel, 'y': ylabel},
                    opacity=0.7,
                    template='plotly_white'
                )
                return fig
            else:
                fig, ax = plt.subplots(figsize=figsize)
                ax.hist(values, bins=bins, alpha=0.7, color='steelblue')
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save figure if enabled
                if self.save_plots:
                    self._save_figure(fig, f"metadata_dist_{metadata_field}")
                    
                return fig
                
        elif all(isinstance(v, str) for v in values):
            # Categorical data - plot bar chart
            from collections import Counter
            counts = Counter(values)
            
            # Sort by counts if too many categories
            if len(counts) > 10:
                counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])
                title += " (Top 10)"
                
            categories = list(counts.keys())
            frequencies = list(counts.values())
            
            if self.interactive:
                import plotly.express as px
                fig = px.bar(
                    x=categories,
                    y=frequencies,
                    title=title,
                    labels={'x': xlabel, 'y': ylabel},
                    opacity=0.7,
                    template='plotly_white'
                )
                return fig
            else:
                fig, ax = plt.subplots(figsize=figsize)
                
                # Rotate x labels if many categories or long category names
                rotation = 45 if len(categories) > 5 or max([len(str(c)) for c in categories]) > 10 else 0
                
                bars = ax.bar(categories, frequencies, alpha=0.7, color='steelblue')
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height}', ha='center', va='bottom')
                
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.xticks(rotation=rotation)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                
                # Save figure if enabled
                if self.save_plots:
                    self._save_figure(fig, f"metadata_dist_{metadata_field}")
                    
                return fig
        else:
            logger.warning(f"Mixed or unsupported data types for metadata field '{metadata_field}'")
            return None
            
    def plot_metadata_correlation(self, results: List[Dict[str, Any]], metadata_field: str, **kwargs) -> Any:
        """
        Plot the correlation between metadata field and retrieval scores.
        
        Args:
            results: List of retrieval results
            metadata_field: Field to correlate with scores
            **kwargs: Additional plot configuration options
            
        Returns:
            Matplotlib figure or Plotly figure depending on interactive setting
        """
        if not results:
            logger.warning("No results to plot")
            return None
            
        if metadata_field not in self.metadata_fields and metadata_field not in results[0].get("metadata", {}):
            logger.warning(f"Metadata field '{metadata_field}' not found")
            return None
            
        # Extract metadata values and scores
        data_points = []
        for result in results:
            value = self._extract_metadata(result, metadata_field)
            score = self._extract_score(result)
            if value is not None and isinstance(value, (int, float)):
                data_points.append((value, score))
                
        if not data_points:
            logger.warning(f"No valid numeric values for metadata field '{metadata_field}'")
            return None
            
        # Unzip data points
        x_values, y_values = zip(*data_points)
        
        # Plot options
        title = kwargs.get('title', f'Correlation: {metadata_field} vs Score')
        xlabel = kwargs.get('xlabel', metadata_field)
        ylabel = kwargs.get('ylabel', 'Score')
        figsize = kwargs.get('figsize', self.default_figsize)
        
        if self.interactive:
            import plotly.express as px
            
            fig = px.scatter(
                x=x_values, 
                y=y_values,
                title=title,
                labels={'x': xlabel, 'y': ylabel},
                opacity=0.7,
                trendline='ols',
                template='plotly_white'
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot scatter points
            ax.scatter(x_values, y_values, alpha=0.7, s=80, color='steelblue')
            
            # Add trendline
            try:
                z = np.polyfit(x_values, y_values, 1)
                p = np.poly1d(z)
                ax.plot(sorted(x_values), p(sorted(x_values)), "r--", alpha=0.7, linewidth=2)
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(x_values, y_values)[0, 1]
                ax.text(0.05, 0.95, f"Correlation: {correlation:.2f}",
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
            except Exception as e:
                logger.warning(f"Error calculating trendline: {e}")
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure if enabled
            if self.save_plots:
                self._save_figure(fig, f"metadata_corr_{metadata_field}")
                
            return fig
    
    def plot_comparison_scores(
        self,
        results_sets: List[List[Dict[str, Any]]],
        names: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Plot a comparison of score distributions for multiple retrieval systems.
        
        Args:
            results_sets: List of result sets to compare
            names: Names of the retrieval systems (default: System 1, System 2, etc.)
            **kwargs: Additional plot configuration options
            
        Returns:
            Matplotlib figure or Plotly figure depending on interactive setting
        """
        if not results_sets:
            logger.warning("No results sets to compare")
            return None
            
        # Prepare names if not provided
        if names is None:
            names = [f"System {i+1}" for i in range(len(results_sets))]
            
        # Extract scores from each system
        all_scores = []
        for results in results_sets:
            scores = [self._extract_score(result) for result in results]
            all_scores.append(scores)
            
        # Plot options
        title = kwargs.get('title', 'Retrieval Score Comparison')
        xlabel = kwargs.get('xlabel', 'Retrieval System')
        ylabel = kwargs.get('ylabel', 'Score')
        figsize = kwargs.get('figsize', self.default_figsize)
        
        if self.interactive:
            import plotly.express as px
            import pandas as pd
            
            # Create dataframe for plotting
            data = []
            for i, (system_name, scores) in enumerate(zip(names, all_scores)):
                for score in scores:
                    data.append({
                        'System': system_name,
                        'Score': score
                    })
                    
            df = pd.DataFrame(data)
            
            fig = px.box(
                df,
                x='System',
                y='Score',
                title=title,
                points='all',
                color='System',
                template='plotly_white'
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot box plots for each system
            box = ax.boxplot(
                all_scores,
                labels=names,
                patch_artist=True,
                showmeans=True,
                showfliers=True,
                widths=0.6
            )
            
            # Customize box appearance
            for i, patch in enumerate(box['boxes']):
                color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                
            # Add individual points for each score (jitter for better visibility)
            for i, scores in enumerate(all_scores):
                # Calculate x position with jitter
                x = np.random.normal(i + 1, 0.08, size=len(scores))
                ax.scatter(x, scores, alpha=0.6, s=30, color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
                
            # Add mean score labels
            for i, scores in enumerate(all_scores):
                if scores:
                    mean_score = sum(scores) / len(scores)
                    ax.text(i + 1, mean_score, f"{mean_score:.2f}",
                            ha='center', va='bottom', fontsize=9,
                            bbox=dict(boxstyle='round', alpha=0.1))
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Save figure if enabled
            if self.save_plots:
                self._save_figure(fig, "comparison_scores")
                
            return fig
            
    def plot_top_k_comparison(
        self,
        query: str,
        results_sets: List[List[Dict[str, Any]]],
        names: Optional[List[str]] = None,
        k: int = 5,
        **kwargs
    ) -> Any:
        """
        Plot a comparison of the top K results across different retrieval systems.
        
        Args:
            query: The search query
            results_sets: List of result sets to compare
            names: Names of the retrieval systems (default: System 1, System 2, etc.)
            k: Number of top results to compare
            **kwargs: Additional plot configuration options
            
        Returns:
            Matplotlib figure or Plotly figure depending on interactive setting
        """
        if not results_sets:
            logger.warning("No results sets to compare")
            return None
            
        # Prepare names if not provided
        if names is None:
            names = [f"System {i+1}" for i in range(len(results_sets))]
            
        # Get top K results for each system
        top_k_results = []
        for results in results_sets:
            # Limit to top K
            system_top_k = results[:k] if len(results) >= k else results
            # Extract scores
            scores = [self._extract_score(result) for result in system_top_k]
            # Pad with zeros if less than K results
            if len(scores) < k:
                scores.extend([0.0] * (k - len(scores)))
            top_k_results.append(scores)
            
        # Plot options
        title = kwargs.get('title', f'Top {k} Results Comparison')
        xlabel = kwargs.get('xlabel', 'Rank')
        ylabel = kwargs.get('ylabel', 'Score')
        figsize = kwargs.get('figsize', self.default_figsize)
        
        # Create rank positions
        ranks = list(range(1, k + 1))
        
        if self.interactive:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add line for each system
            for i, (name, scores) in enumerate(zip(names, top_k_results)):
                fig.add_trace(go.Scatter(
                    x=ranks,
                    y=scores,
                    mode='lines+markers',
                    name=name,
                    line=dict(width=2),
                    marker=dict(size=10)
                ))
                
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                legend_title="Retrieval System",
                template='plotly_white'
            )
            
            # Set x-axis to show integer ranks
            fig.update_xaxes(tickmode='array', tickvals=ranks)
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot line for each system
            for i, (name, scores) in enumerate(zip(names, top_k_results)):
                color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                ax.plot(ranks, scores, marker='o', linewidth=2, label=name, color=color)
                
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks(ranks)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # Save figure if enabled
            if self.save_plots:
                self._save_figure(fig, "top_k_comparison")
                
            return fig
            
    # BaseRetrievalDebugger interface methods
    
    def analyze(self, query: str, results: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Analyze retrieval results and generate visualizations.
        
        Args:
            query: The search query
            results: List of retrieval results
            **kwargs: Additional configuration options
            
        Returns:
            Dictionary containing analysis results and visualization objects
        """
        if not results:
            logger.warning("No results to analyze")
            return {"result_count": 0, "visualizations": {}}
            
        # Default visualizations to generate
        default_visualizations = kwargs.get('visualizations', [
            'score_distribution',
            'rank_vs_score',
            'content_similarity_map'
        ])
        
        # Store visualizations
        visualizations = {}
        
        # Create requested visualizations
        if 'score_distribution' in default_visualizations:
            visualizations['score_distribution'] = self.plot_score_distribution(results)
            
        if 'rank_vs_score' in default_visualizations:
            visualizations['rank_vs_score'] = self.plot_rank_vs_score(results)
            
        if 'content_similarity_matrix' in default_visualizations:
            visualizations['content_similarity_matrix'] = self.plot_content_similarity_matrix(results)
            
        if 'content_similarity_map' in default_visualizations:
            visualizations['content_similarity_map'] = self.plot_content_similarity_map(query, results)
            
        # Metadata visualizations
        metadata_fields = kwargs.get('metadata_fields', self.metadata_fields)
        for field in metadata_fields:
            try:
                visualizations[f'metadata_distribution_{field}'] = self.plot_metadata_distribution(results, field)
                # Check if field has numeric values for correlation plot
                field_values = [self._extract_metadata(r, field) for r in results]
                if any(isinstance(v, (int, float)) for v in field_values if v is not None):
                    visualizations[f'metadata_correlation_{field}'] = self.plot_metadata_correlation(results, field)
            except Exception as e:
                logger.warning(f"Error generating metadata visualization for field '{field}': {e}")
        
        # Calculate basic statistics
        scores = [self._extract_score(result) for result in results]
        
        analysis_results = {
            "result_count": len(results),
            "score_stats": {
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "median_score": sorted(scores)[len(scores) // 2] if scores else 0,
            },
            "visualizations": visualizations
        }
        
        return analysis_results
        
    def compare(
        self,
        query: str,
        results_sets: List[List[Dict[str, Any]]],
        names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare multiple sets of retrieval results and generate comparative visualizations.
        
        Args:
            query: The search query
            results_sets: List of result sets to compare
            names: Names of the retrieval systems (default: System 1, System 2, etc.)
            **kwargs: Additional configuration options
            
        Returns:
            Dictionary containing comparison results and visualization objects
        """
        if not results_sets:
            logger.warning("No results sets to compare")
            return {"comparison_count": 0, "visualizations": {}}
            
        # Default visualizations to generate
        default_visualizations = kwargs.get('visualizations', [
            'comparison_scores',
            'top_k_comparison'
        ])
        
        # Prepare names if not provided
        if names is None:
            names = [f"System {i+1}" for i in range(len(results_sets))]
            
        # Store visualizations
        visualizations = {}
        
        # Create requested visualizations
        if 'comparison_scores' in default_visualizations:
            visualizations['comparison_scores'] = self.plot_comparison_scores(
                results_sets, names
            )
            
        # Top-K comparison
        k = kwargs.get('k', 5)
        if 'top_k_comparison' in default_visualizations:
            visualizations['top_k_comparison'] = self.plot_top_k_comparison(
                query, results_sets, names, k=k
            )
        
        # Calculate basic statistics for each system
        systems_stats = []
        for i, (name, results) in enumerate(zip(names, results_sets)):
            scores = [self._extract_score(result) for result in results]
            systems_stats.append({
                "name": name,
                "result_count": len(results),
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "median_score": sorted(scores)[len(scores) // 2] if scores else 0,
            })
        
        # Find best system based on average score
        best_system = None
        best_avg = -1
        for stats in systems_stats:
            if stats["avg_score"] > best_avg:
                best_avg = stats["avg_score"]
                best_system = stats["name"]
                
        comparison_results = {
            "comparison_count": len(results_sets),
            "systems": systems_stats,
            "best_system": best_system,
            "visualizations": visualizations
        }
        
        return comparison_results
        
    def evaluate(
        self,
        query: str,
        results: List[Dict[str, Any]],
        ground_truth: Union[List[str], List[Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results against ground truth.
        
        Note: This method returns evaluation metrics but doesn't generate
        visualizations directly. Use the analyze/compare methods for visualizations.
        
        Args:
            query: The search query
            results: List of retrieval results
            ground_truth: List of ground truth document IDs or documents
            **kwargs: Additional configuration options
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("RetrievalVisualizer doesn't implement detailed evaluation metrics.")
        logger.info("Use RetrievalInspector for comprehensive retrieval evaluation.")
        
        # Very basic overlap calculation if ground truth is available
        if not ground_truth or not results:
            return {"overlap": 0.0}
            
        # Extract result IDs
        result_ids = []
        for result in results:
            # Try various common ID fields
            for id_field in ['id', 'doc_id', 'document_id', '_id']:
                if id_field in result:
                    result_ids.append(str(result[id_field]))
                    break
        
        # Extract ground truth IDs
        gt_ids = []
        for gt_item in ground_truth:
            if isinstance(gt_item, str):
                gt_ids.append(gt_item)
            elif isinstance(gt_item, dict):
                # Try various common ID fields
                for id_field in ['id', 'doc_id', 'document_id', '_id']:
                    if id_field in gt_item:
                        gt_ids.append(str(gt_item[id_field]))
                        break
        
        # Calculate basic overlap
        overlap_count = len(set(result_ids) & set(gt_ids))
        overlap_ratio = overlap_count / len(gt_ids) if gt_ids else 0.0
        
        return {"overlap": overlap_ratio}
        
    def get_insights(self, query: str, results: List[Dict[str, Any]], **kwargs) -> List[str]:
        """
        Get insights about retrieval results based on visualizations.
        
        Args:
            query: The search query
            results: List of retrieval results
            **kwargs: Additional configuration options
            
        Returns:
            List of insights as strings
        """
        if not results:
            return ["No results to analyze"]
            
        insights = []
        
        # Score distribution insights
        scores = [self._extract_score(result) for result in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if scores:
            if avg_score > 0.8:
                insights.append("Retrieved documents have very high similarity scores, indicating strong relevance.")
            elif avg_score < 0.3:
                insights.append("Retrieved documents have low similarity scores, suggesting potential relevance issues.")
                
            score_range = max(scores) - min(scores)
            if score_range < 0.1:
                insights.append("Very narrow score distribution suggests potential over-optimization for a specific pattern.")
            elif score_range > 0.6:
                insights.append("Wide score distribution indicates varying degrees of relevance in results.")
                
            # Score decay insight
            if len(scores) > 3:
                first_score = scores[0]
                drop_off = first_score - scores[-1]
                if drop_off > 0.5:
                    insights.append(f"Significant score drop-off ({drop_off:.2f}) from first to last result suggests good ranking quality.")
                elif drop_off < 0.1:
                    insights.append("Minimal score drop-off suggests potential issues with result diversity or ranking.")
        
        # Content diversity insights
        if len(results) > 1:
            try:
                contents = [self._extract_content(result) for result in results]
                valid_contents = [content for content in contents if content.strip()]
                
                if valid_contents:
                    # Use TF-IDF to check content similarity
                    tfidf_matrix = self._preprocess_texts(valid_contents)
                    
                    # Calculate pairwise similarities
                    similarities = []
                    for i in range(len(valid_contents)):
                        for j in range(i+1, len(valid_contents)):
                            dot_product = np.dot(tfidf_matrix[i], tfidf_matrix[j])
                            norm_i = np.linalg.norm(tfidf_matrix[i])
                            norm_j = np.linalg.norm(tfidf_matrix[j])
                            if norm_i > 0 and norm_j > 0:
                                sim = dot_product / (norm_i * norm_j)
                                similarities.append(sim)
                    
                    if similarities:
                        avg_sim = sum(similarities) / len(similarities)
                        if avg_sim > 0.8:
                            insights.append("High content similarity between documents suggests potential redundancy.")
                        elif avg_sim < 0.3:
                            insights.append("Low content similarity indicates diverse result set.")
            except Exception as e:
                logger.warning(f"Error analyzing content diversity: {e}")
                
        # Metadata insights
        if self.metadata_fields:
            for field in self.metadata_fields:
                try:
                    values = [self._extract_metadata(result, field) for result in results]
                    values = [v for v in values if v is not None]
                    
                    if values:
                        # For numeric fields, check correlation with scores
                        if all(isinstance(v, (int, float)) for v in values):
                            correlation = np.corrcoef(values, scores[:len(values)])[0, 1]
                            if abs(correlation) > 0.7:
                                direction = "positive" if correlation > 0 else "negative"
                                insights.append(f"Strong {direction} correlation between '{field}' and retrieval scores.")
                                
                        # For categorical fields, check distribution
                        elif all(isinstance(v, str) for v in values):
                            from collections import Counter
                            counts = Counter(values)
                            most_common = counts.most_common(1)[0]
                            if most_common[1] / len(values) > 0.8:
                                insights.append(f"'{field}' shows low diversity with '{most_common[0]}' dominating {most_common[1]} of {len(values)} results.")
                except Exception as e:
                    logger.warning(f"Error analyzing metadata field '{field}': {e}")
        
        # Combine insights
        if not insights:
            insights = ["No significant patterns detected in the retrieval results."]
            
        return insights 