"""
Embedding Analyzers

This module provides tools for analyzing embeddings, including similarity analysis,
clustering analysis, and outlier detection.
"""

import os
import json
import logging
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Callable
import numpy as np
from collections import defaultdict

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class BaseEmbeddingAnalyzer(ABC):
    """
    Abstract base class for embedding analyzers.
    """
    
    @abstractmethod
    def analyze(self, embeddings: List[List[float]], **kwargs) -> Dict[str, Any]:
        """
        Analyze embeddings.
        
        Args:
            embeddings: List of embeddings to analyze
            **kwargs: Additional arguments for analysis
            
        Returns:
            Dictionary containing analysis results
        """
        pass


class SimilarityAnalyzer(BaseEmbeddingAnalyzer):
    """
    Analyze similarity between embeddings.
    """
    
    def __init__(
        self,
        similarity_metric: str = "cosine",
        threshold: float = 0.8,
        top_k: int = 5
    ):
        """
        Initialize a similarity analyzer.
        
        Args:
            similarity_metric: Metric to use for similarity calculation
                              ('cosine', 'euclidean', 'dot')
            threshold: Similarity threshold for identifying similar pairs
            top_k: Number of top similar pairs to return
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for similarity analysis. "
                "Install it with `pip install scikit-learn`."
            )
        
        self.similarity_metric = similarity_metric.lower()
        self.threshold = threshold
        self.top_k = top_k
    
    def analyze(
        self,
        embeddings: List[List[float]],
        texts: Optional[List[str]] = None,
        ids: Optional[List[Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze similarity between embeddings.
        
        Args:
            embeddings: List of embeddings to analyze
            texts: List of texts corresponding to embeddings (optional)
            ids: List of IDs corresponding to embeddings (optional)
            **kwargs: Additional arguments for analysis
            
        Returns:
            Dictionary containing similarity analysis results
        """
        if not embeddings:
            return {"error": "No embeddings to analyze"}
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Calculate similarity matrix
        if self.similarity_metric == "cosine":
            similarity_matrix = cosine_similarity(embeddings_array)
        elif self.similarity_metric == "euclidean":
            # Convert Euclidean distance to similarity
            distances = euclidean_distances(embeddings_array)
            similarity_matrix = 1 / (1 + distances)
        elif self.similarity_metric == "dot":
            similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
        else:
            return {"error": f"Unsupported similarity metric: {self.similarity_metric}"}
        
        # Set diagonal to 0 to ignore self-similarity
        np.fill_diagonal(similarity_matrix, 0)
        
        # Find pairs above threshold
        similar_pairs = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = similarity_matrix[i, j]
                if similarity >= self.threshold:
                    pair_info = {
                        "index1": i,
                        "index2": j,
                        "similarity": float(similarity)
                    }
                    
                    # Add texts if available
                    if texts:
                        pair_info["text1"] = texts[i]
                        pair_info["text2"] = texts[j]
                    
                    # Add IDs if available
                    if ids:
                        pair_info["id1"] = ids[i]
                        pair_info["id2"] = ids[j]
                    
                    similar_pairs.append(pair_info)
        
        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Get top-k pairs
        top_pairs = similar_pairs[:self.top_k]
        
        # Calculate statistics
        all_similarities = similarity_matrix[np.triu_indices(len(embeddings), k=1)]
        
        stats = {
            "mean_similarity": float(np.mean(all_similarities)),
            "median_similarity": float(np.median(all_similarities)),
            "min_similarity": float(np.min(all_similarities)),
            "max_similarity": float(np.max(all_similarities)),
            "std_similarity": float(np.std(all_similarities)),
            "num_pairs_above_threshold": len(similar_pairs),
            "total_pairs": len(all_similarities)
        }
        
        return {
            "similar_pairs": top_pairs,
            "statistics": stats,
            "similarity_metric": self.similarity_metric,
            "threshold": self.threshold
        }


class ClusteringAnalyzer(BaseEmbeddingAnalyzer):
    """
    Analyze clustering of embeddings.
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        min_clusters: int = 2,
        max_clusters: int = 10,
        algorithm: str = "kmeans",
        random_state: int = 42
    ):
        """
        Initialize a clustering analyzer.
        
        Args:
            n_clusters: Number of clusters (None for automatic determination)
            min_clusters: Minimum number of clusters to try (for automatic determination)
            max_clusters: Maximum number of clusters to try (for automatic determination)
            algorithm: Clustering algorithm to use ('kmeans', 'dbscan')
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for clustering analysis. "
                "Install it with `pip install scikit-learn`."
            )
        
        self.n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.algorithm = algorithm.lower()
        self.random_state = random_state
    
    def analyze(
        self,
        embeddings: List[List[float]],
        texts: Optional[List[str]] = None,
        ids: Optional[List[Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze clustering of embeddings.
        
        Args:
            embeddings: List of embeddings to analyze
            texts: List of texts corresponding to embeddings (optional)
            ids: List of IDs corresponding to embeddings (optional)
            **kwargs: Additional arguments for analysis
            
        Returns:
            Dictionary containing clustering analysis results
        """
        if not embeddings:
            return {"error": "No embeddings to analyze"}
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Determine number of clusters if not specified
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(embeddings_array)
        
        # Apply clustering
        if self.algorithm == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            labels = clusterer.fit_predict(embeddings_array)
            centers = clusterer.cluster_centers_
        elif self.algorithm == "dbscan":
            # Estimate eps parameter for DBSCAN
            nn = NearestNeighbors(n_neighbors=min(20, len(embeddings) - 1))
            nn.fit(embeddings_array)
            distances, _ = nn.kneighbors(embeddings_array)
            eps = np.percentile(distances[:, 1:].flatten(), 90)  # 90th percentile of distances
            
            clusterer = DBSCAN(eps=eps, min_samples=5)
            labels = clusterer.fit_predict(embeddings_array)
            
            # For DBSCAN, -1 indicates noise points
            # Convert -1 to a separate cluster number for analysis
            if -1 in labels:
                max_label = max(labels)
                labels = np.array([label if label != -1 else max_label + 1 for label in labels])
            
            # Calculate cluster centers
            centers = []
            for cluster_id in range(max(labels) + 1):
                cluster_points = embeddings_array[labels == cluster_id]
                if len(cluster_points) > 0:
                    centers.append(np.mean(cluster_points, axis=0))
                else:
                    centers.append(np.zeros(embeddings_array.shape[1]))
            centers = np.array(centers)
        else:
            return {"error": f"Unsupported clustering algorithm: {self.algorithm}"}
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_stats(embeddings_array, labels, centers)
        
        # Organize data by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            item = {"index": i}
            
            if texts:
                item["text"] = texts[i]
            
            if ids:
                item["id"] = ids[i]
            
            clusters[int(label)].append(item)
        
        # Convert defaultdict to regular dict for JSON serialization
        clusters_dict = {str(k): v for k, v in clusters.items()}
        
        # Calculate overall clustering quality metrics
        quality_metrics = self._calculate_quality_metrics(embeddings_array, labels)
        
        return {
            "clusters": clusters_dict,
            "cluster_stats": cluster_stats,
            "quality_metrics": quality_metrics,
            "num_clusters": len(clusters),
            "algorithm": self.algorithm
        }
    
    def _find_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """
        Find the optimal number of clusters using silhouette score.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Optimal number of clusters
        """
        # Limit max_clusters to a reasonable value based on data size
        max_clusters = min(self.max_clusters, len(embeddings) // 5)
        max_clusters = max(max_clusters, self.min_clusters)
        
        if len(embeddings) < self.min_clusters * 2:
            return self.min_clusters
        
        best_score = -1
        best_n = self.min_clusters
        
        for n in range(self.min_clusters, max_clusters + 1):
            clusterer = KMeans(n_clusters=n, random_state=self.random_state)
            labels = clusterer.fit_predict(embeddings)
            
            try:
                score = silhouette_score(embeddings, labels)
                
                if score > best_score:
                    best_score = score
                    best_n = n
            except Exception as e:
                logger.warning(f"Error calculating silhouette score for n={n}: {e}")
        
        return best_n
    
    def _calculate_cluster_stats(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each cluster.
        
        Args:
            embeddings: Numpy array of embeddings
            labels: Numpy array of cluster labels
            centers: Numpy array of cluster centers
            
        Returns:
            Dictionary of cluster statistics
        """
        stats = {}
        
        for cluster_id in range(max(labels) + 1):
            cluster_points = embeddings[labels == cluster_id]
            
            if len(cluster_points) == 0:
                continue
            
            # Calculate distances to center
            center = centers[cluster_id]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            
            # Calculate internal similarity
            if len(cluster_points) > 1:
                similarities = cosine_similarity(cluster_points)
                np.fill_diagonal(similarities, 0)  # Ignore self-similarity
                mean_similarity = np.mean(similarities)
            else:
                mean_similarity = 1.0
            
            stats[str(cluster_id)] = {
                "size": int(len(cluster_points)),
                "mean_distance_to_center": float(np.mean(distances)),
                "max_distance_to_center": float(np.max(distances)),
                "mean_internal_similarity": float(mean_similarity)
            }
        
        return stats
    
    def _calculate_quality_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate overall clustering quality metrics.
        
        Args:
            embeddings: Numpy array of embeddings
            labels: Numpy array of cluster labels
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Skip metrics if there's only one cluster
        if len(set(labels)) <= 1:
            return {"note": "Cannot calculate metrics for a single cluster"}
        
        try:
            metrics["silhouette_score"] = float(silhouette_score(embeddings, labels))
        except Exception as e:
            logger.warning(f"Error calculating silhouette score: {e}")
        
        try:
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(embeddings, labels))
        except Exception as e:
            logger.warning(f"Error calculating Calinski-Harabasz score: {e}")
        
        return metrics


class OutlierAnalyzer(BaseEmbeddingAnalyzer):
    """
    Analyze outliers in embeddings.
    """
    
    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        n_neighbors: int = 20,
        random_state: int = 42
    ):
        """
        Initialize an outlier analyzer.
        
        Args:
            method: Method to use for outlier detection
                  ('isolation_forest', 'local_outlier_factor', 'dbscan', 'distance')
            contamination: Expected proportion of outliers (for isolation_forest and LOF)
            n_neighbors: Number of neighbors to consider (for LOF and distance methods)
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for outlier analysis. "
                "Install it with `pip install scikit-learn`."
            )
        
        self.method = method.lower()
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state
    
    def analyze(
        self,
        embeddings: List[List[float]],
        texts: Optional[List[str]] = None,
        ids: Optional[List[Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze outliers in embeddings.
        
        Args:
            embeddings: List of embeddings to analyze
            texts: List of texts corresponding to embeddings (optional)
            ids: List of IDs corresponding to embeddings (optional)
            **kwargs: Additional arguments for analysis
            
        Returns:
            Dictionary containing outlier analysis results
        """
        if not embeddings:
            return {"error": "No embeddings to analyze"}
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Detect outliers
        if self.method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            
            detector = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state
            )
            # -1 for outliers, 1 for inliers
            labels = detector.fit_predict(embeddings_array)
            outlier_indices = np.where(labels == -1)[0]
            
            # Calculate anomaly scores (higher is more anomalous)
            scores = -detector.score_samples(embeddings_array)
        
        elif self.method == "local_outlier_factor":
            from sklearn.neighbors import LocalOutlierFactor
            
            detector = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(embeddings) - 1),
                contamination=self.contamination
            )
            # -1 for outliers, 1 for inliers
            labels = detector.fit_predict(embeddings_array)
            outlier_indices = np.where(labels == -1)[0]
            
            # Calculate anomaly scores (higher is more anomalous)
            scores = -detector.negative_outlier_factor_
        
        elif self.method == "dbscan":
            from sklearn.cluster import DBSCAN
            
            # Estimate eps parameter for DBSCAN
            nn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(embeddings) - 1))
            nn.fit(embeddings_array)
            distances, _ = nn.kneighbors(embeddings_array)
            eps = np.percentile(distances[:, 1:].flatten(), 90)  # 90th percentile of distances
            
            detector = DBSCAN(eps=eps, min_samples=5)
            labels = detector.fit_predict(embeddings_array)
            
            # -1 indicates noise points (outliers)
            outlier_indices = np.where(labels == -1)[0]
            
            # Calculate anomaly scores based on distance to nearest cluster
            scores = np.zeros(len(embeddings))
            
            # For each point, calculate distance to nearest cluster center
            for i, embedding in enumerate(embeddings_array):
                if i in outlier_indices:
                    # For outliers, calculate distance to nearest cluster center
                    min_dist = float('inf')
                    for cluster_id in set(labels) - {-1}:
                        cluster_points = embeddings_array[labels == cluster_id]
                        cluster_center = np.mean(cluster_points, axis=0)
                        dist = np.linalg.norm(embedding - cluster_center)
                        min_dist = min(min_dist, dist)
                    scores[i] = min_dist
                else:
                    # For inliers, use distance to own cluster center
                    cluster_id = labels[i]
                    cluster_points = embeddings_array[labels == cluster_id]
                    cluster_center = np.mean(cluster_points, axis=0)
                    scores[i] = np.linalg.norm(embedding - cluster_center)
        
        elif self.method == "distance":
            # Calculate distance to k nearest neighbors
            nn = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, len(embeddings)))
            nn.fit(embeddings_array)
            distances, _ = nn.kneighbors(embeddings_array)
            
            # Average distance to k nearest neighbors (excluding self)
            scores = np.mean(distances[:, 1:], axis=1)
            
            # Determine outliers based on score threshold
            threshold = np.percentile(scores, 100 * (1 - self.contamination))
            outlier_indices = np.where(scores > threshold)[0]
        
        else:
            return {"error": f"Unsupported outlier detection method: {self.method}"}
        
        # Prepare outlier information
        outliers = []
        for i in outlier_indices:
            outlier_info = {
                "index": int(i),
                "score": float(scores[i])
            }
            
            if texts:
                outlier_info["text"] = texts[i]
            
            if ids:
                outlier_info["id"] = ids[i]
            
            outliers.append(outlier_info)
        
        # Sort outliers by score (descending)
        outliers.sort(key=lambda x: x["score"], reverse=True)
        
        # Calculate statistics
        stats = {
            "num_outliers": len(outliers),
            "outlier_percentage": len(outliers) / len(embeddings) * 100,
            "mean_score": float(np.mean(scores)),
            "median_score": float(np.median(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "std_score": float(np.std(scores))
        }
        
        return {
            "outliers": outliers,
            "statistics": stats,
            "method": self.method,
            "contamination": self.contamination
        }


class DimensionalityAnalyzer(BaseEmbeddingAnalyzer):
    """
    Analyze dimensionality and distribution of embeddings.
    """
    
    def __init__(
        self,
        n_components_to_analyze: int = 10,
        random_state: int = 42
    ):
        """
        Initialize a dimensionality analyzer.
        
        Args:
            n_components_to_analyze: Number of principal components to analyze
            random_state: Random seed for reproducibility
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for dimensionality analysis. "
                "Install it with `pip install scikit-learn`."
            )
        
        self.n_components_to_analyze = n_components_to_analyze
        self.random_state = random_state
    
    def analyze(
        self,
        embeddings: List[List[float]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze dimensionality and distribution of embeddings.
        
        Args:
            embeddings: List of embeddings to analyze
            **kwargs: Additional arguments for analysis
            
        Returns:
            Dictionary containing dimensionality analysis results
        """
        if not embeddings:
            return {"error": "No embeddings to analyze"}
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Get basic statistics
        embedding_dim = embeddings_array.shape[1]
        
        # Calculate mean, std, min, max for each dimension
        dim_stats = {
            "mean": embeddings_array.mean(axis=0).tolist(),
            "std": embeddings_array.std(axis=0).tolist(),
            "min": embeddings_array.min(axis=0).tolist(),
            "max": embeddings_array.max(axis=0).tolist()
        }
        
        # Calculate norms
        norms = np.linalg.norm(embeddings_array, axis=1)
        norm_stats = {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "min": float(np.min(norms)),
            "max": float(np.max(norms)),
            "median": float(np.median(norms))
        }
        
        # Perform PCA to analyze variance explained
        from sklearn.decomposition import PCA
        
        # Limit components to analyze based on embedding dimension
        n_components = min(self.n_components_to_analyze, embedding_dim)
        
        pca = PCA(n_components=n_components, random_state=self.random_state)
        pca.fit(embeddings_array)
        
        # Calculate cumulative explained variance
        explained_variance_ratio = pca.explained_variance_ratio_.tolist()
        cumulative_variance = np.cumsum(explained_variance_ratio).tolist()
        
        # Find number of components needed for different variance thresholds
        variance_thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
        components_needed = {}
        
        for threshold in variance_thresholds:
            components = next((i + 1 for i, v in enumerate(cumulative_variance) if v >= threshold), n_components)
            components_needed[str(threshold)] = components
        
        # Calculate sparsity (percentage of near-zero values)
        epsilon = 1e-6
        sparsity = np.mean(np.abs(embeddings_array) < epsilon) * 100
        
        return {
            "embedding_dimension": embedding_dim,
            "dimension_stats": dim_stats,
            "norm_stats": norm_stats,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance": cumulative_variance,
            "components_needed": components_needed,
            "sparsity_percentage": float(sparsity)
        }


# Factory function to get an analyzer
def get_embedding_analyzer(
    analyzer_type: str = "similarity",
    **kwargs
) -> BaseEmbeddingAnalyzer:
    """
    Get an embedding analyzer based on the specified type.
    
    Args:
        analyzer_type: Type of analyzer to use
                     ('similarity', 'clustering', 'outlier', 'dimensionality')
        **kwargs: Additional arguments for the specific analyzer
        
    Returns:
        Analyzer instance
        
    Raises:
        ValueError: If an unsupported analyzer type is specified
    """
    if analyzer_type == "similarity":
        return SimilarityAnalyzer(**kwargs)
    elif analyzer_type == "clustering":
        return ClusteringAnalyzer(**kwargs)
    elif analyzer_type == "outlier":
        return OutlierAnalyzer(**kwargs)
    elif analyzer_type == "dimensionality":
        return DimensionalityAnalyzer(**kwargs)
    else:
        raise ValueError(
            f"Unsupported analyzer type: {analyzer_type}. "
            "Supported types are: 'similarity', 'clustering', 'outlier', 'dimensionality'."
        ) 