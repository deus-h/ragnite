"""
Dimensionality Reduction

This module provides tools for reducing the dimensionality of embeddings
for visualization, efficiency, and analysis.
"""

import os
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

# Try to import optional dependencies
try:
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class BaseDimensionalityReducer(ABC):
    """
    Abstract base class for dimensionality reducers.
    """
    
    @abstractmethod
    def fit(self, embeddings: List[List[float]]) -> 'BaseDimensionalityReducer':
        """
        Fit the dimensionality reducer to a set of embeddings.
        
        Args:
            embeddings: List of embeddings to fit to
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Transform embeddings to a lower-dimensional space.
        
        Args:
            embeddings: List of embeddings to transform
            
        Returns:
            List of transformed embeddings
        """
        pass
    
    def fit_transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Fit the dimensionality reducer and transform embeddings.
        
        Args:
            embeddings: List of embeddings to fit and transform
            
        Returns:
            List of transformed embeddings
        """
        self.fit(embeddings)
        return self.transform(embeddings)
    
    def save(self, path: str) -> None:
        """
        Save the dimensionality reducer to a file.
        
        Args:
            path: Path to save the reducer to
        """
        raise NotImplementedError("This reducer does not support saving.")
    
    @classmethod
    def load(cls, path: str) -> 'BaseDimensionalityReducer':
        """
        Load a dimensionality reducer from a file.
        
        Args:
            path: Path to load the reducer from
            
        Returns:
            Loaded dimensionality reducer
        """
        raise NotImplementedError("This reducer does not support loading.")


class PCAReducer(BaseDimensionalityReducer):
    """
    Reduce dimensionality using Principal Component Analysis (PCA).
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: int = 42,
        whiten: bool = False
    ):
        """
        Initialize a PCA dimensionality reducer.
        
        Args:
            n_components: Number of components to reduce to
            random_state: Random seed for reproducibility
            whiten: Whether to whiten the data
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for PCA. "
                "Install it with `pip install scikit-learn`."
            )
        
        self.n_components = n_components
        self.random_state = random_state
        self.whiten = whiten
        self.pca = None
    
    def fit(self, embeddings: List[List[float]]) -> 'PCAReducer':
        """
        Fit the PCA reducer to a set of embeddings.
        
        Args:
            embeddings: List of embeddings to fit to
            
        Returns:
            Self
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fitting.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Create and fit PCA
        self.pca = PCA(
            n_components=min(self.n_components, embeddings_array.shape[1], len(embeddings) - 1),
            random_state=self.random_state,
            whiten=self.whiten
        )
        self.pca.fit(embeddings_array)
        
        # Log explained variance
        explained_variance = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA explained variance: {explained_variance:.4f}")
        
        return self
    
    def transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Transform embeddings using PCA.
        
        Args:
            embeddings: List of embeddings to transform
            
        Returns:
            List of transformed embeddings
        """
        if not embeddings:
            return []
        
        if self.pca is None:
            raise ValueError("PCA has not been fitted. Call fit() first.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Transform embeddings
        transformed = self.pca.transform(embeddings_array)
        
        return transformed.tolist()
    
    def save(self, path: str) -> None:
        """
        Save the PCA reducer to a file.
        
        Args:
            path: Path to save the reducer to
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted. Nothing to save.")
        
        import joblib
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save the PCA model
        joblib.dump(self.pca, path)
        logger.info(f"Saved PCA reducer to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PCAReducer':
        """
        Load a PCA reducer from a file.
        
        Args:
            path: Path to load the reducer from
            
        Returns:
            Loaded PCA reducer
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"PCA reducer file not found: {path}")
        
        import joblib
        
        # Load the PCA model
        pca = joblib.load(path)
        
        # Create a new reducer
        reducer = cls(
            n_components=pca.n_components,
            random_state=pca.random_state,
            whiten=pca.whiten
        )
        reducer.pca = pca
        
        logger.info(f"Loaded PCA reducer from {path}")
        return reducer


class SVDReducer(BaseDimensionalityReducer):
    """
    Reduce dimensionality using Truncated Singular Value Decomposition (SVD).
    """
    
    def __init__(
        self,
        n_components: int = 2,
        random_state: int = 42,
        n_iter: int = 5
    ):
        """
        Initialize an SVD dimensionality reducer.
        
        Args:
            n_components: Number of components to reduce to
            random_state: Random seed for reproducibility
            n_iter: Number of iterations for randomized SVD
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for SVD. "
                "Install it with `pip install scikit-learn`."
            )
        
        self.n_components = n_components
        self.random_state = random_state
        self.n_iter = n_iter
        self.svd = None
    
    def fit(self, embeddings: List[List[float]]) -> 'SVDReducer':
        """
        Fit the SVD reducer to a set of embeddings.
        
        Args:
            embeddings: List of embeddings to fit to
            
        Returns:
            Self
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fitting.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Create and fit SVD
        self.svd = TruncatedSVD(
            n_components=min(self.n_components, embeddings_array.shape[1], len(embeddings) - 1),
            random_state=self.random_state,
            n_iter=self.n_iter
        )
        self.svd.fit(embeddings_array)
        
        # Log explained variance
        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info(f"SVD explained variance: {explained_variance:.4f}")
        
        return self
    
    def transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Transform embeddings using SVD.
        
        Args:
            embeddings: List of embeddings to transform
            
        Returns:
            List of transformed embeddings
        """
        if not embeddings:
            return []
        
        if self.svd is None:
            raise ValueError("SVD has not been fitted. Call fit() first.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Transform embeddings
        transformed = self.svd.transform(embeddings_array)
        
        return transformed.tolist()
    
    def save(self, path: str) -> None:
        """
        Save the SVD reducer to a file.
        
        Args:
            path: Path to save the reducer to
        """
        if self.svd is None:
            raise ValueError("SVD has not been fitted. Nothing to save.")
        
        import joblib
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save the SVD model
        joblib.dump(self.svd, path)
        logger.info(f"Saved SVD reducer to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SVDReducer':
        """
        Load an SVD reducer from a file.
        
        Args:
            path: Path to load the reducer from
            
        Returns:
            Loaded SVD reducer
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"SVD reducer file not found: {path}")
        
        import joblib
        
        # Load the SVD model
        svd = joblib.load(path)
        
        # Create a new reducer
        reducer = cls(
            n_components=svd.n_components,
            random_state=svd.random_state,
            n_iter=svd.n_iter
        )
        reducer.svd = svd
        
        logger.info(f"Loaded SVD reducer from {path}")
        return reducer


class TSNEReducer(BaseDimensionalityReducer):
    """
    Reduce dimensionality using t-Distributed Stochastic Neighbor Embedding (t-SNE).
    """
    
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int = 42,
        init: str = "pca"
    ):
        """
        Initialize a t-SNE dimensionality reducer.
        
        Args:
            n_components: Number of components to reduce to
            perplexity: Perplexity parameter for t-SNE
            learning_rate: Learning rate for t-SNE
            n_iter: Number of iterations for t-SNE
            random_state: Random seed for reproducibility
            init: Initialization method ('pca', 'random')
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for t-SNE. "
                "Install it with `pip install scikit-learn`."
            )
        
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.init = init
        self.tsne = None
        self.embedding_ = None
    
    def fit(self, embeddings: List[List[float]]) -> 'TSNEReducer':
        """
        Fit the t-SNE reducer to a set of embeddings.
        
        Args:
            embeddings: List of embeddings to fit to
            
        Returns:
            Self
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fitting.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Create and fit t-SNE
        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=min(self.perplexity, len(embeddings) - 1),
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init=self.init
        )
        self.embedding_ = self.tsne.fit_transform(embeddings_array)
        
        logger.info(f"t-SNE fitted with {self.n_components} components")
        
        return self
    
    def transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Transform embeddings using t-SNE.
        Note: t-SNE does not support transform on new data.
        This will return the fitted embeddings if the input matches the fitted data,
        otherwise it will raise an error.
        
        Args:
            embeddings: List of embeddings to transform
            
        Returns:
            List of transformed embeddings
        """
        if not embeddings:
            return []
        
        if self.embedding_ is None:
            raise ValueError("t-SNE has not been fitted. Call fit() first.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Check if the input is the same as the fitted data
        if len(embeddings) != len(self.embedding_):
            raise ValueError(
                "t-SNE does not support transform on new data. "
                "The number of samples in the input does not match the number of samples in the fitted data."
            )
        
        # Return the fitted embedding
        return self.embedding_.tolist()
    
    def fit_transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Fit the t-SNE reducer and transform embeddings.
        
        Args:
            embeddings: List of embeddings to fit and transform
            
        Returns:
            List of transformed embeddings
        """
        self.fit(embeddings)
        return self.embedding_.tolist()
    
    def save(self, path: str) -> None:
        """
        Save the t-SNE reducer to a file.
        
        Args:
            path: Path to save the reducer to
        """
        if self.embedding_ is None:
            raise ValueError("t-SNE has not been fitted. Nothing to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save the t-SNE embedding and parameters
        data = {
            "embedding": self.embedding_.tolist(),
            "n_components": self.n_components,
            "perplexity": self.perplexity,
            "learning_rate": self.learning_rate,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "init": self.init
        }
        
        with open(path, "w") as f:
            json.dump(data, f)
        
        logger.info(f"Saved t-SNE reducer to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TSNEReducer':
        """
        Load a t-SNE reducer from a file.
        
        Args:
            path: Path to load the reducer from
            
        Returns:
            Loaded t-SNE reducer
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"t-SNE reducer file not found: {path}")
        
        # Load the t-SNE embedding and parameters
        with open(path, "r") as f:
            data = json.load(f)
        
        # Create a new reducer
        reducer = cls(
            n_components=data["n_components"],
            perplexity=data["perplexity"],
            learning_rate=data["learning_rate"],
            n_iter=data["n_iter"],
            random_state=data["random_state"],
            init=data["init"]
        )
        reducer.embedding_ = np.array(data["embedding"])
        
        logger.info(f"Loaded t-SNE reducer from {path}")
        return reducer


class UMAPReducer(BaseDimensionalityReducer):
    """
    Reduce dimensionality using Uniform Manifold Approximation and Projection (UMAP).
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42
    ):
        """
        Initialize a UMAP dimensionality reducer.
        
        Args:
            n_components: Number of components to reduce to
            n_neighbors: Number of neighbors to consider
            min_dist: Minimum distance between points in the embedding
            metric: Distance metric to use
            random_state: Random seed for reproducibility
        """
        if not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP is required for UMAP dimensionality reduction. "
                "Install it with `pip install umap-learn`."
            )
        
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.umap_model = None
    
    def fit(self, embeddings: List[List[float]]) -> 'UMAPReducer':
        """
        Fit the UMAP reducer to a set of embeddings.
        
        Args:
            embeddings: List of embeddings to fit to
            
        Returns:
            Self
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fitting.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Create and fit UMAP
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=min(self.n_neighbors, len(embeddings) - 1),
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )
        self.umap_model.fit(embeddings_array)
        
        logger.info(f"UMAP fitted with {self.n_components} components")
        
        return self
    
    def transform(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Transform embeddings using UMAP.
        
        Args:
            embeddings: List of embeddings to transform
            
        Returns:
            List of transformed embeddings
        """
        if not embeddings:
            return []
        
        if self.umap_model is None:
            raise ValueError("UMAP has not been fitted. Call fit() first.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Transform embeddings
        transformed = self.umap_model.transform(embeddings_array)
        
        return transformed.tolist()
    
    def save(self, path: str) -> None:
        """
        Save the UMAP reducer to a file.
        
        Args:
            path: Path to save the reducer to
        """
        if self.umap_model is None:
            raise ValueError("UMAP has not been fitted. Nothing to save.")
        
        import joblib
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save the UMAP model
        joblib.dump(self.umap_model, path)
        logger.info(f"Saved UMAP reducer to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'UMAPReducer':
        """
        Load a UMAP reducer from a file.
        
        Args:
            path: Path to load the reducer from
            
        Returns:
            Loaded UMAP reducer
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"UMAP reducer file not found: {path}")
        
        import joblib
        
        # Load the UMAP model
        umap_model = joblib.load(path)
        
        # Create a new reducer
        reducer = cls(
            n_components=umap_model.n_components,
            n_neighbors=umap_model.n_neighbors,
            min_dist=umap_model.min_dist,
            metric=umap_model.metric,
            random_state=umap_model.random_state
        )
        reducer.umap_model = umap_model
        
        logger.info(f"Loaded UMAP reducer from {path}")
        return reducer


# Factory function to get a dimensionality reducer
def get_dimensionality_reducer(
    reducer_type: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs
) -> BaseDimensionalityReducer:
    """
    Get a dimensionality reducer based on the specified type.
    
    Args:
        reducer_type: Type of reducer to use ('pca', 'svd', 'tsne', 'umap')
        n_components: Number of components to reduce to
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments for the specific reducer
        
    Returns:
        Dimensionality reducer instance
        
    Raises:
        ValueError: If an unsupported reducer type is specified
    """
    if reducer_type == "pca":
        return PCAReducer(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
    elif reducer_type == "svd":
        return SVDReducer(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
    elif reducer_type == "tsne":
        return TSNEReducer(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
    elif reducer_type == "umap":
        return UMAPReducer(
            n_components=n_components,
            random_state=random_state,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unsupported dimensionality reducer type: {reducer_type}. "
            "Supported types are: 'pca', 'svd', 'tsne', 'umap'."
        ) 