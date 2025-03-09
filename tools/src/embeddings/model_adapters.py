"""
Model Adapters

This module provides tools for adapting between different embedding models,
allowing conversion and compatibility between embeddings from different sources.
"""

import os
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple

# Try to import optional dependencies
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    """
    
    @abstractmethod
    def adapt(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Adapt embeddings from one model to be compatible with another.
        
        Args:
            embeddings: List of embeddings to adapt
            
        Returns:
            List of adapted embeddings
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the adapter to a file.
        
        Args:
            path: Path to save the adapter to
        """
        raise NotImplementedError("This adapter does not support saving.")
    
    @classmethod
    def load(cls, path: str) -> 'BaseModelAdapter':
        """
        Load an adapter from a file.
        
        Args:
            path: Path to load the adapter from
            
        Returns:
            Loaded adapter
        """
        raise NotImplementedError("This adapter does not support loading.")


class DimensionalityAdapter(BaseModelAdapter):
    """
    Adapt embeddings by changing their dimensionality.
    """
    
    def __init__(
        self,
        target_dim: int = 768,
        method: str = "pca",
        random_state: int = 42
    ):
        """
        Initialize a dimensionality adapter.
        
        Args:
            target_dim: Target dimensionality
            method: Method to use for dimensionality change ('pca', 'truncate', 'pad')
            random_state: Random seed for reproducibility
        """
        if method == "pca" and not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for PCA-based dimensionality adaptation. "
                "Install it with `pip install scikit-learn`."
            )
        
        self.target_dim = target_dim
        self.method = method.lower()
        self.random_state = random_state
        self.transformer = None
    
    def fit(self, embeddings: List[List[float]]) -> 'DimensionalityAdapter':
        """
        Fit the adapter to a set of embeddings.
        
        Args:
            embeddings: List of embeddings to fit to
            
        Returns:
            Self
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fitting.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        source_dim = embeddings_array.shape[1]
        
        # If source and target dimensions are the same, no need to fit
        if source_dim == self.target_dim:
            logger.info("Source and target dimensions are the same. No adaptation needed.")
            return self
        
        # Fit transformer based on method
        if self.method == "pca":
            # Only fit PCA if reducing dimensionality
            if source_dim > self.target_dim:
                self.transformer = PCA(
                    n_components=self.target_dim,
                    random_state=self.random_state
                )
                self.transformer.fit(embeddings_array)
                logger.info(f"Fitted PCA transformer from {source_dim} to {self.target_dim} dimensions.")
        
        # For truncate and pad methods, no fitting is needed
        
        return self
    
    def adapt(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Adapt embeddings to the target dimensionality.
        
        Args:
            embeddings: List of embeddings to adapt
            
        Returns:
            List of adapted embeddings
        """
        if not embeddings:
            return []
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        source_dim = embeddings_array.shape[1]
        
        # If source and target dimensions are the same, return as is
        if source_dim == self.target_dim:
            return embeddings
        
        # Adapt based on method
        if self.method == "pca":
            if source_dim > self.target_dim:
                # Reduce dimensionality using PCA
                if self.transformer is None:
                    # Fit on the fly if not already fitted
                    self.fit(embeddings)
                
                adapted = self.transformer.transform(embeddings_array)
            else:
                # Increase dimensionality using PCA
                # This is not ideal, but we'll do our best
                logger.warning(
                    f"Increasing dimensionality from {source_dim} to {self.target_dim} "
                    f"using PCA is not recommended. Consider using 'pad' method instead."
                )
                
                # Fit PCA on the fly
                pca = PCA(n_components=min(source_dim, len(embeddings) - 1), random_state=self.random_state)
                pca_result = pca.fit_transform(embeddings_array)
                
                # Pad with zeros
                adapted = np.zeros((len(embeddings), self.target_dim))
                adapted[:, :pca_result.shape[1]] = pca_result
        
        elif self.method == "truncate":
            if source_dim > self.target_dim:
                # Simply truncate to the target dimension
                adapted = embeddings_array[:, :self.target_dim]
            else:
                # Pad with zeros
                adapted = np.zeros((len(embeddings), self.target_dim))
                adapted[:, :source_dim] = embeddings_array
        
        elif self.method == "pad":
            # Pad with zeros or truncate as needed
            adapted = np.zeros((len(embeddings), self.target_dim))
            min_dim = min(source_dim, self.target_dim)
            adapted[:, :min_dim] = embeddings_array[:, :min_dim]
        
        else:
            raise ValueError(
                f"Unsupported dimensionality adaptation method: {self.method}. "
                "Supported methods are: 'pca', 'truncate', 'pad'."
            )
        
        return adapted.tolist()
    
    def save(self, path: str) -> None:
        """
        Save the adapter to a file.
        
        Args:
            path: Path to save the adapter to
        """
        if self.method == "pca" and self.transformer is not None:
            import joblib
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save the PCA transformer
            joblib.dump(self.transformer, path)
            logger.info(f"Saved PCA transformer to {path}")
        else:
            # For other methods, save the configuration
            config = {
                "target_dim": self.target_dim,
                "method": self.method,
                "random_state": self.random_state
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save the configuration
            with open(path, "w") as f:
                json.dump(config, f)
            
            logger.info(f"Saved adapter configuration to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DimensionalityAdapter':
        """
        Load an adapter from a file.
        
        Args:
            path: Path to load the adapter from
            
        Returns:
            Loaded adapter
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Adapter file not found: {path}")
        
        try:
            # Try to load as a PCA transformer
            import joblib
            transformer = joblib.load(path)
            
            adapter = cls(
                target_dim=transformer.n_components,
                method="pca",
                random_state=transformer.random_state
            )
            adapter.transformer = transformer
            
            logger.info(f"Loaded PCA transformer from {path}")
            return adapter
        except Exception:
            # Try to load as a configuration file
            try:
                with open(path, "r") as f:
                    config = json.load(f)
                
                adapter = cls(
                    target_dim=config.get("target_dim", 768),
                    method=config.get("method", "pad"),
                    random_state=config.get("random_state", 42)
                )
                
                logger.info(f"Loaded adapter configuration from {path}")
                return adapter
            except Exception as e:
                raise ValueError(f"Failed to load adapter from {path}: {e}")


class NormalizationAdapter(BaseModelAdapter):
    """
    Adapt embeddings by normalizing them.
    """
    
    def __init__(
        self,
        normalization: str = "l2",
        scale: bool = False
    ):
        """
        Initialize a normalization adapter.
        
        Args:
            normalization: Normalization method ('l2', 'l1', 'max', 'none')
            scale: Whether to scale the embeddings to have zero mean and unit variance
        """
        if scale and not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for scaling. "
                "Install it with `pip install scikit-learn`."
            )
        
        self.normalization = normalization.lower()
        self.scale = scale
        self.scaler = None
    
    def fit(self, embeddings: List[List[float]]) -> 'NormalizationAdapter':
        """
        Fit the adapter to a set of embeddings.
        
        Args:
            embeddings: List of embeddings to fit to
            
        Returns:
            Self
        """
        if not embeddings:
            raise ValueError("No embeddings provided for fitting.")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Fit scaler if scaling is enabled
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(embeddings_array)
            logger.info("Fitted standard scaler.")
        
        return self
    
    def adapt(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Adapt embeddings by normalizing them.
        
        Args:
            embeddings: List of embeddings to adapt
            
        Returns:
            List of adapted embeddings
        """
        if not embeddings:
            return []
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Apply scaling if enabled
        if self.scale:
            if self.scaler is None:
                # Fit on the fly if not already fitted
                self.fit(embeddings)
            
            embeddings_array = self.scaler.transform(embeddings_array)
        
        # Apply normalization
        if self.normalization == "l2":
            # L2 normalization (unit vector)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.maximum(norms, 1e-12)
            normalized = embeddings_array / norms
        
        elif self.normalization == "l1":
            # L1 normalization
            norms = np.sum(np.abs(embeddings_array), axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.maximum(norms, 1e-12)
            normalized = embeddings_array / norms
        
        elif self.normalization == "max":
            # Max normalization
            max_vals = np.max(np.abs(embeddings_array), axis=1, keepdims=True)
            # Avoid division by zero
            max_vals = np.maximum(max_vals, 1e-12)
            normalized = embeddings_array / max_vals
        
        elif self.normalization == "none":
            # No normalization
            normalized = embeddings_array
        
        else:
            raise ValueError(
                f"Unsupported normalization method: {self.normalization}. "
                "Supported methods are: 'l2', 'l1', 'max', 'none'."
            )
        
        return normalized.tolist()
    
    def save(self, path: str) -> None:
        """
        Save the adapter to a file.
        
        Args:
            path: Path to save the adapter to
        """
        if self.scale and self.scaler is not None:
            import joblib
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save the scaler and configuration
            data = {
                "scaler": self.scaler,
                "normalization": self.normalization,
                "scale": self.scale
            }
            
            joblib.dump(data, path)
            logger.info(f"Saved normalization adapter to {path}")
        else:
            # Save just the configuration
            config = {
                "normalization": self.normalization,
                "scale": self.scale
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            # Save the configuration
            with open(path, "w") as f:
                json.dump(config, f)
            
            logger.info(f"Saved adapter configuration to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NormalizationAdapter':
        """
        Load an adapter from a file.
        
        Args:
            path: Path to load the adapter from
            
        Returns:
            Loaded adapter
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Adapter file not found: {path}")
        
        try:
            # Try to load as a joblib file with scaler
            import joblib
            data = joblib.load(path)
            
            adapter = cls(
                normalization=data.get("normalization", "l2"),
                scale=data.get("scale", False)
            )
            adapter.scaler = data.get("scaler")
            
            logger.info(f"Loaded normalization adapter from {path}")
            return adapter
        except Exception:
            # Try to load as a configuration file
            try:
                with open(path, "r") as f:
                    config = json.load(f)
                
                adapter = cls(
                    normalization=config.get("normalization", "l2"),
                    scale=config.get("scale", False)
                )
                
                logger.info(f"Loaded adapter configuration from {path}")
                return adapter
            except Exception as e:
                raise ValueError(f"Failed to load adapter from {path}: {e}")


class LinearAdapter(BaseModelAdapter):
    """
    Adapt embeddings using a linear transformation.
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        random_state: int = 42
    ):
        """
        Initialize a linear adapter.
        
        Args:
            source_dim: Source dimensionality
            target_dim: Target dimensionality
            random_state: Random seed for reproducibility
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for linear adaptation. "
                "Install it with `pip install torch`."
            )
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.random_state = random_state
        
        # Set random seed
        torch.manual_seed(random_state)
        
        # Initialize linear transformation
        self.linear = nn.Linear(source_dim, target_dim, bias=False)
        
        # Initialize with identity-like matrix if possible
        if source_dim <= target_dim:
            # Initialize with identity for the common dimensions
            weight = torch.zeros(target_dim, source_dim)
            for i in range(source_dim):
                weight[i, i] = 1.0
            self.linear.weight.data = weight
        else:
            # Initialize with PCA-like projection
            # This is just a random orthogonal matrix for now
            nn.init.orthogonal_(self.linear.weight)
    
    def fit(
        self,
        source_embeddings: List[List[float]],
        target_embeddings: List[List[float]],
        learning_rate: float = 0.01,
        num_epochs: int = 100,
        batch_size: int = 32
    ) -> 'LinearAdapter':
        """
        Fit the adapter to map source embeddings to target embeddings.
        
        Args:
            source_embeddings: List of source embeddings
            target_embeddings: List of target embeddings (must have the same length as source_embeddings)
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Self
        """
        if len(source_embeddings) != len(target_embeddings):
            raise ValueError(
                f"Source and target embeddings must have the same length. "
                f"Got {len(source_embeddings)} and {len(target_embeddings)}."
            )
        
        if not source_embeddings:
            raise ValueError("No embeddings provided for fitting.")
        
        # Convert embeddings to PyTorch tensors
        source_tensor = torch.tensor(source_embeddings, dtype=torch.float32)
        target_tensor = torch.tensor(target_embeddings, dtype=torch.float32)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(source_tensor, target_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Set up optimizer
        optimizer = torch.optim.Adam(self.linear.parameters(), lr=learning_rate)
        
        # Train the linear transformation
        self.linear.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for source_batch, target_batch in dataloader:
                # Forward pass
                output = self.linear(source_batch)
                
                # Compute loss (mean squared error)
                loss = torch.nn.functional.mse_loss(output, target_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.6f}")
        
        # Set to evaluation mode
        self.linear.eval()
        
        return self
    
    def adapt(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Adapt embeddings using the learned linear transformation.
        
        Args:
            embeddings: List of embeddings to adapt
            
        Returns:
            List of adapted embeddings
        """
        if not embeddings:
            return []
        
        # Convert embeddings to PyTorch tensor
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        
        # Apply linear transformation
        with torch.no_grad():
            adapted_tensor = self.linear(embeddings_tensor)
        
        # Convert back to list
        return adapted_tensor.tolist()
    
    def save(self, path: str) -> None:
        """
        Save the adapter to a file.
        
        Args:
            path: Path to save the adapter to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save the model state and configuration
        state = {
            "linear_state_dict": self.linear.state_dict(),
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "random_state": self.random_state
        }
        
        torch.save(state, path)
        logger.info(f"Saved linear adapter to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'LinearAdapter':
        """
        Load an adapter from a file.
        
        Args:
            path: Path to load the adapter from
            
        Returns:
            Loaded adapter
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Adapter file not found: {path}")
        
        # Load the model state and configuration
        state = torch.load(path)
        
        # Create a new adapter
        adapter = cls(
            source_dim=state["source_dim"],
            target_dim=state["target_dim"],
            random_state=state["random_state"]
        )
        
        # Load the model state
        adapter.linear.load_state_dict(state["linear_state_dict"])
        adapter.linear.eval()
        
        logger.info(f"Loaded linear adapter from {path}")
        return adapter


class CompositeAdapter(BaseModelAdapter):
    """
    Combine multiple adapters into a single adapter.
    """
    
    def __init__(self, adapters: List[BaseModelAdapter]):
        """
        Initialize a composite adapter.
        
        Args:
            adapters: List of adapters to apply in sequence
        """
        self.adapters = adapters
    
    def adapt(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Adapt embeddings using all adapters in sequence.
        
        Args:
            embeddings: List of embeddings to adapt
            
        Returns:
            List of adapted embeddings
        """
        if not embeddings:
            return []
        
        # Apply adapters in sequence
        adapted = embeddings
        for adapter in self.adapters:
            adapted = adapter.adapt(adapted)
        
        return adapted
    
    def save(self, path: str) -> None:
        """
        Save the adapter to a directory.
        
        Args:
            path: Directory to save the adapter to
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save each adapter
        for i, adapter in enumerate(self.adapters):
            adapter_path = os.path.join(path, f"adapter_{i}.pt")
            adapter.save(adapter_path)
        
        # Save the configuration
        config = {
            "num_adapters": len(self.adapters),
            "adapter_paths": [f"adapter_{i}.pt" for i in range(len(self.adapters))]
        }
        
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        logger.info(f"Saved composite adapter to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CompositeAdapter':
        """
        Load an adapter from a directory.
        
        Args:
            path: Directory to load the adapter from
            
        Returns:
            Loaded adapter
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Adapter directory not found: {path}")
        
        # Load the configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Load each adapter
        adapters = []
        for adapter_path in config["adapter_paths"]:
            full_path = os.path.join(path, adapter_path)
            
            # Try to determine the adapter type
            try:
                # Try loading as a linear adapter
                adapter = LinearAdapter.load(full_path)
            except Exception:
                try:
                    # Try loading as a dimensionality adapter
                    adapter = DimensionalityAdapter.load(full_path)
                except Exception:
                    try:
                        # Try loading as a normalization adapter
                        adapter = NormalizationAdapter.load(full_path)
                    except Exception as e:
                        raise ValueError(f"Failed to load adapter from {full_path}: {e}")
            
            adapters.append(adapter)
        
        logger.info(f"Loaded composite adapter from {path}")
        return cls(adapters)


# Factory function to get a model adapter
def get_model_adapter(
    adapter_type: str = "dimensionality",
    **kwargs
) -> BaseModelAdapter:
    """
    Get a model adapter based on the specified type.
    
    Args:
        adapter_type: Type of adapter to use
                    ('dimensionality', 'normalization', 'linear', 'composite')
        **kwargs: Additional arguments for the specific adapter
        
    Returns:
        Adapter instance
        
    Raises:
        ValueError: If an unsupported adapter type is specified
    """
    if adapter_type == "dimensionality":
        return DimensionalityAdapter(**kwargs)
    elif adapter_type == "normalization":
        return NormalizationAdapter(**kwargs)
    elif adapter_type == "linear":
        return LinearAdapter(**kwargs)
    elif adapter_type == "composite":
        # For composite adapter, 'adapters' must be provided
        if "adapters" not in kwargs:
            raise ValueError("'adapters' must be provided for composite adapter.")
        return CompositeAdapter(adapters=kwargs["adapters"])
    else:
        raise ValueError(
            f"Unsupported adapter type: {adapter_type}. "
            "Supported types are: 'dimensionality', 'normalization', 'linear', 'composite'."
        ) 