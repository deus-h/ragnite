"""
Embedding Generators

This module provides tools for generating embeddings from text using
various embedding models, including local and remote models.
"""

import os
import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class BaseEmbeddingGenerator(ABC):
    """
    Abstract base class for embedding generators.
    """
    
    @abstractmethod
    def generate(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            List of embeddings
        """
        pass
    
    def generate_single(self, text: str, **kwargs) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to generate an embedding for
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            Embedding
        """
        embeddings = self.generate([text], **kwargs)
        return embeddings[0] if embeddings else []
    
    def batch_generate(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in batches.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Size of each batch
            show_progress: Whether to show a progress bar
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Process in batches
        all_embeddings = []
        
        # Set up progress tracking if requested
        if show_progress:
            try:
                from tqdm import tqdm
                batches = list(range(0, len(texts), batch_size))
                pbar = tqdm(total=len(batches))
            except ImportError:
                logger.warning(
                    "tqdm is required for progress tracking. "
                    "Install it with `pip install tqdm`."
                )
                show_progress = False
        
        # Process each batch
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.generate(batch_texts, **kwargs)
            all_embeddings.extend(batch_embeddings)
            
            # Update progress if tracking
            if show_progress:
                pbar.update(1)
        
        # Close progress bar if used
        if show_progress:
            pbar.close()
        
        return all_embeddings
    
    def save_embeddings(
        self,
        embeddings: List[List[float]],
        path: str,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Save embeddings to a file.
        
        Args:
            embeddings: List of embeddings to save
            path: Path to save the embeddings to
            metadata: Optional list of metadata for each embedding
        """
        if not embeddings:
            logger.warning("No embeddings to save.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Prepare data to save
        data = {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings)
        }
        
        # Add metadata if provided
        if metadata:
            if len(metadata) != len(embeddings):
                logger.warning(
                    f"Metadata length ({len(metadata)}) does not match "
                    f"embeddings length ({len(embeddings)}). Metadata will not be saved."
                )
            else:
                data["metadata"] = metadata
        
        # Save to file
        with open(path, "w") as f:
            json.dump(data, f)
        
        logger.info(f"Saved {len(embeddings)} embeddings to {path}")
    
    @staticmethod
    def load_embeddings(
        path: str
    ) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """
        Load embeddings from a file.
        
        Args:
            path: Path to load the embeddings from
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        
        # Load from file
        with open(path, "r") as f:
            data = json.load(f)
        
        embeddings = data.get("embeddings", [])
        metadata = data.get("metadata", None)
        
        logger.info(f"Loaded {len(embeddings)} embeddings from {path}")
        
        return embeddings, metadata


class SentenceTransformerGenerator(BaseEmbeddingGenerator):
    """
    Generate embeddings using Sentence Transformers.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize a Sentence Transformer embedding generator.
        
        Args:
            model_name: Name of the Sentence Transformer model
            device: Device to use for inference ('cpu', 'cuda', 'cuda:0', etc.)
            normalize_embeddings: Whether to normalize embeddings
            cache_dir: Directory to cache models
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerGenerator. "
                "Install it with `pip install sentence-transformers`."
            )
        
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.cache_dir = cache_dir
        
        # Load model
        try:
            self.model = SentenceTransformer(
                model_name_or_path=model_name,
                device=device,
                cache_folder=cache_dir
            )
            logger.info(f"Loaded Sentence Transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer model: {e}")
            raise
    
    def generate(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            batch_size: Size of each batch
            show_progress_bar: Whether to show a progress bar
            convert_to_numpy: Whether to convert embeddings to numpy arrays
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.model.encode(
            sentences=texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings,
            **kwargs
        )
        
        # Convert to list of lists
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        
        return embeddings


class HuggingFaceGenerator(BaseEmbeddingGenerator):
    """
    Generate embeddings using Hugging Face Transformers.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        pooling_strategy: str = "mean",
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize a Hugging Face embedding generator.
        
        Args:
            model_name: Name of the Hugging Face model
            tokenizer_name: Name of the tokenizer (defaults to model_name)
            device: Device to use for inference ('cpu', 'cuda', 'cuda:0', etc.)
            max_length: Maximum sequence length
            pooling_strategy: Strategy for pooling token embeddings ('mean', 'cls', 'max')
            normalize_embeddings: Whether to normalize embeddings
            cache_dir: Directory to cache models
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for HuggingFaceGenerator. "
                "Install it with `pip install transformers`."
            )
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for HuggingFaceGenerator. "
                "Install it with `pip install torch`."
            )
        
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy.lower()
        self.normalize_embeddings = normalize_embeddings
        self.cache_dir = cache_dir
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded Hugging Face model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {e}")
            raise
    
    def generate(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            # Use [CLS] token embedding
            embeddings = model_output.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            # Mean pooling
            attention_mask = encoded_input["attention_mask"]
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            # Max pooling
            attention_mask = encoded_input["attention_mask"]
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            embeddings = torch.max(token_embeddings, 1)[0]
        else:
            raise ValueError(
                f"Unsupported pooling strategy: {self.pooling_strategy}. "
                "Supported strategies are: 'mean', 'cls', 'max'."
            )
        
        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to list of lists
        embeddings = embeddings.cpu().numpy().tolist()
        
        return embeddings


class OpenAIGenerator(BaseEmbeddingGenerator):
    """
    Generate embeddings using OpenAI API.
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        dimensions: Optional[int] = None,
        normalize_embeddings: bool = True
    ):
        """
        Initialize an OpenAI embedding generator.
        
        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            api_base: OpenAI API base URL
            dimensions: Number of dimensions for the embeddings (model-specific)
            normalize_embeddings: Whether to normalize embeddings
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required for OpenAIGenerator. "
                "Install it with `pip install requests`."
            )
        
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base or "https://api.openai.com/v1"
        self.dimensions = dimensions
        self.normalize_embeddings = normalize_embeddings
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set it in the constructor or as the OPENAI_API_KEY environment variable."
            )
    
    def generate(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "input": texts,
            "model": self.model_name
        }
        
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        
        # Send request
        try:
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract embeddings
            embeddings = [item["embedding"] for item in result["data"]]
            
            # Normalize if requested
            if self.normalize_embeddings:
                embeddings = [
                    self._normalize_embedding(embedding)
                    for embedding in embeddings
                ]
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI API: {e}")
            raise
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize an embedding to unit length.
        
        Args:
            embedding: Embedding to normalize
            
        Returns:
            Normalized embedding
        """
        norm = np.sqrt(sum(x * x for x in embedding))
        return [x / norm for x in embedding]


class TensorFlowGenerator(BaseEmbeddingGenerator):
    """
    Generate embeddings using TensorFlow Hub models.
    """
    
    def __init__(
        self,
        model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4",
        normalize_embeddings: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize a TensorFlow Hub embedding generator.
        
        Args:
            model_url: URL of the TensorFlow Hub model
            normalize_embeddings: Whether to normalize embeddings
            cache_dir: Directory to cache models
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "tensorflow is required for TensorFlowGenerator. "
                "Install it with `pip install tensorflow tensorflow-hub`."
            )
        
        try:
            import tensorflow_hub as hub
            self.hub = hub
        except ImportError:
            raise ImportError(
                "tensorflow-hub is required for TensorFlowGenerator. "
                "Install it with `pip install tensorflow-hub`."
            )
        
        self.model_url = model_url
        self.normalize_embeddings = normalize_embeddings
        
        # Set cache directory if provided
        if cache_dir:
            os.environ["TFHUB_CACHE_DIR"] = cache_dir
        
        # Load model
        try:
            self.model = self.hub.load(model_url)
            logger.info(f"Loaded TensorFlow Hub model: {model_url}")
        except Exception as e:
            logger.error(f"Error loading TensorFlow Hub model: {e}")
            raise
    
    def generate(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.model(texts).numpy()
        
        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = tf.nn.l2_normalize(embeddings, axis=1).numpy()
        
        # Convert to list of lists
        embeddings = embeddings.tolist()
        
        return embeddings


class CustomGenerator(BaseEmbeddingGenerator):
    """
    Generate embeddings using a custom function.
    """
    
    def __init__(
        self,
        embedding_function: Callable[[List[str]], List[List[float]]],
        normalize_embeddings: bool = True
    ):
        """
        Initialize a custom embedding generator.
        
        Args:
            embedding_function: Function that takes a list of texts and returns a list of embeddings
            normalize_embeddings: Whether to normalize embeddings
        """
        self.embedding_function = embedding_function
        self.normalize_embeddings = normalize_embeddings
    
    def generate(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            **kwargs: Additional arguments for embedding generation
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.embedding_function(texts)
        
        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = [
                self._normalize_embedding(embedding)
                for embedding in embeddings
            ]
        
        return embeddings
    
    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        Normalize an embedding to unit length.
        
        Args:
            embedding: Embedding to normalize
            
        Returns:
            Normalized embedding
        """
        norm = np.sqrt(sum(x * x for x in embedding))
        return [x / norm for x in embedding]


# Factory function to get an embedding generator
def get_embedding_generator(
    generator_type: str = "sentence-transformers",
    model_name: Optional[str] = None,
    normalize_embeddings: bool = True,
    **kwargs
) -> BaseEmbeddingGenerator:
    """
    Get an embedding generator based on the specified type.
    
    Args:
        generator_type: Type of generator to use
                      ('sentence-transformers', 'huggingface', 'openai', 'tensorflow', 'custom')
        model_name: Name of the model to use (model-specific)
        normalize_embeddings: Whether to normalize embeddings
        **kwargs: Additional arguments for the specific generator
        
    Returns:
        Embedding generator instance
        
    Raises:
        ValueError: If an unsupported generator type is specified
    """
    if generator_type == "sentence-transformers":
        return SentenceTransformerGenerator(
            model_name=model_name or "all-MiniLM-L6-v2",
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )
    elif generator_type == "huggingface":
        return HuggingFaceGenerator(
            model_name=model_name or "bert-base-uncased",
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )
    elif generator_type == "openai":
        return OpenAIGenerator(
            model_name=model_name or "text-embedding-3-small",
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )
    elif generator_type == "tensorflow":
        return TensorFlowGenerator(
            model_url=model_name or "https://tfhub.dev/google/universal-sentence-encoder/4",
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )
    elif generator_type == "custom":
        if "embedding_function" not in kwargs:
            raise ValueError(
                "embedding_function is required for custom generator."
            )
        return CustomGenerator(
            embedding_function=kwargs.pop("embedding_function"),
            normalize_embeddings=normalize_embeddings
        )
    else:
        raise ValueError(
            f"Unsupported generator type: {generator_type}. "
            "Supported types are: 'sentence-transformers', 'huggingface', 'openai', 'tensorflow', 'custom'."
        ) 