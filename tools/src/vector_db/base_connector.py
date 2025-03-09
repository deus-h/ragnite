"""
Base Connector for Vector Databases

This module provides the abstract base class for vector database connectors.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple


class BaseVectorDBConnector(ABC):
    """
    Abstract base class for vector database connectors.
    
    This class defines the interface that all vector database connectors must implement.
    It ensures consistent behavior across different vector database implementations.
    """
    
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """
        Connect to the vector database.
        
        Args:
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the vector database.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to the vector database.
        
        Returns:
            bool: True if connected, False otherwise
        """
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """
        List all collections in the vector database.
        
        Returns:
            List[str]: List of collection names
        """
        pass
    
    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the vector database.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict[str, Any]: Collection information
        """
        pass
    
    @abstractmethod
    def create_collection(
        self, 
        collection_name: str, 
        dimension: int,
        distance_metric: str = "cosine",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a new collection in the vector database.
        
        Args:
            collection_name: Name of the collection to create
            dimension: Dimensionality of vectors to store
            distance_metric: Distance metric to use ("cosine", "euclidean", "dot")
            metadata: Optional metadata to associate with the collection
            
        Returns:
            Any: Collection object or ID
        """
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the vector database.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None
    ) -> bool:
        """
        Add vectors to a collection.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vectors to add
            ids: List of IDs for the vectors
            metadata: Optional list of metadata for each vector
            documents: Optional list of document texts for each vector
            
        Returns:
            bool: True if addition was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_vector(
        self,
        collection_name: str,
        id: str
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Get a vector and its metadata by ID.
        
        Args:
            collection_name: Name of the collection
            id: ID of the vector to get
            
        Returns:
            Tuple[List[float], Dict[str, Any]]: Tuple of (vector, metadata)
        """
        pass
    
    @abstractmethod
    def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """
        Delete vectors from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs of vectors to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            top_k: Number of results to return
            filter: Optional filter to apply
            
        Returns:
            List[Dict[str, Any]]: List of search results with id, score, and metadata
        """
        pass
    
    @abstractmethod
    def count_vectors(
        self,
        collection_name: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            filter: Optional filter to apply
            
        Returns:
            int: Number of vectors
        """
        pass
    
    def __enter__(self):
        """
        Enter context manager.
        
        Returns:
            BaseVectorDBConnector: Self
        """
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        self.disconnect() 