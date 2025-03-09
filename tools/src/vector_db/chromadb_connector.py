"""
ChromaDB Connector

This module provides a connector for ChromaDB vector database.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from .base_connector import BaseVectorDBConnector

# Configure logging
logger = logging.getLogger(__name__)

class ChromaDBConnector(BaseVectorDBConnector):
    """
    Connector for ChromaDB vector database.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        ssl: bool = False,
        headers: Optional[Dict[str, str]] = None,
        persistent: bool = True,
        path: Optional[str] = "./chroma",
        in_memory: bool = False,
    ):
        """
        Initialize ChromaDB connector.
        
        Args:
            host: Host address for HTTP connection
            port: Port for HTTP connection
            ssl: Whether to use SSL for HTTP connection
            headers: Headers for HTTP requests
            persistent: Whether to persist data to disk
            path: Path for persistent storage
            in_memory: Whether to use in-memory storage
        """
        self.host = host
        self.port = port
        self.ssl = ssl
        self.headers = headers
        self.persistent = persistent
        self.path = path
        self.in_memory = in_memory
        self.client = None
        self.collections_cache = {}
    
    def connect(self, **kwargs) -> bool:
        """
        Connect to ChromaDB.
        
        Args:
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            import chromadb
            from chromadb.config import Settings
            
            if self.in_memory:
                # Connect to in-memory client
                self.client = chromadb.Client(Settings(anonymized_telemetry=False))
                logger.info("Connected to ChromaDB in-memory client")
            elif self.host == "localhost" and self.port == 8000:
                # Connect to HTTP client
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    ssl=self.ssl,
                    headers=self.headers
                )
                logger.info(f"Connected to ChromaDB HTTP client at {self.host}:{self.port}")
            else:
                # Connect to persistent client
                self.client = chromadb.PersistentClient(
                    path=self.path,
                    settings=Settings(anonymized_telemetry=False)
                )
                logger.info(f"Connected to ChromaDB persistent client at {self.path}")
            
            return True
        
        except ImportError:
            logger.error("ChromaDB package not installed. Install with: pip install chromadb")
            return False
        
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from ChromaDB.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        try:
            # Reset client reference and collections cache
            self.client = None
            self.collections_cache = {}
            return True
        
        except Exception as e:
            logger.error(f"Error disconnecting from ChromaDB: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to ChromaDB.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.client is not None
    
    def _check_connection(self):
        """
        Check if connected to ChromaDB and raise exception if not.
        
        Raises:
            ConnectionError: If not connected to ChromaDB
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to ChromaDB. Call connect() first.")
    
    def _get_collection(self, collection_name: str):
        """
        Get a collection by name, using cache if available.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection object
            
        Raises:
            ValueError: If collection does not exist
        """
        self._check_connection()
        
        # Check cache first
        if collection_name in self.collections_cache:
            return self.collections_cache[collection_name]
        
        # Check if collection exists
        try:
            collection = self.client.get_collection(name=collection_name)
            self.collections_cache[collection_name] = collection
            return collection
        except Exception as e:
            raise ValueError(f"Collection '{collection_name}' does not exist: {str(e)}")
    
    def list_collections(self) -> List[str]:
        """
        List all collections in ChromaDB.
        
        Returns:
            List[str]: List of collection names
        """
        self._check_connection()
        
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in ChromaDB.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        self._check_connection()
        
        try:
            collections = self.list_collections()
            return collection_name in collections
        
        except Exception as e:
            logger.error(f"Error checking if collection exists: {str(e)}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection in ChromaDB.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict[str, Any]: Collection information
        """
        self._check_connection()
        
        try:
            collection = self._get_collection(collection_name)
            # Basic information
            info = {
                "name": collection.name,
                "metadata": collection.metadata,
            }
            
            # Count
            count_result = collection.count()
            info["count"] = count_result
            
            # Get a sample vector to determine dimensions
            if count_result > 0:
                sample = collection.peek(1)
                if sample and "embeddings" in sample and sample["embeddings"]:
                    info["dimension"] = len(sample["embeddings"][0])
            
            return info
        
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"name": collection_name, "error": str(e)}
    
    def create_collection(
        self, 
        collection_name: str, 
        dimension: int,
        distance_metric: str = "cosine",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a new collection in ChromaDB.
        
        Args:
            collection_name: Name of the collection to create
            dimension: Dimensionality of vectors to store (not needed for ChromaDB but kept for API consistency)
            distance_metric: Distance metric to use ("cosine", "euclidean", "dot")
            metadata: Optional metadata to associate with the collection
            
        Returns:
            Any: Collection object or ID
        """
        self._check_connection()
        
        # Convert distance metric to ChromaDB format
        distance_map = {
            "cosine": "cosine",
            "euclidean": "l2",
            "dot": "ip"  # Inner product
        }
        
        chroma_distance = distance_map.get(distance_metric.lower(), "cosine")
        
        if metadata is None:
            metadata = {}
        
        # Add distance metric to metadata
        metadata_with_distance = {
            **metadata,
            "hnsw:space": chroma_distance
        }
        
        try:
            # Delete collection if it exists
            if self.collection_exists(collection_name):
                self.delete_collection(collection_name)
            
            # Create collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata_with_distance
            )
            
            # Update cache
            self.collections_cache[collection_name] = collection
            
            logger.info(f"Created collection {collection_name} in ChromaDB")
            return collection
        
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from ChromaDB.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        try:
            # Delete collection
            self.client.delete_collection(name=collection_name)
            
            # Remove from cache
            if collection_name in self.collections_cache:
                del self.collections_cache[collection_name]
            
            logger.info(f"Deleted collection {collection_name} from ChromaDB")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None
    ) -> bool:
        """
        Add vectors to a collection in ChromaDB.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vectors to add
            ids: List of IDs for the vectors
            metadata: Optional list of metadata for each vector
            documents: Optional list of document texts for each vector
            
        Returns:
            bool: True if addition was successful, False otherwise
        """
        self._check_connection()
        
        try:
            collection = self._get_collection(collection_name)
            
            # Prepare arguments for add
            add_kwargs = {
                "embeddings": vectors,
                "ids": ids
            }
            
            if metadata is not None:
                add_kwargs["metadatas"] = metadata
            
            if documents is not None:
                add_kwargs["documents"] = documents
            
            # Add vectors
            collection.add(**add_kwargs)
            
            logger.info(f"Added {len(vectors)} vectors to collection {collection_name} in ChromaDB")
            return True
        
        except Exception as e:
            logger.error(f"Error adding vectors: {str(e)}")
            return False
    
    def get_vector(
        self,
        collection_name: str,
        id: str
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Get a vector and its metadata by ID from ChromaDB.
        
        Args:
            collection_name: Name of the collection
            id: ID of the vector to get
            
        Returns:
            Tuple[List[float], Dict[str, Any]]: Tuple of (vector, metadata)
        """
        self._check_connection()
        
        try:
            collection = self._get_collection(collection_name)
            
            # Get vector by ID
            result = collection.get(
                ids=[id],
                include=["embeddings", "metadatas", "documents"]
            )
            
            if not result or not result["ids"]:
                raise ValueError(f"Vector with ID '{id}' not found")
            
            vector = result["embeddings"][0]
            metadata = {}
            
            if "metadatas" in result and result["metadatas"]:
                metadata = result["metadatas"][0]
            
            if "documents" in result and result["documents"]:
                metadata["document"] = result["documents"][0]
            
            return vector, metadata
        
        except Exception as e:
            logger.error(f"Error getting vector: {str(e)}")
            raise
    
    def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """
        Delete vectors from a collection in ChromaDB.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs of vectors to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        try:
            collection = self._get_collection(collection_name)
            
            # Delete vectors by IDs
            collection.delete(ids=ids)
            
            logger.info(f"Deleted {len(ids)} vectors from collection {collection_name} in ChromaDB")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection in ChromaDB.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            top_k: Number of results to return
            filter: Optional filter to apply
            
        Returns:
            List[Dict[str, Any]]: List of search results with id, score, and metadata
        """
        self._check_connection()
        
        try:
            collection = self._get_collection(collection_name)
            
            # Search with query vector
            query_kwargs = {
                "query_embeddings": [query_vector],
                "n_results": top_k,
                "include": ["distances", "metadatas", "documents"]
            }
            
            if filter is not None:
                query_kwargs["where"] = filter
            
            results = collection.query(**query_kwargs)
            
            # Format results
            formatted_results = []
            
            if results and "ids" in results and results["ids"]:
                for i, id in enumerate(results["ids"][0]):
                    result = {
                        "id": id,
                        "score": 1.0 - results["distances"][0][i]  # Convert distance to similarity score
                    }
                    
                    if "metadatas" in results and results["metadatas"] and results["metadatas"][0]:
                        result["metadata"] = results["metadatas"][0][i]
                    
                    if "documents" in results and results["documents"] and results["documents"][0]:
                        result["document"] = results["documents"][0][i]
                    
                    formatted_results.append(result)
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            return []
    
    def count_vectors(
        self,
        collection_name: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count vectors in a collection in ChromaDB.
        
        Args:
            collection_name: Name of the collection
            filter: Optional filter to apply
            
        Returns:
            int: Number of vectors
        """
        self._check_connection()
        
        try:
            collection = self._get_collection(collection_name)
            
            # If no filter, use simple count
            if filter is None:
                return collection.count()
            
            # Otherwise, use get with filter to count
            result = collection.get(where=filter)
            
            if result and "ids" in result:
                return len(result["ids"])
            
            return 0
        
        except Exception as e:
            logger.error(f"Error counting vectors: {str(e)}")
            return 0 