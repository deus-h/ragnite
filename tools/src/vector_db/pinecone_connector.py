"""
Pinecone Connector

This module provides a connector for Pinecone vector database.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union

from .base_connector import BaseVectorDBConnector

# Configure logging
logger = logging.getLogger(__name__)

class PineconeConnector(BaseVectorDBConnector):
    """
    Connector for Pinecone vector database.
    """
    
    def __init__(
        self,
        api_key: str,
        environment: str,
        project_id: Optional[str] = None,
        timeout: int = 60,
        pool_threads: int = 10
    ):
        """
        Initialize Pinecone connector.
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            project_id: Pinecone project ID (optional)
            timeout: Timeout for API requests in seconds
            pool_threads: Number of threads for connection pool
        """
        self.api_key = api_key
        self.environment = environment
        self.project_id = project_id
        self.timeout = timeout
        self.pool_threads = pool_threads
        self.client = None
        self.indexes = {}
    
    def connect(self, **kwargs) -> bool:
        """
        Connect to Pinecone.
        
        Args:
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            import pinecone
            
            # Update connection parameters with kwargs
            api_key = kwargs.get("api_key", self.api_key)
            environment = kwargs.get("environment", self.environment)
            project_id = kwargs.get("project_id", self.project_id)
            
            # Initialize Pinecone client
            init_params = {
                "api_key": api_key,
                "environment": environment
            }
            
            if project_id:
                init_params["project_id"] = project_id
                
            pinecone.init(**init_params)
            
            # Store client reference
            self.client = pinecone
            
            # Test connection by listing indexes
            _ = self.list_collections()
            
            logger.info(f"Connected to Pinecone environment: {environment}")
            return True
            
        except ImportError:
            logger.error("Pinecone Python package not installed. Install with 'pip install pinecone-client'")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Pinecone.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        try:
            # Clear cached indexes
            self.indexes = {}
            
            # Set client to None
            self.client = None
            
            logger.info("Disconnected from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Pinecone: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to Pinecone.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.client is not None
    
    def _check_connection(self):
        """
        Check if connected to Pinecone, raise exception if not.
        
        Raises:
            RuntimeError: If not connected to Pinecone
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Pinecone. Call connect() first.")
    
    def _get_index(self, index_name: str):
        """
        Get a Pinecone index by name, with caching.
        
        Args:
            index_name: Name of the index
            
        Returns:
            Index: Pinecone index
            
        Raises:
            ValueError: If index does not exist
        """
        self._check_connection()
        
        if index_name not in self.indexes:
            if not self.collection_exists(index_name):
                raise ValueError(f"Index '{index_name}' does not exist")
                
            # Connect to the index
            self.indexes[index_name] = self.client.Index(index_name)
            
        return self.indexes[index_name]
    
    def list_collections(self) -> List[str]:
        """
        List all collections (indexes) in Pinecone.
        
        Returns:
            List[str]: List of collection names
        """
        self._check_connection()
        
        try:
            # List all indexes
            indexes = self.client.list_indexes()
            return [index.name for index in indexes]
        except Exception as e:
            logger.error(f"Failed to list Pinecone indexes: {str(e)}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection (index) exists in Pinecone.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        self._check_connection()
        
        try:
            indexes = self.list_collections()
            return collection_name in indexes
        except Exception as e:
            logger.error(f"Failed to check if index exists: {str(e)}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection (index).
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict[str, Any]: Collection information
        """
        self._check_connection()
        
        try:
            # Get index description
            index_description = self.client.describe_index(collection_name)
            
            # Extract relevant information
            info = {
                "name": collection_name,
                "dimension": index_description.dimension,
                "metric": index_description.metric,
                "pods": index_description.pods,
                "status": index_description.status,
                "total_vector_count": index_description.total_vector_count
            }
            
            # Add index stats
            index = self._get_index(collection_name)
            stats = index.describe_index_stats()
            
            if hasattr(stats, "namespaces") and stats.namespaces:
                info["namespaces"] = stats.namespaces
            if hasattr(stats, "dimension") and stats.dimension:
                info["dimension"] = stats.dimension
            if hasattr(stats, "index_fullness") and stats.index_fullness:
                info["index_fullness"] = stats.index_fullness
            if hasattr(stats, "total_vector_count") and stats.total_vector_count:
                info["total_vector_count"] = stats.total_vector_count
            
            return info
        except Exception as e:
            logger.error(f"Failed to get index info: {str(e)}")
            return {
                "name": collection_name,
                "error": str(e)
            }
    
    def create_collection(
        self, 
        collection_name: str, 
        dimension: int,
        distance_metric: str = "cosine",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a new collection (index) in Pinecone.
        
        Args:
            collection_name: Name of the collection to create
            dimension: Dimension of vectors
            distance_metric: Distance metric to use (cosine, euclidean, or dotproduct)
            metadata: Additional metadata for the collection
        
        Returns:
            Any: Created collection
            
        Raises:
            ValueError: If collection already exists or if parameters are invalid
        """
        self._check_connection()
        
        if self.collection_exists(collection_name):
            raise ValueError(f"Index '{collection_name}' already exists")
        
        # Map distance metric to Pinecone metric
        metric_map = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "dotproduct": "dotproduct",
            "dot": "dotproduct"
        }
        
        if distance_metric.lower() not in metric_map:
            raise ValueError(f"Unsupported distance metric: {distance_metric}. "
                             f"Supported metrics: {', '.join(metric_map.keys())}")
        
        pinecone_metric = metric_map[distance_metric.lower()]
        
        try:
            # Create index
            create_args = {
                "name": collection_name,
                "dimension": dimension,
                "metric": pinecone_metric
            }
            
            # Add metadata if provided
            if metadata:
                if "pod_type" in metadata:
                    create_args["pod_type"] = metadata["pod_type"]
                if "pods" in metadata:
                    create_args["pods"] = metadata["pods"]
                if "replicas" in metadata:
                    create_args["replicas"] = metadata["replicas"]
                if "shards" in metadata:
                    create_args["shards"] = metadata["shards"]
                if "metadata_config" in metadata:
                    create_args["metadata_config"] = metadata["metadata_config"]
            
            self.client.create_index(**create_args)
            
            # Wait for index to be ready
            while not self.client.describe_index(collection_name).status.ready:
                logger.info(f"Waiting for index '{collection_name}' to be ready...")
                time.sleep(1)
            
            # Get the index
            index = self._get_index(collection_name)
            
            logger.info(f"Created Pinecone index: {collection_name}")
            return index
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise ValueError(f"Failed to create index: {str(e)}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection (index) from Pinecone.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            logger.warning(f"Index '{collection_name}' does not exist")
            return False
        
        try:
            # Delete the index
            self.client.delete_index(collection_name)
            
            # Remove from cache
            if collection_name in self.indexes:
                del self.indexes[collection_name]
            
            logger.info(f"Deleted Pinecone index: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index: {str(e)}")
            return False
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
        batch_size: int = 100,
        namespace: str = ""
    ) -> bool:
        """
        Add vectors to a collection (index) in Pinecone.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vectors to add
            ids: List of IDs for the vectors
            metadata: List of metadata for the vectors
            documents: List of documents for the vectors
            batch_size: Batch size for adding vectors
            namespace: Namespace to add vectors to
            
        Returns:
            bool: True if addition was successful, False otherwise
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")
        
        if metadata and len(metadata) != len(vectors):
            raise ValueError("Number of metadata items must match number of vectors")
        
        if documents and len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")
        
        # Get the index
        index = self._get_index(collection_name)
        
        try:
            # Prepare items for upsert
            items = []
            for i, (vector, id) in enumerate(zip(vectors, ids)):
                item = {
                    "id": id,
                    "values": vector
                }
                
                # Add metadata if provided
                if metadata and i < len(metadata):
                    meta = metadata[i]
                    # Add document as metadata if provided
                    if documents and i < len(documents):
                        meta["document"] = documents[i]
                    item["metadata"] = meta
                elif documents and i < len(documents):
                    # Add document as metadata if no other metadata is provided
                    item["metadata"] = {"document": documents[i]}
                
                items.append(item)
            
            # Upsert in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                index.upsert(vectors=batch, namespace=namespace)
            
            logger.info(f"Added {len(vectors)} vectors to Pinecone index: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {str(e)}")
            return False
    
    def get_vector(
        self,
        collection_name: str,
        id: str,
        namespace: str = ""
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Get a vector from a collection (index) in Pinecone.
        
        Args:
            collection_name: Name of the collection
            id: ID of the vector to get
            namespace: Namespace to get vector from
            
        Returns:
            Tuple[List[float], Dict[str, Any]]: Vector and metadata
            
        Raises:
            ValueError: If vector does not exist
        """
        # Get the index
        index = self._get_index(collection_name)
        
        try:
            # Fetch the vector
            result = index.fetch(ids=[id], namespace=namespace)
            
            if not result.vectors or id not in result.vectors:
                raise ValueError(f"Vector with ID '{id}' not found")
            
            vector_data = result.vectors[id]
            vector = vector_data.values
            metadata = vector_data.metadata if hasattr(vector_data, "metadata") else {}
            
            return vector, metadata
            
        except Exception as e:
            logger.error(f"Failed to get vector: {str(e)}")
            raise ValueError(f"Failed to get vector: {str(e)}")
    
    def delete_vectors(
        self,
        collection_name: str,
        ids: List[str],
        namespace: str = ""
    ) -> bool:
        """
        Delete vectors from a collection (index) in Pinecone.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete
            namespace: Namespace to delete vectors from
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        # Get the index
        index = self._get_index(collection_name)
        
        try:
            # Delete the vectors
            index.delete(ids=ids, namespace=namespace)
            
            logger.info(f"Deleted {len(ids)} vectors from Pinecone index: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        include_metadata: bool = True,
        include_values: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection (index) in Pinecone.
        
        Args:
            collection_name: Name of the collection
            query_vector: Vector to search for
            top_k: Number of results to return
            filter: Filter for the search
            namespace: Namespace to search in
            include_metadata: Whether to include metadata in results
            include_values: Whether to include vector values in results
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        # Get the index
        index = self._get_index(collection_name)
        
        try:
            # Perform the search
            result = index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                namespace=namespace,
                include_metadata=include_metadata,
                include_values=include_values
            )
            
            # Format the results
            formatted_results = []
            for match in result.matches:
                item = {
                    "id": match.id,
                    "score": match.score
                }
                
                if include_values and hasattr(match, "values"):
                    item["vector"] = match.values
                
                if include_metadata and hasattr(match, "metadata"):
                    item["metadata"] = match.metadata
                
                formatted_results.append(item)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            return []
    
    def count_vectors(
        self,
        collection_name: str,
        filter: Optional[Dict[str, Any]] = None,
        namespace: str = ""
    ) -> int:
        """
        Count vectors in a collection (index) in Pinecone.
        
        Args:
            collection_name: Name of the collection
            filter: Filter for counting
            namespace: Namespace to count vectors in
            
        Returns:
            int: Number of vectors
        """
        # Get the index
        index = self._get_index(collection_name)
        
        try:
            # Get index stats
            stats = index.describe_index_stats()
            
            # Count vectors in namespace
            if namespace:
                if hasattr(stats, "namespaces") and namespace in stats.namespaces:
                    return stats.namespaces[namespace].vector_count
                return 0
            
            # Count total vectors
            return stats.total_vector_count
            
        except Exception as e:
            logger.error(f"Failed to count vectors: {str(e)}")
            return 0 