"""
Grok Connector

This module provides a connector for xAI's Grok vector database operations.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Union

from .base_connector import BaseVectorDBConnector

# Configure logging
logger = logging.getLogger(__name__)

class GrokConnector(BaseVectorDBConnector):
    """
    Connector for xAI's Grok vector database operations.
    
    This connector leverages Grok's embeddings API to provide vector database functionality.
    It implements the BaseVectorDBConnector interface for standardized vector operations.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        embedding_model: str = "grok-embedding",
        timeout: float = 60.0,
        request_interval: float = 0.1,
        max_retries: int = 3,
        verify_ssl: bool = True,
        proxies: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Grok connector.
        
        Args:
            api_key: xAI API key
            base_url: Base URL for the xAI API
            embedding_model: Name of the embedding model to use
            timeout: Timeout for API requests in seconds
            request_interval: Interval between requests to prevent rate limiting
            max_retries: Maximum number of retries for failed requests
            verify_ssl: Whether to verify SSL certificates
            proxies: Proxy configuration for requests
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.request_interval = request_interval
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.proxies = proxies
        self.connected = False
        self.client = None
        self.collection_cache = {}
        
    def connect(self, **kwargs) -> bool:
        """
        Connect to Grok API.
        
        Args:
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            # Import requests here to avoid dependency issues
            import requests
            
            # Override init parameters with kwargs if provided
            api_key = kwargs.get("api_key", self.api_key)
            base_url = kwargs.get("base_url", self.base_url)
            verify_ssl = kwargs.get("verify_ssl", self.verify_ssl)
            proxies = kwargs.get("proxies", self.proxies)
            
            # Test connection by making a request to the models endpoint
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{base_url}/models",
                headers=headers,
                timeout=self.timeout,
                verify=verify_ssl,
                proxies=proxies
            )
            
            if response.status_code == 200:
                self.connected = True
                self.client = requests.Session()
                self.client.headers.update(headers)
                
                logger.info("Successfully connected to Grok API")
                return True
            else:
                error_message = f"Failed to connect to Grok API: {response.status_code} - {response.text}"
                logger.error(error_message)
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to Grok API: {str(e)}")
            self.connected = False
            return False
            
    def disconnect(self) -> bool:
        """
        Disconnect from Grok API.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        try:
            if self.client:
                self.client.close()
                self.client = None
                
            self.connected = False
            self.collection_cache = {}
            
            logger.info("Disconnected from Grok API")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Grok API: {str(e)}")
            return False
            
    def is_connected(self) -> bool:
        """
        Check if connected to Grok API.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected and self.client is not None
        
    def _check_connection(self):
        """
        Check if connected to Grok API, raise exception if not.
        
        Raises:
            ConnectionError: If not connected to Grok API
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Grok API. Call connect() first.")
            
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[List[float]]: List of embeddings
            
        Raises:
            Exception: If embedding creation fails
        """
        self._check_connection()
        
        try:
            embeddings = []
            
            # Process in batches to avoid rate limits
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                data = {
                    "model": self.embedding_model,
                    "input": batch_texts
                }
                
                response = self.client.post(
                    f"{self.base_url}/embeddings",
                    json=data,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    proxies=self.proxies
                )
                
                if response.status_code != 200:
                    raise Exception(f"Error creating embeddings: {response.status_code} - {response.text}")
                
                response_data = response.json()
                batch_embeddings = [item["embedding"] for item in response_data["data"]]
                embeddings.extend(batch_embeddings)
                
                # Add delay to avoid rate limiting
                time.sleep(self.request_interval)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
            
    def _list_vector_stores(self) -> List[str]:
        """
        List all vector stores.
        
        Returns:
            List[str]: List of vector store names
            
        Raises:
            Exception: If listing vector stores fails
        """
        self._check_connection()
        
        try:
            response = self.client.get(
                f"{self.base_url}/vector_stores",
                timeout=self.timeout,
                verify=self.verify_ssl,
                proxies=self.proxies
            )
            
            if response.status_code != 200:
                if response.status_code == 404:
                    # Vector stores endpoint might not be available yet
                    logger.warning("Vector stores endpoint not found. Returning empty list.")
                    return []
                else:
                    raise Exception(f"Error listing vector stores: {response.status_code} - {response.text}")
            
            response_data = response.json()
            return [store["id"] for store in response_data.get("data", [])]
            
        except Exception as e:
            logger.error(f"Error listing vector stores: {str(e)}")
            # Return empty list in case of error
            return []
            
    def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List[str]: List of collection names
        """
        try:
            # Grok's API uses vector_stores as collections
            return self._list_vector_stores()
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return []
            
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            collections = self.list_collections()
            return collection_name in collections
        except Exception as e:
            logger.error(f"Error checking if collection exists: {str(e)}")
            return False
            
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict[str, Any]: Collection information
        """
        self._check_connection()
        
        try:
            if not self.collection_exists(collection_name):
                return {"name": collection_name, "error": "Collection does not exist"}
                
            response = self.client.get(
                f"{self.base_url}/vector_stores/{collection_name}",
                timeout=self.timeout,
                verify=self.verify_ssl,
                proxies=self.proxies
            )
            
            if response.status_code != 200:
                return {
                    "name": collection_name,
                    "error": f"Failed to get collection info: {response.status_code} - {response.text}"
                }
                
            collection_data = response.json()
            
            # Format the response to match the expected schema
            info = {
                "name": collection_name,
                "id": collection_data.get("id", collection_name),
                "dimension": collection_data.get("dimension", 0),
                "vector_count": collection_data.get("vector_count", 0),
                "metadata_config": collection_data.get("metadata_config", {}),
                "created_at": collection_data.get("created_at", ""),
                "status": collection_data.get("status", "unknown")
            }
            
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
        Create a new collection.
        
        Args:
            collection_name: Name of the collection to create
            dimension: Dimension of vectors
            distance_metric: Distance metric to use (cosine, euclidean, dot)
            metadata: Additional metadata for the collection
            
        Returns:
            Any: Created collection
            
        Raises:
            ValueError: If collection already exists or if parameters are invalid
        """
        self._check_connection()
        
        if self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")
            
        try:
            # Prepare metadata configuration
            metadata_config = {}
            if metadata:
                metadata_config = metadata.get("metadata_config", {})
                
            # Map distance metric to API format
            metric_map = {
                "cosine": "cosine",
                "euclidean": "l2",
                "l2": "l2",
                "dot": "dot",
                "inner_product": "dot"
            }
            
            api_distance_metric = metric_map.get(distance_metric.lower(), "cosine")
            
            data = {
                "name": collection_name,
                "dimension": dimension,
                "metric": api_distance_metric,
                "metadata_config": metadata_config
            }
            
            response = self.client.post(
                f"{self.base_url}/vector_stores",
                json=data,
                timeout=self.timeout,
                verify=self.verify_ssl,
                proxies=self.proxies
            )
            
            if response.status_code not in (200, 201):
                raise ValueError(f"Failed to create collection: {response.status_code} - {response.text}")
                
            # Store collection in cache
            response_data = response.json()
            self.collection_cache[collection_name] = response_data
            
            logger.info(f"Created collection: {collection_name}")
            return response_data
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise ValueError(f"Failed to create collection: {str(e)}")
            
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return False
            
        try:
            response = self.client.delete(
                f"{self.base_url}/vector_stores/{collection_name}",
                timeout=self.timeout,
                verify=self.verify_ssl,
                proxies=self.proxies
            )
            
            if response.status_code not in (200, 202, 204):
                logger.error(f"Failed to delete collection: {response.status_code} - {response.text}")
                return False
                
            # Remove from cache if exists
            if collection_name in self.collection_cache:
                del self.collection_cache[collection_name]
                
            logger.info(f"Deleted collection: {collection_name}")
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
        Add vectors to a collection.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vectors to add
            ids: List of IDs for the vectors
            metadata: List of metadata for the vectors
            documents: List of documents for the vectors
            
        Returns:
            bool: True if addition was successful, False otherwise
        """
        self._check_connection()
        
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")
            
        if metadata and len(metadata) != len(vectors):
            raise ValueError("Number of metadata items must match number of vectors")
            
        if documents and len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")
            
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
            
        try:
            # Process in batches to avoid hitting API limits
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                end_idx = min(i + batch_size, len(vectors))
                batch_vectors = vectors[i:end_idx]
                batch_ids = ids[i:end_idx]
                batch_metadata = metadata[i:end_idx] if metadata else [{}] * len(batch_vectors)
                
                # Add document content to metadata if provided
                if documents:
                    batch_documents = documents[i:end_idx]
                    for j, doc in enumerate(batch_documents):
                        if doc:
                            batch_metadata[j]["text"] = doc
                
                # Prepare data for API
                vectors_data = []
                for j in range(len(batch_vectors)):
                    vector_entry = {
                        "id": batch_ids[j],
                        "embedding": batch_vectors[j],
                        "metadata": batch_metadata[j]
                    }
                    vectors_data.append(vector_entry)
                
                # Make API request
                response = self.client.post(
                    f"{self.base_url}/vector_stores/{collection_name}/vectors/batch",
                    json={"vectors": vectors_data},
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    proxies=self.proxies
                )
                
                if response.status_code not in (200, 201):
                    # Try individual insertions if batch fails
                    success = self._add_vectors_individually(
                        collection_name, 
                        batch_vectors, 
                        batch_ids, 
                        batch_metadata
                    )
                    if not success:
                        return False
                
                # Add delay to avoid rate limiting
                time.sleep(self.request_interval)
            
            logger.info(f"Added {len(vectors)} vectors to collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding vectors: {str(e)}")
            return False
            
    def _add_vectors_individually(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """
        Add vectors individually when batch add fails.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vectors to add
            ids: List of IDs for the vectors
            metadata: List of metadata for the vectors
            
        Returns:
            bool: True if all additions were successful, False otherwise
        """
        all_successful = True
        
        for i in range(len(vectors)):
            try:
                data = {
                    "id": ids[i],
                    "embedding": vectors[i],
                    "metadata": metadata[i]
                }
                
                response = self.client.post(
                    f"{self.base_url}/vector_stores/{collection_name}/vectors",
                    json=data,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    proxies=self.proxies
                )
                
                if response.status_code not in (200, 201):
                    logger.error(f"Failed to add vector {ids[i]}: {response.status_code} - {response.text}")
                    all_successful = False
                
                # Add delay to avoid rate limiting
                time.sleep(self.request_interval)
                
            except Exception as e:
                logger.error(f"Error adding vector {ids[i]}: {str(e)}")
                all_successful = False
                
        return all_successful
            
    def get_vector(
        self,
        collection_name: str,
        id: str
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Get a vector from a collection.
        
        Args:
            collection_name: Name of the collection
            id: ID of the vector to get
            
        Returns:
            Tuple[List[float], Dict[str, Any]]: Vector and metadata
            
        Raises:
            ValueError: If vector does not exist
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
            
        try:
            response = self.client.get(
                f"{self.base_url}/vector_stores/{collection_name}/vectors/{id}",
                timeout=self.timeout,
                verify=self.verify_ssl,
                proxies=self.proxies
            )
            
            if response.status_code != 200:
                raise ValueError(f"Failed to get vector: {response.status_code} - {response.text}")
                
            vector_data = response.json()
            
            # Extract vector and metadata
            vector = vector_data.get("embedding", [])
            metadata = vector_data.get("metadata", {})
            
            return vector, metadata
            
        except Exception as e:
            logger.error(f"Error getting vector: {str(e)}")
            raise ValueError(f"Failed to get vector: {str(e)}")
            
    def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """
        Delete vectors from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return False
            
        try:
            # Try batch deletion first
            response = self.client.post(
                f"{self.base_url}/vector_stores/{collection_name}/vectors/delete",
                json={"ids": ids},
                timeout=self.timeout,
                verify=self.verify_ssl,
                proxies=self.proxies
            )
            
            if response.status_code in (200, 202, 204):
                logger.info(f"Deleted {len(ids)} vectors from collection: {collection_name}")
                return True
                
            # If batch deletion fails, try individual deletions
            all_successful = True
            for id in ids:
                try:
                    response = self.client.delete(
                        f"{self.base_url}/vector_stores/{collection_name}/vectors/{id}",
                        timeout=self.timeout,
                        verify=self.verify_ssl,
                        proxies=self.proxies
                    )
                    
                    if response.status_code not in (200, 202, 204):
                        logger.error(f"Failed to delete vector {id}: {response.status_code} - {response.text}")
                        all_successful = False
                    
                    # Add delay to avoid rate limiting
                    time.sleep(self.request_interval)
                    
                except Exception as e:
                    logger.error(f"Error deleting vector {id}: {str(e)}")
                    all_successful = False
                    
            return all_successful
            
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
        Search for similar vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Vector to search for
            top_k: Number of results to return
            filter: Filter for the search
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return []
            
        try:
            data = {
                "embedding": query_vector,
                "k": top_k
            }
            
            # Add filter if provided
            if filter:
                data["filter"] = filter
                
            response = self.client.post(
                f"{self.base_url}/vector_stores/{collection_name}/query",
                json=data,
                timeout=self.timeout,
                verify=self.verify_ssl,
                proxies=self.proxies
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to search vectors: {response.status_code} - {response.text}")
                return []
                
            search_results = response.json().get("matches", [])
            
            # Format results to match expected schema
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    "id": result.get("id"),
                    "score": result.get("score", 0.0),
                    "metadata": result.get("metadata", {})
                }
                formatted_results.append(formatted_result)
                
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
        Count vectors in a collection.
        
        Args:
            collection_name: Name of the collection
            filter: Filter for counting
            
        Returns:
            int: Number of vectors
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return 0
            
        try:
            # Get collection info first as it might contain the count
            collection_info = self.get_collection_info(collection_name)
            
            # If filter is not provided, return the count from collection info
            if not filter and "vector_count" in collection_info:
                return collection_info["vector_count"]
                
            # If filter is provided or vector_count not available, make a specific count request
            url = f"{self.base_url}/vector_stores/{collection_name}/count"
            
            # If filter is provided, include it in the request
            if filter:
                response = self.client.post(
                    url,
                    json={"filter": filter},
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    proxies=self.proxies
                )
            else:
                response = self.client.get(
                    url,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    proxies=self.proxies
                )
                
            if response.status_code != 200:
                logger.error(f"Failed to count vectors: {response.status_code} - {response.text}")
                return 0
                
            count_data = response.json()
            return count_data.get("count", 0)
            
        except Exception as e:
            logger.error(f"Error counting vectors: {str(e)}")
            return 0 