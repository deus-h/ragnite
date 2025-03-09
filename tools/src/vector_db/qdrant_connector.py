"""
Qdrant Connector

This module provides a connector for Qdrant vector database.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from .base_connector import BaseVectorDBConnector

# Configure logging
logger = logging.getLogger(__name__)

class QdrantConnector(BaseVectorDBConnector):
    """
    Connector for Qdrant vector database.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        api_key: Optional[str] = None,
        timeout: int = 10,
        path: Optional[str] = None,
    ):
        """
        Initialize Qdrant connector.
        
        Args:
            url: URL for REST client (overrides host and port if provided)
            host: Host address for REST or gRPC client
            port: Port for REST client
            grpc_port: Port for gRPC client
            prefer_grpc: Whether to prefer gRPC over REST
            api_key: API key for Qdrant Cloud or authentication
            timeout: Connection timeout in seconds
            path: Path for local Qdrant storage, for embedded mode
        """
        self.url = url
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.api_key = api_key
        self.timeout = timeout
        self.path = path
        self.client = None
    
    def connect(self, **kwargs) -> bool:
        """
        Connect to Qdrant.
        
        Args:
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance
            
            # Override connection parameters if provided
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            
            # Connect based on provided parameters
            if self.path:
                # Local persistent storage (embedded mode)
                self.client = QdrantClient(path=self.path)
                logger.info(f"Connected to Qdrant in embedded mode at {self.path}")
            
            elif self.url:
                # Connect via URL (typically for Qdrant Cloud)
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
                logger.info(f"Connected to Qdrant via URL: {self.url}")
            
            else:
                # Connect to host:port
                if self.prefer_grpc:
                    # gRPC client
                    self.client = QdrantClient(
                        host=self.host,
                        port=self.grpc_port,
                        api_key=self.api_key,
                        timeout=self.timeout,
                        grpc_options=(('grpc.enable_http_proxy', 0),),
                        prefer_grpc=True
                    )
                    logger.info(f"Connected to Qdrant via gRPC at {self.host}:{self.grpc_port}")
                else:
                    # REST client
                    self.client = QdrantClient(
                        host=self.host,
                        port=self.port,
                        api_key=self.api_key,
                        timeout=self.timeout
                    )
                    logger.info(f"Connected to Qdrant via REST at {self.host}:{self.port}")
            
            return True
        
        except ImportError:
            logger.error("Qdrant client package not installed. Install with: pip install qdrant-client")
            return False
        
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Qdrant.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        try:
            # Close the client (if needed)
            self.client = None
            return True
        
        except Exception as e:
            logger.error(f"Error disconnecting from Qdrant: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to Qdrant.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.client:
            return False
        
        try:
            # Test connection with a simple request
            collections = self.client.get_collections()
            return True
        
        except Exception:
            return False
    
    def _check_connection(self):
        """
        Check if connected to Qdrant and raise exception if not.
        
        Raises:
            ConnectionError: If not connected to Qdrant
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to Qdrant. Call connect() first.")
    
    def _get_distance_type(self, distance_metric: str):
        """
        Convert distance metric string to Qdrant Distance enum value.
        
        Args:
            distance_metric: Distance metric name
            
        Returns:
            Distance: Qdrant Distance enum value
        """
        from qdrant_client.http.models import Distance
        
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        
        if distance_metric.lower() not in distance_map:
            raise ValueError(f"Unsupported distance metric: {distance_metric}. " 
                           f"Use one of: {list(distance_map.keys())}")
        
        return distance_map[distance_metric.lower()]
    
    def list_collections(self) -> List[str]:
        """
        List all collections in Qdrant.
        
        Returns:
            List[str]: List of collection names
        """
        self._check_connection()
        
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Qdrant.
        
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
        Get information about a collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict[str, Any]: Collection information
        """
        self._check_connection()
        
        try:
            # Get collection info
            collection_info = self.client.get_collection(collection_name=collection_name)
            
            # Format response
            info = {
                "name": collection_name,
                "dimension": collection_info.config.params.vectors.size,
                "distance_metric": str(collection_info.config.params.vectors.distance).lower(),
                "count": collection_info.vectors_count,
                "status": str(collection_info.status)
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
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection to create
            dimension: Dimensionality of vectors to store
            distance_metric: Distance metric to use ("cosine", "euclidean", "dot")
            metadata: Optional metadata to associate with the collection
            
        Returns:
            Any: True if creation was successful
        """
        self._check_connection()
        
        try:
            from qdrant_client.http.models import VectorParams
            from qdrant_client.http.models import Distance
            from qdrant_client.http.models import OptimizersConfigDiff
            
            # Get distance type
            distance_type = self._get_distance_type(distance_metric)
            
            # Check if collection already exists
            if self.collection_exists(collection_name):
                # Delete existing collection
                self.delete_collection(collection_name)
            
            # Create vector params
            vector_params = VectorParams(
                size=dimension,
                distance=distance_type
            )
            
            # Create optimizers config
            optimizers_config = OptimizersConfigDiff(
                indexing_threshold=20000,  # Start indexing after 20k vectors
            )
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_params,
                optimizers_config=optimizers_config
            )
            
            # Add metadata if provided
            if metadata:
                from qdrant_client.http.models import CollectionMetaSchema
                self.client.set_collection_meta(
                    collection_name=collection_name,
                    metadata=metadata
                )
            
            logger.info(f"Created collection '{collection_name}' in Qdrant")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Qdrant.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        try:
            if not self.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return True
            
            # Delete collection
            self.client.delete_collection(collection_name=collection_name)
            
            logger.info(f"Deleted collection '{collection_name}' from Qdrant")
            
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
        Add vectors to a collection in Qdrant.
        
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
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:
            from qdrant_client.http.models import PointStruct
            
            # Get vectors info
            num_vectors = len(vectors)
            if len(ids) != num_vectors:
                raise ValueError("Number of vectors and IDs must match")
            
            # Prepare payloads (metadata)
            payloads = []
            
            for i in range(num_vectors):
                payload = {}
                
                # Add metadata if provided
                if metadata and i < len(metadata) and metadata[i]:
                    payload.update(metadata[i])
                
                # Add document if provided
                if documents and i < len(documents) and documents[i]:
                    payload["document"] = documents[i]
                
                payloads.append(payload)
            
            # Convert string ids to integers if needed
            points = []
            for i in range(num_vectors):
                # Use original ID as string in payload
                if "id" not in payloads[i]:
                    payloads[i]["id_str"] = ids[i]
                
                # Create point
                try:
                    # Try to use ID as integer
                    point_id = int(ids[i])
                except (ValueError, TypeError):
                    # Use hash of string ID if not convertible to int
                    point_id = hash(ids[i]) & 0xffffffff  # Ensure positive 32-bit integer
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vectors[i],
                    payload=payloads[i]
                ))
            
            # Add points to collection
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Added {len(vectors)} vectors to collection '{collection_name}' in Qdrant")
            
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
        Get a vector and its metadata by ID from Qdrant.
        
        Args:
            collection_name: Name of the collection
            id: ID of the vector to get
            
        Returns:
            Tuple[List[float], Dict[str, Any]]: Tuple of (vector, metadata)
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:
            # Try to convert ID to integer
            try:
                point_id = int(id)
            except (ValueError, TypeError):
                # Use hash of string ID if not convertible to int
                point_id = hash(id) & 0xffffffff  # Ensure positive 32-bit integer
            
            # Get vector by ID
            point = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_vectors=True,
                with_payload=True
            )
            
            if not point:
                # Try to find by original string ID in payload
                filter_query = {
                    "must": [
                        {
                            "key": "id_str",
                            "match": {
                                "value": id
                            }
                        }
                    ]
                }
                
                search_result = self.client.search(
                    collection_name=collection_name,
                    query_filter=filter_query,
                    limit=1,
                    with_vectors=True,
                    with_payload=True
                )
                
                if not search_result:
                    raise ValueError(f"Vector with ID '{id}' not found")
                
                point = search_result
            
            # Extract vector and metadata
            vector = list(point[0].vector)
            metadata = dict(point[0].payload)
            
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
        Delete vectors from a collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs of vectors to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:
            # Convert string IDs to integers
            point_ids = []
            for id_str in ids:
                try:
                    point_id = int(id_str)
                    point_ids.append(point_id)
                except (ValueError, TypeError):
                    # Use hash of string ID
                    point_id = hash(id_str) & 0xffffffff
                    point_ids.append(point_id)
                    
                    # Also try to delete by filter for string IDs
                    filter_query = {
                        "must": [
                            {
                                "key": "id_str",
                                "match": {
                                    "value": id_str
                                }
                            }
                        ]
                    }
                    
                    self.client.delete(
                        collection_name=collection_name,
                        points_filter=filter_query
                    )
            
            # Delete vectors by IDs
            if point_ids:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=point_ids
                )
            
            logger.info(f"Deleted {len(ids)} vectors from collection '{collection_name}' in Qdrant")
            
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
        Search for similar vectors in a collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            top_k: Number of results to return
            filter: Optional filter to apply (as a dictionary)
            
        Returns:
            List[Dict[str, Any]]: List of search results with id, score, and metadata
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:            
            # Perform search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter,
                with_payload=True
            )
            
            # Format results
            results = []
            for item in search_result:
                result = {
                    "id": item.payload.get("id_str", str(item.id)),
                    "score": item.score
                }
                
                # Extract metadata (exclude id_str)
                metadata = {k: v for k, v in item.payload.items() if k != "id_str"}
                
                if metadata:
                    result["metadata"] = metadata
                
                if "document" in item.payload:
                    result["document"] = item.payload["document"]
                
                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            return []
    
    def count_vectors(
        self,
        collection_name: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count vectors in a collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            filter: Optional filter to apply
            
        Returns:
            int: Number of vectors
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        try:
            # Get count with optional filter
            count = self.client.count(
                collection_name=collection_name,
                count_filter=filter
            )
            
            return count.count
        
        except Exception as e:
            logger.error(f"Error counting vectors: {str(e)}")
            return 0 