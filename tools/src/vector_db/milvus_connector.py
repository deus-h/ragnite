"""
Milvus Connector

This module provides a connector for Milvus vector database.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Union, cast

from .base_connector import BaseVectorDBConnector

# Configure logging
logger = logging.getLogger(__name__)

class MilvusConnector(BaseVectorDBConnector):
    """
    Connector for Milvus vector database.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        uri: Optional[str] = None,
        user: str = "",
        password: str = "",
        secure: bool = False,
        db_name: str = "default",
        timeout: Optional[float] = None
    ):
        """
        Initialize Milvus connector.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            uri: Milvus server URI (alternative to host and port)
            user: Username for authentication
            password: Password for authentication
            secure: Whether to use TLS connection
            db_name: Database name
            timeout: Timeout for operations in seconds
        """
        self.host = host
        self.port = port
        self.uri = uri
        self.user = user
        self.password = password
        self.secure = secure
        self.db_name = db_name
        self.timeout = timeout
        self.client = None
        self.utility = None
        self.collection_cache = {}
    
    def connect(self, **kwargs) -> bool:
        """
        Connect to Milvus.
        
        Args:
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            from pymilvus import connections, utility, Collection
            
            # Update connection parameters with kwargs
            host = kwargs.get("host", self.host)
            port = kwargs.get("port", self.port)
            uri = kwargs.get("uri", self.uri)
            user = kwargs.get("user", self.user)
            password = kwargs.get("password", self.password)
            secure = kwargs.get("secure", self.secure)
            db_name = kwargs.get("db_name", self.db_name)
            timeout = kwargs.get("timeout", self.timeout)
            
            # Configure connection parameters
            conn_params = {
                "user": user,
                "password": password,
                "secure": secure
            }
            
            if timeout is not None:
                conn_params["timeout"] = timeout
            
            # Connect using URI if provided, otherwise use host and port
            if uri:
                conn_params["uri"] = uri
            else:
                conn_params["host"] = host
                conn_params["port"] = port
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                **conn_params
            )
            
            # Store utility reference
            self.utility = utility
            
            # Use the specified database
            if db_name != "default":
                try:
                    from pymilvus import db
                    if db_name not in db.list_database():
                        db.create_database(db_name)
                    db.using_database(db_name)
                except Exception as e:
                    logger.warning(f"Failed to use database {db_name}: {str(e)}")
            
            logger.info(f"Connected to Milvus server")
            return True
            
        except ImportError:
            logger.error("PyMilvus package not installed. Install with 'pip install pymilvus'")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Milvus.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        try:
            from pymilvus import connections
            
            # Disconnect from Milvus
            connections.disconnect("default")
            
            # Clear cache
            self.utility = None
            self.collection_cache = {}
            
            logger.info("Disconnected from Milvus")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to Milvus.
        
        Returns:
            bool: True if connected, False otherwise
        """
        try:
            from pymilvus import connections
            
            return connections.has_connection("default")
        except Exception:
            return False
    
    def _check_connection(self):
        """
        Check if connected to Milvus, raise exception if not.
        
        Raises:
            RuntimeError: If not connected to Milvus
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Milvus. Call connect() first.")
    
    def _get_collection(self, collection_name: str):
        """
        Get a Milvus collection by name, with caching.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection: Milvus collection
            
        Raises:
            ValueError: If collection does not exist
        """
        self._check_connection()
        
        from pymilvus import Collection
        
        if collection_name not in self.collection_cache:
            if not self.collection_exists(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist")
            
            # Get the collection
            self.collection_cache[collection_name] = Collection(collection_name)
        
        return self.collection_cache[collection_name]
    
    def _check_and_load_collection(self, collection_name: str):
        """
        Check if a collection exists and load it into memory if needed.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection: Milvus collection
            
        Raises:
            ValueError: If collection does not exist
        """
        collection = self._get_collection(collection_name)
        
        # Load collection if not loaded
        if not collection.is_loaded:
            collection.load()
        
        return collection
    
    def list_collections(self) -> List[str]:
        """
        List all collections in Milvus.
        
        Returns:
            List[str]: List of collection names
        """
        self._check_connection()
        
        try:
            from pymilvus import utility
            
            return utility.list_collections()
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Milvus.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        self._check_connection()
        
        try:
            from pymilvus import utility
            
            return utility.has_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to check if collection exists: {str(e)}")
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
        
        if not self.collection_exists(collection_name):
            return {"name": collection_name, "error": "Collection does not exist"}
        
        try:
            # Get the collection
            collection = self._get_collection(collection_name)
            
            # Get collection information
            info = {
                "name": collection_name,
                "schema": collection.schema,
                "description": collection.description,
                "num_entities": collection.num_entities,
                "is_empty": collection.is_empty,
                "indexes": collection.indexes,
                "index_status": collection.index_status,
                "loaded": collection.is_loaded
            }
            
            return info
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
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
        Create a new collection in Milvus.
        
        Args:
            collection_name: Name of the collection to create
            dimension: Dimension of vectors
            distance_metric: Distance metric to use (cosine, euclidean, ip, l2, etc.)
            metadata: Additional metadata for the collection, including:
                - description: Description of the collection
                - schema: Schema for the collection
                - index_params: Parameters for index creation
                - consistency_level: Consistency level for the collection
        
        Returns:
            Any: Created collection
            
        Raises:
            ValueError: If collection already exists or if parameters are invalid
        """
        self._check_connection()
        
        if self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")
        
        try:
            from pymilvus import (
                Collection, 
                FieldSchema, 
                CollectionSchema, 
                DataType
            )
            
            # Map distance metric to Milvus metric
            metric_map = {
                "cosine": "COSINE",
                "l2": "L2",
                "euclidean": "L2",
                "ip": "IP",
                "dotproduct": "IP",
                "dot": "IP",
                "manhattan": "MANHATTAN",
                "hamming": "HAMMING",
                "jaccard": "JACCARD",
                "tanimoto": "TANIMOTO"
            }
            
            if distance_metric.lower() not in metric_map:
                raise ValueError(f"Unsupported distance metric: {distance_metric}. "
                                f"Supported metrics: {', '.join(metric_map.keys())}")
            
            milvus_metric = metric_map[distance_metric.lower()]
            
            # Define fields
            fields = [
                # Primary key field
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=100
                ),
                # Vector field
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dimension
                ),
                # Metadata field (JSON)
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON
                ),
                # Document field
                FieldSchema(
                    name="document",
                    dtype=DataType.VARCHAR,
                    max_length=65535
                )
            ]
            
            # Add custom fields if provided
            if metadata and "fields" in metadata:
                for field in metadata.get("fields", []):
                    fields.append(field)
            
            # Create schema
            schema = CollectionSchema(
                fields=fields,
                description=metadata.get("description", "") if metadata else ""
            )
            
            # Create collection
            collection = Collection(
                name=collection_name,
                schema=schema,
                consistency_level=metadata.get("consistency_level", "Strong") if metadata else "Strong"
            )
            
            # Create index
            index_params = {
                "metric_type": milvus_metric,
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
            
            # Override index params if provided
            if metadata and "index_params" in metadata:
                index_params.update(metadata.get("index_params", {}))
            
            # Create index
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # Add to cache
            self.collection_cache[collection_name] = collection
            
            logger.info(f"Created Milvus collection: {collection_name}")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise ValueError(f"Failed to create collection: {str(e)}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from Milvus.
        
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
            from pymilvus import utility
            
            # Remove from cache
            if collection_name in self.collection_cache:
                del self.collection_cache[collection_name]
            
            # Delete the collection
            utility.drop_collection(collection_name)
            
            logger.info(f"Deleted Milvus collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
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
        Add vectors to a collection in Milvus.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vectors to add
            ids: List of IDs for the vectors
            metadata: List of metadata for the vectors
            documents: List of documents for the vectors
            
        Returns:
            bool: True if addition was successful, False otherwise
        """
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")
        
        if metadata and len(metadata) != len(vectors):
            raise ValueError("Number of metadata items must match number of vectors")
        
        if documents and len(documents) != len(vectors):
            raise ValueError("Number of documents must match number of vectors")
        
        self._check_connection()
        
        # Get the collection
        collection = self._get_collection(collection_name)
        
        try:
            # Prepare data
            data = {
                "id": ids,
                "embedding": vectors
            }
            
            # Add metadata if provided
            if metadata:
                data["metadata"] = metadata
            else:
                # Use empty metadata if not provided
                data["metadata"] = [{} for _ in range(len(vectors))]
            
            # Add documents if provided
            if documents:
                data["document"] = documents
            else:
                # Use empty documents if not provided
                data["document"] = ["" for _ in range(len(vectors))]
            
            # Insert data
            collection.insert(data)
            
            logger.info(f"Added {len(vectors)} vectors to Milvus collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {str(e)}")
            return False
    
    def get_vector(
        self,
        collection_name: str,
        id: str
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Get a vector from a collection in Milvus.
        
        Args:
            collection_name: Name of the collection
            id: ID of the vector to get
            
        Returns:
            Tuple[List[float], Dict[str, Any]]: Vector and metadata
            
        Raises:
            ValueError: If vector does not exist
        """
        self._check_connection()
        
        # Get the collection
        collection = self._check_and_load_collection(collection_name)
        
        try:
            # Query the vector
            results = collection.query(
                expr=f'id == "{id}"',
                output_fields=["embedding", "metadata", "document"]
            )
            
            if not results or len(results) == 0:
                raise ValueError(f"Vector with ID '{id}' not found")
            
            # Extract vector and metadata
            result = results[0]
            vector = result.get("embedding", [])
            
            # Prepare metadata
            metadata = result.get("metadata", {})
            
            # Add document to metadata if available
            if "document" in result:
                metadata["document"] = result["document"]
            
            return vector, metadata
            
        except Exception as e:
            logger.error(f"Failed to get vector: {str(e)}")
            raise ValueError(f"Failed to get vector: {str(e)}")
    
    def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """
        Delete vectors from a collection in Milvus.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        # Get the collection
        collection = self._get_collection(collection_name)
        
        try:
            # Prepare expression for deletion
            expr = f'id in ["{ids[0]}"'
            for id in ids[1:]:
                expr += f', "{id}"'
            expr += "]"
            
            # Delete the vectors
            collection.delete(expr)
            
            logger.info(f"Deleted {len(ids)} vectors from Milvus collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in a collection in Milvus.
        
        Args:
            collection_name: Name of the collection
            query_vector: Vector to search for
            top_k: Number of results to return
            filter: Filter for the search
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        self._check_connection()
        
        # Get the collection and load it if needed
        collection = self._check_and_load_collection(collection_name)
        
        try:
            # Prepare search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Prepare expression from filter
            expr = None
            if filter:
                expr = self._parse_filter(filter)
            
            # Perform the search
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["metadata", "document"]
            )
            
            # Format the results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    # Extract metadata
                    metadata = hit.entity.get("metadata", {})
                    
                    # Add document to metadata if available
                    if "document" in hit.entity:
                        metadata["document"] = hit.entity["document"]
                    
                    formatted_result = {
                        "id": hit.id,
                        "score": hit.score,
                        "metadata": metadata
                    }
                    
                    formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            return []
    
    def _parse_filter(self, filter: Dict[str, Any]) -> str:
        """
        Parse a filter dictionary into a Milvus expression string.
        
        Args:
            filter: Filter dictionary
            
        Returns:
            str: Milvus expression string
        """
        if not filter:
            return ""
        
        # Handle field-operator-value format
        if "field" in filter and "operator" in filter and "value" in filter:
            field = filter["field"]
            operator = filter["operator"]
            value = filter["value"]
            
            # Handle metadata field
            if "." in field:
                # For metadata fields, use JSON path
                parts = field.split(".")
                if parts[0] == "metadata":
                    field_path = ".".join(parts[1:])
                    return self._build_metadata_expr(field_path, operator, value)
            
            # Map operators
            op_map = {
                "eq": "==",
                "==": "==",
                "=": "==",
                "neq": "!=",
                "!=": "!=",
                "gt": ">",
                ">": ">",
                "gte": ">=",
                ">=": ">=",
                "lt": "<",
                "<": "<",
                "lte": "<=",
                "<=": "<=",
                "in": "in",
                "not in": "not in"
            }
            
            milvus_op = op_map.get(str(operator).lower(), "==")
            
            # Format value based on type
            if isinstance(value, str):
                # Escape quotes in string values
                value = value.replace('"', '\\"')
                return f'{field} {milvus_op} "{value}"'
            elif isinstance(value, (int, float, bool)):
                return f"{field} {milvus_op} {value}"
            elif isinstance(value, list):
                # For 'in' operator with list values
                if milvus_op in ["in", "not in"]:
                    if all(isinstance(v, str) for v in value):
                        # String list
                        values_str = ", ".join([f'"{v.replace('"', '\\"')}"' for v in value])
                    else:
                        # Numeric list
                        values_str = ", ".join([str(v) for v in value])
                    return f"{field} {milvus_op} [{values_str}]"
        
        # Handle AND/OR operators
        if "AND" in filter or "and" in filter:
            operands = filter.get("AND", filter.get("and", []))
            if operands:
                return " && ".join([f"({self._parse_filter(op)})" for op in operands if self._parse_filter(op)])
        
        if "OR" in filter or "or" in filter:
            operands = filter.get("OR", filter.get("or", []))
            if operands:
                return " || ".join([f"({self._parse_filter(op)})" for op in operands if self._parse_filter(op)])
        
        # If we can't parse the filter, return empty string
        logger.warning(f"Could not parse filter: {filter}")
        return ""
    
    def _build_metadata_expr(self, field_path: str, operator: str, value: Any) -> str:
        """
        Build a Milvus expression for filtering on metadata fields.
        
        Args:
            field_path: Field path within metadata
            operator: Operator for comparison
            value: Value to compare against
            
        Returns:
            str: Milvus expression string
        """
        # Map operators
        op_map = {
            "eq": "==",
            "==": "==",
            "=": "==",
            "neq": "!=",
            "!=": "!=",
            "gt": ">",
            ">": ">",
            "gte": ">=",
            ">=": ">=",
            "lt": "<",
            "<": "<",
            "lte": "<=",
            "<=": "<=",
            "in": "in",
            "not in": "not in"
        }
        
        milvus_op = op_map.get(str(operator).lower(), "==")
        
        # Format value based on type
        if isinstance(value, str):
            # Escape quotes in string values
            value = value.replace('"', '\\"')
            return f'json_contains(metadata, "{{\"{field_path}\": \"{value}\"}}")'
        elif isinstance(value, (int, float, bool)):
            return f'json_contains(metadata, "{{\"{field_path}\": {value}}}")'
        elif isinstance(value, list):
            # For array values, we need a different approach
            if all(isinstance(v, str) for v in value):
                # String array
                values_str = ", ".join([f'"{v.replace('"', '\\"')}"' for v in value])
                return f'json_contains(metadata, "{{\"{field_path}\": [{values_str}]}}")'
            else:
                # Numeric array
                values_str = ", ".join([str(v) for v in value])
                return f'json_contains(metadata, "{{\"{field_path}\": [{values_str}]}}")'
        
        # Default case
        return f'json_contains(metadata, "{{\"{field_path}\": null}}")'
    
    def count_vectors(
        self,
        collection_name: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count vectors in a collection in Milvus.
        
        Args:
            collection_name: Name of the collection
            filter: Filter for counting
            
        Returns:
            int: Number of vectors
        """
        self._check_connection()
        
        # Get the collection
        collection = self._get_collection(collection_name)
        
        try:
            # If no filter, return total count
            if not filter:
                return collection.num_entities
            
            # Parse filter
            expr = self._parse_filter(filter)
            
            if not expr:
                return collection.num_entities
            
            # Count with filter
            results = collection.query(
                expr=expr,
                output_fields=["count(*)"]
            )
            
            if results and len(results) > 0:
                return results[0].get("count(*)", 0)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to count vectors: {str(e)}")
            
            # Fall back to total count
            try:
                return collection.num_entities
            except Exception:
                return 0 