"""
Weaviate Connector

This module provides a connector for Weaviate vector database.
"""

import logging
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple, Union, cast

from .base_connector import BaseVectorDBConnector

# Configure logging
logger = logging.getLogger(__name__)

class WeaviateConnector(BaseVectorDBConnector):
    """
    Connector for Weaviate vector database.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        auth_client_secret: Optional[Dict[str, str]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        timeout_config: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Weaviate connector.
        
        Args:
            url: URL of the Weaviate instance
            api_key: API key for authentication
            auth_client_secret: Client secret for authentication
            additional_headers: Additional headers for API requests
            timeout_config: Configuration for request timeouts
        """
        self.url = url
        self.api_key = api_key
        self.auth_client_secret = auth_client_secret
        self.additional_headers = additional_headers or {}
        self.timeout_config = timeout_config or {
            "query": 30.0,
            "get": 30.0,
            "create": 60.0,
            "delete": 60.0,
            "update": 60.0
        }
        self.client = None
        self.schema = None
    
    def connect(self, **kwargs) -> bool:
        """
        Connect to Weaviate.
        
        Args:
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            import weaviate
            from weaviate.auth import AuthApiKey, AuthClientCredentials
            
            # Update connection parameters with kwargs
            url = kwargs.get("url", self.url)
            api_key = kwargs.get("api_key", self.api_key)
            auth_client_secret = kwargs.get("auth_client_secret", self.auth_client_secret)
            additional_headers = kwargs.get("additional_headers", self.additional_headers)
            timeout_config = kwargs.get("timeout_config", self.timeout_config)
            
            # Configure authentication
            auth = None
            if api_key:
                auth = AuthApiKey(api_key=api_key)
            elif auth_client_secret:
                auth = AuthClientCredentials(
                    client_id=auth_client_secret.get("client_id", ""),
                    client_secret=auth_client_secret.get("client_secret", ""),
                    scope=auth_client_secret.get("scope", "")
                )
            
            # Create client
            client = weaviate.Client(
                url=url,
                auth_client_secret=auth if isinstance(auth, AuthClientCredentials) else None,
                auth_client_password=None,
                headers=additional_headers,
                timeout_config=timeout_config
            )
            
            # Set API key if provided
            if isinstance(auth, AuthApiKey):
                client.timeout_config = timeout_config
                client.add_header("Authorization", f"Bearer {api_key}")
            
            # Check if server is ready
            if not client.is_ready():
                logger.error("Weaviate server is not ready")
                return False
            
            # Store client reference
            self.client = client
            
            # Get schema
            self.schema = client.schema.get()
            
            logger.info(f"Connected to Weaviate at {url}")
            return True
            
        except ImportError:
            logger.error("Weaviate Python package not installed. Install with 'pip install weaviate-client'")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from Weaviate.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        try:
            # Set client to None
            self.client = None
            self.schema = None
            
            logger.info("Disconnected from Weaviate")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Weaviate: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to Weaviate.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if self.client is None:
            return False
        
        try:
            return self.client.is_ready()
        except Exception:
            return False
    
    def _check_connection(self):
        """
        Check if connected to Weaviate, raise exception if not.
        
        Raises:
            RuntimeError: If not connected to Weaviate
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to Weaviate. Call connect() first.")
    
    def _sanitize_class_name(self, name: str) -> str:
        """
        Sanitize a collection name to a valid Weaviate class name.
        
        Args:
            name: Name to sanitize
            
        Returns:
            str: Sanitized name
        """
        # Replace hyphens and underscores with spaces
        sanitized = name.replace("-", " ").replace("_", " ")
        
        # Split by spaces, capitalize each part, and join
        capitalized = "".join([part.capitalize() for part in sanitized.split()])
        
        # Ensure first letter is uppercase
        if capitalized and capitalized[0].islower():
            capitalized = capitalized[0].upper() + capitalized[1:]
        
        return capitalized
    
    def _get_class_name(self, collection_name: str) -> str:
        """
        Get a valid Weaviate class name from a collection name.
        
        Args:
            collection_name: Original collection name
            
        Returns:
            str: Valid Weaviate class name
        """
        return self._sanitize_class_name(collection_name)
    
    def list_collections(self) -> List[str]:
        """
        List all collections (classes) in Weaviate.
        
        Returns:
            List[str]: List of collection names
        """
        self._check_connection()
        
        try:
            schema = self.client.schema.get()
            return [cls.get("class") for cls in schema.get("classes", [])]
        except Exception as e:
            logger.error(f"Failed to list Weaviate classes: {str(e)}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection (class) exists in Weaviate.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        self._check_connection()
        
        class_name = self._get_class_name(collection_name)
        
        try:
            schema = self.client.schema.get()
            classes = [cls.get("class") for cls in schema.get("classes", [])]
            return class_name in classes
        except Exception as e:
            logger.error(f"Failed to check if class exists: {str(e)}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection (class).
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict[str, Any]: Collection information
        """
        self._check_connection()
        
        class_name = self._get_class_name(collection_name)
        
        if not self.collection_exists(class_name):
            return {"name": class_name, "error": "Collection does not exist"}
        
        try:
            # Get class schema
            schema = self.client.schema.get(class_name)
            
            # Get count of objects in the class
            count_result = self.client.query.aggregate(class_name).with_meta_count().do()
            count = count_result.get("data", {}).get("Aggregate", {}).get(class_name, [{}])[0].get("meta", {}).get("count", 0)
            
            info = {
                "name": class_name,
                "original_name": collection_name,
                "count": count,
                "schema": schema
            }
            
            return info
        except Exception as e:
            logger.error(f"Failed to get class info: {str(e)}")
            return {
                "name": class_name,
                "original_name": collection_name,
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
        Create a new collection (class) in Weaviate.
        
        Args:
            collection_name: Name of the collection to create
            dimension: Dimension of vectors
            distance_metric: Distance metric to use (cosine, euclidean, or dot)
            metadata: Additional metadata for the collection, including:
                - properties: List of property definitions
                - description: Description of the class
                - vectorizer: Name of the vectorizer to use
                - module_config: Configuration for modules
                - inverted_index_config: Configuration for inverted index
                
        Returns:
            Any: Created collection
            
        Raises:
            ValueError: If collection already exists or if parameters are invalid
        """
        self._check_connection()
        
        class_name = self._get_class_name(collection_name)
        
        if self.collection_exists(class_name):
            raise ValueError(f"Class '{class_name}' already exists")
        
        # Map distance metric to Weaviate distance
        metric_map = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "dot": "dot",
            "dotproduct": "dot",
            "l2": "euclidean",
            "l2-squared": "euclidean"
        }
        
        if distance_metric.lower() not in metric_map:
            raise ValueError(f"Unsupported distance metric: {distance_metric}. "
                             f"Supported metrics: {', '.join(metric_map.keys())}")
        
        weaviate_metric = metric_map[distance_metric.lower()]
        
        try:
            # Create class configuration
            class_config = {
                "class": class_name,
                "vectorizer": "none",  # We'll add vectors directly
                "vectorIndexConfig": {
                    "distance": weaviate_metric,
                    "vectorCacheMaxObjects": 500000,
                    "skip": False,
                    "ef": 256,
                    "efConstruction": 128,
                    "maxConnections": 64,
                    "dynamicEfMin": 100,
                    "dynamicEfMax": 500
                }
            }
            
            # Add properties
            properties = []
            
            # Add document property
            properties.append({
                "name": "document",
                "dataType": ["text"],
                "description": "The document text",
                "indexInverted": True
            })
            
            # Include metadata properties if provided
            if metadata and "properties" in metadata:
                for prop in metadata.get("properties", []):
                    properties.append(prop)
            
            # Add properties to class config
            if properties:
                class_config["properties"] = properties
            
            # Add description if provided
            if metadata and "description" in metadata:
                class_config["description"] = metadata["description"]
            
            # Add vectorizer if provided
            if metadata and "vectorizer" in metadata:
                class_config["vectorizer"] = metadata["vectorizer"]
            
            # Add module_config if provided
            if metadata and "module_config" in metadata:
                class_config["moduleConfig"] = metadata["module_config"]
            
            # Add inverted_index_config if provided
            if metadata and "inverted_index_config" in metadata:
                class_config["invertedIndexConfig"] = metadata["inverted_index_config"]
            
            # Create the class
            self.client.schema.create_class(class_config)
            
            logger.info(f"Created Weaviate class: {class_name}")
            
            # Refresh schema
            self.schema = self.client.schema.get()
            
            return class_config
            
        except Exception as e:
            logger.error(f"Failed to create class: {str(e)}")
            raise ValueError(f"Failed to create class: {str(e)}")
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection (class) from Weaviate.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        class_name = self._get_class_name(collection_name)
        
        if not self.collection_exists(class_name):
            logger.warning(f"Class '{class_name}' does not exist")
            return False
        
        try:
            # Delete the class
            self.client.schema.delete_class(class_name)
            
            # Refresh schema
            self.schema = self.client.schema.get()
            
            logger.info(f"Deleted Weaviate class: {class_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete class: {str(e)}")
            return False
    
    def add_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> bool:
        """
        Add vectors to a collection (class) in Weaviate.
        
        Args:
            collection_name: Name of the collection
            vectors: List of vectors to add
            ids: List of IDs for the vectors
            metadata: List of metadata for the vectors
            documents: List of documents for the vectors
            batch_size: Batch size for adding vectors
            
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
        
        class_name = self._get_class_name(collection_name)
        
        if not self.collection_exists(class_name):
            raise ValueError(f"Class '{class_name}' does not exist")
        
        try:
            # Configure batch
            with self.client.batch as batch:
                batch.batch_size = batch_size
                
                # Add objects in batch
                for i, (vector, id_str) in enumerate(zip(vectors, ids)):
                    # Prepare properties
                    properties = {}
                    
                    # Add metadata if provided
                    if metadata and i < len(metadata):
                        for key, value in metadata[i].items():
                            properties[key] = value
                    
                    # Add document if provided
                    if documents and i < len(documents):
                        properties["document"] = documents[i]
                    
                    # Generate UUID from id_str if it's not a valid UUID
                    try:
                        obj_uuid = uuid.UUID(id_str)
                    except ValueError:
                        # Create a deterministic UUID from the string
                        obj_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, id_str)
                    
                    # Add the object
                    batch.add_data_object(
                        data_object=properties,
                        class_name=class_name,
                        uuid=str(obj_uuid),
                        vector=vector
                    )
            
            logger.info(f"Added {len(vectors)} vectors to Weaviate class: {class_name}")
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
        Get a vector from a collection (class) in Weaviate.
        
        Args:
            collection_name: Name of the collection
            id: ID of the vector to get
            
        Returns:
            Tuple[List[float], Dict[str, Any]]: Vector and metadata
            
        Raises:
            ValueError: If vector does not exist
        """
        self._check_connection()
        
        class_name = self._get_class_name(collection_name)
        
        if not self.collection_exists(class_name):
            raise ValueError(f"Class '{class_name}' does not exist")
        
        try:
            # Try to parse id as UUID
            try:
                obj_uuid = uuid.UUID(id)
            except ValueError:
                # Create a deterministic UUID from the string
                obj_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, id)
            
            # Get the object
            result = self.client.data_object.get_by_id(
                class_name=class_name,
                uuid=str(obj_uuid),
                with_vector=True
            )
            
            if not result:
                raise ValueError(f"Object with ID '{id}' not found")
            
            # Extract vector and properties
            vector = result.get("vector", [])
            
            # Remove system properties
            properties = {
                k: v for k, v in result.items() 
                if k not in ["id", "vector", "class", "creationTimeUnix", "lastUpdateTimeUnix"]
            }
            
            return vector, properties
            
        except Exception as e:
            logger.error(f"Failed to get vector: {str(e)}")
            raise ValueError(f"Failed to get vector: {str(e)}")
    
    def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """
        Delete vectors from a collection (class) in Weaviate.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        class_name = self._get_class_name(collection_name)
        
        if not self.collection_exists(class_name):
            logger.warning(f"Class '{class_name}' does not exist")
            return False
        
        try:
            # Delete objects in batch
            with self.client.batch as batch:
                for id_str in ids:
                    # Try to parse id as UUID
                    try:
                        obj_uuid = uuid.UUID(id_str)
                    except ValueError:
                        # Create a deterministic UUID from the string
                        obj_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, id_str)
                    
                    # Delete the object
                    batch.delete_objects(class_name=class_name, where={"path": ["id"], "operator": "Equal", "valueString": str(obj_uuid)})
            
            logger.info(f"Deleted {len(ids)} vectors from Weaviate class: {class_name}")
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
        Search for similar vectors in a collection (class) in Weaviate.
        
        Args:
            collection_name: Name of the collection
            query_vector: Vector to search for
            top_k: Number of results to return
            filter: Filter for the search
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        self._check_connection()
        
        class_name = self._get_class_name(collection_name)
        
        if not self.collection_exists(class_name):
            logger.warning(f"Class '{class_name}' does not exist")
            return []
        
        try:
            # Configure query
            query = self.client.query.get(class_name, ["document", "id"])
            
            # Add vector search
            query = query.with_near_vector({
                "vector": query_vector,
                "certainty": 0.7  # Default certainty threshold
            })
            
            # Add limit
            query = query.with_limit(top_k)
            
            # Add filter if provided
            if filter:
                where_filter = self._convert_filter_to_weaviate(filter)
                if where_filter:
                    query = query.with_where(where_filter)
            
            # Execute query
            result = query.do()
            
            # Format the results
            formatted_results = []
            if "data" in result and "Get" in result["data"] and class_name in result["data"]["Get"]:
                results = result["data"]["Get"][class_name]
                
                for item in results:
                    formatted_result = {
                        "id": item.get("id", ""),
                        "score": item.get("_additional", {}).get("certainty", 0.0),
                        "metadata": {
                            k: v for k, v in item.items() 
                            if k not in ["id", "_additional", "vector"]
                        }
                    }
                    
                    formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {str(e)}")
            return []
    
    def _convert_filter_to_weaviate(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a generic filter to a Weaviate filter.
        
        Args:
            filter: Generic filter
            
        Returns:
            Dict[str, Any]: Weaviate filter
        """
        if not filter:
            return {}
        
        # Simple filter conversion for common operators
        if "field" in filter and "operator" in filter and "value" in filter:
            # Map operators
            operator_map = {
                "eq": "Equal",
                "==": "Equal",
                "=": "Equal",
                "neq": "NotEqual",
                "!=": "NotEqual",
                "gt": "GreaterThan",
                ">": "GreaterThan",
                "gte": "GreaterThanEqual",
                ">=": "GreaterThanEqual",
                "lt": "LessThan",
                "<": "LessThan",
                "lte": "LessThanEqual",
                "<=": "LessThanEqual",
                "like": "Like",
                "contains": "ContainsAll",
                "in": "ContainsAny"
            }
            
            field = filter["field"]
            operator = filter["operator"]
            value = filter["value"]
            
            weaviate_operator = operator_map.get(operator.lower(), operator)
            
            # Determine value type
            value_field = None
            if isinstance(value, str):
                value_field = "valueString"
            elif isinstance(value, int):
                value_field = "valueInt"
            elif isinstance(value, float):
                value_field = "valueNumber"
            elif isinstance(value, bool):
                value_field = "valueBoolean"
            elif isinstance(value, list):
                value_field = "valueStringArray" if all(isinstance(v, str) for v in value) else None
            
            if value_field:
                return {
                    "path": [field],
                    "operator": weaviate_operator,
                    value_field: value
                }
        
        # Handle AND/OR operators
        if "AND" in filter or "and" in filter:
            operands = filter.get("AND", filter.get("and", []))
            return {
                "operator": "And",
                "operands": [self._convert_filter_to_weaviate(operand) for operand in operands]
            }
        
        if "OR" in filter or "or" in filter:
            operands = filter.get("OR", filter.get("or", []))
            return {
                "operator": "Or",
                "operands": [self._convert_filter_to_weaviate(operand) for operand in operands]
            }
        
        return filter  # Return the original filter if we can't convert it
    
    def count_vectors(
        self,
        collection_name: str,
        filter: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count vectors in a collection (class) in Weaviate.
        
        Args:
            collection_name: Name of the collection
            filter: Filter for counting
            
        Returns:
            int: Number of vectors
        """
        self._check_connection()
        
        class_name = self._get_class_name(collection_name)
        
        if not self.collection_exists(class_name):
            logger.warning(f"Class '{class_name}' does not exist")
            return 0
        
        try:
            # Configure query
            query = self.client.query.aggregate(class_name).with_meta_count()
            
            # Add filter if provided
            if filter:
                where_filter = self._convert_filter_to_weaviate(filter)
                if where_filter:
                    query = query.with_where(where_filter)
            
            # Execute query
            result = query.do()
            
            # Extract count
            count = 0
            if "data" in result and "Aggregate" in result["data"] and class_name in result["data"]["Aggregate"]:
                count = result["data"]["Aggregate"][class_name][0]["meta"]["count"]
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to count vectors: {str(e)}")
            return 0 