"""
PostgreSQL Connector with pgvector

This module provides a connector for PostgreSQL vector database with pgvector extension.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union

from .base_connector import BaseVectorDBConnector

# Configure logging
logger = logging.getLogger(__name__)

class PostgresVectorConnector(BaseVectorDBConnector):
    """
    Connector for PostgreSQL vector database with pgvector extension.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        schema: str = "vector_store",
        ssl_mode: Optional[str] = None,
        timeout: int = 10
    ):
        """
        Initialize PostgreSQL vector connector.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            schema: Schema for vector tables
            ssl_mode: SSL mode for connection
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema
        self.ssl_mode = ssl_mode
        self.timeout = timeout
        self.conn = None
        self.cursor = None
    
    def connect(self, **kwargs) -> bool:
        """
        Connect to PostgreSQL.
        
        Args:
            **kwargs: Additional connection parameters
            
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            import psycopg2
            import psycopg2.extras
            
            # Override connection parameters if provided
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            
            # Set connection parameters
            conn_params = {
                "host": self.host,
                "port": self.port,
                "dbname": self.database,
                "user": self.user,
                "password": self.password,
                "connect_timeout": self.timeout
            }
            
            if self.ssl_mode:
                conn_params["sslmode"] = self.ssl_mode
            
            # Connect to database
            self.conn = psycopg2.connect(**conn_params)
            self.conn.autocommit = True
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            # Check if pgvector extension exists
            self.cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            if not self.cursor.fetchone()[0]:
                logger.warning("pgvector extension not found in database. Attempting to create...")
                self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Ensure schema exists
            self.cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            
            # Check for collections table
            self.cursor.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = '{self.schema}'
                    AND table_name = 'collections'
                )
            """)
            
            if not self.cursor.fetchone()[0]:
                logger.info("Collections table not found. Creating...")
                self._create_collections_table()
            
            logger.info(f"Connected to PostgreSQL database at {self.host}:{self.port}/{self.database}")
            return True
        
        except ImportError:
            logger.error("psycopg2 package not installed. Install with: pip install psycopg2-binary")
            return False
        
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            return False
    
    def _create_collections_table(self):
        """
        Create the collections table if it doesn't exist.
        """
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.collections (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                dimension INTEGER NOT NULL,
                distance_metric VARCHAR(50) NOT NULL DEFAULT 'cosine',
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
    
    def disconnect(self) -> bool:
        """
        Disconnect from PostgreSQL.
        
        Returns:
            bool: True if disconnection was successful, False otherwise
        """
        try:
            if self.cursor:
                self.cursor.close()
            
            if self.conn:
                self.conn.close()
            
            self.cursor = None
            self.conn = None
            
            return True
        
        except Exception as e:
            logger.error(f"Error disconnecting from PostgreSQL: {str(e)}")
            return False
    
    def is_connected(self) -> bool:
        """
        Check if connected to PostgreSQL.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.conn or not self.cursor:
            return False
        
        try:
            # Simple query to check connection
            self.cursor.execute("SELECT 1")
            return True
        
        except Exception:
            return False
    
    def _check_connection(self):
        """
        Check if connected to PostgreSQL and raise exception if not.
        
        Raises:
            ConnectionError: If not connected to PostgreSQL
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to PostgreSQL. Call connect() first.")
    
    def _collection_table_name(self, collection_name: str) -> str:
        """
        Get the name of the table for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            str: Table name for the collection
        """
        # Sanitize collection name for table name
        sanitized = collection_name.replace(" ", "_").replace("-", "_").lower()
        return f"{self.schema}.vectors_{sanitized}"
    
    def list_collections(self) -> List[str]:
        """
        List all collections in PostgreSQL.
        
        Returns:
            List[str]: List of collection names
        """
        self._check_connection()
        
        try:
            self.cursor.execute(f"SELECT name FROM {self.schema}.collections ORDER BY name")
            return [row[0] for row in self.cursor.fetchall()]
        
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in PostgreSQL.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        self._check_connection()
        
        try:
            self.cursor.execute(f"""
                SELECT EXISTS(
                    SELECT 1 FROM {self.schema}.collections 
                    WHERE name = %s
                )
            """, (collection_name,))
            
            return self.cursor.fetchone()[0]
        
        except Exception as e:
            logger.error(f"Error checking if collection exists: {str(e)}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection in PostgreSQL.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict[str, Any]: Collection information
        """
        self._check_connection()
        
        try:
            # Get collection info from collections table
            self.cursor.execute(f"""
                SELECT id, name, dimension, distance_metric, metadata, 
                       created_at, updated_at
                FROM {self.schema}.collections
                WHERE name = %s
            """, (collection_name,))
            
            result = self.cursor.fetchone()
            
            if not result:
                return {"name": collection_name, "error": "Collection does not exist"}
            
            # Get vector count from collection table
            table_name = self._collection_table_name(collection_name)
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self.cursor.fetchone()[0]
            
            # Convert result to dict
            info = dict(result)
            info["count"] = count
            
            # Handle date serialization
            for k, v in info.items():
                if hasattr(v, 'isoformat'):  # For datetime objects
                    info[k] = v.isoformat()
            
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
        Create a new collection in PostgreSQL.
        
        Args:
            collection_name: Name of the collection to create
            dimension: Dimensionality of vectors to store
            distance_metric: Distance metric to use ("cosine", "euclidean", "dot")
            metadata: Optional metadata to associate with the collection
            
        Returns:
            Any: Collection ID
        """
        self._check_connection()
        
        # Validate distance metric
        valid_metrics = ["cosine", "euclidean", "dot"]
        if distance_metric not in valid_metrics:
            raise ValueError(f"Distance metric must be one of {valid_metrics}")
        
        # Check if collection already exists
        if self.collection_exists(collection_name):
            # Delete existing collection
            self.delete_collection(collection_name)
        
        table_name = self._collection_table_name(collection_name)
        
        try:
            # Start transaction
            self.cursor.execute("BEGIN")
            
            # Insert collection record
            self.cursor.execute(f"""
                INSERT INTO {self.schema}.collections
                (name, dimension, distance_metric, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (collection_name, dimension, distance_metric, json.dumps(metadata) if metadata else None))
            
            collection_id = self.cursor.fetchone()[0]
            
            # Create table for collection vectors
            self.cursor.execute(f"""
                CREATE TABLE {table_name} (
                    id SERIAL PRIMARY KEY,
                    external_id VARCHAR(255) UNIQUE NOT NULL,
                    embedding vector({dimension}) NOT NULL,
                    metadata JSONB,
                    document TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Create vector index based on distance metric
            if distance_metric == "cosine":
                self.cursor.execute(f"""
                    CREATE INDEX ON {table_name}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            elif distance_metric == "euclidean":
                self.cursor.execute(f"""
                    CREATE INDEX ON {table_name}
                    USING ivfflat (embedding vector_l2_ops)
                    WITH (lists = 100)
                """)
            else:  # dot product
                self.cursor.execute(f"""
                    CREATE INDEX ON {table_name}
                    USING ivfflat (embedding vector_ip_ops)
                    WITH (lists = 100)
                """)
            
            # Create index on external_id
            self.cursor.execute(f"""
                CREATE INDEX ON {table_name} (external_id)
            """)
            
            # Commit transaction
            self.cursor.execute("COMMIT")
            
            logger.info(f"Created collection '{collection_name}' in PostgreSQL")
            
            return collection_id
        
        except Exception as e:
            # Rollback transaction
            self.cursor.execute("ROLLBACK")
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from PostgreSQL.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection '{collection_name}' does not exist")
            return True
        
        table_name = self._collection_table_name(collection_name)
        
        try:
            # Start transaction
            self.cursor.execute("BEGIN")
            
            # Drop the collection table
            self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Delete collection record
            self.cursor.execute(f"""
                DELETE FROM {self.schema}.collections
                WHERE name = %s
            """, (collection_name,))
            
            # Commit transaction
            self.cursor.execute("COMMIT")
            
            logger.info(f"Deleted collection '{collection_name}' from PostgreSQL")
            
            return True
        
        except Exception as e:
            # Rollback transaction
            self.cursor.execute("ROLLBACK")
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
        Add vectors to a collection in PostgreSQL.
        
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
        
        table_name = self._collection_table_name(collection_name)
        
        try:
            # Get vectors info
            num_vectors = len(vectors)
            if len(ids) != num_vectors:
                raise ValueError("Number of vectors and IDs must match")
            
            # Prepare metadata and documents
            if metadata is None:
                metadata = [None] * num_vectors
            
            if documents is None:
                documents = [None] * num_vectors
            
            # Start transaction
            self.cursor.execute("BEGIN")
            
            import psycopg2.extras
            
            # Prepare values for batch insert
            values = []
            for i in range(num_vectors):
                values.append((
                    ids[i],
                    vectors[i],
                    json.dumps(metadata[i]) if metadata[i] else None,
                    documents[i]
                ))
            
            # Execute batch insert
            psycopg2.extras.execute_values(
                self.cursor,
                f"""
                INSERT INTO {table_name}
                (external_id, embedding, metadata, document)
                VALUES %s
                ON CONFLICT (external_id) 
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    document = EXCLUDED.document,
                    updated_at = NOW()
                """,
                values,
                template="(%s, %s::vector, %s, %s)"
            )
            
            # Commit transaction
            self.cursor.execute("COMMIT")
            
            logger.info(f"Added {len(vectors)} vectors to collection '{collection_name}' in PostgreSQL")
            
            return True
        
        except Exception as e:
            # Rollback transaction
            self.cursor.execute("ROLLBACK")
            logger.error(f"Error adding vectors: {str(e)}")
            return False
    
    def get_vector(
        self,
        collection_name: str,
        id: str
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Get a vector and its metadata by ID from PostgreSQL.
        
        Args:
            collection_name: Name of the collection
            id: ID of the vector to get
            
        Returns:
            Tuple[List[float], Dict[str, Any]]: Tuple of (vector, metadata)
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        table_name = self._collection_table_name(collection_name)
        
        try:
            self.cursor.execute(f"""
                SELECT embedding, metadata, document
                FROM {table_name}
                WHERE external_id = %s
            """, (id,))
            
            result = self.cursor.fetchone()
            
            if not result:
                raise ValueError(f"Vector with ID '{id}' not found")
            
            # Convert vector to list
            vector = list(result["embedding"])
            
            # Prepare metadata
            metadata = {}
            
            if result["metadata"]:
                metadata = result["metadata"]
            
            if result["document"]:
                metadata["document"] = result["document"]
            
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
        Delete vectors from a collection in PostgreSQL.
        
        Args:
            collection_name: Name of the collection
            ids: List of IDs of vectors to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        table_name = self._collection_table_name(collection_name)
        
        try:
            # Convert ids to tuple for SQL
            ids_tuple = tuple(ids)
            
            # Handle single id case
            if len(ids) == 1:
                ids_clause = f"('{ids[0]}')"
            else:
                ids_clause = str(ids_tuple)
            
            # Delete vectors
            self.cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE external_id IN {ids_clause}
            """)
            
            count = self.cursor.rowcount
            
            logger.info(f"Deleted {count} vectors from collection '{collection_name}' in PostgreSQL")
            
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
        Search for similar vectors in a collection in PostgreSQL.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            top_k: Number of results to return
            filter: Optional filter to apply (as a JSON object)
            
        Returns:
            List[Dict[str, Any]]: List of search results with id, score, and metadata
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        table_name = self._collection_table_name(collection_name)
        
        try:
            # Get distance metric for collection
            self.cursor.execute(f"""
                SELECT distance_metric FROM {self.schema}.collections
                WHERE name = %s
            """, (collection_name,))
            
            distance_metric = self.cursor.fetchone()["distance_metric"]
            
            # Choose distance operator based on metric
            if distance_metric == "cosine":
                distance_op = "<->"
            elif distance_metric == "euclidean":
                distance_op = "<->"
            else:  # dot product
                distance_op = "<#>"
            
            # Prepare filter clause if needed
            filter_clause = ""
            if filter:
                filter_conditions = []
                for key, value in filter.items():
                    filter_conditions.append(f"metadata->'{key}' = '{json.dumps(value)}'")
                
                if filter_conditions:
                    filter_clause = "AND " + " AND ".join(filter_conditions)
            
            # Perform search
            query = f"""
                SELECT 
                    external_id,
                    embedding {distance_op} %s::vector AS distance,
                    metadata,
                    document
                FROM {table_name}
                WHERE TRUE {filter_clause}
                ORDER BY distance
                LIMIT %s
            """
            
            self.cursor.execute(query, (query_vector, top_k))
            
            results = []
            for row in self.cursor.fetchall():
                # For cosine and euclidean, lower distance is better
                # For dot product, higher is better
                score = 1.0 - float(row["distance"]) if distance_metric in ["cosine", "euclidean"] else float(row["distance"])
                
                result = {
                    "id": row["external_id"],
                    "score": score
                }
                
                if row["metadata"]:
                    result["metadata"] = row["metadata"]
                
                if row["document"]:
                    result["document"] = row["document"]
                
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
        Count vectors in a collection in PostgreSQL.
        
        Args:
            collection_name: Name of the collection
            filter: Optional filter to apply
            
        Returns:
            int: Number of vectors
        """
        self._check_connection()
        
        if not self.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        table_name = self._collection_table_name(collection_name)
        
        try:
            # Prepare filter clause if needed
            filter_clause = ""
            if filter:
                filter_conditions = []
                for key, value in filter.items():
                    filter_conditions.append(f"metadata->'{key}' = '{json.dumps(value)}'")
                
                if filter_conditions:
                    filter_clause = "WHERE " + " AND ".join(filter_conditions)
            
            # Count vectors
            query = f"SELECT COUNT(*) FROM {table_name} {filter_clause}"
            self.cursor.execute(query)
            
            return self.cursor.fetchone()[0]
        
        except Exception as e:
            logger.error(f"Error counting vectors: {str(e)}")
            return 0 