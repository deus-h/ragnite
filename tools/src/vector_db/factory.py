"""
Vector Database Connector Factory

This module provides a factory function to get a database connector based on the database type.
"""

import logging
from typing import Dict, Any, Optional, Union

from .base_connector import BaseVectorDBConnector
from .chromadb_connector import ChromaDBConnector
from .postgres_connector import PostgresVectorConnector
from .qdrant_connector import QdrantConnector
from .pinecone_connector import PineconeConnector
from .weaviate_connector import WeaviateConnector
from .milvus_connector import MilvusConnector
from .grok_connector import GrokConnector

# Configure logging
logger = logging.getLogger(__name__)

def get_database_connector(
    db_type: str,
    connection_params: Optional[Dict[str, Any]] = None,
    auto_connect: bool = True
) -> BaseVectorDBConnector:
    """
    Get a database connector based on the database type.
    
    Args:
        db_type: Type of database ("chromadb", "postgres", "qdrant", "pinecone", "weaviate", "milvus", "grok")
        connection_params: Connection parameters for the database
        auto_connect: Whether to automatically connect to the database
        
    Returns:
        BaseVectorDBConnector: Database connector
        
    Raises:
        ValueError: If the database type is not supported
    """
    if connection_params is None:
        connection_params = {}
    
    # Normalize database type
    db_type = db_type.lower().strip()
    
    # Create connector based on database type
    if db_type in ["chromadb", "chroma"]:
        connector = ChromaDBConnector(**connection_params)
    
    elif db_type in ["postgres", "postgresql", "pgvector"]:
        connector = PostgresVectorConnector(**connection_params)
    
    elif db_type in ["qdrant"]:
        connector = QdrantConnector(**connection_params)
    
    elif db_type in ["pinecone"]:
        connector = PineconeConnector(**connection_params)
    
    elif db_type in ["weaviate"]:
        connector = WeaviateConnector(**connection_params)
    
    elif db_type in ["milvus"]:
        connector = MilvusConnector(**connection_params)
    
    elif db_type in ["grok", "xai"]:
        connector = GrokConnector(**connection_params)
    
    else:
        supported_dbs = ["chromadb", "postgres", "qdrant", "pinecone", "weaviate", "milvus", "grok"]
        raise ValueError(f"Unsupported database type: {db_type}. Supported types: {supported_dbs}")
    
    # Connect to database if auto_connect is True
    if auto_connect:
        success = connector.connect()
        if not success:
            logger.warning(f"Failed to connect to {db_type} database. Call connect() with appropriate parameters.")
    
    return connector 