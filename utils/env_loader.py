"""
Environment Variable Loader

This module provides utilities for loading environment variables for the RAGNITE project.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_env(env_file: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file. If None, will look for .env in current 
                 directory and parent directories.
                 
    Returns:
        Dictionary of environment variables
    """
    if env_file is None:
        # Look for .env in current directory and parent directories
        current_dir = Path.cwd()
        env_path = None
        
        # Check current directory and up to 3 parent directories
        for _ in range(4):
            test_path = current_dir / ".env"
            if test_path.exists():
                env_path = test_path
                break
            current_dir = current_dir.parent
        
        if env_path is None:
            logger.warning("No .env file found. Using environment variables from the system.")
            return dict(os.environ)
        
        load_dotenv(env_path)
    else:
        env_path = Path(env_file)
        if not env_path.exists():
            logger.warning(f".env file not found at {env_path}. Using environment variables from the system.")
            return dict(os.environ)
        
        load_dotenv(env_path)
    
    logger.info(f"Loaded environment variables from {env_path}")
    return dict(os.environ)

def get_env(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Get environment variable with type conversion.
    
    Args:
        key: Environment variable name
        default: Default value if environment variable is not set
        required: Whether the environment variable is required
        
    Returns:
        Environment variable value or default
        
    Raises:
        ValueError: If environment variable is required but not set
    """
    value = os.environ.get(key)
    
    if value is None:
        if required:
            raise ValueError(f"Required environment variable {key} is not set")
        return default
    
    # Convert boolean strings
    if value.lower() in ('true', 'yes', '1', 'y'):
        return True
    if value.lower() in ('false', 'no', '0', 'n'):
        return False
    
    # Convert integer strings
    try:
        return int(value)
    except ValueError:
        pass
    
    # Convert float strings
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value

def get_api_key(provider: str) -> str:
    """
    Get API key for a specific provider.
    
    Args:
        provider: Provider name (openai, anthropic, cohere, etc.)
        
    Returns:
        API key for the provider
        
    Raises:
        ValueError: If API key is not set
    """
    provider = provider.upper()
    key_name = f"{provider}_API_KEY"
    
    api_key = get_env(key_name)
    if not api_key:
        raise ValueError(f"API key for {provider} is not set. Please set {key_name} environment variable.")
    
    return api_key

def get_database_config(database: str) -> Dict[str, Any]:
    """
    Get configuration for a specific vector database.
    
    Args:
        database: Database name (chromadb, postgres, qdrant, etc.)
        
    Returns:
        Dictionary of database configuration parameters
    """
    database = database.upper()
    
    if database == "CHROMADB":
        return {
            "host": get_env("CHROMADB_HOST", "localhost"),
            "port": get_env("CHROMADB_PORT", 8000),
            "collection": get_env("CHROMADB_COLLECTION", "ragnite"),
        }
    elif database == "POSTGRES":
        return {
            "host": get_env("POSTGRES_HOST", "localhost"),
            "port": get_env("POSTGRES_PORT", 5432),
            "user": get_env("POSTGRES_USER", "postgres"),
            "password": get_env("POSTGRES_PASSWORD", "postgres"),
            "database": get_env("POSTGRES_DB", "vectordb"),
        }
    elif database == "QDRANT":
        return {
            "host": get_env("QDRANT_HOST", "localhost"),
            "http_port": get_env("QDRANT_HTTP_PORT", 6333),
            "grpc_port": get_env("QDRANT_GRPC_PORT", 6334),
            "collection": get_env("QDRANT_COLLECTION", "ragnite"),
        }
    elif database == "PINECONE":
        return {
            "api_key": get_env("PINECONE_API_KEY"),
            "environment": get_env("PINECONE_ENV"),
            "index": get_env("PINECONE_INDEX", "ragnite"),
        }
    elif database == "WEAVIATE":
        return {
            "url": get_env("WEAVIATE_URL", "http://localhost:8080"),
            "api_key": get_env("WEAVIATE_API_KEY"),
            "class_name": get_env("WEAVIATE_CLASS", "RAGDocument"),
        }
    else:
        raise ValueError(f"Unknown database: {database}")

def get_cache_config() -> Dict[str, Any]:
    """
    Get cache configuration.
    
    Returns:
        Dictionary of cache configuration parameters
    """
    return {
        "cache_dir": get_env("CACHE_DIR", "~/.ragnite/cache"),
        "ttl": get_env("CACHE_TTL", 86400),
        "embedding_cache_enabled": get_env("EMBEDDING_CACHE_ENABLED", True),
        "result_cache_enabled": get_env("RESULT_CACHE_ENABLED", True),
        "semantic_cache_enabled": get_env("SEMANTIC_CACHE_ENABLED", True),
        "prompt_cache_enabled": get_env("PROMPT_CACHE_ENABLED", True),
        "dashboard_port": get_env("CACHE_DASHBOARD_PORT", 8088),
    }

def get_logging_config() -> Dict[str, Any]:
    """
    Get logging configuration.
    
    Returns:
        Dictionary of logging configuration parameters
    """
    return {
        "level": get_env("LOG_LEVEL", "INFO"),
        "monitoring_enabled": get_env("MONITORING_ENABLED", True),
        "monitoring_port": get_env("MONITORING_PORT", 8089),
        "monitoring_host": get_env("MONITORING_HOST", "localhost"),
    }

def get_ollama_config() -> Dict[str, str]:
    """
    Get Ollama configuration.
    
    Returns:
        Dictionary of Ollama configuration parameters
    """
    return {
        "host": get_env("OLLAMA_HOST", "http://localhost:11434"),
        "model": get_env("OLLAMA_MODEL", "llama3"),
    }

def get_xai_config() -> Dict[str, str]:
    """
    Get xAI (Grok) configuration.
    
    Returns:
        Dictionary of xAI configuration parameters
    """
    return {
        "api_key": get_env("XAI_API_KEY", ""),
    }

def get_google_config() -> Dict[str, str]:
    """
    Get Google AI (Gemini) configuration.
    
    Returns:
        Dictionary of Google AI configuration parameters
    """
    return {
        "api_key": get_env("GOOGLE_API_KEY", ""),
        "project_id": get_env("GOOGLE_PROJECT_ID", ""),
    }

def setup_logging():
    """
    Set up logging configuration based on environment variables.
    """
    log_config = get_logging_config()
    log_level = log_config["level"]
    
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Load environment variables when module is imported
load_env()

if __name__ == "__main__":
    # Example usage
    setup_logging()
    logger.info("Environment variables loaded")
    
    # Get API key
    try:
        openai_key = get_api_key("openai")
        logger.info(f"OpenAI API key loaded: {openai_key[:5]}...")
    except ValueError as e:
        logger.error(e)
    
    # Get database config
    try:
        chroma_config = get_database_config("chromadb")
        logger.info(f"ChromaDB configuration: {chroma_config}")
    except ValueError as e:
        logger.error(e)
    
    # Get cache config
    cache_config = get_cache_config()
    logger.info(f"Cache configuration: {cache_config}") 