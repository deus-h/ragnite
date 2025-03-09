#!/usr/bin/env python3
"""
Test Environment Setup Script

This script validates the environment configuration and tests connections
to various required services.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import environment loader
from utils.env_loader import (
    load_env,
    get_env,
    get_api_key,
    get_database_config,
    get_cache_config,
    get_ollama_config,
    setup_logging,
)

# Configure logging
setup_logging()
logger = logging.getLogger("setup_test_env")

def check_api_key(provider):
    """Check if API key is available for a provider"""
    try:
        key = get_api_key(provider)
        logger.info(f"✅ {provider.upper()} API key is available")
        return True
    except ValueError as e:
        logger.warning(f"❌ {provider.upper()} API key is not available: {e}")
        return False

def check_database_connection(database):
    """Check connection to a vector database"""
    try:
        config = get_database_config(database)
        logger.info(f"⚙️ {database.upper()} configuration: {config}")
        
        # Placeholder for actual connection testing
        # In a full implementation, we would test the actual connection here
        # This would require importing specific libraries for each database
        logger.info(f"✅ {database.upper()} configuration is available")
        return True
    except Exception as e:
        logger.warning(f"❌ {database.upper()} configuration check failed: {e}")
        return False

def check_ollama():
    """Check Ollama connection"""
    try:
        import requests
        
        config = get_ollama_config()
        ollama_url = f"{config['host']}/api/tags"
        
        # Test connection to Ollama API
        response = requests.get(ollama_url)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            logger.info(f"✅ Ollama is running with models: {', '.join(model_names)}")
            
            # Check if specified model is available
            if config["model"] in model_names:
                logger.info(f"✅ Specified model '{config['model']}' is available")
            else:
                logger.warning(f"❌ Specified model '{config['model']}' is not available in Ollama")
                logger.info(f"Available models: {', '.join(model_names)}")
                
            return True
        else:
            logger.warning(f"❌ Ollama responded with status code {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"❌ Ollama connection check failed: {e}")
        return False

def check_cache_settings():
    """Check cache settings"""
    try:
        config = get_cache_config()
        logger.info(f"⚙️ Cache configuration: {config}")
        
        # Check if cache directory exists or can be created
        cache_dir = Path(os.path.expanduser(config["cache_dir"]))
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created cache directory: {cache_dir}")
            except Exception as e:
                logger.warning(f"❌ Failed to create cache directory: {e}")
                return False
        else:
            logger.info(f"✅ Cache directory exists: {cache_dir}")
        
        return True
    except Exception as e:
        logger.warning(f"❌ Cache configuration check failed: {e}")
        return False

def main():
    """Main function to run all tests"""
    logger.info("Starting environment validation...")
    
    # Check API keys
    api_providers = ["openai", "anthropic", "cohere", "mistral", "xai", "google"]
    api_results = {}
    for provider in api_providers:
        api_results[provider] = check_api_key(provider)
    
    # Check database configurations
    databases = ["chromadb", "postgres", "qdrant", "pinecone", "weaviate"]
    db_results = {}
    for db in databases:
        db_results[db] = check_database_connection(db)
    
    # Check Ollama
    ollama_result = check_ollama()
    
    # Check cache settings
    cache_result = check_cache_settings()
    
    # Print summary
    logger.info("\n\n" + "="*50)
    logger.info("Environment Validation Summary")
    logger.info("="*50)
    
    logger.info("\nAPI Keys:")
    for provider, result in api_results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {provider.upper()}")
    
    logger.info("\nDatabase Configurations:")
    for db, result in db_results.items():
        status = "✅" if result else "❌"
        logger.info(f"{status} {db.upper()}")
    
    logger.info("\nOllama:")
    status = "✅" if ollama_result else "❌"
    logger.info(f"{status} Ollama connection")
    
    logger.info("\nCache Settings:")
    status = "✅" if cache_result else "❌"
    logger.info(f"{status} Cache configuration")
    
    logger.info("\n" + "="*50)
    logger.info("Next Steps:")
    
    # Provide recommendations based on results
    next_steps = []
    
    if not any(api_results.values()):
        next_steps.append("Set at least one LLM API key in your .env file")
    
    if not any(db_results.values()):
        next_steps.append("Configure at least one vector database in your .env file")
        next_steps.append("Consider running 'docker-compose -f docker-compose.dev.yml up -d'")
    
    if not ollama_result:
        next_steps.append("Install Ollama or configure it in your .env file")
        next_steps.append("If you're using Docker, ensure the Ollama service is running")
    
    if not cache_result:
        next_steps.append("Configure cache settings in your .env file")
    
    if next_steps:
        for i, step in enumerate(next_steps, 1):
            logger.info(f"{i}. {step}")
    else:
        logger.info("Your environment is ready for developing and testing RAGNITE!")
    
    logger.info("="*50)

if __name__ == "__main__":
    main() 