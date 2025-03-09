#!/usr/bin/env python3
"""
Environment Setup Check Script

This script checks if either:
1. At least one LLM API key (OpenAI, Anthropic, Mistral) is available, or
2. Ollama is running with the specified model

If neither condition is met, it prompts the user to add at least one API key
or ensure Ollama is running.
"""

import os
import sys
import logging
from pathlib import Path

# Try to import requests, but don't fail if not available
try:
    import requests
    has_requests = True
except ImportError:
    has_requests = False

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("setup_env_check")

def parse_env_file(env_path):
    """Simple function to parse .env file without external dependencies"""
    env_vars = {}
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Extract key and value
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    except Exception as e:
        logger.error(f"Error parsing .env file: {e}")
    
    return env_vars

def check_api_keys(env_vars):
    """Check if any API keys are available in the env vars"""
    api_providers = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY", "GOOGLE_API_KEY"]
    available_keys = []
    
    for key in api_providers:
        value = env_vars.get(key)
        # Check if key exists and isn't the template placeholder
        if value and not value.startswith("your_") and value != "":
            available_keys.append(key)
    
    return available_keys

def check_ollama(env_vars):
    """Check if Ollama is running and has the specified model"""
    if not has_requests:
        logger.warning("❌ Python requests package not installed. Cannot check Ollama.")
        logger.info("To install: pip install requests")
        return False

    try:
        ollama_host = env_vars.get("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = env_vars.get("OLLAMA_MODEL", "llama3")
        
        # Make request to Ollama API
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if ollama_model in model_names:
                logger.info(f"✅ Ollama is running with the required model: {ollama_model}")
                return True
            else:
                logger.warning(f"❌ Ollama is running but the required model '{ollama_model}' is not available")
                logger.info(f"Available models: {', '.join(model_names)}")
                return False
        else:
            logger.warning(f"❌ Ollama API returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"❌ Cannot connect to Ollama: {e}")
        return False

def main():
    """Main function to check environment setup"""
    # Load .env file
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        logger.error(f"❌ .env file not found at {env_path}")
        sys.exit(1)
    
    # Parse .env file
    env_vars = parse_env_file(env_path)
    
    # Check for API keys
    available_keys = check_api_keys(env_vars)
    has_api_keys = len(available_keys) > 0
    
    # Check for Ollama
    has_ollama = check_ollama(env_vars)
    
    # Provide feedback based on checks
    if has_api_keys:
        logger.info(f"✅ Found {len(available_keys)} valid API key(s): {', '.join(available_keys)}")
    
    # Overall status
    if has_api_keys or has_ollama:
        logger.info("✅ Environment setup successful! You have either valid API keys or Ollama configured.")
    else:
        logger.warning("❌ Neither valid API keys nor a running Ollama instance with the required model were found.")
        logger.info("\nPlease do ONE of the following:")
        logger.info("1. Add at least one API key to your .env file:")
        logger.info("   - OPENAI_API_KEY")
        logger.info("   - ANTHROPIC_API_KEY")
        logger.info("   - MISTRAL_API_KEY")
        logger.info("   - XAI_API_KEY")
        logger.info("   - GOOGLE_API_KEY")
        logger.info("\n2. OR start Ollama with the required model:")
        ollama_model = env_vars.get("OLLAMA_MODEL", "llama3")
        logger.info(f"   - Run: ollama pull {ollama_model}")
        logger.info(f"   - Run: ollama serve")

if __name__ == "__main__":
    main() 