#!/usr/bin/env python3
"""
Google AI (Gemini) Provider Example

This example demonstrates how to use the Google AI provider with RAGNITE.
"""

import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import environment loader
from utils.env_loader import load_env, get_google_config

# Import model factory and message classes
from tools.src.models.base_model import Message, Role
from tools.src.models.model_factory import get_model_provider


def main():
    """Main function demonstrating Google AI (Gemini) provider usage."""
    # Load environment variables
    load_env()
    
    # Get Google AI configuration
    google_config = get_google_config()
    
    # Check if API key is set
    if not google_config.get("api_key"):
        logger.error("Google API key is not set. Please set GOOGLE_API_KEY in your .env file.")
        sys.exit(1)
    
    # Create Google AI provider
    try:
        provider = get_model_provider(
            "google",
            api_key=google_config["api_key"],
            model="gemini-pro",  # Use Gemini Pro by default
        )
        logger.info("Successfully created Google AI provider")
    except ImportError as e:
        logger.error(f"Failed to create Google AI provider: {e}")
        logger.error("Make sure to install the google-generativeai package: pip install google-generativeai")
        sys.exit(1)
    
    # Create a conversation
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful AI assistant that provides accurate and concise information."),
        Message(role=Role.USER, content="What is Retrieval-Augmented Generation (RAG) and why is it useful?"),
    ]
    
    # Generate a response
    try:
        logger.info("Generating response...")
        response = provider.generate(messages, temperature=0.7)
        
        # Print the response
        print("\n" + "="*80)
        print("Google AI (Gemini) Response:")
        print("="*80)
        print(response["content"])
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        sys.exit(1)
    
    # Demonstrate streaming (optional)
    try_streaming = input("\nWould you like to see streaming in action? (y/n): ").strip().lower()
    if try_streaming.startswith('y'):
        try:
            logger.info("Generating streaming response...")
            print("\n" + "="*80)
            print("Google AI (Gemini) Streaming Response:")
            print("="*80)
            
            # Add a follow-up question
            messages.append(Message(role=Role.ASSISTANT, content=response["content"]))
            messages.append(Message(role=Role.USER, content="What are some common challenges when implementing RAG systems?"))
            
            # Stream the response
            for chunk in provider.generate_stream(messages, temperature=0.7):
                # Print the chunk without a newline
                print(chunk["content"], end="", flush=True)
                
            # Print a newline at the end
            print("\n" + "="*80)
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")


if __name__ == "__main__":
    main() 