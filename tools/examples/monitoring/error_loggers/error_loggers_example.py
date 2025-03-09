#!/usr/bin/env python3
"""
Example script demonstrating the usage of error loggers in RAG systems.

This script shows how to use the various error loggers to record, store,
and notify about errors that occur during RAG system operation.
"""

import sys
import os
import json
import datetime
import logging
import time
import random
from typing import Dict, List, Any, Optional

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import error loggers
from tools.src.monitoring.error_loggers import (
    BaseErrorLogger,
    ConsoleErrorLogger,
    FileErrorLogger,
    DatabaseErrorLogger,
    CloudErrorLogger,
    AlertErrorLogger,
    get_error_logger
)


# Define sample error classes
class RAGError(Exception):
    """Base class for RAG-specific errors."""
    pass


class RetrieverError(RAGError):
    """Error related to the retriever component."""
    pass


class EmbeddingError(RAGError):
    """Error related to the embedding component."""
    pass


class GenerationError(RAGError):
    """Error related to the generation component."""
    pass


# Helper functions
def setup_log_dir():
    """Set up log directory for examples."""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def generate_sample_errors():
    """Generate sample errors of different types."""
    errors = [
        # Simple string errors
        "Connection failed",
        "Timeout occurred while waiting for response",
        "Authorization error: Invalid API key",
        
        # Standard exceptions
        ValueError("Invalid parameter: top_k must be a positive integer"),
        TypeError("Expected dict but received list"),
        KeyError("Missing required parameter: 'query'"),
        
        # Custom RAG exceptions
        RetrieverError("Failed to retrieve documents from vector database"),
        EmbeddingError("Embedding generation failed for text input"),
        GenerationError("LLM generation failed with status code 429"),
    ]
    return errors


def generate_error_context():
    """Generate sample error context."""
    contexts = [
        {
            "component": "retriever",
            "operation": "vector_search",
            "parameters": {"query": "sample query", "top_k": 5},
            "user_id": f"user_{random.randint(1000, 9999)}"
        },
        {
            "component": "embedder",
            "operation": "generate_embeddings",
            "parameters": {"batch_size": 32, "model": "sentence-transformers/all-MiniLM-L6-v2"},
            "user_id": f"user_{random.randint(1000, 9999)}"
        },
        {
            "component": "generator",
            "operation": "generate_response",
            "parameters": {"model": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 150},
            "user_id": f"user_{random.randint(1000, 9999)}"
        }
    ]
    return random.choice(contexts)


# Example functions for each logger
def console_logger_example():
    """Example usage of ConsoleErrorLogger."""
    print("\n=== Console Error Logger Example ===")
    
    # Create a console logger
    logger = ConsoleErrorLogger(
        name="console_example",
        level=logging.ERROR,
        config={
            "use_colors": True,
            "output_stream": "stderr",
            "json_format": False
        }
    )
    
    # Log errors of different levels
    errors = generate_sample_errors()
    for error in errors:
        level = random.choice([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL])
        context = generate_error_context()
        error_id = logger.log_error(error, level, context)
        print(f"Logged error with ID: {error_id}")
    
    # Get and print filtered errors
    print("\nFiltered errors (ERROR level and above):")
    filtered_errors = logger.get_errors(level=logging.ERROR, limit=5)
    for error in filtered_errors:
        print(f"- {error['level_name']}: {error['message']} (ID: {error['error_id']})")
    
    print("\nConsole logger also supports different levels:")
    logger.log_debug("This is a debug message")
    logger.log_info("This is an info message")
    logger.log_warning("This is a warning message")
    logger.log_error("This is an error message")
    logger.log_critical("This is a critical message")


def file_logger_example():
    """Example usage of FileErrorLogger."""
    print("\n=== File Error Logger Example ===")
    
    # Set up log directory
    log_dir = setup_log_dir()
    
    # Create a file logger
    logger = FileErrorLogger(
        name="file_example",
        level=logging.WARNING,  # Only log WARNING and above
        config={
            "log_dir": log_dir,
            "log_file": "example_errors.log",
            "rotation_type": "size",
            "max_bytes": 1024 * 1024,  # 1 MB
            "backup_count": 3,
            "json_format": True
        }
    )
    
    # Log errors
    errors = generate_sample_errors()
    for error in errors:
        level = random.choice([logging.WARNING, logging.ERROR, logging.CRITICAL])
        context = generate_error_context()
        error_id = logger.log_error(error, level, context)
        print(f"Logged error with ID: {error_id}")
    
    # Get and print filtered errors
    print("\nFiltered errors (ERROR level and above):")
    try:
        filtered_errors = logger.get_errors(level=logging.ERROR, limit=5)
        for error in filtered_errors:
            print(f"- {error['level_name']}: {error['message']} (ID: {error['error_id']})")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Print log file location
    log_file = os.path.join(log_dir, "example_errors.log")
    print(f"\nErrors logged to file: {log_file}")
    
    # Show file content (first few lines)
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print("\nLog file content (first 3 lines):")
            for line in lines[:3]:
                print(line.strip())
    except Exception as e:
        print(f"Error reading log file: {e}")


def database_logger_example():
    """Example usage of DatabaseErrorLogger."""
    print("\n=== Database Error Logger Example ===")
    print("Note: This example uses SQLite for demonstration purposes.")
    
    # Set up log directory
    log_dir = setup_log_dir()
    
    # Create a database logger
    db_path = os.path.join(log_dir, "errors.db")
    logger = None
    
    try:
        logger = DatabaseErrorLogger(
            name="db_example",
            level=logging.ERROR,
            config={
                "db_type": "sqlite",
                "db_path": db_path,
                "table_name": "error_logs",
                "auto_create_table": True
            }
        )
        
        # Log errors
        errors = generate_sample_errors()
        for error in errors[:5]:  # Log fewer errors for database example
            level = random.choice([logging.ERROR, logging.CRITICAL])
            context = generate_error_context()
            error_id = logger.log_error(error, level, context)
            print(f"Logged error with ID: {error_id}")
        
        # Get and print filtered errors
        print("\nFiltered errors from database:")
        filtered_errors = logger.get_errors(limit=3)
        for error in filtered_errors:
            print(f"- {error['level_name']}: {error['message']} (ID: {error['error_id']})")
        
        print(f"\nErrors logged to SQLite database: {db_path}")
        print("You can connect to this database using any SQLite client.")
        print("For example: sqlite3 <db_path>")
        print("Then query the table: SELECT * FROM error_logs;")
    
    except Exception as e:
        print(f"Error in database logger example: {e}")
    
    finally:
        # Close database connection
        if logger and hasattr(logger, 'close'):
            logger.close()


def cloud_logger_example():
    """Example usage of CloudErrorLogger."""
    print("\n=== Cloud Error Logger Example ===")
    print("Note: This example simulates cloud logging using local fallback.")
    
    # Set up log directory
    log_dir = setup_log_dir()
    
    # Create a cloud logger (with local fallback)
    logger = CloudErrorLogger(
        name="cloud_example",
        level=logging.ERROR,
        config={
            "cloud_provider": "aws",  # This would be a real provider in production
            "aws_region": "us-east-1",
            "aws_log_group": "rag_errors",
            "aws_log_stream": "example_stream",
            "local_fallback": True,
            "local_log_dir": log_dir
        }
    )
    
    # Log errors
    errors = generate_sample_errors()
    for error in errors[:3]:  # Log fewer errors for cloud example
        level = random.choice([logging.ERROR, logging.CRITICAL])
        context = generate_error_context()
        
        # Since we likely don't have actual cloud credentials, this will use fallback
        try:
            error_id = logger.log_error(error, level, context)
            print(f"Logged error with ID: {error_id}")
        except Exception as e:
            print(f"Error logging to cloud: {e}")
            print("This is expected without real cloud credentials.")
            print("Errors should be written to local fallback file.")
    
    # Check local fallback file
    fallback_file = None
    for file in os.listdir(log_dir):
        if file.startswith("cloud_error_fallback_"):
            fallback_file = os.path.join(log_dir, file)
            break
    
    if fallback_file:
        print(f"\nErrors logged to local fallback file: {fallback_file}")
        try:
            with open(fallback_file, 'r') as f:
                lines = f.readlines()
                print("\nFallback file content (first 2 lines):")
                for line in lines[:2]:
                    print(line.strip())
        except Exception as e:
            print(f"Error reading fallback file: {e}")
    else:
        print("\nNo fallback file found. Check if any errors were logged.")


def alert_logger_example():
    """Example usage of AlertErrorLogger."""
    print("\n=== Alert Error Logger Example ===")
    print("Note: This example simulates alert configuration without sending actual alerts.")
    
    # Create an alert logger
    logger = AlertErrorLogger(
        name="alert_example",
        level=logging.WARNING,
        config={
            "alert_channel": "email",  # Would be real channel in production
            "alert_threshold": logging.CRITICAL,  # Only CRITICAL errors trigger alerts
            "alert_cooldown": 60,  # 1 minute between alerts
            "alert_conditions": {
                "error_types": ["RetrieverError", "GenerationError"],
                "error_patterns": [r"timeout", r"failed", r"error"]
            },
            # Email configuration (would be real in production)
            "email_smtp_server": "smtp.example.com",
            "email_from": "alerts@example.com",
            "email_to": ["admin@example.com"]
        }
    )
    
    # Log some errors (some will match alert conditions, others won't)
    errors = [
        "Simple warning message",  # Not critical, no alert
        ValueError("Invalid value"),  # Not critical, no alert
        GenerationError("LLM generation failed with timeout"),  # Critical + matches conditions
        RetrieverError("Retrieval operation failed"),  # Critical + matches conditions
        KeyError("Missing key"),  # Critical but wrong type, no alert
        "Critical system error that doesn't match patterns"  # Critical but no pattern match
    ]
    
    # Log errors with explanations
    for error in errors:
        if isinstance(error, Exception):
            error_type = error.__class__.__name__
        else:
            error_type = "Message"
        
        level = logging.CRITICAL if any(t in error_type for t in ["Error", "KeyError"]) else logging.WARNING
        
        context = generate_error_context()
        error_id = logger.log_error(error, level, context)
        
        would_alert = (
            level >= logger.config["alert_threshold"] and
            (error_type in logger.config["alert_conditions"].get("error_types", []) or
             any(re.search(pattern, str(error)) for pattern in logger.config["alert_conditions"].get("error_patterns", [])))
        )
        
        print(f"Logged: {error_type} - {error}")
        print(f"Level: {logging.getLevelName(level)}, Would trigger alert: {would_alert}")
        print(f"Error ID: {error_id}")
        print("-" * 40)


def factory_function_example():
    """Example usage of the factory function."""
    print("\n=== Factory Function Example ===")
    
    # Set up log directory
    log_dir = setup_log_dir()
    
    # Create different loggers using the factory function
    loggers = []
    
    # Console logger
    console_logger = get_error_logger(
        logger_type="console",
        name="factory_console_logger",
        level=logging.ERROR,
        config={"use_colors": True}
    )
    loggers.append(("Console Logger", console_logger))
    
    # File logger
    file_logger = get_error_logger(
        logger_type="file",
        name="factory_file_logger",
        level=logging.WARNING,
        config={
            "log_dir": log_dir,
            "log_file": "factory_example.log"
        }
    )
    loggers.append(("File Logger", file_logger))
    
    # Print logger information
    for name, logger in loggers:
        print(f"- Created {name}: {logger.__class__.__name__}")
        print(f"  - Name: {logger.name}")
        print(f"  - Level: {logging.getLevelName(logger.level)}")
        print(f"  - Config: {list(logger.config.keys())}")
    
    # Log an error to all loggers
    error = RetrieverError("This error was logged via the factory function")
    context = {"source": "factory_function_example"}
    
    print("\nLogging error to all loggers:")
    for name, logger in loggers:
        error_id = logger.log_error(error, logging.ERROR, context)
        print(f"- Logged to {name} with ID: {error_id}")


def main():
    """Run all examples."""
    try:
        # Run examples
        console_logger_example()
        file_logger_example()
        database_logger_example()
        cloud_logger_example()
        alert_logger_example()
        factory_function_example()
        
        print("\nAll examples completed successfully!")
    
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import re  # Import here for the alert example
    main() 