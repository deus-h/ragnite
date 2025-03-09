#!/usr/bin/env python3
"""
Factory function for error loggers in RAG systems.

This module provides a factory function for creating error logger instances
based on the specified type and configuration.
"""

from typing import Dict, Any, Optional
import logging
from .base import BaseErrorLogger
from .console_logger import ConsoleErrorLogger
from .file_logger import FileErrorLogger
from .database_logger import DatabaseErrorLogger
from .cloud_logger import CloudErrorLogger
from .alert_logger import AlertErrorLogger


def get_error_logger(
    logger_type: str,
    name: Optional[str] = None,
    level: int = logging.ERROR,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    config: Optional[Dict[str, Any]] = None
) -> BaseErrorLogger:
    """
    Create an error logger instance based on the specified type and configuration.
    
    Args:
        logger_type (str): Type of error logger to create.
            Valid values: 'console', 'file', 'database', 'cloud', 'alert'.
        name (Optional[str]): Name of the logger.
            If not provided, a default name will be used based on the logger type.
        level (int): Minimum logging level. Defaults to logging.ERROR.
        format (str): Log message format.
        config (Optional[Dict[str, Any]]): Configuration options for the logger.
            Defaults to an empty dictionary.
            
    Returns:
        BaseErrorLogger: Error logger instance.
        
    Raises:
        ValueError: If an unsupported logger type is specified.
    """
    # Normalize logger type
    logger_type = logger_type.lower().strip()
    
    # Define mapping of logger types to classes
    logger_classes = {
        "console": ConsoleErrorLogger,
        "file": FileErrorLogger,
        "database": DatabaseErrorLogger,
        "cloud": CloudErrorLogger,
        "alert": AlertErrorLogger
    }
    
    # Check if logger type is supported
    if logger_type not in logger_classes:
        valid_types = ", ".join(f"'{t}'" for t in logger_classes.keys())
        raise ValueError(
            f"Unsupported logger type: '{logger_type}'. "
            f"Valid types are: {valid_types}."
        )
    
    # Get logger class
    logger_class = logger_classes[logger_type]
    
    # Set default name if not provided
    if name is None:
        name = f"{logger_type}_error_logger"
    
    # Create and return logger instance
    return logger_class(name=name, level=level, format=format, config=config) 