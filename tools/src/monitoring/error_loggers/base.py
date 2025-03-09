#!/usr/bin/env python3
"""
Base class for error loggers in RAG systems.

This module provides the abstract base class for all error loggers,
which are responsible for recording, storing, and notifying about
errors that occur during system operation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import datetime
import logging
import os
import json
import uuid


class BaseErrorLogger(ABC):
    """
    Abstract base class for error loggers.
    
    Error loggers record, store, and notify about errors that occur
    during RAG system operation, providing detailed information for
    debugging and system improvement.
    
    Attributes:
        name (str): Name of the logger.
        level (int): Minimum logging level (default: logging.ERROR).
        format (str): Log message format.
        config (Dict[str, Any]): Configuration options for the logger.
    """
    
    # Standard log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    def __init__(
        self,
        name: str,
        level: int = logging.ERROR,
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the error logger.
        
        Args:
            name (str): Name of the logger.
            level (int): Minimum logging level. Defaults to logging.ERROR.
            format (str): Log message format.
            config (Optional[Dict[str, Any]]): Configuration options for the logger.
                Defaults to an empty dictionary.
        """
        self.name = name
        self.level = level
        self.format = format
        self.config = config or {}
        
        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Check if logger already has handlers
        if not self.logger.handlers:
            # Create a formatter
            formatter = logging.Formatter(format)
            
            # Create a handler based on the logger type
            handler = self._create_handler()
            
            # Set the formatter for the handler
            if handler:
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
    
    @abstractmethod
    def _create_handler(self) -> Optional[logging.Handler]:
        """
        Create a handler for the logger.
        
        Returns:
            Optional[logging.Handler]: Logger handler or None.
        """
        pass
    
    @abstractmethod
    def log_error(
        self,
        error: Union[str, Exception],
        level: int = logging.ERROR,
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log an error.
        
        Args:
            error (Union[str, Exception]): The error message or exception to log.
            level (int): Log level. Defaults to logging.ERROR.
            context (Optional[Dict[str, Any]]): Additional context about the error.
            error_id (Optional[str]): Unique identifier for the error.
                If not provided, a UUID will be generated.
                
        Returns:
            str: The error ID.
        """
        pass
    
    @abstractmethod
    def get_errors(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        level: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get errors from the log.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering errors.
            end_time (Optional[datetime.datetime]): End time for filtering errors.
            level (Optional[int]): Minimum log level for filtering errors.
            limit (int): Maximum number of errors to return. Defaults to 100.
            
        Returns:
            List[Dict[str, Any]]: List of error records.
        """
        pass
    
    def _format_error_record(
        self,
        error: Union[str, Exception],
        level: int,
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format an error record.
        
        Args:
            error (Union[str, Exception]): The error message or exception to log.
            level (int): Log level.
            context (Optional[Dict[str, Any]]): Additional context about the error.
            error_id (Optional[str]): Unique identifier for the error.
                
        Returns:
            Dict[str, Any]: Formatted error record.
        """
        # Generate error ID if not provided
        if error_id is None:
            error_id = str(uuid.uuid4())
        
        # Get timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Get error message
        if isinstance(error, Exception):
            error_message = str(error)
            error_type = error.__class__.__name__
        else:
            error_message = error
            error_type = "LogMessage"
        
        # Create error record
        record = {
            "error_id": error_id,
            "timestamp": timestamp,
            "level": level,
            "level_name": logging.getLevelName(level),
            "message": error_message,
            "error_type": error_type,
            "logger_name": self.name
        }
        
        # Add traceback if available
        if isinstance(error, Exception) and hasattr(error, "__traceback__"):
            import traceback
            record["traceback"] = traceback.format_exception(
                type(error), error, error.__traceback__
            )
        
        # Add context if provided
        if context:
            record["context"] = context
        
        return record
    
    def log_debug(
        self,
        message: Union[str, Exception],
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log a debug message.
        
        Args:
            message (Union[str, Exception]): The message or exception to log.
            context (Optional[Dict[str, Any]]): Additional context about the message.
            error_id (Optional[str]): Unique identifier for the message.
                
        Returns:
            str: The message ID.
        """
        return self.log_error(message, self.DEBUG, context, error_id)
    
    def log_info(
        self,
        message: Union[str, Exception],
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log an info message.
        
        Args:
            message (Union[str, Exception]): The message or exception to log.
            context (Optional[Dict[str, Any]]): Additional context about the message.
            error_id (Optional[str]): Unique identifier for the message.
                
        Returns:
            str: The message ID.
        """
        return self.log_error(message, self.INFO, context, error_id)
    
    def log_warning(
        self,
        message: Union[str, Exception],
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log a warning message.
        
        Args:
            message (Union[str, Exception]): The message or exception to log.
            context (Optional[Dict[str, Any]]): Additional context about the message.
            error_id (Optional[str]): Unique identifier for the message.
                
        Returns:
            str: The message ID.
        """
        return self.log_error(message, self.WARNING, context, error_id)
    
    def log_critical(
        self,
        error: Union[str, Exception],
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log a critical error.
        
        Args:
            error (Union[str, Exception]): The error message or exception to log.
            context (Optional[Dict[str, Any]]): Additional context about the error.
            error_id (Optional[str]): Unique identifier for the error.
                
        Returns:
            str: The error ID.
        """
        return self.log_error(error, self.CRITICAL, context, error_id)
    
    def set_level(self, level: int) -> None:
        """
        Set the minimum logging level.
        
        Args:
            level (int): Minimum logging level.
        """
        self.level = level
        self.logger.setLevel(level) 