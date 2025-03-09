#!/usr/bin/env python3
"""
Console error logger for RAG systems.

This module provides the ConsoleErrorLogger class, which is responsible for
logging errors to the console.
"""

from typing import Dict, List, Any, Optional, Union
import datetime
import logging
import json
import sys
from .base import BaseErrorLogger


class ConsoleErrorLogger(BaseErrorLogger):
    """
    Logger for errors to the console.
    
    This logger outputs errors to the console (stdout/stderr), with
    customizable formatting and coloring options.
    
    Attributes:
        name (str): Name of the logger.
        level (int): Minimum logging level.
        format (str): Log message format.
        config (Dict[str, Any]): Configuration options for the logger.
    """
    
    def __init__(
        self,
        name: str = "console_error_logger",
        level: int = logging.ERROR,
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the console error logger.
        
        Args:
            name (str): Name of the logger. Defaults to "console_error_logger".
            level (int): Minimum logging level. Defaults to logging.ERROR.
            format (str): Log message format.
            config (Optional[Dict[str, Any]]): Configuration options for the logger.
                Supported options:
                - use_colors (bool): Whether to use colors for different log levels.
                - output_stream (str): Output stream to use ("stdout", "stderr").
                - json_format (bool): Whether to output logs in JSON format.
        """
        self.stored_errors = []
        super().__init__(name, level, format, config)
        
        # Set default config values
        self.config.setdefault("use_colors", True)
        self.config.setdefault("output_stream", "stderr")
        self.config.setdefault("json_format", False)
    
    def _create_handler(self) -> logging.Handler:
        """
        Create a handler for the logger.
        
        Returns:
            logging.Handler: Console handler.
        """
        # Determine output stream
        if self.config["output_stream"] == "stdout":
            stream = sys.stdout
        else:
            stream = sys.stderr
        
        # Create handler
        handler = logging.StreamHandler(stream)
        
        # Set level
        handler.setLevel(self.level)
        
        # Set colors if enabled
        if self.config["use_colors"]:
            self._set_colors(handler)
        
        return handler
    
    def _set_colors(self, handler: logging.Handler) -> None:
        """
        Set colors for different log levels.
        
        Args:
            handler (logging.Handler): The handler to set colors for.
        """
        if hasattr(handler, 'setFormatter'):
            # Define colors
            colors = {
                logging.DEBUG: '\033[94m',     # Blue
                logging.INFO: '\033[92m',      # Green
                logging.WARNING: '\033[93m',   # Yellow
                logging.ERROR: '\033[91m',     # Red
                logging.CRITICAL: '\033[1;91m' # Bold Red
            }
            reset = '\033[0m'
            
            # Create a custom formatter with colors
            class ColoredFormatter(logging.Formatter):
                def format(self, record):
                    levelno = record.levelno
                    if levelno in colors:
                        record.levelname = f"{colors[levelno]}{record.levelname}{reset}"
                        record.msg = f"{colors[levelno]}{record.msg}{reset}"
                    return super().format(record)
            
            # Set the formatter
            handler.setFormatter(ColoredFormatter(self.format))
    
    def log_error(
        self,
        error: Union[str, Exception],
        level: int = logging.ERROR,
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log an error to the console.
        
        Args:
            error (Union[str, Exception]): The error message or exception to log.
            level (int): Log level. Defaults to logging.ERROR.
            context (Optional[Dict[str, Any]]): Additional context about the error.
            error_id (Optional[str]): Unique identifier for the error.
                If not provided, a UUID will be generated.
                
        Returns:
            str: The error ID.
        """
        # Format error record
        record = self._format_error_record(error, level, context, error_id)
        error_id = record["error_id"]
        
        # Store error record
        self.stored_errors.append(record)
        if len(self.stored_errors) > 1000:  # Limit in-memory storage
            self.stored_errors = self.stored_errors[-1000:]
        
        # Log using standard logging
        if self.config["json_format"]:
            # Log as JSON
            self.logger.log(level, json.dumps(record))
        else:
            # Log as text
            message = record["message"]
            if context:
                context_str = " | Context: " + json.dumps(context)
                message += context_str
            
            self.logger.log(level, message)
        
        return error_id
    
    def get_errors(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        level: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get errors from the stored records.
        
        Note: Console logger only stores errors in memory, so errors will be
        lost when the program exits.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering errors.
            end_time (Optional[datetime.datetime]): End time for filtering errors.
            level (Optional[int]): Minimum log level for filtering errors.
            limit (int): Maximum number of errors to return. Defaults to 100.
            
        Returns:
            List[Dict[str, Any]]: List of error records.
        """
        filtered_errors = self.stored_errors.copy()
        
        # Filter by time
        if start_time or end_time:
            filtered_errors = self._filter_by_time(
                filtered_errors, start_time, end_time
            )
        
        # Filter by level
        if level is not None:
            filtered_errors = [
                error for error in filtered_errors
                if error["level"] >= level
            ]
        
        # Apply limit
        return filtered_errors[-limit:]
    
    def _filter_by_time(
        self,
        errors: List[Dict[str, Any]],
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter errors by time range.
        
        Args:
            errors (List[Dict[str, Any]]): List of error records.
            start_time (Optional[datetime.datetime]): Start time for filtering.
            end_time (Optional[datetime.datetime]): End time for filtering.
            
        Returns:
            List[Dict[str, Any]]: Filtered list of error records.
        """
        filtered = []
        
        for error in errors:
            timestamp = datetime.datetime.fromisoformat(error["timestamp"])
            
            # Check start_time
            if start_time and timestamp < start_time:
                continue
            
            # Check end_time
            if end_time and timestamp > end_time:
                continue
            
            filtered.append(error)
        
        return filtered 