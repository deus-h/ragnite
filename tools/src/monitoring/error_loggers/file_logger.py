#!/usr/bin/env python3
"""
File error logger for RAG systems.

This module provides the FileErrorLogger class, which is responsible for
logging errors to files.
"""

from typing import Dict, List, Any, Optional, Union
import datetime
import logging
import json
import os
import glob
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from .base import BaseErrorLogger


class FileErrorLogger(BaseErrorLogger):
    """
    Logger for errors to files.
    
    This logger writes errors to log files, with support for file rotation,
    different formats, and structured logging.
    
    Attributes:
        name (str): Name of the logger.
        level (int): Minimum logging level.
        format (str): Log message format.
        config (Dict[str, Any]): Configuration options for the logger.
    """
    
    def __init__(
        self,
        name: str = "file_error_logger",
        level: int = logging.ERROR,
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the file error logger.
        
        Args:
            name (str): Name of the logger. Defaults to "file_error_logger".
            level (int): Minimum logging level. Defaults to logging.ERROR.
            format (str): Log message format.
            config (Optional[Dict[str, Any]]): Configuration options for the logger.
                Supported options:
                - log_dir (str): Directory to store log files. Defaults to "./logs".
                - log_file (str): Name of the log file. Defaults to "error.log".
                - rotation_type (str): Type of log rotation ("size", "time", "none").
                - max_bytes (int): Maximum file size for size-based rotation.
                - backup_count (int): Number of backup files to keep.
                - rotation_interval (str): Interval for time-based rotation.
                - json_format (bool): Whether to output logs in JSON format.
        """
        super().__init__(name, level, format, config)
        
        # Set default config values
        self.config.setdefault("log_dir", "./logs")
        self.config.setdefault("log_file", "error.log")
        self.config.setdefault("rotation_type", "size")
        self.config.setdefault("max_bytes", 10 * 1024 * 1024)  # 10 MB
        self.config.setdefault("backup_count", 5)
        self.config.setdefault("rotation_interval", "midnight")  # daily at midnight
        self.config.setdefault("json_format", True)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.config["log_dir"], exist_ok=True)
    
    def _create_handler(self) -> logging.Handler:
        """
        Create a handler for the logger.
        
        Returns:
            logging.Handler: File handler.
        """
        log_file = os.path.join(self.config["log_dir"], self.config["log_file"])
        
        # Create handler based on rotation type
        if self.config["rotation_type"] == "size":
            handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config["max_bytes"],
                backupCount=self.config["backup_count"]
            )
        elif self.config["rotation_type"] == "time":
            handler = TimedRotatingFileHandler(
                log_file,
                when=self.config["rotation_interval"],
                backupCount=self.config["backup_count"]
            )
        else:
            handler = logging.FileHandler(log_file)
        
        # Set level
        handler.setLevel(self.level)
        
        return handler
    
    def log_error(
        self,
        error: Union[str, Exception],
        level: int = logging.ERROR,
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log an error to a file.
        
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
        Get errors from log files.
        
        This method reads errors from log files in the specified date range.
        Note: This method only works when logs are in JSON format.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering errors.
            end_time (Optional[datetime.datetime]): End time for filtering errors.
            level (Optional[int]): Minimum log level for filtering errors.
            limit (int): Maximum number of errors to return. Defaults to 100.
            
        Returns:
            List[Dict[str, Any]]: List of error records.
        """
        # Check if JSON format is enabled
        if not self.config["json_format"]:
            raise ValueError(
                "get_errors() requires JSON format. "
                "Set config['json_format'] = True"
            )
        
        # Get log files
        log_files = self._get_log_files()
        
        # Read and parse errors
        errors = []
        for log_file in log_files:
            file_errors = self._read_errors_from_file(
                log_file, start_time, end_time, level
            )
            errors.extend(file_errors)
        
        # Sort by timestamp
        errors.sort(key=lambda e: e["timestamp"])
        
        # Apply limit
        if len(errors) > limit:
            errors = errors[-limit:]
        
        return errors
    
    def _get_log_files(self) -> List[str]:
        """
        Get list of log files.
        
        Returns:
            List[str]: List of log file paths.
        """
        log_file = os.path.join(self.config["log_dir"], self.config["log_file"])
        
        # Get base log file and rotated files
        files = [log_file]
        
        # Get rotated files
        if self.config["rotation_type"] == "size":
            # For size-based rotation, files are named file.1, file.2, etc.
            pattern = log_file + ".*"
            rotated_files = glob.glob(pattern)
            files.extend(rotated_files)
        elif self.config["rotation_type"] == "time":
            # For time-based rotation, files are named file.YYYY-MM-DD
            pattern = log_file + ".*"
            rotated_files = glob.glob(pattern)
            files.extend(rotated_files)
        
        # Filter out non-existing files
        files = [f for f in files if os.path.exists(f)]
        
        return files
    
    def _read_errors_from_file(
        self,
        file_path: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        min_level: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Read errors from a log file.
        
        Args:
            file_path (str): Path to the log file.
            start_time (Optional[datetime.datetime]): Start time for filtering.
            end_time (Optional[datetime.datetime]): End time for filtering.
            min_level (Optional[int]): Minimum log level for filtering.
            
        Returns:
            List[Dict[str, Any]]: List of error records.
        """
        errors = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Extract JSON from the line
                        # In case the log format includes non-JSON content
                        json_start = line.find('{')
                        json_end = line.rfind('}')
                        
                        if json_start != -1 and json_end != -1:
                            json_str = line[json_start:json_end + 1]
                            record = json.loads(json_str)
                            
                            # Apply filters
                            if self._filter_record(record, start_time, end_time, min_level):
                                errors.append(record)
                    except (json.JSONDecodeError, ValueError):
                        # Skip lines that don't contain valid JSON
                        continue
        except Exception as e:
            # Log error but don't crash
            self.logger.error(f"Error reading log file {file_path}: {e}")
        
        return errors
    
    def _filter_record(
        self,
        record: Dict[str, Any],
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        min_level: Optional[int] = None
    ) -> bool:
        """
        Check if a record passes the filters.
        
        Args:
            record (Dict[str, Any]): Error record.
            start_time (Optional[datetime.datetime]): Start time for filtering.
            end_time (Optional[datetime.datetime]): End time for filtering.
            min_level (Optional[int]): Minimum log level for filtering.
            
        Returns:
            bool: True if the record passes the filters, False otherwise.
        """
        # Check if record contains required fields
        if "timestamp" not in record or "level" not in record:
            return False
        
        # Check level
        if min_level is not None:
            try:
                record_level = record["level"]
                if record_level < min_level:
                    return False
            except (TypeError, ValueError):
                # If level is not a number, skip record
                return False
        
        # Check timestamp
        if start_time or end_time:
            try:
                timestamp = datetime.datetime.fromisoformat(record["timestamp"])
                
                if start_time and timestamp < start_time:
                    return False
                
                if end_time and timestamp > end_time:
                    return False
            except (TypeError, ValueError):
                # If timestamp is not valid, skip record
                return False
        
        return True 