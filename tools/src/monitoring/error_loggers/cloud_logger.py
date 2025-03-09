#!/usr/bin/env python3
"""
Cloud error logger for RAG systems.

This module provides the CloudErrorLogger class, which is responsible for
logging errors to cloud services such as AWS CloudWatch, Google Cloud Logging,
and Azure Monitor.
"""

from typing import Dict, List, Any, Optional, Union
import datetime
import logging
import json
import importlib.util
import os
import uuid
from .base import BaseErrorLogger


class CloudErrorLogger(BaseErrorLogger):
    """
    Logger for errors to cloud services.
    
    This logger sends errors to cloud logging services, with support for
    AWS CloudWatch, Google Cloud Logging, and Azure Monitor.
    
    Attributes:
        name (str): Name of the logger.
        level (int): Minimum logging level.
        format (str): Log message format.
        config (Dict[str, Any]): Configuration options for the logger.
    """
    
    CLOUD_PROVIDERS = ["aws", "gcp", "azure"]
    
    def __init__(
        self,
        name: str = "cloud_error_logger",
        level: int = logging.ERROR,
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the cloud error logger.
        
        Args:
            name (str): Name of the logger. Defaults to "cloud_error_logger".
            level (int): Minimum logging level. Defaults to logging.ERROR.
            format (str): Log message format.
            config (Optional[Dict[str, Any]]): Configuration options for the logger.
                Supported options:
                - cloud_provider (str): Cloud provider to use. Supported values: "aws", "gcp", "azure".
                - aws_region (str): AWS region (for AWS CloudWatch).
                - aws_log_group (str): AWS CloudWatch log group name.
                - aws_log_stream (str): AWS CloudWatch log stream name.
                - gcp_project (str): GCP project ID (for Google Cloud Logging).
                - gcp_log_name (str): GCP log name.
                - azure_connection_string (str): Azure connection string (for Azure Monitor).
                - azure_workspace_id (str): Azure Log Analytics workspace ID.
                - azure_primary_key (str): Azure Log Analytics primary key.
                - azure_log_type (str): Azure Log Analytics log type.
                - local_fallback (bool): Whether to fall back to local logging if cloud logging fails.
                - local_log_dir (str): Directory for local fallback logs.
        """
        self.stored_errors = []
        super().__init__(name, level, format, config)
        
        # Set default config values
        self.config.setdefault("cloud_provider", "aws")
        self.config.setdefault("aws_region", "us-east-1")
        self.config.setdefault("aws_log_group", "rag_logs")
        self.config.setdefault("aws_log_stream", f"errors_{datetime.datetime.now().strftime('%Y%m%d')}")
        self.config.setdefault("gcp_project", "")
        self.config.setdefault("gcp_log_name", "rag_logs")
        self.config.setdefault("azure_connection_string", "")
        self.config.setdefault("azure_workspace_id", "")
        self.config.setdefault("azure_primary_key", "")
        self.config.setdefault("azure_log_type", "RAGErrors")
        self.config.setdefault("local_fallback", True)
        self.config.setdefault("local_log_dir", "./logs")
        
        # Set up cloud client
        self.cloud_client = None
        self._setup_cloud_client()
    
    def _create_handler(self) -> Optional[logging.Handler]:
        """
        Create a handler for the logger.
        
        For cloud logger, we don't need a standard logging handler
        if we're using the cloud APIs directly. We'll handle logging ourselves.
        
        Returns:
            Optional[logging.Handler]: Logger handler or None.
        """
        # If we're using AWS CloudWatch, we can use the CloudWatch Logs handler
        if self.config["cloud_provider"] == "aws" and self.cloud_client:
            try:
                from watchtower import CloudWatchLogHandler
                return CloudWatchLogHandler(
                    log_group=self.config["aws_log_group"],
                    stream_name=self.config["aws_log_stream"],
                    boto3_client=self.cloud_client
                )
            except ImportError:
                pass
        
        return None
    
    def _setup_cloud_client(self) -> None:
        """Set up cloud client."""
        provider = self.config["cloud_provider"].lower()
        
        if provider not in self.CLOUD_PROVIDERS:
            raise ValueError(
                f"Unsupported cloud provider: {provider}. "
                f"Supported providers: {', '.join(self.CLOUD_PROVIDERS)}"
            )
        
        try:
            if provider == "aws":
                self._setup_aws_client()
            elif provider == "gcp":
                self._setup_gcp_client()
            elif provider == "azure":
                self._setup_azure_client()
        except ImportError as e:
            self.logger.warning(
                f"Failed to set up {provider} client: {e}. "
                "Will use local logging instead."
            )
            
            # Set up local log directory for fallback
            if self.config["local_fallback"]:
                os.makedirs(self.config["local_log_dir"], exist_ok=True)
    
    def _setup_aws_client(self) -> None:
        """Set up AWS CloudWatch Logs client."""
        # Check if boto3 is installed
        if not importlib.util.find_spec("boto3"):
            raise ImportError(
                "AWS CloudWatch support requires boto3. "
                "Install it with: pip install boto3"
            )
        
        import boto3
        
        # Create CloudWatch Logs client
        self.cloud_client = boto3.client(
            "logs",
            region_name=self.config["aws_region"]
        )
        
        # Create log group and stream if they don't exist
        try:
            self.cloud_client.create_log_group(
                logGroupName=self.config["aws_log_group"]
            )
        except self.cloud_client.exceptions.ResourceAlreadyExistsException:
            pass
        
        try:
            self.cloud_client.create_log_stream(
                logGroupName=self.config["aws_log_group"],
                logStreamName=self.config["aws_log_stream"]
            )
        except self.cloud_client.exceptions.ResourceAlreadyExistsException:
            pass
    
    def _setup_gcp_client(self) -> None:
        """Set up Google Cloud Logging client."""
        # Check if google-cloud-logging is installed
        if not importlib.util.find_spec("google.cloud.logging"):
            raise ImportError(
                "Google Cloud Logging support requires google-cloud-logging. "
                "Install it with: pip install google-cloud-logging"
            )
        
        from google.cloud import logging as gcp_logging
        
        # Create Cloud Logging client
        gcp_client = gcp_logging.Client(project=self.config["gcp_project"])
        self.cloud_client = gcp_client.logger(self.config["gcp_log_name"])
    
    def _setup_azure_client(self) -> None:
        """Set up Azure Monitor client."""
        # Check if azure-monitor-ingestion is installed
        if not importlib.util.find_spec("azure.monitor.ingestion"):
            raise ImportError(
                "Azure Monitor support requires azure-monitor-ingestion. "
                "Install it with: pip install azure-monitor-ingestion"
            )
        
        from azure.monitor.ingestion import LogsIngestionClient
        from azure.core.credentials import AzureKeyCredential
        
        # Create Azure Monitor client
        if self.config["azure_connection_string"]:
            # Use connection string
            self.cloud_client = LogsIngestionClient.from_connection_string(
                self.config["azure_connection_string"]
            )
        elif self.config["azure_workspace_id"] and self.config["azure_primary_key"]:
            # Use workspace ID and key
            self.cloud_client = LogsIngestionClient(
                endpoint=f"https://{self.config['azure_workspace_id']}.ods.opinsights.azure.com",
                credential=AzureKeyCredential(self.config["azure_primary_key"])
            )
        else:
            raise ValueError(
                "Azure Monitor requires either connection_string or "
                "workspace_id and primary_key"
            )
    
    def log_error(
        self,
        error: Union[str, Exception],
        level: int = logging.ERROR,
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log an error to the cloud service.
        
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
        
        # Log to cloud service
        try:
            self._log_to_cloud(record)
        except Exception as e:
            self.logger.error(f"Failed to log error to cloud: {e}")
            
            # Fall back to local logging if enabled
            if self.config["local_fallback"]:
                self._log_to_local_file(record)
        
        return error_id
    
    def _log_to_cloud(self, record: Dict[str, Any]) -> None:
        """
        Log an error record to the cloud service.
        
        Args:
            record (Dict[str, Any]): Error record to log.
        """
        provider = self.config["cloud_provider"].lower()
        
        if provider == "aws" and self.cloud_client:
            self._log_to_aws(record)
        elif provider == "gcp" and self.cloud_client:
            self._log_to_gcp(record)
        elif provider == "azure" and self.cloud_client:
            self._log_to_azure(record)
        else:
            raise ValueError(f"Cloud provider not set up: {provider}")
    
    def _log_to_aws(self, record: Dict[str, Any]) -> None:
        """
        Log an error record to AWS CloudWatch Logs.
        
        Args:
            record (Dict[str, Any]): Error record to log.
        """
        # Convert timestamp to Unix timestamp in milliseconds
        timestamp = int(datetime.datetime.fromisoformat(record["timestamp"]).timestamp() * 1000)
        
        # Log to CloudWatch Logs
        self.cloud_client.put_log_events(
            logGroupName=self.config["aws_log_group"],
            logStreamName=self.config["aws_log_stream"],
            logEvents=[
                {
                    "timestamp": timestamp,
                    "message": json.dumps(record)
                }
            ]
        )
    
    def _log_to_gcp(self, record: Dict[str, Any]) -> None:
        """
        Log an error record to Google Cloud Logging.
        
        Args:
            record (Dict[str, Any]): Error record to log.
        """
        # Determine severity based on level
        severity = "DEFAULT"
        if record["level"] == logging.DEBUG:
            severity = "DEBUG"
        elif record["level"] == logging.INFO:
            severity = "INFO"
        elif record["level"] == logging.WARNING:
            severity = "WARNING"
        elif record["level"] == logging.ERROR:
            severity = "ERROR"
        elif record["level"] == logging.CRITICAL:
            severity = "CRITICAL"
        
        # Log to Cloud Logging
        self.cloud_client.log_struct(record, severity=severity)
    
    def _log_to_azure(self, record: Dict[str, Any]) -> None:
        """
        Log an error record to Azure Monitor.
        
        Args:
            record (Dict[str, Any]): Error record to log.
        """
        # Log to Azure Monitor
        from azure.monitor.ingestion import LogsIngestionClient
        
        self.cloud_client.upload(
            rule_id=self.config["azure_workspace_id"],
            stream_name=self.config["azure_log_type"],
            logs=[record]
        )
    
    def _log_to_local_file(self, record: Dict[str, Any]) -> None:
        """
        Log an error record to a local file as fallback.
        
        Args:
            record (Dict[str, Any]): Error record to log.
        """
        # Create log directory if it doesn't exist
        os.makedirs(self.config["local_log_dir"], exist_ok=True)
        
        # Log to local file
        log_file = os.path.join(
            self.config["local_log_dir"],
            f"cloud_error_fallback_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        )
        
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to log error to local file: {e}")
    
    def get_errors(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        level: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get errors from the cloud service.
        
        Note: This method only returns errors that have been stored in memory.
        It does not retrieve errors from the cloud service.
        
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