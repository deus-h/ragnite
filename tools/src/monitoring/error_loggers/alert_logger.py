#!/usr/bin/env python3
"""
Alert error logger for RAG systems.

This module provides the AlertErrorLogger class, which is responsible for
sending alerts on errors to various channels such as email, Slack, and webhook.
"""

from typing import Dict, List, Any, Optional, Union, Callable
import datetime
import logging
import json
import smtplib
import importlib.util
import re
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from .base import BaseErrorLogger


class AlertErrorLogger(BaseErrorLogger):
    """
    Logger for sending alerts on errors.
    
    This logger sends alerts on errors to various channels such as email,
    Slack, and webhook, with customizable alert conditions.
    
    Attributes:
        name (str): Name of the logger.
        level (int): Minimum logging level.
        format (str): Log message format.
        config (Dict[str, Any]): Configuration options for the logger.
    """
    
    ALERT_CHANNELS = ["email", "slack", "webhook", "custom"]
    
    def __init__(
        self,
        name: str = "alert_error_logger",
        level: int = logging.ERROR,
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the alert error logger.
        
        Args:
            name (str): Name of the logger. Defaults to "alert_error_logger".
            level (int): Minimum logging level. Defaults to logging.ERROR.
            format (str): Log message format.
            config (Optional[Dict[str, Any]]): Configuration options for the logger.
                Supported options:
                - alert_channel (str): Alert channel to use. Supported values: "email", "slack", "webhook", "custom".
                - alert_threshold (int): Minimum log level for sending alerts. Defaults to logging.CRITICAL.
                - alert_cooldown (int): Cooldown period between alerts in seconds. Defaults to 300 (5 minutes).
                - alert_conditions (Dict[str, Any]): Conditions for sending alerts.
                  Can include "error_types", "error_patterns", and custom conditions.
                
                For Email:
                - email_smtp_server (str): SMTP server for sending emails.
                - email_smtp_port (int): SMTP port. Defaults to 587.
                - email_username (str): SMTP username for authentication.
                - email_password (str): SMTP password for authentication.
                - email_use_tls (bool): Whether to use TLS. Defaults to True.
                - email_from (str): Sender email address.
                - email_to (List[str]): List of recipient email addresses.
                - email_subject_template (str): Template for email subject.
                - email_body_template (str): Template for email body.
                
                For Slack:
                - slack_webhook_url (str): Slack webhook URL.
                - slack_channel (str): Slack channel to send alerts to.
                - slack_username (str): Username to display in Slack. Defaults to logger name.
                - slack_icon_emoji (str): Emoji to display as icon in Slack. Defaults to ":warning:".
                - slack_message_template (str): Template for Slack message.
                
                For Webhook:
                - webhook_url (str): Webhook URL.
                - webhook_method (str): HTTP method to use for webhook. Defaults to "POST".
                - webhook_headers (Dict[str, str]): HTTP headers to include in webhook request.
                - webhook_payload_template (str): Template for webhook payload.
                
                For Custom:
                - custom_alert_function (Callable): Custom function for sending alerts.
                  Function signature: (record: Dict[str, Any]) -> bool
        """
        self.stored_errors = []
        self.last_alert_time = None
        self.custom_alert_function = None
        super().__init__(name, level, format, config)
        
        # Set default config values
        self.config.setdefault("alert_channel", "email")
        self.config.setdefault("alert_threshold", logging.CRITICAL)
        self.config.setdefault("alert_cooldown", 300)  # 5 minutes
        self.config.setdefault("alert_conditions", {})
        
        # Set default values for email
        self.config.setdefault("email_smtp_port", 587)
        self.config.setdefault("email_use_tls", True)
        self.config.setdefault("email_subject_template", "[ALERT] Error in {logger_name}: {error_type}")
        self.config.setdefault("email_body_template", """
        <html>
        <body>
            <h2>Error Alert</h2>
            <p><strong>Error ID:</strong> {error_id}</p>
            <p><strong>Time:</strong> {timestamp}</p>
            <p><strong>Level:</strong> {level_name}</p>
            <p><strong>Error Type:</strong> {error_type}</p>
            <p><strong>Message:</strong> {message}</p>
            <p><strong>Logger:</strong> {logger_name}</p>
            {traceback_html}
            {context_html}
        </body>
        </html>
        """)
        
        # Set default values for Slack
        self.config.setdefault("slack_username", name)
        self.config.setdefault("slack_icon_emoji", ":warning:")
        self.config.setdefault("slack_message_template", """
        *Error Alert*
        *Error ID:* {error_id}
        *Time:* {timestamp}
        *Level:* {level_name}
        *Error Type:* {error_type}
        *Message:* {message}
        *Logger:* {logger_name}
        """)
        
        # Set default values for webhook
        self.config.setdefault("webhook_method", "POST")
        self.config.setdefault("webhook_headers", {
            "Content-Type": "application/json"
        })
        self.config.setdefault("webhook_payload_template", """
        {{
            "error_id": "{error_id}",
            "timestamp": "{timestamp}",
            "level": {level},
            "level_name": "{level_name}",
            "error_type": "{error_type}",
            "message": "{message}",
            "logger_name": "{logger_name}"
        }}
        """)
        
        # Register custom alert function if provided
        if "custom_alert_function" in self.config:
            self.custom_alert_function = self.config["custom_alert_function"]
    
    def _create_handler(self) -> Optional[logging.Handler]:
        """
        Create a handler for the logger.
        
        For alert logger, we don't need a standard logging handler.
        We'll handle logging and alerts ourselves.
        
        Returns:
            Optional[logging.Handler]: Always returns None.
        """
        return None
    
    def log_error(
        self,
        error: Union[str, Exception],
        level: int = logging.ERROR,
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log an error and send an alert if conditions are met.
        
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
        message = record["message"]
        if context:
            context_str = " | Context: " + json.dumps(context)
            message += context_str
        
        self.logger.log(level, message)
        
        # Check if alert should be sent
        if self._should_send_alert(record):
            try:
                self._send_alert(record)
                # Update last alert time
                self.last_alert_time = datetime.datetime.now()
            except Exception as e:
                self.logger.error(f"Failed to send alert: {e}")
        
        return error_id
    
    def _should_send_alert(self, record: Dict[str, Any]) -> bool:
        """
        Check if an alert should be sent for an error.
        
        Args:
            record (Dict[str, Any]): Error record.
            
        Returns:
            bool: True if an alert should be sent, False otherwise.
        """
        # Check level threshold
        if record["level"] < self.config["alert_threshold"]:
            return False
        
        # Check cooldown period
        if self.last_alert_time:
            elapsed = (datetime.datetime.now() - self.last_alert_time).total_seconds()
            if elapsed < self.config["alert_cooldown"]:
                return False
        
        # Check error type conditions
        if "error_types" in self.config["alert_conditions"]:
            error_types = self.config["alert_conditions"]["error_types"]
            if record["error_type"] not in error_types:
                return False
        
        # Check error pattern conditions
        if "error_patterns" in self.config["alert_conditions"]:
            error_patterns = self.config["alert_conditions"]["error_patterns"]
            message = record["message"]
            
            # Check if message matches any pattern
            for pattern in error_patterns:
                if re.search(pattern, message):
                    return True
            
            # If patterns are specified but none match, don't alert
            return False
        
        # If no conditions are specified or all conditions are met, send alert
        return True
    
    def _send_alert(self, record: Dict[str, Any]) -> None:
        """
        Send an alert for an error.
        
        Args:
            record (Dict[str, Any]): Error record.
        """
        channel = self.config["alert_channel"].lower()
        
        if channel not in self.ALERT_CHANNELS:
            raise ValueError(
                f"Unsupported alert channel: {channel}. "
                f"Supported channels: {', '.join(self.ALERT_CHANNELS)}"
            )
        
        if channel == "email":
            self._send_email_alert(record)
        elif channel == "slack":
            self._send_slack_alert(record)
        elif channel == "webhook":
            self._send_webhook_alert(record)
        elif channel == "custom" and self.custom_alert_function:
            self.custom_alert_function(record)
    
    def _send_email_alert(self, record: Dict[str, Any]) -> None:
        """
        Send an email alert for an error.
        
        Args:
            record (Dict[str, Any]): Error record.
        """
        # Check if email configuration is provided
        required_fields = ["email_smtp_server", "email_from", "email_to"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Email alert requires {field} configuration")
        
        # Format email subject and body
        subject = self._format_template(self.config["email_subject_template"], record)
        body = self._format_template(self.config["email_body_template"], record)
        
        # Create email message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.config["email_from"]
        msg["To"] = ", ".join(self.config["email_to"])
        
        # Add HTML body
        msg.attach(MIMEText(body, "html"))
        
        # Send email
        with smtplib.SMTP(
            self.config["email_smtp_server"],
            self.config["email_smtp_port"]
        ) as server:
            if self.config["email_use_tls"]:
                server.starttls()
            
            if "email_username" in self.config and "email_password" in self.config:
                server.login(
                    self.config["email_username"],
                    self.config["email_password"]
                )
            
            server.send_message(msg)
    
    def _send_slack_alert(self, record: Dict[str, Any]) -> None:
        """
        Send a Slack alert for an error.
        
        Args:
            record (Dict[str, Any]): Error record.
        """
        # Check if Slack configuration is provided
        if "slack_webhook_url" not in self.config:
            raise ValueError("Slack alert requires slack_webhook_url configuration")
        
        # Format Slack message
        message = self._format_template(self.config["slack_message_template"], record)
        
        # Create payload
        payload = {
            "text": message,
            "username": self.config["slack_username"],
            "icon_emoji": self.config["slack_icon_emoji"]
        }
        
        if "slack_channel" in self.config:
            payload["channel"] = self.config["slack_channel"]
        
        # Send to Slack
        response = requests.post(
            self.config["slack_webhook_url"],
            json=payload
        )
        response.raise_for_status()
    
    def _send_webhook_alert(self, record: Dict[str, Any]) -> None:
        """
        Send a webhook alert for an error.
        
        Args:
            record (Dict[str, Any]): Error record.
        """
        # Check if webhook configuration is provided
        if "webhook_url" not in self.config:
            raise ValueError("Webhook alert requires webhook_url configuration")
        
        # Format webhook payload
        payload_str = self._format_template(self.config["webhook_payload_template"], record)
        
        try:
            # Parse payload as JSON if possible
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            # Use raw string as payload
            payload = payload_str
        
        # Send to webhook
        response = requests.request(
            method=self.config["webhook_method"],
            url=self.config["webhook_url"],
            headers=self.config["webhook_headers"],
            json=payload if isinstance(payload, dict) else None,
            data=payload if not isinstance(payload, dict) else None
        )
        response.raise_for_status()
    
    def _format_template(self, template: str, record: Dict[str, Any]) -> str:
        """
        Format a template string with values from the error record.
        
        Args:
            template (str): Template string.
            record (Dict[str, Any]): Error record.
            
        Returns:
            str: Formatted string.
        """
        # Create a copy of the record for formatting
        format_data = record.copy()
        
        # Add HTML-formatted traceback and context if they exist
        if "traceback" in record and record["traceback"]:
            traceback_html = "<p><strong>Traceback:</strong></p><pre>" + \
                             "\n".join(record["traceback"]) + \
                             "</pre>"
            format_data["traceback_html"] = traceback_html
        else:
            format_data["traceback_html"] = ""
        
        if "context" in record and record["context"]:
            context_html = "<p><strong>Context:</strong></p><pre>" + \
                          json.dumps(record["context"], indent=2) + \
                          "</pre>"
            format_data["context_html"] = context_html
        else:
            format_data["context_html"] = ""
        
        # Format the template
        return template.format(**format_data)
    
    def get_errors(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        level: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get errors from the stored records.
        
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