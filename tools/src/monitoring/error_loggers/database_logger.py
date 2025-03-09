#!/usr/bin/env python3
"""
Database error logger for RAG systems.

This module provides the DatabaseErrorLogger class, which is responsible for
logging errors to databases such as SQLite, PostgreSQL, MySQL, etc.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import datetime
import logging
import json
import os
import sqlite3
import uuid
try:
    import psycopg2
    import psycopg2.extras
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

try:
    import mysql.connector
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False

from .base import BaseErrorLogger


class DatabaseErrorLogger(BaseErrorLogger):
    """
    Logger for errors to databases.
    
    This logger writes errors to a database, with support for SQLite,
    PostgreSQL, and MySQL.
    
    Attributes:
        name (str): Name of the logger.
        level (int): Minimum logging level.
        format (str): Log message format.
        config (Dict[str, Any]): Configuration options for the logger.
    """
    
    def __init__(
        self,
        name: str = "database_error_logger",
        level: int = logging.ERROR,
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the database error logger.
        
        Args:
            name (str): Name of the logger. Defaults to "database_error_logger".
            level (int): Minimum logging level. Defaults to logging.ERROR.
            format (str): Log message format.
            config (Optional[Dict[str, Any]]): Configuration options for the logger.
                Supported options:
                - db_type (str): Database type. Supported values: "sqlite", "postgres", "mysql".
                - db_path (str): Path to SQLite database file (for SQLite only).
                - db_host (str): Database host (for PostgreSQL/MySQL).
                - db_port (int): Database port (for PostgreSQL/MySQL).
                - db_name (str): Database name (for PostgreSQL/MySQL).
                - db_user (str): Database user (for PostgreSQL/MySQL).
                - db_password (str): Database password (for PostgreSQL/MySQL).
                - table_name (str): Name of the error table. Defaults to "error_logs".
                - auto_create_table (bool): Whether to create the table if it doesn't exist.
        """
        super().__init__(name, level, format, config)
        
        # Set default config values
        self.config.setdefault("db_type", "sqlite")
        self.config.setdefault("db_path", "./logs/errors.db")
        self.config.setdefault("db_host", "localhost")
        self.config.setdefault("db_port", 5432)  # PostgreSQL default port
        self.config.setdefault("db_name", "rag_logs")
        self.config.setdefault("db_user", "postgres")
        self.config.setdefault("db_password", "")
        self.config.setdefault("table_name", "error_logs")
        self.config.setdefault("auto_create_table", True)
        
        # Set up database connection
        self.connection = None
        self._setup_database()
    
    def _create_handler(self) -> Optional[logging.Handler]:
        """
        Create a handler for the logger.
        
        For database logger, we don't need a standard logging handler.
        We'll handle logging ourselves.
        
        Returns:
            Optional[logging.Handler]: Always returns None.
        """
        return None
    
    def _setup_database(self) -> None:
        """Set up database connection and table."""
        db_type = self.config["db_type"].lower()
        
        if db_type == "sqlite":
            self._setup_sqlite()
        elif db_type == "postgres":
            self._setup_postgres()
        elif db_type == "mysql":
            self._setup_mysql()
        else:
            raise ValueError(
                f"Unsupported database type: {db_type}. "
                "Supported types: sqlite, postgres, mysql"
            )
    
    def _setup_sqlite(self) -> None:
        """Set up SQLite database."""
        # Create directory for database file if it doesn't exist
        db_path = self.config["db_path"]
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect to database
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        
        # Create table if it doesn't exist
        if self.config["auto_create_table"]:
            cursor = self.connection.cursor()
            table_name = self.config["table_name"]
            
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    level_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    logger_name TEXT NOT NULL,
                    traceback TEXT,
                    context TEXT
                )
            """)
            self.connection.commit()
    
    def _setup_postgres(self) -> None:
        """Set up PostgreSQL database."""
        if not HAS_POSTGRES:
            raise ImportError(
                "PostgreSQL support requires psycopg2. "
                "Install it with: pip install psycopg2-binary"
            )
        
        # Connect to database
        self.connection = psycopg2.connect(
            host=self.config["db_host"],
            port=self.config["db_port"],
            dbname=self.config["db_name"],
            user=self.config["db_user"],
            password=self.config["db_password"]
        )
        
        # Create table if it doesn't exist
        if self.config["auto_create_table"]:
            cursor = self.connection.cursor()
            table_name = self.config["table_name"]
            
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    level INTEGER NOT NULL,
                    level_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    logger_name TEXT NOT NULL,
                    traceback TEXT,
                    context JSONB
                )
            """)
            self.connection.commit()
    
    def _setup_mysql(self) -> None:
        """Set up MySQL database."""
        if not HAS_MYSQL:
            raise ImportError(
                "MySQL support requires mysql-connector-python. "
                "Install it with: pip install mysql-connector-python"
            )
        
        # Connect to database
        self.connection = mysql.connector.connect(
            host=self.config["db_host"],
            port=self.config["db_port"],
            database=self.config["db_name"],
            user=self.config["db_user"],
            password=self.config["db_password"]
        )
        
        # Create table if it doesn't exist
        if self.config["auto_create_table"]:
            cursor = self.connection.cursor()
            table_name = self.config["table_name"]
            
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id VARCHAR(36) PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    level INT NOT NULL,
                    level_name VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    error_type VARCHAR(100) NOT NULL,
                    logger_name VARCHAR(100) NOT NULL,
                    traceback TEXT,
                    context JSON
                )
            """)
            self.connection.commit()
    
    def log_error(
        self,
        error: Union[str, Exception],
        level: int = logging.ERROR,
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None
    ) -> str:
        """
        Log an error to the database.
        
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
        
        # Insert record into database
        try:
            self._insert_record(record)
            self.connection.commit()
        except Exception as e:
            # Log error using standard logging
            self.logger.error(f"Failed to log error to database: {e}")
            
            # Try to reconnect and insert again
            try:
                self._setup_database()
                self._insert_record(record)
                self.connection.commit()
            except Exception as e2:
                self.logger.error(f"Failed to reconnect and log error: {e2}")
        
        return error_id
    
    def _insert_record(self, record: Dict[str, Any]) -> None:
        """
        Insert a record into the database.
        
        Args:
            record (Dict[str, Any]): Error record.
        """
        db_type = self.config["db_type"].lower()
        table_name = self.config["table_name"]
        
        # Prepare record for database
        db_record = {
            "id": record["error_id"],
            "timestamp": record["timestamp"],
            "level": record["level"],
            "level_name": record["level_name"],
            "message": record["message"],
            "error_type": record["error_type"],
            "logger_name": record["logger_name"],
            "traceback": json.dumps(record["traceback"]) if "traceback" in record else None,
            "context": json.dumps(record["context"]) if "context" in record else None
        }
        
        if db_type == "sqlite":
            cursor = self.connection.cursor()
            
            # In SQLite, we need to use placeholders
            placeholders = ", ".join(["?"] * len(db_record))
            columns = ", ".join(db_record.keys())
            values = tuple(db_record.values())
            
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, values)
            
        elif db_type == "postgres":
            cursor = self.connection.cursor()
            
            # In PostgreSQL, we can use named parameters
            placeholders = ", ".join([f"%({key})s" for key in db_record.keys()])
            columns = ", ".join(db_record.keys())
            
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, db_record)
            
        elif db_type == "mysql":
            cursor = self.connection.cursor()
            
            # In MySQL, we need to use placeholders
            placeholders = ", ".join(["%s"] * len(db_record))
            columns = ", ".join(db_record.keys())
            values = tuple(db_record.values())
            
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(query, values)
    
    def get_errors(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        level: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get errors from the database.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering errors.
            end_time (Optional[datetime.datetime]): End time for filtering errors.
            level (Optional[int]): Minimum log level for filtering errors.
            limit (int): Maximum number of errors to return. Defaults to 100.
            
        Returns:
            List[Dict[str, Any]]: List of error records.
        """
        db_type = self.config["db_type"].lower()
        table_name = self.config["table_name"]
        
        # Build query
        query, params = self._build_query(start_time, end_time, level, limit)
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            
            # Fetch results
            if db_type == "sqlite":
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
            else:
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            # Log error using standard logging
            self.logger.error(f"Failed to get errors from database: {e}")
            
            # Try to reconnect and query again
            try:
                self._setup_database()
                cursor = self.connection.cursor()
                cursor.execute(query, params)
                
                # Fetch results
                if db_type == "sqlite":
                    rows = cursor.fetchall()
                    return [self._row_to_dict(row) for row in rows]
                else:
                    return [dict(row) for row in cursor.fetchall()]
                    
            except Exception as e2:
                self.logger.error(f"Failed to reconnect and get errors: {e2}")
                return []
    
    def _build_query(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        level: Optional[int] = None,
        limit: int = 100
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a query for getting errors.
        
        Args:
            start_time (Optional[datetime.datetime]): Start time for filtering.
            end_time (Optional[datetime.datetime]): End time for filtering.
            level (Optional[int]): Minimum log level for filtering.
            limit (int): Maximum number of errors to return.
            
        Returns:
            Tuple[str, Dict[str, Any]]: Query string and parameters.
        """
        db_type = self.config["db_type"].lower()
        table_name = self.config["table_name"]
        
        # Build WHERE clause
        conditions = []
        params = {}
        
        if start_time:
            if db_type == "sqlite":
                conditions.append("timestamp >= :start_time")
                params["start_time"] = start_time.isoformat()
            else:
                conditions.append("timestamp >= %(start_time)s")
                params["start_time"] = start_time
        
        if end_time:
            if db_type == "sqlite":
                conditions.append("timestamp <= :end_time")
                params["end_time"] = end_time.isoformat()
            else:
                conditions.append("timestamp <= %(end_time)s")
                params["end_time"] = end_time
        
        if level is not None:
            if db_type == "sqlite":
                conditions.append("level >= :level")
                params["level"] = level
            else:
                conditions.append("level >= %(level)s")
                params["level"] = level
        
        # Build full query
        query = f"SELECT * FROM {table_name}"
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        # Add limit
        if db_type == "sqlite" or db_type == "postgres":
            query += f" LIMIT {limit}"
        elif db_type == "mysql":
            query += f" LIMIT {limit}"
        
        return query, params
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """
        Convert a SQLite row to a dictionary.
        
        Args:
            row (sqlite3.Row): SQLite row.
            
        Returns:
            Dict[str, Any]: Dictionary representation of the row.
        """
        record = {}
        for key in row.keys():
            value = row[key]
            
            # Parse JSON fields
            if key in ["context", "traceback"] and value is not None:
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            record[key] = value
        
        return record
    
    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __del__(self) -> None:
        """Destructor to ensure connection is closed."""
        self.close() 