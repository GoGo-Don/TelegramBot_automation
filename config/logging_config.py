"""
Advanced Multi-Level Logging Configuration Module

This module provides comprehensive logging configuration with multiple handlers,
formatters, and logging levels. It supports structured logging, performance monitoring,
error tracking, and integration with external monitoring systems.

Author: Development Team
Version: 1.0.0
Date: 2025-09-16
"""

import json
import logging
import logging.handlers
import os
import sys
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import colorlog
import structlog
from prometheus_client import Counter, Gauge, Histogram
from pythonjsonlogger import jsonlogger

# Context variables for request tracking
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_context: ContextVar[Optional[int]] = ContextVar('user_id', default=None)
session_context: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


# Metrics for monitoring
log_messages_total = Counter('log_messages_total', 'Total log messages', ['level', 'module'])
error_count = Counter('errors_total', 'Total errors', ['error_type', 'module'])
response_time = Histogram('request_duration_seconds', 'Request duration', ['endpoint'])
active_sessions = Gauge('active_sessions_total', 'Active user sessions')


@dataclass
class LogContext:
    """
    Structured logging context data.
    
    This class encapsulates all contextual information that should be
    included with log messages for comprehensive debugging and monitoring.
    """
    
    # Request tracking
    request_id: Optional[str] = None
    user_id: Optional[int] = None
    session_id: Optional[str] = None
    
    # Performance tracking
    start_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Application context
    module: Optional[str] = None
    function: Optional[str] = None
    endpoint: Optional[str] = None
    
    # Business context
    operation: Optional[str] = None
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    
    # Technical context
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    hostname: Optional[str] = None
    
    # Custom fields
    custom_fields: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values for context."""
        if self.custom_fields is None:
            self.custom_fields = {}
        
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)
        
        if self.thread_id is None:
            import threading
            self.thread_id = threading.get_ident()
        
        if self.process_id is None:
            self.process_id = os.getpid()
        
        if self.hostname is None:
            import socket
            self.hostname = socket.gethostname()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        result = asdict(self)
        
        # Format datetime fields
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        
        # Remove None values
        result = {k: v for k, v in result.items() if v is not None}
        
        return result
    
    def add_field(self, key: str, value: Any) -> None:
        """Add custom field to context."""
        if self.custom_fields is None:
            self.custom_fields = {}
        self.custom_fields[key] = value
    
    def update_duration(self) -> None:
        """Update duration based on current time."""
        if self.start_time:
            delta = datetime.now(timezone.utc) - self.start_time
            self.duration_ms = delta.total_seconds() * 1000


class ContextualFilter(logging.Filter):
    """
    Logging filter that adds contextual information to log records.
    
    This filter enriches log records with context variables and
    performance metrics for comprehensive logging.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add contextual information to log record.
        
        Args:
            record: Log record to enhance
            
        Returns:
            True to include record in logging output
        """
        # Add context variables
        record.request_id = request_id_context.get()
        record.user_id = user_id_context.get()
        record.session_id = session_context.get()
        
        # Add performance information
        record.timestamp_iso = datetime.now(timezone.utc).isoformat()
        record.process_id = os.getpid()
        
        # Add module information if available
        if hasattr(record, 'module') and record.module:
            record.module_name = record.module
        else:
            # Extract module from logger name
            record.module_name = record.name.split('.')[0] if '.' in record.name else record.name
        
        # Update metrics
        log_messages_total.labels(level=record.levelname, module=record.module_name).inc()
        
        if record.levelno >= logging.ERROR:
            error_type = getattr(record, 'error_type', 'unknown')
            error_count.labels(error_type=error_type, module=record.module_name).inc()
        
        return True


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging output.
    
    This formatter creates consistent, parseable log output with
    structured data and contextual information.
    """
    
    def __init__(self, include_context: bool = True, json_format: bool = False):
        """
        Initialize structured formatter.
        
        Args:
            include_context: Whether to include contextual information
            json_format: Whether to format as JSON
        """
        self.include_context = include_context
        self.json_format = json_format
        
        if json_format:
            super().__init__()
        else:
            fmt = (
                "%(timestamp_iso)s | %(levelname)-8s | "
                "%(module_name)-12s | %(funcName)-20s | "
                "%(message)s"
            )
            if include_context:
                fmt += " | REQ:%(request_id)s USER:%(user_id)s"
            super().__init__(fmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with structured data.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message
        """
        if self.json_format:
            return self._format_json(record)
        else:
            return self._format_text(record)
    
    def _format_json(self, record: logging.LogRecord) -> str:
        """Format record as JSON."""
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'module': getattr(record, 'module_name', record.name),
            'function': record.funcName,
            'message': record.getMessage(),
            'line': record.lineno,
            'file': record.pathname,
        }
        
        # Add context if available
        if self.include_context:
            if hasattr(record, 'request_id') and record.request_id:
                log_data['request_id'] = record.request_id
            if hasattr(record, 'user_id') and record.user_id:
                log_data['user_id'] = record.user_id
            if hasattr(record, 'session_id') and record.session_id:
                log_data['session_id'] = record.session_id
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'stack_info', 'exc_info',
                          'exc_text', 'message'):
                if not key.startswith('_'):
                    log_data[f'extra_{key}'] = value
        
        return json.dumps(log_data)
    
    def _format_text(self, record: logging.LogRecord) -> str:
        """Format record as text."""
        # Set default values for missing attributes
        for attr in ['request_id', 'user_id', 'session_id', 'module_name']:
            if not hasattr(record, attr):
                setattr(record, attr, '-')
        
        if not hasattr(record, 'timestamp_iso'):
            record.timestamp_iso = datetime.now(timezone.utc).isoformat()
        
        return super().format(record)


class PerformanceLogger:
    """
    Performance logging utility for tracking execution times and metrics.
    
    This class provides decorators and context managers for automatic
    performance tracking and logging of function execution times.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize performance logger.
        
        Args:
            logger: Logger instance to use for output
        """
        self.logger = logger
    
    def log_function_performance(self, function_name: Optional[str] = None):
        """
        Decorator for logging function performance.
        
        Args:
            function_name: Custom function name for logging
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = function_name or func.__name__
                start_time = datetime.now(timezone.utc)
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Calculate duration
                    end_time = datetime.now(timezone.utc)
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    
                    # Log performance
                    self.logger.info(
                        f"Function {func_name} completed successfully",
                        extra={
                            'duration_ms': duration_ms,
                            'function': func_name,
                            'performance': True,
                            'status': 'success'
                        }
                    )
                    
                    # Update metrics
                    response_time.labels(endpoint=func_name).observe(duration_ms / 1000)
                    
                    return result
                
                except Exception as e:
                    # Calculate duration even on failure
                    end_time = datetime.now(timezone.utc)
                    duration_ms = (end_time - start_time).total_seconds() * 1000
                    
                    # Log error with performance data
                    self.logger.error(
                        f"Function {func_name} failed with error: {str(e)}",
                        extra={
                            'duration_ms': duration_ms,
                            'function': func_name,
                            'performance': True,
                            'status': 'error',
                            'error_type': type(e).__name__,
                            'error_message': str(e)
                        },
                        exc_info=True
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def log_block_performance(self, block_name: str):
        """
        Context manager for logging code block performance.
        
        Args:
            block_name: Name of the code block
            
        Returns:
            Performance context manager
        """
        return PerformanceContext(self.logger, block_name)


class PerformanceContext:
    """
    Context manager for performance logging of code blocks.
    """
    
    def __init__(self, logger: logging.Logger, block_name: str):
        """
        Initialize performance context.
        
        Args:
            logger: Logger instance
            block_name: Name of the code block
        """
        self.logger = logger
        self.block_name = block_name
        self.start_time = None
    
    def __enter__(self):
        """Enter performance monitoring context."""
        self.start_time = datetime.now(timezone.utc)
        self.logger.debug(f"Starting execution of block: {self.block_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit performance monitoring context."""
        if self.start_time:
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - self.start_time).total_seconds() * 1000
            
            if exc_type is None:
                # Success
                self.logger.info(
                    f"Block {self.block_name} completed successfully",
                    extra={
                        'duration_ms': duration_ms,
                        'block': self.block_name,
                        'performance': True,
                        'status': 'success'
                    }
                )
            else:
                # Error
                self.logger.error(
                    f"Block {self.block_name} failed with {exc_type.__name__}: {exc_val}",
                    extra={
                        'duration_ms': duration_ms,
                        'block': self.block_name,
                        'performance': True,
                        'status': 'error',
                        'error_type': exc_type.__name__ if exc_type else 'unknown'
                    }
                )


class LoggingManager:
    """
    Centralized logging manager for the application.
    
    This class provides a unified interface for configuring and managing
    all logging aspects of the application including handlers, formatters,
    and performance monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize logging manager with configuration.
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config
        self.loggers: Dict[str, logging.Logger] = {}
        self.performance_loggers: Dict[str, PerformanceLogger] = {}
        self._setup_root_logger()
        self._setup_handlers()
    
    def _setup_root_logger(self) -> None:
        """Configure the root logger with basic settings."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # Remove default handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    def _setup_handlers(self) -> None:
        """Configure logging handlers based on configuration."""
        # Console handler
        if self.config.get('log_to_console', True):
            self._setup_console_handler()
        
        # File handler
        if self.config.get('log_to_file', True):
            self._setup_file_handler()
        
        # Rotating file handler for errors
        self._setup_error_file_handler()
        
        # JSON file handler for structured logging
        if self.config.get('enable_structured_logging', True):
            self._setup_json_handler()
        
        # Performance log handler
        if self.config.get('enable_performance_logging', True):
            self._setup_performance_handler()
    
    def _setup_console_handler(self) -> None:
        """Setup colored console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.config.get('console_color', True):
            # Colored console formatter
            color_formatter = colorlog.ColoredFormatter(
                fmt='%(log_color)s%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green', 
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(color_formatter)
        else:
            # Plain console formatter
            plain_formatter = StructuredFormatter(include_context=False, json_format=False)
            console_handler.setFormatter(plain_formatter)
        
        # Add contextual filter
        console_handler.addFilter(ContextualFilter())
        console_handler.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # Add to root logger
        logging.getLogger().addHandler(console_handler)
    
    def _setup_file_handler(self) -> None:
        """Setup rotating file handler for general logging."""
        log_file_path = Path(self.config.get('log_file_path', 'data/logs/app.log'))
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=self.config.get('max_file_size_mb', 10) * 1024 * 1024,
            backupCount=self.config.get('backup_count', 5),
            encoding='utf-8'
        )
        
        # Text formatter for file logs
        file_formatter = StructuredFormatter(
            include_context=self.config.get('log_context', True),
            json_format=False
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(ContextualFilter())
        file_handler.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        logging.getLogger().addHandler(file_handler)
    
    def _setup_error_file_handler(self) -> None:
        """Setup dedicated error log file handler."""
        error_log_path = Path('data/logs/errors.log')
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        error_handler = logging.handlers.RotatingFileHandler(
            filename=str(error_log_path),
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        
        error_formatter = StructuredFormatter(include_context=True, json_format=False)
        error_handler.setFormatter(error_formatter)
        error_handler.addFilter(ContextualFilter())
        error_handler.setLevel(logging.ERROR)
        
        logging.getLogger().addHandler(error_handler)
    
    def _setup_json_handler(self) -> None:
        """Setup JSON structured logging handler."""
        json_log_path = Path('data/logs/structured.log')
        json_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        json_handler = logging.handlers.RotatingFileHandler(
            filename=str(json_log_path),
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=15,
            encoding='utf-8'
        )
        
        json_formatter = StructuredFormatter(include_context=True, json_format=True)
        json_handler.setFormatter(json_formatter)
        json_handler.addFilter(ContextualFilter())
        json_handler.setLevel(logging.DEBUG)
        
        logging.getLogger().addHandler(json_handler)
    
    def _setup_performance_handler(self) -> None:
        """Setup performance logging handler."""
        perf_log_path = Path('data/logs/performance.log')
        perf_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=str(perf_log_path),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Only log performance-related messages
        class PerformanceFilter(logging.Filter):
            def filter(self, record):
                return getattr(record, 'performance', False)
        
        perf_formatter = StructuredFormatter(include_context=True, json_format=True)
        perf_handler.setFormatter(perf_formatter)
        perf_handler.addFilter(PerformanceFilter())
        perf_handler.setLevel(logging.INFO)
        
        logging.getLogger().addHandler(perf_handler)
    
    def get_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with specified name and level.
        
        Args:
            name: Logger name (typically module name)
            level: Optional logging level override
            
        Returns:
            Configured logger instance
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            
            # Set module-specific level if configured
            module_levels = self.config.get('module_levels', {})
            if level:
                logger.setLevel(getattr(logging, level))
            elif name in module_levels:
                logger.setLevel(getattr(logging, module_levels[name]))
            
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def get_performance_logger(self, name: str) -> PerformanceLogger:
        """
        Get performance logger for specified module.
        
        Args:
            name: Module name
            
        Returns:
            Performance logger instance
        """
        if name not in self.performance_loggers:
            logger = self.get_logger(name)
            self.performance_loggers[name] = PerformanceLogger(logger)
        
        return self.performance_loggers[name]
    
    def set_context(self, request_id: Optional[str] = None, 
                   user_id: Optional[int] = None,
                   session_id: Optional[str] = None) -> None:
        """
        Set logging context variables for request tracking.
        
        Args:
            request_id: Unique request identifier
            user_id: User identifier
            session_id: Session identifier
        """
        if request_id:
            request_id_context.set(request_id)
        if user_id:
            user_id_context.set(user_id)
        if session_id:
            session_context.set(session_id)
    
    def clear_context(self) -> None:
        """Clear all context variables."""
        request_id_context.set(None)
        user_id_context.set(None)
        session_context.set(None)
    
    def log_health_check(self) -> Dict[str, Any]:
        """
        Perform and log system health check.
        
        Returns:
            Health check results
        """
        health_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'status': 'healthy',
            'checks': {}
        }
        
        # Check log file accessibility
        try:
            log_path = Path(self.config.get('log_file_path', 'data/logs/app.log'))
            if log_path.exists():
                health_data['checks']['log_file'] = {'status': 'ok', 'path': str(log_path)}
            else:
                health_data['checks']['log_file'] = {'status': 'warning', 'message': 'Log file not found'}
        except Exception as e:
            health_data['checks']['log_file'] = {'status': 'error', 'error': str(e)}
            health_data['status'] = 'degraded'
        
        # Check handler count
        handler_count = len(logging.getLogger().handlers)
        health_data['checks']['handlers'] = {'status': 'ok', 'count': handler_count}
        
        # Log health check
        health_logger = self.get_logger('health')
        health_logger.info("System health check completed", extra={'health_check': health_data})
        
        return health_data


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def setup_logging(config: Dict[str, Any]) -> LoggingManager:
    """
    Setup application logging with provided configuration.
    
    Args:
        config: Logging configuration dictionary
        
    Returns:
        Configured logging manager instance
    """
    global _logging_manager
    _logging_manager = LoggingManager(config)
    return _logging_manager


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance for specified module.
    
    Args:
        name: Logger name
        level: Optional logging level
        
    Returns:
        Logger instance
        
    Raises:
        RuntimeError: If logging not initialized
    """
    if _logging_manager is None:
        raise RuntimeError("Logging not initialized. Call setup_logging() first.")
    
    return _logging_manager.get_logger(name, level)


def get_performance_logger(name: str) -> PerformanceLogger:
    """
    Get performance logger for specified module.
    
    Args:
        name: Module name
        
    Returns:
        Performance logger instance
        
    Raises:
        RuntimeError: If logging not initialized
    """
    if _logging_manager is None:
        raise RuntimeError("Logging not initialized. Call setup_logging() first.")
    
    return _logging_manager.get_performance_logger(name)


def set_logging_context(**kwargs) -> None:
    """
    Set logging context variables.
    
    Args:
        **kwargs: Context variables (request_id, user_id, session_id)
    """
    if _logging_manager is None:
        return
    
    _logging_manager.set_context(**kwargs)


def clear_logging_context() -> None:
    """Clear all logging context variables."""
    if _logging_manager is None:
        return
    
    _logging_manager.clear_context()


# Export commonly used functions and classes
__all__ = [
    'LogContext',
    'ContextualFilter',
    'StructuredFormatter',
    'PerformanceLogger',
    'PerformanceContext',
    'LoggingManager',
    'setup_logging',
    'get_logger',
    'get_performance_logger',
    'set_logging_context',
    'clear_logging_context'
]
