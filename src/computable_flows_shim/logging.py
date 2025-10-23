import logging
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps
import time


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record) -> str:
        """Format log record as JSON with structured data."""
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
        }

        # Add extra fields if present (fields beyond standard LogRecord attributes)
        standard_fields = {
            'name', 'msg', 'args', 'level', 'levelno', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
            'processName', 'process', 'message', 'asctime'
        }

        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in standard_fields:
                extra_fields[key] = value

        if extra_fields:
            log_entry['extra'] = extra_fields  # type: ignore

        return json.dumps(log_entry)


class SDKLogger:
    """Centralized logging configuration for AFS SDK."""

    _configured = False

    @staticmethod
    def configure(
        level: str = "WARNING",
        format: str = "json",
        output: str = "stderr",
        log_file: Optional[str] = None,
        enable_performance_logging: bool = False
    ):
        """Configure SDK-wide logging with validation."""
        # Input validation (lightweight contracts)
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")

        valid_formats = ["json", "text"]
        if format not in valid_formats:
            raise ValueError(f"Invalid log format: {format}. Must be one of {valid_formats}")

        valid_outputs = ["stderr", "stdout", "file", "null"]
        if output not in valid_outputs:
            raise ValueError(f"Invalid log output: {output}. Must be one of {valid_outputs}")

        if output == "file" and log_file and not os.path.isabs(log_file) and not log_file.startswith("logs/"):
            raise ValueError("log_file must be an absolute path or start with 'logs/' when output='file'")

        # Set root logger level
        numeric_level = getattr(logging, level.upper(), logging.WARNING)
        logger = logging.getLogger('afs')
        logger.setLevel(numeric_level)

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add new handler based on output
        if output == "stderr":
            handler = logging.StreamHandler()
        elif output == "stdout":
            handler = logging.StreamHandler(sys.stdout)
        elif output == "file":
            # Generate timestamped log file in logs directory if no specific file given
            if not log_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"logs/afs_{timestamp}.log"
            handler = logging.FileHandler(log_file)
        elif output == "null":
            handler = logging.NullHandler()
        else:
            handler = logging.NullHandler()

        # Set formatter
        if format == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))

        logger.addHandler(handler)
        logger.propagate = False  # Don't propagate to root logger

        SDKLogger._configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for a specific module."""
    return logging.getLogger(f'afs.{name}')


def log_performance(logger: logging.Logger, operation: str):
    """Decorator for timing critical operations with lazy evaluation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Only log if DEBUG level is enabled (lazy evaluation)
            if not logger.isEnabledFor(logging.DEBUG):
                return func(*args, **kwargs)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                logger.debug(f"{operation} completed",
                           extra={'duration_ms': duration * 1000, 'success': True})
                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(f"{operation} failed",
                           extra={'duration_ms': duration * 1000, 'error': str(e), 'success': False})
                raise
        return wrapper
    return decorator


def configure_logging(
    level: str = "WARNING",
    format: str = "json",
    output: str = "stderr",
    log_file: Optional[str] = None,
    enable_performance_logging: bool = False
):
    """Public API for configuring SDK logging."""
    SDKLogger.configure(
        level=level,
        format=format,
        output=output,
        log_file=log_file,
        enable_performance_logging=enable_performance_logging
    )


# Environment variable configuration
def _configure_from_env():
    """Configure logging from environment variables if present."""
    if os.getenv('AFS_LOG_LEVEL'):
        configure_logging(
            level=os.getenv('AFS_LOG_LEVEL', 'WARNING'),
            format=os.getenv('AFS_LOG_FORMAT', 'json'),
            output=os.getenv('AFS_LOG_OUTPUT', 'stderr'),
            log_file=os.getenv('AFS_LOG_FILE'),
            enable_performance_logging=os.getenv('AFS_LOG_PERFORMANCE', '').lower() == 'true'
        )


# Auto-configure from environment on import
_configure_from_env()