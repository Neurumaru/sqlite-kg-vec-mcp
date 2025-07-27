"""
Logging configuration and setup.
"""

import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import structlog


class LogLevel(Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LoggingConfig:
    """
    Logging configuration settings.
    """

    level: LogLevel = LogLevel.INFO
    format: str = "json"  # json or text
    output: str = "console"  # console or file
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    include_trace: bool = True
    include_caller: bool = False
    sanitize_sensitive_data: bool = True


def configure_structured_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure structured logging for the application.

    Args:
        config: Logging configuration (uses defaults if None)
    """
    if config is None:
        config = LoggingConfig()

    _configure_structlog(config)


def _configure_structlog(config: LoggingConfig) -> None:
    """Configure structlog for structured logging."""
    import logging

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if config.output == "console" else None,
        level=getattr(logging, config.level.value),
    )

    # Build processor chain
    processors = [
        # Filter by level
        structlog.stdlib.filter_by_level,
        # Add log level and logger name
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Handle positional arguments
        structlog.stdlib.PositionalArgumentsFormatter(),
        # Add stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Handle unicode
        structlog.processors.UnicodeDecoder(),
    ]

    # Add caller info if requested
    if config.include_caller:
        processors.append(structlog.processors.CallsiteParameterAdder())

    # Add sanitization processor
    if config.sanitize_sensitive_data:
        processors.append(_sanitize_processor)

    # Add final renderer
    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _configure_stdlib_logging(config: LoggingConfig) -> None:
    """Configure standard library logging as fallback."""
    import logging

    # Create formatter
    if config.format == "json":
        formatter = _JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.value))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add appropriate handler
    if config.output == "console":
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(config.file_path or "app.log")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def _sanitize_processor(logger, method_name, event_dict):
    """Processor to sanitize sensitive data from logs."""
    sensitive_keys = {
        "password",
        "token",
        "secret",
        "key",
        "auth",
        "credential",
        "api_key",
        "private_key",
    }

    def sanitize_dict(d):
        if not isinstance(d, dict):
            return d

        sanitized = {}
        for key, value in d.items():
            key_lower = str(key).lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    sanitize_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    return sanitize_dict(event_dict)


class _JSONFormatter(object):
    """JSON formatter for standard library logging."""

    def __init__(self):
        import json

        self.json = json

    def format(self, record):
        """Format log record as JSON."""
        import traceback

        log_entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = traceback.format_exception(*record.exc_info)

        return self.json.dumps(log_entry)


def get_logging_config_from_env() -> LoggingConfig:
    """
    Get logging configuration from environment variables.

    Environment variables:
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - LOG_FORMAT: Output format (json, text)
    - LOG_OUTPUT: Output destination (console, file)
    - LOG_FILE: Log file path (when output=file)
    - LOG_INCLUDE_TRACE: Include trace information (true/false)
    - LOG_INCLUDE_CALLER: Include caller information (true/false)
    - LOG_SANITIZE: Sanitize sensitive data (true/false)

    Returns:
        LoggingConfig instance
    """
    return LoggingConfig(
        level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
        format=os.getenv("LOG_FORMAT", "json"),
        output=os.getenv("LOG_OUTPUT", "console"),
        file_path=os.getenv("LOG_FILE"),
        include_trace=os.getenv("LOG_INCLUDE_TRACE", "true").lower() == "true",
        include_caller=os.getenv("LOG_INCLUDE_CALLER", "false").lower() == "true",
        sanitize_sensitive_data=os.getenv("LOG_SANITIZE", "true").lower() == "true",
    )
