"""
Logging configuration and setup.
"""

import json
import logging as stdlib_logging
import os
import sys
import traceback
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from ..config.observability import LoggingObservabilityConfig


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
    file_path: str | None = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    include_trace: bool = True
    include_caller: bool = False
    sanitize_sensitive_data: bool = True


def configure_structured_logging(
    config: LoggingConfig | None = None,
    observability_config: LoggingObservabilityConfig | None = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        config: Logging configuration (uses defaults if None, deprecated)
        observability_config: New observability logging configuration
    """
    if observability_config is not None:
        # Convert observability config to logging config
        config = LoggingConfig(
            level=LogLevel(observability_config.level),
            format=observability_config.format,
            output=observability_config.output,
            file_path=observability_config.file_path,
            include_trace=observability_config.include_trace,
            include_caller=observability_config.include_caller,
            sanitize_sensitive_data=observability_config.sanitize_sensitive_data,
        )
    elif config is None:
        config = LoggingConfig()

    _configure_structlog(config)


def _configure_structlog(config: LoggingConfig) -> None:
    """Configure structlog for structured logging."""

    # Configure stdlib logging
    stdlib_logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if config.output == "console" else None,
        level=getattr(stdlib_logging, config.level.value),
    )

    # Build processor chain
    processors: list[
        Callable[
            [Any, str, MutableMapping[str, Any]],
            Mapping[str, Any] | str | bytes | bytearray | tuple,
        ]
    ] = [
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

    # Create formatter
    formatter: Any
    if config.format == "json":
        formatter = _JSONFormatter()
    else:
        formatter = stdlib_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure root logger
    root_logger = stdlib_logging.getLogger()
    root_logger.setLevel(getattr(stdlib_logging, config.level.value))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add appropriate handler
    if config.output == "console":
        handler = stdlib_logging.StreamHandler(sys.stdout)
    else:
        handler = stdlib_logging.FileHandler(config.file_path or "app.log")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def _sanitize_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
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

    def sanitize_dict(d: Any) -> Any:
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
                sanitized[key] = str(
                    [sanitize_dict(item) if isinstance(item, dict) else item for item in value]
                )
            else:
                sanitized[key] = value
        return sanitized

    result = sanitize_dict(event_dict)
    return result if isinstance(result, dict) else event_dict


class _JSONFormatter:
    """JSON formatter for standard library logging."""

    def __init__(self):
        self.json = json

    def format(self, record):
        """Format log record as JSON."""
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

    Deprecated: Use LoggingObservabilityConfig.from_env() instead.

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


def get_observability_logging_config_from_env() -> LoggingObservabilityConfig:
    """
    Get observability logging configuration from environment variables.

    Returns:
        LoggingObservabilityConfig instance
    """
    return LoggingObservabilityConfig()
