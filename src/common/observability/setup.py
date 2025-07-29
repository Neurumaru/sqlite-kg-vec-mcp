"""
Setup and initialization utilities for the observability system.
"""

import os
from typing import Any, Dict, Optional

from ..logging.config import (
    LoggingConfig,
    configure_structured_logging,
    get_logging_config_from_env,
)
from .integration import initialize_observability
from .logger import get_observable_logger


def setup_observability(
    logging_config: Optional[Dict[str, Any]] = None,
    observability_config: Optional[Dict[str, Any]] = None,
    auto_configure: bool = True,
) -> None:
    """
    Complete setup for the unified logging and observability system.

    This function should be called at application startup to configure:
    - Structured logging (with structlog if available)
    - Trace context management
    - External observability service integration

    Args:
        logging_config: Logging configuration override
        observability_config: Observability service configuration
        auto_configure: Whether to auto-configure from environment variables
    """

    # 1. Configure structured logging
    if auto_configure and logging_config is None:
        # Get configuration from environment variables
        config = get_logging_config_from_env()
        configure_structured_logging(config)
    elif logging_config:
        config = LoggingConfig(**logging_config)
        configure_structured_logging(config)
    else:
        # Use defaults
        configure_structured_logging()

    # 2. Initialize observability integration
    if auto_configure and observability_config is None:
        observability_config = get_observability_config_from_env()

    if observability_config:
        initialize_observability(observability_config)

    # Log successful setup
    logger = get_observable_logger("observability_setup", "common")
    logger.info(
        "observability_system_initialized",
        structured_logging=True,
        external_integration=bool(observability_config),
        auto_configured=auto_configure,
    )


def get_observability_config_from_env() -> Dict[str, Any]:
    """
    Get observability configuration from environment variables.

    Environment variables:
    - OBSERVABILITY_SERVICE: Service type (langfuse, opentelemetry, none)
    - LANGFUSE_SECRET_KEY: Langfuse secret key
    - LANGFUSE_PUBLIC_KEY: Langfuse public key
    - LANGFUSE_HOST: Langfuse host (default: https://cloud.langfuse.com)
    - JAEGER_HOST: Jaeger host for OpenTelemetry (default: localhost)
    - JAEGER_PORT: Jaeger port for OpenTelemetry (default: 6831)

    Returns:
        Configuration dictionary
    """
    service_type = os.getenv("OBSERVABILITY_SERVICE", "none").lower()

    if service_type == "none":
        return {}

    config: Dict[str, Any] = {"service_type": service_type}

    if service_type == "langfuse":
        langfuse_config = {
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        }
        config["langfuse"] = langfuse_config

        # Only include if keys are provided
        if not langfuse_config["secret_key"] or not langfuse_config["public_key"]:
            return {}

    elif service_type == "opentelemetry":
        jaeger_port_str = os.getenv("JAEGER_PORT", "6831")
        config["opentelemetry"] = {
            "jaeger_host": os.getenv("JAEGER_HOST", "localhost"),
            "jaeger_port": int(jaeger_port_str),
        }

    return config


def quick_setup() -> None:
    """
    Quick setup with sensible defaults for development.

    This is a convenience function for getting started quickly.
    It configures:
    - JSON logging to console
    - INFO level logging
    - Automatic environment variable detection
    """
    setup_observability(
        logging_config={
            "level": "INFO",
            "format": "json",
            "output": "console",
            "include_trace": True,
            "sanitize_sensitive_data": True,
        },
        auto_configure=True,
    )


def production_setup() -> None:
    """
    Production setup with appropriate settings.

    This configures:
    - Structured JSON logging
    - WARNING level logging (reduce noise)
    - Full observability integration
    - Sensitive data sanitization
    """
    setup_observability(
        logging_config={
            "level": "WARNING",
            "format": "json",
            "output": "console",
            "include_trace": True,
            "include_caller": False,
            "sanitize_sensitive_data": True,
        },
        auto_configure=True,
    )


def development_setup() -> None:
    """
    Development setup with verbose logging.

    This configures:
    - Human-readable text logging
    - DEBUG level logging
    - Full trace information
    - No external services (for faster development)
    """
    setup_observability(
        logging_config={
            "level": "DEBUG",
            "format": "text",  # Human-readable for development
            "output": "console",
            "include_trace": True,
            "include_caller": True,
            "sanitize_sensitive_data": False,
        },
        observability_config={},  # No external services
        auto_configure=False,
    )
