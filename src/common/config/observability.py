"""
Observability and monitoring configuration settings.
"""

from typing import Dict, Optional

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class LangfuseConfig(BaseSettings):
    """Langfuse observability service configuration."""
    
    enabled: bool = Field(
        default=False,
        description="Enable Langfuse integration"
    )
    
    host: Optional[str] = Field(
        default=None,
        description="Langfuse server host"
    )
    
    public_key: Optional[str] = Field(
        default=None,
        description="Langfuse public key"
    )
    
    secret_key: Optional[str] = Field(
        default=None,
        description="Langfuse secret key"
    )
    
    project_name: Optional[str] = Field(
        default=None,
        description="Langfuse project name"
    )
    
    flush_interval: float = Field(
        default=5.0,
        description="Flush interval for batched events"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug logging for Langfuse"
    )

    model_config = {
        "env_prefix": "LANGFUSE_",
        "extra": "ignore"
    }


class PrometheusConfig(BaseSettings):
    """Prometheus metrics configuration."""
    
    enabled: bool = Field(
        default=False,
        description="Enable Prometheus metrics"
    )
    
    port: int = Field(
        default=8080,
        description="Prometheus metrics server port"
    )
    
    host: str = Field(
        default="0.0.0.0",
        description="Prometheus metrics server host"
    )
    
    path: str = Field(
        default="/metrics",
        description="Metrics endpoint path"
    )
    
    namespace: str = Field(
        default="sqlite_kg_vec",
        description="Metrics namespace"
    )
    
    job_name: str = Field(
        default="knowledge_graph",
        description="Job name for metrics"
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    model_config = {
        "env_prefix": "PROMETHEUS_",
        "extra": "ignore"
    }


class OpenTelemetryConfig(BaseSettings):
    """OpenTelemetry tracing configuration."""
    
    enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing"
    )
    
    service_name: str = Field(
        default="sqlite-kg-vec-mcp",
        description="Service name for tracing"
    )
    
    service_version: str = Field(
        default="0.2.0",
        description="Service version for tracing"
    )
    
    endpoint: Optional[str] = Field(
        default=None,
        description="OpenTelemetry collector endpoint"
    )
    
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers for tracing export"
    )
    
    compression: str = Field(
        default="gzip",
        description="Compression for trace export"
    )
    
    timeout: float = Field(
        default=30.0,
        description="Export timeout in seconds"
    )
    
    insecure: bool = Field(
        default=True,
        description="Use insecure connection for tracing"
    )

    model_config = {
        "env_prefix": "OTEL_",
        "extra": "ignore"
    }


class LoggingObservabilityConfig(BaseSettings):
    """Logging-specific observability configuration."""
    
    level: str = Field(
        default="INFO",
        description="Default logging level"
    )
    
    format: str = Field(
        default="json",
        description="Log format (json, text)"
    )
    
    output: str = Field(
        default="console",
        description="Log output (console, file)"
    )
    
    file_path: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    
    include_trace: bool = Field(
        default=True,
        description="Include trace information in logs"
    )
    
    include_caller: bool = Field(
        default=False,
        description="Include caller information in logs"
    )
    
    sanitize_sensitive_data: bool = Field(
        default=True,
        description="Sanitize sensitive data in logs"
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        """Validate log format."""
        valid_formats = {"json", "text"}
        if v not in valid_formats:
            raise ValueError(f"Format must be one of {valid_formats}")
        return v

    model_config = {
        "env_prefix": "LOG_",
        "extra": "ignore"
    }


class ObservabilityConfig(BaseSettings):
    """
    Combined observability configuration settings.
    
    Includes settings for logging, tracing, metrics, and external services.
    """
    
    # Enable/disable observability
    enabled: bool = Field(
        default=True,
        description="Enable observability features"
    )
    
    # Service identification
    service_name: str = Field(
        default="sqlite-kg-vec-mcp",
        description="Service name for observability"
    )
    
    service_version: str = Field(
        default="0.2.0",
        description="Service version for observability"
    )
    
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)"
    )
    
    # Component configurations
    logging: LoggingObservabilityConfig = Field(
        default_factory=LoggingObservabilityConfig,
        description="Logging configuration"
    )
    
    langfuse: LangfuseConfig = Field(
        default_factory=LangfuseConfig,
        description="Langfuse configuration"
    )
    
    prometheus: PrometheusConfig = Field(
        default_factory=PrometheusConfig,
        description="Prometheus configuration"
    )
    
    opentelemetry: OpenTelemetryConfig = Field(
        default_factory=OpenTelemetryConfig,
        description="OpenTelemetry configuration"
    )
    
    # Sampling and performance
    trace_sampling_ratio: float = Field(
        default=1.0,
        description="Trace sampling ratio (0.0-1.0)"
    )
    
    metrics_interval: float = Field(
        default=60.0,
        description="Metrics collection interval in seconds"
    )

    @field_validator("trace_sampling_ratio")
    @classmethod
    def validate_trace_sampling_ratio(cls, v):
        """Validate trace sampling ratio."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Trace sampling ratio must be between 0.0 and 1.0")
        return v
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = {"development", "staging", "production"}
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v

    model_config = {
        "env_prefix": "OBSERVABILITY_",
        "env_file": ".env",
        "extra": "ignore"
    }