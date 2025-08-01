"""
Integration module for connecting observability with external services.
"""

from typing import Any

from langfuse import Langfuse
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from .logger import get_observable_logger


class ObservabilityIntegration:
    """
    Integration class for connecting with external observability services.

    This class can be extended to integrate with services like:
    - Langfuse for LLM observability
    - OpenTelemetry for distributed tracing
    - Prometheus for metrics
    - Custom monitoring solutions
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize observability integration.

        Args:
            config: Configuration for external services
        """
        self.config = config or {}
        self.logger = get_observable_logger("observability_integration", "common")
        self._external_service: Any | None = None

        # Initialize external service if configured
        self._initialize_external_service()

    def _initialize_external_service(self) -> None:
        """Initialize external observability service."""
        service_type = self.config.get("service_type")

        if service_type == "langfuse":
            self._initialize_langfuse()
        elif service_type == "opentelemetry":
            self._initialize_opentelemetry()
        else:
            self.logger.info(
                "observability_service_not_configured",
                available_types=["langfuse", "opentelemetry"],
            )

    def _initialize_langfuse(self) -> None:
        """Initialize Langfuse integration."""
        try:
            langfuse_config = self.config.get("langfuse", {})
            self._external_service = Langfuse(
                secret_key=langfuse_config.get("secret_key"),
                public_key=langfuse_config.get("public_key"),
                host=langfuse_config.get("host", "https://cloud.langfuse.com"),
            )

            self.logger.info("langfuse_initialized", host=langfuse_config.get("host"))

        except ImportError:
            self.logger.warning(
                "langfuse_not_available",
                message="Install langfuse package to enable Langfuse integration",
            )
        except Exception as exception:
            self.logger.error(
                "langfuse_initialization_failed",
                error_type=type(exception).__name__,
                error_message=str(exception),
            )

    def _initialize_opentelemetry(self) -> None:
        """Initialize OpenTelemetry integration with tracing and metrics."""
        try:
            otel_config = self.config.get("opentelemetry", {})

            # Create resource with service information
            resource = Resource(
                attributes={
                    SERVICE_NAME: otel_config.get("service_name", "sqlite-kg-vec-mcp"),
                    SERVICE_VERSION: otel_config.get("service_version", "0.2.0"),
                }
            )

            # Configure tracing
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)

            # Add span processors
            if otel_config.get("endpoint"):
                # OTLP exporter for production
                otlp_exporter = OTLPSpanExporter(
                    endpoint=otel_config["endpoint"],
                    headers=otel_config.get("headers", {}),
                    insecure=otel_config.get("insecure", True),
                    timeout=otel_config.get("timeout", 30),
                )
                tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            else:
                # Console exporter for development
                console_exporter = ConsoleSpanExporter()
                tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

            # Configure metrics
            if otel_config.get("endpoint"):
                # OTLP metrics exporter
                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(
                        endpoint=otel_config["endpoint"].replace("/v1/traces", "/v1/metrics"),
                        headers=otel_config.get("headers", {}),
                        insecure=otel_config.get("insecure", True),
                        timeout=otel_config.get("timeout", 30),
                    ),
                    export_interval_millis=30000,  # 30 seconds
                )
            else:
                # Console metrics exporter for development
                metric_reader = PeriodicExportingMetricReader(
                    ConsoleMetricExporter(), export_interval_millis=30000
                )

            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)

            # Auto-instrumentation
            RequestsInstrumentor().instrument()
            SQLite3Instrumentor().instrument()

            # Store tracer and meter for use
            tracer = trace.get_tracer(__name__)
            meter = metrics.get_meter(__name__)

            self._external_service = {
                "tracer": tracer,
                "meter": meter,
                "tracer_provider": tracer_provider,
                "meter_provider": meter_provider,
            }

            self.logger.info(
                "opentelemetry_initialized",
                service_name=otel_config.get("service_name", "sqlite-kg-vec-mcp"),
                endpoint=otel_config.get("endpoint", "console"),
                auto_instrumentation=True,
            )

        except ImportError as e:
            self.logger.warning(
                "opentelemetry_not_available",
                message="Install opentelemetry packages to enable tracing",
                missing_package=str(e),
            )
        except Exception as exception:
            self.logger.error(
                "opentelemetry_initialization_failed",
                error_type=type(exception).__name__,
                error_message=str(exception),
            )

    def get_external_service(self) -> Any | None:
        """
        Get the external observability service instance.

        Returns:
            External service instance or None if not configured
        """
        return self._external_service

    def create_trace(self, name: str, **metadata) -> str | None:
        """
        Create a trace in the external service.

        Args:
            name: Trace name
            **metadata: Additional metadata

        Returns:
            Trace ID if successful
        """
        if not self._external_service:
            return None

        try:
            if hasattr(self._external_service, "trace"):
                # Langfuse-style
                trace_obj = self._external_service.trace(name=name, **metadata)
                return str(trace_obj.id)
            if isinstance(self._external_service, dict) and "tracer" in self._external_service:
                # OpenTelemetry-style (new format)
                tracer = self._external_service["tracer"]
                span = tracer.start_span(name=name)
                for key, value in metadata.items():
                    span.set_attribute(key, str(value))
                trace_id = str(span.get_span_context().trace_id)
                span.end()
                return trace_id
            if hasattr(self._external_service, "start_span"):
                # OpenTelemetry-style (legacy)
                span = self._external_service.start_span(
                    name=name
                )  # pylint: disable=too-many-function-args
                for key, value in metadata.items():
                    span.set_attribute(key, str(value))
                trace_id = str(span.get_span_context().trace_id)
                span.end()
                return trace_id
        except Exception as exception:
            self.logger.error(
                "external_trace_creation_failed",
                trace_name=name,
                error_type=type(exception).__name__,
                error_message=str(exception),
            )

        return None

    def log_llm_generation(
        self, trace_id: str, model: str, prompt: str, response: str, **metadata
    ) -> None:
        """
        Log LLM generation to external service.

        Args:
            trace_id: Trace identifier
            model: Model name
            prompt: Input prompt
            response: Generated response
            **metadata: Additional metadata
        """
        if not self._external_service:
            return

        try:
            if hasattr(self._external_service, "generation"):
                # Langfuse-style
                self._external_service.generation(
                    trace_id=trace_id,
                    name="llm_generation",
                    model=model,
                    input=prompt,
                    output=response,
                    **metadata,
                )

                self.logger.debug(
                    "llm_generation_logged",
                    trace_id=trace_id,
                    model=model,
                    prompt_length=len(prompt),
                    response_length=len(response),
                )
        except Exception as exception:
            self.logger.error(
                "llm_generation_logging_failed",
                trace_id=trace_id,
                error_type=type(exception).__name__,
                error_message=str(exception),
            )

    def record_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """
        Record a metric to external service.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        if not self._external_service:
            return

        try:
            if isinstance(self._external_service, dict) and "meter" in self._external_service:
                # OpenTelemetry metrics
                meter = self._external_service["meter"]

                # Create counter or gauge based on metric name pattern
                if "count" in name.lower() or "total" in name.lower():
                    counter = meter.create_counter(name, description=f"Counter for {name}")
                    counter.add(value, attributes=tags or {})
                else:
                    # For other metrics, use histogram
                    histogram = meter.create_histogram(name, description=f"Histogram for {name}")
                    histogram.record(value, attributes=tags or {})

                self.logger.debug(
                    "opentelemetry_metric_recorded",
                    metric_name=name,
                    metric_value=value,
                    metric_tags=tags or {},
                )
            else:
                # Fallback: just log the metric
                self.logger.debug(
                    "metric_recorded",
                    metric_name=name,
                    metric_value=value,
                    metric_tags=tags or {},
                )
        except Exception as exception:
            self.logger.error(
                "metric_recording_failed",
                metric_name=name,
                error_type=type(exception).__name__,
                error_message=str(exception),
            )

    def start_span(self, name: str, **attributes):
        """
        Start a new span with OpenTelemetry context manager.

        Args:
            name: Span name
            **attributes: Span attributes

        Returns:
            Context manager for the span
        """
        if isinstance(self._external_service, dict) and "tracer" in self._external_service:
            tracer = self._external_service["tracer"]
            span = tracer.start_span(name)
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            return span
        return None

    def flush(self) -> None:
        """Flush any pending data to external service."""
        if not self._external_service:
            return

        try:
            if hasattr(self._external_service, "flush"):
                self._external_service.flush()
                self.logger.debug("observability_data_flushed")
            elif isinstance(self._external_service, dict):
                # Flush OpenTelemetry providers
                if "tracer_provider" in self._external_service:
                    self._external_service["tracer_provider"].force_flush(30)
                if "meter_provider" in self._external_service:
                    self._external_service["meter_provider"].force_flush(30)
                self.logger.debug("opentelemetry_data_flushed")
        except Exception as exception:
            self.logger.error(
                "observability_flush_failed",
                error_type=type(exception).__name__,
                error_message=str(exception),
            )


class ObservabilityManager:
    """Manages observability integration instances."""

    _instance: ObservabilityIntegration | None = None

    @classmethod
    def initialize(cls, config: dict[str, Any] | None = None) -> ObservabilityIntegration:
        """
        Initialize observability integration.

        Args:
            config: Configuration for observability services

        Returns:
            ObservabilityIntegration instance
        """
        cls._instance = ObservabilityIntegration(config)
        return cls._instance

    @classmethod
    def get_instance(cls) -> ObservabilityIntegration | None:
        """
        Get the observability integration instance.

        Returns:
            ObservabilityIntegration instance or None if not initialized
        """
        return cls._instance


# Backward compatibility functions
def initialize_observability(
    config: dict[str, Any] | None = None,
) -> ObservabilityIntegration:
    """Initialize global observability integration (deprecated)."""
    return ObservabilityManager.initialize(config)


def get_observability_integration() -> ObservabilityIntegration | None:
    """Get the global observability integration instance (deprecated)."""
    return ObservabilityManager.get_instance()
