"""
Integration module for connecting observability with external services.
"""

from typing import Optional, Any, Dict
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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize observability integration.
        
        Args:
            config: Configuration for external services
        """
        self.config = config or {}
        self.logger = get_observable_logger("observability_integration", "common")
        self._external_service = None
        
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
            self.logger.info("observability_service_not_configured",
                           available_types=["langfuse", "opentelemetry"])
    
    def _initialize_langfuse(self) -> None:
        """Initialize Langfuse integration."""
        try:
            from langfuse import Langfuse
            
            langfuse_config = self.config.get("langfuse", {})
            self._external_service = Langfuse(
                secret_key=langfuse_config.get("secret_key"),
                public_key=langfuse_config.get("public_key"),
                host=langfuse_config.get("host", "https://cloud.langfuse.com")
            )
            
            self.logger.info("langfuse_initialized",
                           host=langfuse_config.get("host"))
            
        except ImportError:
            self.logger.warning("langfuse_not_available",
                              message="Install langfuse package to enable Langfuse integration")
        except Exception as e:
            self.logger.error("langfuse_initialization_failed",
                            error_type=type(e).__name__,
                            error_message=str(e))
    
    def _initialize_opentelemetry(self) -> None:
        """Initialize OpenTelemetry integration."""
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            
            # Configure OpenTelemetry
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            # Configure Jaeger exporter
            jaeger_config = self.config.get("opentelemetry", {})
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_config.get("jaeger_host", "localhost"),
                agent_port=jaeger_config.get("jaeger_port", 6831),
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self._external_service = tracer
            
            self.logger.info("opentelemetry_initialized",
                           jaeger_host=jaeger_config.get("jaeger_host"))
            
        except ImportError:
            self.logger.warning("opentelemetry_not_available",
                              message="Install opentelemetry packages to enable tracing")
        except Exception as e:
            self.logger.error("opentelemetry_initialization_failed",
                            error_type=type(e).__name__,
                            error_message=str(e))
    
    def get_external_service(self) -> Optional[Any]:
        """
        Get the external observability service instance.
        
        Returns:
            External service instance or None if not configured
        """
        return self._external_service
    
    def create_trace(self, name: str, **metadata) -> Optional[str]:
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
            if hasattr(self._external_service, 'trace'):
                # Langfuse-style
                trace = self._external_service.trace(name=name, **metadata)
                return trace.id
            elif hasattr(self._external_service, 'start_span'):
                # OpenTelemetry-style
                span = self._external_service.start_span(name)
                for key, value in metadata.items():
                    span.set_attribute(key, str(value))
                return span.get_span_context().trace_id
        except Exception as e:
            self.logger.error("external_trace_creation_failed",
                            trace_name=name,
                            error_type=type(e).__name__,
                            error_message=str(e))
        
        return None
    
    def log_llm_generation(
        self,
        trace_id: str,
        model: str,
        prompt: str,
        response: str,
        **metadata
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
            if hasattr(self._external_service, 'generation'):
                # Langfuse-style
                self._external_service.generation(
                    trace_id=trace_id,
                    name="llm_generation",
                    model=model,
                    input=prompt,
                    output=response,
                    **metadata
                )
                
                self.logger.debug("llm_generation_logged",
                                trace_id=trace_id,
                                model=model,
                                prompt_length=len(prompt),
                                response_length=len(response))
        except Exception as e:
            self.logger.error("llm_generation_logging_failed",
                            trace_id=trace_id,
                            error_type=type(e).__name__,
                            error_message=str(e))
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
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
            # Custom metric recording logic can be added here
            # For now, just log the metric
            self.logger.debug("metric_recorded",
                            metric_name=name,
                            metric_value=value,
                            metric_tags=tags or {})
        except Exception as e:
            self.logger.error("metric_recording_failed",
                            metric_name=name,
                            error_type=type(e).__name__,
                            error_message=str(e))
    
    def flush(self) -> None:
        """Flush any pending data to external service."""
        if not self._external_service:
            return
        
        try:
            if hasattr(self._external_service, 'flush'):
                self._external_service.flush()
                self.logger.debug("observability_data_flushed")
        except Exception as e:
            self.logger.error("observability_flush_failed",
                            error_type=type(e).__name__,
                            error_message=str(e))


# Global integration instance
_observability_integration: Optional[ObservabilityIntegration] = None


def initialize_observability(config: Optional[Dict[str, Any]] = None) -> ObservabilityIntegration:
    """
    Initialize global observability integration.
    
    Args:
        config: Configuration for observability services
        
    Returns:
        ObservabilityIntegration instance
    """
    global _observability_integration
    
    _observability_integration = ObservabilityIntegration(config)
    return _observability_integration


def get_observability_integration() -> Optional[ObservabilityIntegration]:
    """
    Get the global observability integration instance.
    
    Returns:
        ObservabilityIntegration instance or None if not initialized
    """
    return _observability_integration