"""
관찰 가능성을 외부 서비스와 연결하기 위한 통합 모듈.
"""

from typing import Any, Optional

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

from src.domain.config.timeout_config import TimeoutConfig

from .logger import get_observable_logger


class ObservabilityIntegration:
    """
    외부 관찰 가능성 서비스와 연결하기 위한 통합 클래스.

    이 클래스는 다음과 같은 서비스와 통합하도록 확장될 수 있습니다:
    - LLM 관찰 가능성을 위한 Langfuse
    - 분산 트레이싱을 위한 OpenTelemetry
    - 메트릭스를 위한 Prometheus
    - 커스텀 모니터링 솔루션
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        timeout_config: Optional[TimeoutConfig] = None,
    ):
        """
        관찰 가능성 통합을 초기화합니다.

        인자:
            config: 외부 서비스 구성
            timeout_config: 타임아웃 설정 객체
        """
        self.config = config or {}
        self.timeout_config = timeout_config or TimeoutConfig.from_env()
        self.logger = get_observable_logger("observability_integration", "common")
        self._external_service: Optional[Any] = None

        # 구성된 경우 외부 서비스 초기화
        self._initialize_external_service()

    def _initialize_external_service(self) -> None:
        """외부 관찰 가능성 서비스를 초기화합니다."""
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
        """Langfuse 통합을 초기화합니다."""
        try:
            langfuse_config = self.config.get("langfuse", {})
            self._external_service = Langfuse(
                secret_key=langfuse_config.get("secret_key"),
                public_key=langfuse_config.get("public_key"),
                host=langfuse_config.get("host", "https://cloud.langfuse.com"),
            )

            self.logger.info("langfuse_initialized", host=langfuse_config.get("host"))

        except ImportError:  # exception 변수명으로 변경
            self.logger.warning(
                "langfuse_not_available",
                message="Langfuse 통합을 활성화하려면 langfuse 패키지를 설치하세요",
            )
        except Exception as exception:
            self.logger.error(
                "langfuse_initialization_failed",
                error_type=type(exception).__name__,
                error_message=str(exception),
            )

    def _initialize_opentelemetry(self) -> None:
        """트레이싱 및 메트릭스를 사용하여 OpenTelemetry 통합을 초기화합니다."""
        try:
            otel_config = self.config.get("opentelemetry", {})

            # 서비스 정보로 리소스 생성
            resource = Resource(
                attributes={
                    SERVICE_NAME: otel_config.get("service_name", "sqlite-kg-vec-mcp"),
                    SERVICE_VERSION: otel_config.get("service_version", "0.2.0"),
                }
            )

            # 트레이싱 구성
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)

            # 스팬 프로세서 추가
            if otel_config.get("endpoint"):
                # 프로덕션을 위한 OTLP 익스포터
                otlp_exporter = OTLPSpanExporter(
                    endpoint=otel_config["endpoint"],
                    headers=otel_config.get("headers", {}),
                    insecure=otel_config.get("insecure", True),
                    timeout=int(self.timeout_config.observability_export_timeout),
                )
                tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            else:
                # 개발을 위한 콘솔 익스포터
                console_exporter = ConsoleSpanExporter()
                tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

            # 메트릭스 구성
            if otel_config.get("endpoint"):
                # OTLP 메트릭스 익스포터
                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(
                        endpoint=otel_config["endpoint"].replace("/v1/traces", "/v1/metrics"),
                        headers=otel_config.get("headers", {}),
                        insecure=otel_config.get("insecure", True),
                        timeout=int(self.timeout_config.observability_export_timeout),
                    ),
                    export_interval_millis=int(
                        self.timeout_config.metrics_collection_interval * 1000
                    ),
                )
            else:
                # 개발을 위한 콘솔 메트릭스 익스포터
                metric_reader = PeriodicExportingMetricReader(
                    ConsoleMetricExporter(),
                    export_interval_millis=int(
                        self.timeout_config.metrics_collection_interval * 1000
                    ),
                )

            meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
            metrics.set_meter_provider(meter_provider)

            # 자동 계측
            RequestsInstrumentor().instrument()
            SQLite3Instrumentor().instrument()

            # 사용을 위해 트레이서 및 미터 저장
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

        except ImportError as exception:
            self.logger.warning(
                "opentelemetry_not_available",
                message="트레이싱 활성화를 위해 opentelemetry 패키지를 설치하세요",
                missing_package=str(exception),
            )
        except Exception as exception:
            self.logger.error(
                "opentelemetry_initialization_failed",
                error_type=type(exception).__name__,
                error_message=str(exception),
            )

    def get_external_service(self) -> Any:
        """
        외부 관찰 가능성 서비스 인스턴스를 가져옵니다.

        반환:
            외부 서비스 인스턴스 또는 구성되지 않은 경우 None
        """
        return self._external_service

    def create_trace(self, name: str, **metadata) -> Optional[str]:
        """
        외부 서비스에 트레이스를 생성합니다.

        인자:
            name: 트레이스 이름
            **metadata: 추가 메타데이터

        반환:
            성공한 경우 트레이스 ID
        """
        if not self._external_service:
            return None

        try:
            if hasattr(self._external_service, "trace"):
                # Langfuse 스타일
                trace_obj = self._external_service.trace(name=name, **metadata)
                return str(trace_obj.id)
            if isinstance(self._external_service, dict) and "tracer" in self._external_service:
                # OpenTelemetry 스타일 (새 형식)
                tracer = self._external_service["tracer"]
                span = tracer.start_span(name=name)
                for key, value in metadata.items():
                    span.set_attribute(key, str(value))
                trace_id = str(span.get_span_context().trace_id)
                span.end()
                return trace_id
            if hasattr(self._external_service, "start_span"):
                # OpenTelemetry 스타일 (레거시)
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
        LLM 생성을 외부 서비스에 로깅합니다.

        인자:
            trace_id: 트레이스 식별자
            model: 모델 이름
            prompt: 입력 프롬프트
            response: 생성된 응답
            **metadata: 추가 메타데이터
        """
        if not self._external_service:
            return

        try:
            if hasattr(self._external_service, "generation"):
                # Langfuse 스타일
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

    def record_metric(self, name: str, value: float, tags: Optional[dict[str, str]] = None) -> None:
        """
        외부 서비스에 메트릭을 기록합니다.

        인자:
            name: 메트릭 이름
            value: 메트릭 값
            tags: 선택적 태그
        """
        if not self._external_service:
            return

        try:
            if isinstance(self._external_service, dict) and "meter" in self._external_service:
                # OpenTelemetry 메트릭스
                meter = self._external_service["meter"]

                # 메트릭 이름 패턴에 따라 카운터 또는 게이지 생성
                if "count" in name.lower() or "total" in name.lower():
                    counter = meter.create_counter(name, description=f"{name}에 대한 카운터")
                    counter.add(value, attributes=tags or {})
                else:
                    # 다른 메트릭의 경우 히스토그램 사용
                    histogram = meter.create_histogram(
                        name, description=f"{name}에 대한 히스토그램"
                    )
                    histogram.record(value, attributes=tags or {})

                self.logger.debug(
                    "opentelemetry_metric_recorded",
                    metric_name=name,
                    metric_value=value,
                    metric_tags=tags or {},
                )
            else:
                # 대체: 메트릭만 로깅
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
        OpenTelemetry 컨텍스트 관리자를 사용하여 새 스팬을 시작합니다.

        인자:
            name: 스팬 이름
            **attributes: 스팬 속성

        반환:
            스팬에 대한 컨텍스트 관리자
        """
        if isinstance(self._external_service, dict) and "tracer" in self._external_service:
            tracer = self._external_service["tracer"]
            span = tracer.start_span(name)
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            return span
        return None

    def flush(self) -> None:
        """
        모든 보류 중인 데이터를 외부 서비스로 플러시합니다.
        """
        if not self._external_service:
            return

        try:
            if hasattr(self._external_service, "flush"):
                self._external_service.flush()
                self.logger.debug("observability_data_flushed")
            elif isinstance(self._external_service, dict):
                # OpenTelemetry 프로바이더 플러시
                if "tracer_provider" in self._external_service:
                    self._external_service["tracer_provider"].force_flush(
                        int(self.timeout_config.observability_flush_timeout)
                    )
                if "meter_provider" in self._external_service:
                    self._external_service["meter_provider"].force_flush(
                        int(self.timeout_config.observability_flush_timeout)
                    )
                self.logger.debug("opentelemetry_data_flushed")
        except Exception as exception:
            self.logger.error(
                "observability_flush_failed",
                error_type=type(exception).__name__,
                error_message=str(exception),
            )


class ObservabilityManager:
    """
    의존성 주입 패턴을 사용하여 관찰 가능성 통합 인스턴스를 관리합니다.

    이것은 테스트 용이성과 격리를 개선하기 위해 이전 싱글톤 패턴을 대체합니다.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        구성으로 관찰 가능성 관리자를 초기화합니다.

        인자:
            config: 관찰 가능성 서비스 구성
        """
        self._integration = ObservabilityIntegration(config)

    def get_integration(self) -> ObservabilityIntegration:
        """
        관찰 가능성 통합 인스턴스를 가져옵니다.

        반환:
            ObservabilityIntegration 인스턴스
        """
        return self._integration

    def create_new_integration(
        self, config: Optional[dict[str, Any]] = None
    ) -> ObservabilityIntegration:
        """
        새로운 관찰 가능성 통합 인스턴스를 생성합니다.

        인자:
            config: 관찰 가능성 서비스 구성

        반환:
            새로운 ObservabilityIntegration 인스턴스
        """
        return ObservabilityIntegration(config)


# ObservabilityManager 인스턴스 생성을 위한 팩토리 함수
def create_observability_manager(config: Optional[dict[str, Any]] = None) -> ObservabilityManager:
    """
    새로운 ObservabilityManager 인스턴스를 생성합니다.

    인자:
        config: 관찰 가능성 서비스 구성

    반환:
        ObservabilityManager 인스턴스
    """
    return ObservabilityManager(config)


# 이전 버전 호환성 함수 (deprecated)
def initialize_observability(
    config: Optional[dict[str, Any]] = None,
) -> ObservabilityIntegration:
    """전역 관찰 가능성 통합을 초기화합니다 (deprecated)."""
    # 이 함수는 더 이상 사용되지 않으므로 새 코드에서는 사용하지 마세요
    # 대신 create_observability_manager를 사용하세요
    return ObservabilityIntegration(config)


def get_observability_integration() -> Optional[ObservabilityIntegration]:
    """전역 관찰 가능성 통합 인스턴스를 가져옵니다 (deprecated)."""
    # 이 함수는 더 이상 사용되지 않으므로 새 코드에서는 사용하지 마세요
    # 대신 의존성 주입 패턴을 사용하세요
    return None
