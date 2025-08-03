"""
트레이스 컨텍스트 및 구조화된 로깅과 통합되는 관찰 가능한 로거.
"""

import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

from .context import get_current_trace_context


class LogLevel(Enum):
    """로그 레벨 열거형."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ObservableLogger:
    """
    관찰 가능성 컨텍스트 및 구조화된 로깅과 통합되는 로거.

    이 로거는 자동으로 트레이스 정보를 포함하고 모든 컴포넌트에서
    일관된 구조화된 로깅을 제공합니다.
    """

    def __init__(self, component: Optional[str, layer: str, observability_service: Any] = None):
        """
        관찰 가능한 로거를 초기화합니다.

        인자:
            component: 컴포넌트 이름 (예: "sqlite_repository")
            layer: 계층 이름 ("domain", "port", "adapter")
            observability_service: 메트릭/트레이싱을 위한 선택적 관찰 가능성 서비스
        """
        self.component = component
        self.layer = layer
        self.observability_service = observability_service

        # 기본 로거 초기화
        self.logger = structlog.get_logger(component)

    def _get_base_context(self) -> dict[str, Any]:
        """트레이스 정보가 포함된 기본 로깅 컨텍스트를 가져옵니다."""
        context = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "layer": self.layer,
            "component": self.component,
        }

        trace_context = get_current_trace_context()
        if trace_context:
            context.update(
                {
                    "trace_id": trace_context.trace_id,
                    "span_id": trace_context.span_id,
                    "parent_span_id": trace_context.parent_span_id or "",
                    "operation": trace_context.operation or "",
                }
            )

        return context

    def _log(self, level: LogLevel, event: str, **kwargs) -> None:
        """
        내부 로깅 메서드.

        인자:
            level: 로그 레벨
            event: 이벤트 이름/설명
            **kwargs: 추가 컨텍스트
        """
        log_data = self._get_base_context()
        log_data["event"] = event
        log_data["level"] = level.value
        log_data.update(kwargs)

        log_method = getattr(self.logger, level.value.lower())
        log_method(**log_data)

        if self.observability_service and hasattr(self.observability_service, "log_event"):
            trace_context = get_current_trace_context()
            if trace_context:
                self.observability_service.log_event(
                    span_id=trace_context.span_id, name=event, data=log_data
                )

    def debug(self, event: str, **kwargs) -> None:
        """디버그 메시지를 로깅합니다."""
        self._log(LogLevel.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs) -> None:
        """정보 메시지를 로깅합니다."""
        self._log(LogLevel.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        """경고 메시지를 로깅합니다."""
        self._log(LogLevel.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        """오류 메시지를 로깅합니다."""
        self._log(LogLevel.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs) -> None:
        """심각 메시지를 로깅합니다."""
        self._log(LogLevel.CRITICAL, event, **kwargs)

    def exception_occurred(self, exception: Exception, operation: str, **kwargs) -> None:
        """
        풍부한 컨텍스트와 함께 예외를 로깅합니다.

        인자:
            exception: 발생한 예외
            operation: 수행 중인 작업
            **kwargs: 추가 컨텍스트
        """
        self.error(
            "exception_occurred",
            operation=operation,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            **kwargs,
        )

        if self.observability_service and hasattr(self.observability_service, "record_metric"):
            self.observability_service.record_metric(
                "exception_count",
                1,
                tags={
                    "layer": self.layer,
                    "component": self.component,
                    "exception_type": type(exception).__name__,
                    "operation": operation,
                },
            )

    def operation_started(self, operation: str, **kwargs) -> float:
        """
        작업 시작을 로깅하고 시작 시간을 반환합니다.

        인자:
            operation: 작업 이름
            **kwargs: 추가 컨텍스트

        반환:
            기간 계산을 위한 시작 시간
        """
        start_time = time.time()

        self.info("operation_started", operation=operation, start_time=start_time, **kwargs)

        return start_time

    def operation_completed(self, operation: str, start_time: float, **kwargs) -> None:
        """
        기간과 함께 작업 완료를 로깅합니다.

        인자:
            operation: 작업 이름
            start_time: operation_started에서 가져온 시작 시간
            **kwargs: 추가 컨텍스트
        """
        duration_ms = (time.time() - start_time) * 1000

        self.info(
            "operation_completed",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            **kwargs,
        )

        if self.observability_service and hasattr(self.observability_service, "record_metric"):
            self.observability_service.record_metric(
                "operation_duration_ms",
                duration_ms,
                tags={
                    "layer": self.layer,
                    "component": self.component,
                    "operation": operation,
                },
            )

    def operation_failed(
        self, operation: str, start_time: float, exception: Exception, **kwargs
    ) -> None:
        """
        기간 및 예외와 함께 작업 실패를 로깅합니다.

        인자:
            operation: 작업 이름
            start_time: operation_started에서 가져온 시작 시간
            exception: 실패를 유발한 예외
            **kwargs: 추가 컨텍스트
        """
        duration_ms = (time.time() - start_time) * 1000

        self.error(
            "operation_failed",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            **kwargs,
        )

        if self.observability_service:
            if hasattr(self.observability_service, "record_metric"):
                self.observability_service.record_metric(
                    "operation_failure_count",
                    1,
                    tags={
                        "layer": self.layer,
                        "component": self.component,
                        "operation": operation,
                        "exception_type": type(exception).__name__,
                    },
                )


_logger_registry: dict[str, ObservableLogger] = {}


def get_observable_logger(
    component: Optional[str, layer: str, observability_service: Any] = None
) -> ObservableLogger:
    """
    컴포넌트에 대한 관찰 가능한 로거를 가져오거나 생성합니다.

    인자:
        component: 컴포넌트 이름
        layer: 계층 이름
        observability_service: 선택적 관찰 가능성 서비스

    반환:
        관찰 가능한 로거 인스턴스
    """
    key = f"{layer}.{component}"

    if key not in _logger_registry:
        _logger_registry[key] = ObservableLogger(
            component=component,
            layer=layer,
            observability_service=observability_service,
        )

    return _logger_registry[key]


def configure_structured_logging() -> None:
    """
    애플리케이션을 위한 구조화된 로깅을 구성합니다.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
