"""
OpenTelemetry 데코레이터 (공식 패턴 기반).

공식 문서: https://opentelemetry.io/docs/languages/python/instrumentation/
"""

import functools
import time
from collections.abc import Callable
from typing import Any, Optional

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


def traced(
    operation_name: Optional[str] = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    OpenTelemetry 트레이싱 데코레이터 (공식 패턴).

    사용법:
        @traced("document_processing")
        def process_document(doc):
            return processed_doc

    인자:
        operation_name: 스팬 이름 (기본값: 함수명)
        attributes: 스팬 속성
    """

    def decorator(func: Callable) -> Callable:
        if not OTEL_AVAILABLE:
            return func

        span_name = operation_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)

            with tracer.start_as_current_span(span_name) as span:
                # 기본 속성 설정
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # 사용자 정의 속성 설정
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))

                try:
                    # 함수 실행
                    result = func(*args, **kwargs)

                    # 성공 상태 설정
                    span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as exception:
                    # 실패 상태 설정
                    span.set_status(Status(StatusCode.ERROR, str(exception)))
                    span.record_exception(exception)
                    raise

        return wrapper

    return decorator


def measured(
    metric_name: Optional[str] = None,
    track_duration: bool = True,
    track_calls: bool = True,
    unit: str = "1",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    OpenTelemetry 메트릭 데코레이터.

    사용법:
        @measured("document_processing")
        def process_document(doc):
            return processed_doc

    인자:
        metric_name: 메트릭 이름 (기본값: 함수명)
        track_duration: 실행 시간 추적
        track_calls: 호출 횟수 추적
        unit: 메트릭 단위
    """

    def decorator(func: Callable) -> Callable:
        if not OTEL_AVAILABLE:
            return func

        meter = metrics.get_meter(__name__)
        # 메트릭 이름 길이 제한 (OpenTelemetry 63자 제한)
        if metric_name:
            base_name = metric_name[:50]  # 여유분 확보
        else:
            # 모듈명을 단축하고 함수명만 사용
            base_name = func.__name__[:50]

        # 메트릭 인스트루먼트 생성
        call_counter = None
        duration_histogram = None

        if track_calls:
            call_counter = meter.create_counter(
                name=f"{base_name}.calls",
                description=f"{func.__name__} 호출 횟수",
                unit=unit,
            )

        if track_duration:
            duration_histogram = meter.create_histogram(
                name=f"{base_name}.duration",
                description=f"{func.__name__} 실행 시간",
                unit="s",
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 공통 속성
            common_attributes = {
                "function.name": func.__name__,
                "function.module": func.__module__,
            }

            # 호출 횟수 증가
            if call_counter:
                call_counter.add(1, attributes=common_attributes)

            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # 성공 메트릭
                if call_counter:
                    success_attributes = {**common_attributes, "status": "success"}
                    call_counter.add(1, attributes=success_attributes)

                return result

            except Exception as exception:
                # 실패 메트릭
                if call_counter:
                    error_attributes = {
                        **common_attributes,
                        "status": "error",
                        "error.type": type(exception).__name__,
                    }
                    call_counter.add(1, attributes=error_attributes)
                raise

            finally:
                # 실행 시간 기록
                if duration_histogram:
                    duration = time.time() - start_time
                    duration_histogram.record(duration, attributes=common_attributes)

        return wrapper

    return decorator


def observed(
    operation_name: Optional[str] = None,
    metric_name: Optional[str] = None,
    span_attributes: dict[str, Any] | None = None,
    track_duration: bool = True,
    track_calls: bool = True,
) -> Callable:
    """
    트레이싱과 메트릭을 모두 포함하는 통합 데코레이터.

    사용법:
        @observed("document_processing")
        def process_document(doc):
            return processed_doc

    인자:
        operation_name: 스팬 이름
        metric_name: 메트릭 이름
        span_attributes: 스팬 속성
        track_duration: 실행 시간 추적
        track_calls: 호출 횟수 추적
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # 트레이싱과 메트릭 데코레이터 합성
        traced_func = traced(operation_name, span_attributes)(func)
        measured_func = measured(metric_name, track_duration, track_calls)(traced_func)
        return measured_func

    return decorator


# 편의 데코레이터들
def trace_database_operation(table_name: Optional[str] = None):
    """데이터베이스 작업 전용 트레이싱 데코레이터."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        attributes = {"db.operation.type": "query"}
        if table_name:
            attributes["db.sql.table"] = table_name

        return traced(f"db.{func.__name__}", attributes)(func)

    return decorator


def trace_llm_operation(model_name: Optional[str] = None):
    """
    LLM 작업 전용 트레이싱 데코레이터.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        attributes = {"llm.request.type": "completion"}
        if model_name:
            attributes["llm.request.model"] = model_name

        return traced(f"llm.{func.__name__}", attributes)(func)

    return decorator


def trace_vector_operation(operation_type: str = "search"):
    """
    벡터 작업 전용 트레이싱 데코레이터.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        attributes = {"vector.operation.type": operation_type, "vector.engine": "hnsw"}
        return traced(f"vector.{func.__name__}", attributes)(func)

    return decorator
