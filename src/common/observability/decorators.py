"""
자동 관찰 가능성 통합을 위한 데코레이터.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any

from .context import (
    TraceContextManager,
    create_trace_context,
    get_current_trace_context,
)
from .logger import get_observable_logger


def with_observability(
    operation: Optional[str] = None,
    layer: Optional[str] = None,
    component: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False,
):
    """
    함수에 자동 관찰 가능성을 추가하는 데코레이터.

    이 데코레이터는 다음을 수행합니다:
    - 추적 컨텍스트가 없으면 생성합니다.
    - 작업 시작/완료/실패를 로깅합니다.
    - 실행 시간을 측정합니다.
    - 구조화된 로깅으로 예외를 처리합니다.

    인자:
        operation: 작업 이름 (기본값은 함수 이름)
        layer: 계층 이름 (모듈에서 추론 시도)
        component: 컴포넌트 이름 (클래스/모듈에서 추론 시도)
        include_args: 함수 인자를 로깅할지 여부
        include_result: 함수 결과를 로깅할지 여부
    """

    def decorator(func: Callable) -> Callable:
        # 제공되지 않은 경우 메타데이터 추론
        func_operation = operation or func.__name__
        func_layer = layer or _infer_layer(func)
        func_component = component or _infer_component(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 추적 컨텍스트 가져오기 또는 생성
            current_context = get_current_trace_context()
            if current_context is None:
                trace_context = create_trace_context(
                    operation=func_operation, layer=func_layer, component=func_component
                )
                use_context_manager = True
            else:
                # 자식 스팬 생성
                trace_context = create_trace_context(
                    operation=func_operation,
                    layer=func_layer,
                    component=func_component,
                    parent_context=current_context,
                )
                use_context_manager = True

            # 로거 가져오기
            logger = get_observable_logger(func_component, func_layer)

            # 로깅 컨텍스트 준비
            log_context: dict[str, Any] = {}
            if include_args and args:
                log_context["args"] = _sanitize_args(args)
            if include_args and kwargs:
                log_context["kwargs"] = _sanitize_kwargs(kwargs)

            def execute_function():
                # 작업 시작 로깅
                start_time = logger.operation_started(func_operation, **log_context)

                try:
                    # 함수 실행
                    result = func(*args, **kwargs)

                    # 성공 로깅
                    success_context = log_context.copy()
                    if include_result and result is not None:
                        success_context["result"] = _sanitize_result(result)

                    logger.operation_completed(func_operation, start_time, **success_context)

                    return result

                except Exception as exception:  # exception 변수명으로 변경
                    # 실패 로깅
                    logger.operation_failed(func_operation, start_time, exception, **log_context)
                    raise

            # 추적 컨텍스트를 사용하거나 사용하지 않고 실행
            if use_context_manager:
                with TraceContextManager(trace_context):
                    return execute_function()
            else:
                return execute_function()

        return wrapper

    return decorator


def with_trace(
    operation: Optional[str] = None,
    layer: Optional[str] = None,
    component: Optional[str] = None,
    metadata: dict[str, Any]] = None,
):
    """
    함수에 추적 컨텍스트를 추가하는 데코레이터.

    인자:
        operation: 작업 이름
        layer: 계층 이름
        component: 컴포넌트 이름
        metadata: 추가 메타데이터
    """

    def decorator(func: Callable) -> Callable:
        func_operation = operation or func.__name__
        func_layer = layer or _infer_layer(func)
        func_component = component or _infer_component(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_context = get_current_trace_context()
            trace_context = create_trace_context(
                operation=func_operation,
                layer=func_layer,
                component=func_component,
                parent_context=current_context,
                metadata=metadata,
            )

            with TraceContextManager(trace_context):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def with_metrics(metric_name: Optional[str] = None, tags: dict[str, str]] = None):
    """
    자동 메트릭 수집을 추가하는 데코레이터.

    인자:
        metric_name: 메트릭 이름 (기본값은 함수 이름)
        tags: 추가 메트릭 태그
    """

    def decorator(func: Callable) -> Callable:
        func_metric_name = metric_name or f"{func.__name__}_calls"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 관찰 가능성 서비스가 있는 로거 가져오기
            component = _infer_component(func)
            layer = _infer_layer(func)
            logger = get_observable_logger(component, layer)

            # 메트릭 기록
            if logger.observability_service and hasattr(
                logger.observability_service, "record_metric"
            ):
                metric_tags = {
                    "layer": layer,
                    "component": component,
                    "function": func.__name__,
                }
                if tags:
                    metric_tags.update(tags)

                logger.observability_service.record_metric(func_metric_name, 1, tags=metric_tags)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _infer_layer(func: Callable) -> str:
    """함수 모듈 경로에서 계층을 추론합니다."""
    module = inspect.getmodule(func)
    if module and module.__name__:
        module_path = module.__name__
        if "adapters" in module_path:
            return "adapter"
        if "ports" in module_path:
            return "port"
        if "domain" in module_path:
            return "domain"
        if "application" in module_path:
            return "application"
    return "unknown"


def _infer_component(func: Callable) -> str:
    """함수 컨텍스트에서 컴포넌트를 추론합니다."""
    # 메서드인 경우 클래스 이름 가져오기 시도
    if hasattr(func, "__self__"):
        return str(func.__self__.__class__.__name__.lower())

    # 모듈 이름에서 가져오기
    module = inspect.getmodule(func)
    if module and module.__name__:
        parts = module.__name__.split(".")
        if len(parts) > 0:
            return str(parts[-1])

    return str(func.__name__)


def _sanitize_args(args: tuple) -> list[str]:
    """로깅을 위해 함수 인자를 삭제합니다."""
    sanitized: list[str] = []
    for arg in args:
        if hasattr(arg, "__dict__"):
            # 객체 - 타입 이름만 포함
            sanitized.append(f"<{type(arg).__name__}>")
        elif isinstance(arg, str | int | float | bool | type(None)):
            sanitized.append(str(arg))
        else:
            sanitized.append(f"<{type(arg).__name__}>")
    return sanitized


def _sanitize_kwargs(kwargs: dict) -> dict:
    """로깅을 위해 함수 키워드 인자를 삭제합니다."""
    sanitized = {}
    for key, value in kwargs.items():
        if isinstance(value, str | int | float | bool | type(None)):
            sanitized[key] = value
        else:
            sanitized[key] = f"<{type(value).__name__}>"
    return sanitized


def _sanitize_result(result: Any) -> Any:
    """로깅을 위해 함수 결과를 삭제합니다."""
    if isinstance(result, str | int | float | bool | type(None)):
        return result
    if hasattr(result, "__dict__"):
        return f"<{type(result).__name__}>"
    return f"<{type(result).__name__}>"
