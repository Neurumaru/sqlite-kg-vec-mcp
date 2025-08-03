"""
관찰 가능성을 위한 트레이스 및 스팬 컨텍스트 관리.
"""

import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceContext:
    """
    관찰 가능성을 위한 트레이스 및 스팬 정보를 포함합니다.
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation: Optional[str] = None
    layer: Optional[str] = None
    component: Optional[str] = None
    start_time: Optional[datetime] = None
    metadata: dict[str, Any]] = None

    def __post_init__(self):
        """기본값을 초기화합니다."""
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}

    def add_metadata(self, key: str, value: Any) -> None:
        """트레이스 컨텍스트에 메타데이터를 추가합니다."""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value

    def to_dict(self) -> dict[str, Any]:
        """로깅을 위한 딕셔너리로 변환합니다."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "layer": self.layer,
            "component": self.component,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "metadata": self.metadata,
        }


# 트레이스 전파를 위한 컨텍스트 변수
_trace_context: ContextVar[TraceContext]] = ContextVar("trace_context", default=None)


def get_current_trace_id() -> str]:
    """
    현재 트레이스 ID를 컨텍스트에서 가져옵니다.

    반환:
        현재 트레이스 ID 또는 활성 트레이스가 없는 경우 None
    """
    context = _trace_context.get()
    return context.trace_id if context else None


def get_current_span_id() -> str]:
    """
    현재 스팬 ID를 컨텍스트에서 가져옵니다.

    반환:
        현재 스팬 ID 또는 활성 트레이스가 없는 경우 None
    """
    context = _trace_context.get()
    return context.span_id if context else None


def get_current_trace_context() -> TraceContext]:
    """
    현재 트레이스 컨텍스트를 가져옵니다.

    반환:
        현재 트레이스 컨텍스트 또는 활성 트레이스가 없는 경우 None
    """
    return _trace_context.get()


def create_trace_context(
    operation: str,
    layer: str,
    component: str,
    parent_context: Optional[TraceContext] = None,
    metadata: dict[str, Any]] = None,
) -> TraceContext:
    """
    새로운 트레이스 컨텍스트를 생성합니다.

    인자:
        operation: 작업 이름
        layer: 계층 이름 (도메인, 포트, 어댑터)
        component: 컴포넌트 이름
        parent_context: 부모 트레이스 컨텍스트
        metadata: 추가 메타데이터

    반환:
        새로운 트레이스 컨텍스트
    """
    if parent_context:
        trace_id = parent_context.trace_id
        parent_span_id = parent_context.span_id
    else:
        trace_id = str(uuid.uuid4())
        parent_span_id = None

    span_id = str(uuid.uuid4())

    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        operation=operation,
        layer=layer,
        component=component,
        metadata=metadata or {},
    )


def set_trace_context(context: TraceContext]) -> None:
    """
    현재 트레이스 컨텍스트를 설정합니다.

    인자:
        context: 설정할 트레이스 컨텍스트
    """
    _trace_context.set(context)


class TraceContextManager:
    """
    트레이스 컨텍스트를 위한 컨텍스트 관리자.
    """

    def __init__(self, trace_context: TraceContext):
        """
        컨텍스트 관리자를 초기화합니다.

        인자:
            trace_context: 사용할 트레이스 컨텍스트
        """
        self.trace_context = trace_context
        self.previous_context: Optional[TraceContext] = None

    def __enter__(self) -> TraceContext:
        """컨텍스트에 진입합니다."""
        self.previous_context = _trace_context.get()
        _trace_context.set(self.trace_context)
        return self.trace_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트를 종료합니다."""
        _trace_context.set(self.previous_context)


def with_trace_context(trace_context: TraceContext):
    """
    특정 트레이스 컨텍스트와 함께 함수를 실행하는 데코레이터.

    인자:
        trace_context: 사용할 트레이스 컨텍스트

    반환:
        데코레이터 함수
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with TraceContextManager(trace_context):
                return func(*args, **kwargs)

        return wrapper

    return decorator
