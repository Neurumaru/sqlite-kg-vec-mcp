"""
공통 유틸리티 및 공유 컴포넌트.

이 모듈은 순환 종속성을 생성하지 않고 애플리케이션의 여러 계층에서
사용될 수 있는 공유 유틸리티를 포함합니다.
"""

from .logging import (
    LoggingConfig,
    LogLevel,
    ObservableLogger,
    configure_structured_logging,
    get_observable_logger,
)
from .observability import (
    TraceContext,
    TraceContextManager,
    create_trace_context,
    development_setup,
    get_current_span_id,
    get_current_trace_id,
    get_logger,
    get_observability_integration,
)
from .observability import get_observable_logger as get_obs_logger
from .observability import (
    initialize_observability,
    production_setup,
    quick_setup,
    setup_observability,
    with_observability,
)

__all__ = [
    # 핵심 관찰 가능성
    "get_logger",
    "get_observable_logger",
    "with_observability",
    "get_current_trace_id",
    "get_current_span_id",
    "create_trace_context",
    "TraceContext",
    "TraceContextManager",
    # 설정 및 구성
    "setup_observability",
    "quick_setup",
    "production_setup",
    "development_setup",
    # 로깅 구성
    "LogLevel",
    "LoggingConfig",
    "ObservableLogger",
    "configure_structured_logging",
    # 통합
    "initialize_observability",
    "get_observability_integration",
]
