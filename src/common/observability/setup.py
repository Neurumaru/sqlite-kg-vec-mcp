"""
관찰 가능성 시스템을 위한 설정 및 초기화 유틸리티.
"""

import os
from typing import Any

from ..logging.config import (
    LoggingConfig,
    configure_structured_logging,
    get_logging_config_from_env,
)
from .integration import initialize_observability
from .logger import get_observable_logger


def setup_observability(
    logging_config: dict[str, Any] | None = None,
    observability_config: dict[str, Any] | None = None,
    auto_configure: bool = True,
) -> None:
    """
    통합 로깅 및 관찰 가능성 시스템을 위한 완전한 설정.

    이 함수는 애플리케이션 시작 시 호출되어 다음을 구성해야 합니다:
    - 구조화된 로깅 (structlog 사용 가능 시)
    - 트레이스 컨텍스트 관리
    - 외부 관찰 가능성 서비스 통합

    인자:
        logging_config: 로깅 구성 재정의
        observability_config: 관찰 가능성 서비스 구성
        auto_configure: 환경 변수로부터 자동 구성할지 여부
    """

    # 1. 구조화된 로깅 구성
    if auto_configure and logging_config is None:
        # 환경 변수에서 구성 가져오기
        config = get_logging_config_from_env()
        configure_structured_logging(config)
    elif logging_config:
        config = LoggingConfig(**logging_config)
        configure_structured_logging(config)
    else:
        # 기본값 사용
        configure_structured_logging()

    # 2. 관찰 가능성 통합 초기화
    if auto_configure and observability_config is None:
        observability_config = get_observability_config_from_env()

    if observability_config:
        initialize_observability(observability_config)

    # 성공적인 설정 로깅
    logger = get_observable_logger("observability_setup", "common")
    logger.info(
        "observability_system_initialized",
        structured_logging=True,
        external_integration=bool(observability_config),
        auto_configured=auto_configure,
    )


def get_observability_config_from_env() -> dict[str, Any]:
    """
    환경 변수로부터 관찰 가능성 구성을 가져옵니다.

    환경 변수:
    - OBSERVABILITY_SERVICE: 서비스 타입 (langfuse, opentelemetry, none)
    - LANGFUSE_SECRET_KEY: Langfuse 비밀 키
    - LANGFUSE_PUBLIC_KEY: Langfuse 공개 키
    - LANGFUSE_HOST: Langfuse 호스트 (기본값: https://cloud.langfuse.com)
    - JAEGER_HOST: OpenTelemetry를 위한 Jaeger 호스트 (기본값: localhost)
    - JAEGER_PORT: OpenTelemetry를 위한 Jaeger 포트 (기본값: 6831)

    반환:
        구성 딕셔너리
    """
    service_type = os.getenv("OBSERVABILITY_SERVICE", "none").lower()

    if service_type == "none":
        return {}

    config: dict[str, Any] = {"service_type": service_type}

    if service_type == "langfuse":
        langfuse_config = {
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        }
        config["langfuse"] = langfuse_config

        # 키가 제공된 경우에만 포함
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
    개발을 위한 합리적인 기본값으로 빠른 설정.

    이것은 빠르게 시작하기 위한 편의 함수입니다.
    다음과 같이 구성합니다:
    - 콘솔로 JSON 로깅
    - INFO 레벨 로깅
    - 자동 환경 변수 감지
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
    적절한 설정으로 프로덕션 환경 구성.

    이것은 다음을 구성합니다:
    - 구조화된 JSON 로깅
    - WARNING 레벨 로깅 (노이즈 감소)
    - 전체 관찰 가능성 통합
    - 민감한 데이터 삭제
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
    자세한 로깅이 포함된 개발 환경 설정.

    이것은 다음을 구성합니다:
    - 사람이 읽을 수 있는 텍스트 로깅
    - DEBUG 레벨 로깅
    - 전체 트레이스 정보
    - 외부 서비스 없음 (더 빠른 개발을 위해)
    """
    setup_observability(
        logging_config={
            "level": "DEBUG",
            "format": "text",  # 개발을 위해 사람이 읽을 수 있는 형식
            "output": "console",
            "include_trace": True,
            "include_caller": True,
            "sanitize_sensitive_data": False,
        },
        observability_config={},  # 외부 서비스 없음
        auto_configure=False,
    )
