"""
로깅 설정 및 구성.
"""

import json
import logging as stdlib_logging
import os
import sys
import traceback
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import structlog

from ..config.observability import LoggingObservabilityConfig


class LogLevel(Enum):
    """지원되는 로그 레벨."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LoggingConfig:
    """
    로깅 구성 설정.
    """

    level: LogLevel = LogLevel.INFO
    format: str = "json"  # json 또는 text
    output: str = "console"  # console 또는 file
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    include_trace: bool = True
    include_caller: bool = False
    sanitize_sensitive_data: bool = True


def configure_structured_logging(
    config: Optional[LoggingConfig] = None,
    observability_config: Optional[LoggingObservabilityConfig] = None,
) -> None:
    """
    애플리케이션을 위한 구조화된 로깅을 구성합니다.

    인자:
        config: 로깅 구성 (None인 경우 기본값 사용, deprecated)
        observability_config: 새로운 관찰 가능성 로깅 구성
    """
    if observability_config is not None:
        # 관찰 가능성 구성을 로깅 구성으로 변환
        config = LoggingConfig(
            level=LogLevel(observability_config.level),
            format=observability_config.format,
            output=observability_config.output,
            file_path=observability_config.file_path,
            include_trace=observability_config.include_trace,
            include_caller=observability_config.include_caller,
            sanitize_sensitive_data=observability_config.sanitize_sensitive_data,
        )
    elif config is None:
        config = LoggingConfig()

    _configure_structlog(config)


def _configure_structlog(config: LoggingConfig) -> None:
    """구조화된 로깅을 위해 structlog를 구성합니다."""

    # 표준 라이브러리 로깅 구성
    stdlib_logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if config.output == "console" else None,
        level=getattr(stdlib_logging, config.level.value),
    )

    # 프로세서 체인 빌드
    processors: list[
        Callable[
            [Any, str, MutableMapping[str, Any]],
            Mapping[str, Any] | str | bytes | bytearray | tuple,
        ]
    ] = [
        # 레벨별 필터링
        structlog.stdlib.filter_by_level,
        # 로그 레벨 및 로거 이름 추가
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        # 타임스탬프 추가
        structlog.processors.TimeStamper(fmt="iso"),
        # 위치 인자 처리
        structlog.stdlib.PositionalArgumentsFormatter(),
        # 예외를 위한 스택 정보 추가
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # 유니코드 처리
        structlog.processors.UnicodeDecoder(),
    ]

    # 요청 시 호출자 정보 추가
    if config.include_caller:
        processors.append(structlog.processors.CallsiteParameterAdder())

    # 민감 데이터 삭제 프로세서 추가
    if config.sanitize_sensitive_data:
        processors.append(_sanitize_processor)

    # 최종 렌더러 추가
    if config.format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    # structlog 구성
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _configure_stdlib_logging(config: LoggingConfig) -> None:
    """표준 라이브러리 로깅을 대체 메커니즘으로 구성합니다."""

    # 포매터 생성
    formatter: Any
    if config.format == "json":
        formatter = _JSONFormatter()
    else:
        formatter = stdlib_logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 루트 로거 구성
    root_logger = stdlib_logging.getLogger()
    root_logger.setLevel(getattr(stdlib_logging, config.level.value))

    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 적절한 핸들러 추가
    if config.output == "console":
        handler = stdlib_logging.StreamHandler(sys.stdout)
    else:
        handler = stdlib_logging.FileHandler(config.file_path or "app.log")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def _sanitize_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """로그에서 민감한 데이터를 삭제하는 프로세서."""
    sensitive_keys = {
        "password",
        "token",
        "secret",
        "key",
        "auth",
        "credential",
        "api_key",
        "private_key",
    }

    def sanitize_dict(d: Any) -> Any:
        if not isinstance(d, dict):
            return d

        sanitized = {}
        for key, value in d.items():
            key_lower = str(key).lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = str(
                    [sanitize_dict(item) if isinstance(item, dict) else item for item in value]
                )
            else:
                sanitized[key] = value
        return sanitized

    result = sanitize_dict(event_dict)
    return result if isinstance(result, dict) else event_dict


class _JSONFormatter:
    """표준 라이브러리 로깅을 위한 JSON 포매터."""

    def __init__(self):
        self.json = json

    def format(self, record):
        """로그 레코드를 JSON으로 포맷합니다."""
        log_entry = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = traceback.format_exception(*record.exc_info)

        return self.json.dumps(log_entry)


def get_logging_config_from_env() -> LoggingConfig:
    """
    환경 변수로부터 로깅 구성을 가져옵니다.

    Deprecated: 대신 LoggingObservabilityConfig.from_env()를 사용하세요.

    환경 변수:
    - LOG_LEVEL: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - LOG_FORMAT: 출력 형식 (json, text)
    - LOG_OUTPUT: 출력 대상 (console, file)
    - LOG_FILE: 로그 파일 경로 (output=file인 경우)
    - LOG_INCLUDE_TRACE: 트레이스 정보 포함 여부 (true/false)
    - LOG_INCLUDE_CALLER: 호출자 정보 포함 여부 (true/false)
    - LOG_SANITIZE: 민감한 데이터 삭제 여부 (true/false)

    반환:
        LoggingConfig 인스턴스
    """
    return LoggingConfig(
        level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
        format=os.getenv("LOG_FORMAT", "json"),
        output=os.getenv("LOG_OUTPUT", "console"),
        file_path=os.getenv("LOG_FILE"),
        include_trace=os.getenv("LOG_INCLUDE_TRACE", "true").lower() == "true",
        include_caller=os.getenv("LOG_INCLUDE_CALLER", "false").lower() == "true",
        sanitize_sensitive_data=os.getenv("LOG_SANITIZE", "true").lower() == "true",
    )


def get_observability_logging_config_from_env() -> LoggingObservabilityConfig:
    """
    환경 변수로부터 관찰 가능성 로깅 구성을 가져옵니다.

    반환:
        LoggingObservabilityConfig 인스턴스
    """
    return LoggingObservabilityConfig()
