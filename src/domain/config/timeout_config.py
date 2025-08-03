"""
타임아웃 설정 관리.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TimeoutConfig:
    """타임아웃 설정."""

    # HTTP/API 관련 타임아웃 (초)
    default_request_timeout: float
    ollama_quick_timeout: float  # 간단한 요청 (모델 목록 등)
    ollama_standard_timeout: float  # 일반적인 생성 요청
    ollama_download_timeout: float  # 모델 다운로드

    # 데이터베이스 관련 타임아웃 (초)
    database_connection_timeout: float
    database_busy_timeout: float  # SQLite busy_timeout (밀리초로 변환됨)

    # 관찰성/모니터링 관련 타임아웃 (초)
    observability_export_timeout: float
    observability_flush_timeout: float
    metrics_collection_interval: float

    # 기타 타임아웃
    event_validation_timeout: float  # 이벤트 타임스탬프 검증

    @classmethod
    def from_env(cls) -> "TimeoutConfig":
        """환경 변수에서 타임아웃 설정을 로드합니다."""
        return cls(
            # HTTP/API 타임아웃
            default_request_timeout=float(os.getenv("TIMEOUT_DEFAULT_REQUEST", "30.0")),
            ollama_quick_timeout=float(os.getenv("TIMEOUT_OLLAMA_QUICK", "5.0")),
            ollama_standard_timeout=float(os.getenv("TIMEOUT_OLLAMA_STANDARD", "30.0")),
            ollama_download_timeout=float(os.getenv("TIMEOUT_OLLAMA_DOWNLOAD", "300.0")),
            # 데이터베이스 타임아웃
            database_connection_timeout=float(os.getenv("TIMEOUT_DATABASE_CONNECTION", "30.0")),
            database_busy_timeout=float(os.getenv("TIMEOUT_DATABASE_BUSY", "5.0")),
            # 관찰성 타임아웃
            observability_export_timeout=float(os.getenv("TIMEOUT_OBSERVABILITY_EXPORT", "30.0")),
            observability_flush_timeout=float(os.getenv("TIMEOUT_OBSERVABILITY_FLUSH", "30.0")),
            metrics_collection_interval=float(os.getenv("INTERVAL_METRICS_COLLECTION", "60.0")),
            # 기타
            event_validation_timeout=float(os.getenv("TIMEOUT_EVENT_VALIDATION", "60.0")),
        )

    @classmethod
    def default(cls) -> "TimeoutConfig":
        """기본 타임아웃 설정을 반환합니다."""
        return cls(
            # HTTP/API 타임아웃
            default_request_timeout=30.0,
            ollama_quick_timeout=5.0,
            ollama_standard_timeout=30.0,
            ollama_download_timeout=300.0,
            # 데이터베이스 타임아웃
            database_connection_timeout=30.0,
            database_busy_timeout=5.0,
            # 관찰성 타임아웃
            observability_export_timeout=30.0,
            observability_flush_timeout=30.0,
            metrics_collection_interval=60.0,
            # 기타
            event_validation_timeout=60.0,
        )

    def validate(self) -> None:
        """타임아웃 설정을 검증합니다."""
        timeout_values = [
            ("default_request_timeout", self.default_request_timeout),
            ("ollama_quick_timeout", self.ollama_quick_timeout),
            ("ollama_standard_timeout", self.ollama_standard_timeout),
            ("ollama_download_timeout", self.ollama_download_timeout),
            ("database_connection_timeout", self.database_connection_timeout),
            ("database_busy_timeout", self.database_busy_timeout),
            ("observability_export_timeout", self.observability_export_timeout),
            ("observability_flush_timeout", self.observability_flush_timeout),
            ("metrics_collection_interval", self.metrics_collection_interval),
            ("event_validation_timeout", self.event_validation_timeout),
        ]

        for name, value in timeout_values:
            if value <= 0:
                raise ValueError(f"{name}은 0보다 커야 합니다: {value}")

        # 논리적 관계 검증
        if self.ollama_quick_timeout > self.ollama_standard_timeout:
            raise ValueError("ollama_quick_timeout은 ollama_standard_timeout보다 작아야 합니다")

        if self.ollama_standard_timeout > self.ollama_download_timeout:
            raise ValueError("ollama_standard_timeout은 ollama_download_timeout보다 작아야 합니다")

    def get_database_busy_timeout_ms(self) -> int:
        """SQLite PRAGMA busy_timeout용 밀리초 값을 반환합니다."""
        return int(self.database_busy_timeout * 1000)
