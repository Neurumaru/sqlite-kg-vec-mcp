"""
문서 검증 설정 관리.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ValidationConfig:
    """문서 검증 설정."""

    # 컨텐츠 관련 설정
    min_content_length: int
    max_content_length: int
    allow_empty_title: bool

    # 상태 관련 설정
    allow_reprocessing: bool
    allow_processing_while_processing: bool

    # 메타데이터 관련 설정
    required_metadata_keys: Optional[list[str]]
    max_metadata_size: int

    @classmethod
    def from_env(cls) -> "ValidationConfig":
        """환경 변수에서 설정을 로드합니다."""
        return cls(
            min_content_length=int(os.getenv("DOC_MIN_CONTENT_LENGTH", "1")),
            max_content_length=int(os.getenv("DOC_MAX_CONTENT_LENGTH", "1000000")),
            allow_empty_title=os.getenv("DOC_ALLOW_EMPTY_TITLE", "false").lower() == "true",
            allow_reprocessing=os.getenv("DOC_ALLOW_REPROCESSING", "true").lower() == "true",
            allow_processing_while_processing=os.getenv(
                "DOC_ALLOW_PROCESSING_WHILE_PROCESSING", "false"
            ).lower()
            == "true",
            required_metadata_keys=_parse_required_metadata_keys(
                os.getenv("DOC_REQUIRED_METADATA_KEYS")
            ),
            max_metadata_size=int(os.getenv("DOC_MAX_METADATA_SIZE", "10000")),
        )

    @classmethod
    def default(cls) -> "ValidationConfig":
        """기본 설정을 반환합니다."""
        return cls(
            min_content_length=1,
            max_content_length=1_000_000,
            allow_empty_title=False,
            allow_reprocessing=True,
            allow_processing_while_processing=False,
            required_metadata_keys=None,
            max_metadata_size=10_000,
        )

    def validate(self) -> None:
        """설정 값들을 검증합니다."""
        if self.min_content_length < 0:
            raise ValueError("min_content_length는 0 이상이어야 합니다")

        if self.max_content_length <= 0:
            raise ValueError("max_content_length는 0보다 커야 합니다")

        if self.min_content_length > self.max_content_length:
            raise ValueError("min_content_length는 max_content_length보다 작아야 합니다")

        if self.max_metadata_size <= 0:
            raise ValueError("max_metadata_size는 0보다 커야 합니다")


def _parse_required_metadata_keys(value: Optional[str]) -> Optional[list[str]]:
    """환경 변수에서 필수 메타데이터 키 목록을 파싱합니다."""
    if not value or not value.strip():
        return None

    # 콤마로 구분된 키 목록을 파싱
    keys = [key.strip() for key in value.split(",") if key.strip()]
    return keys if keys else None
