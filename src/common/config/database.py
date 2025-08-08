"""
데이터베이스 구성 설정.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from ..validation.field_validators import (
    validate_dimension,
    validate_file_path,
    validate_positive_integer,
    validate_timeout,
)


class DatabaseConfig(BaseSettings):
    """
    데이터베이스 구성 설정.

    지식 그래프 데이터베이스 및 벡터 저장소 작업을 위한 SQLite 특정 설정을 지원합니다.
    """

    # SQLite 데이터베이스 설정
    db_path: str = Field(
        default="data/knowledge_graph.db", description="SQLite 데이터베이스 파일 경로"
    )

    optimize: bool = Field(default=True, description="SQLite 최적화 PRAGMA 적용 여부")

    # 연결 설정
    timeout: float = Field(default=30.0, description="초 단위 데이터베이스 연결 타임아웃")

    check_same_thread: bool = Field(default=False, description="SQLite check_same_thread 매개변수")

    # 벡터 저장소 설정
    vector_dimension: int = Field(default=384, description="벡터 임베딩의 차원")

    max_connections: int = Field(default=10, description="최대 데이터베이스 연결 수")

    # 백업 설정
    backup_enabled: bool = Field(default=False, description="자동 백업 활성화 여부")

    backup_interval: int = Field(default=3600, description="초 단위 백업 간격")

    backup_path: Optional[str] = Field(default=None, description="데이터베이스 백업 경로")

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        """데이터베이스 경로 형식 유효성 검사."""
        return validate_file_path(v, "Database path")

    @field_validator("vector_dimension")
    @classmethod
    def validate_vector_dimension(cls, v: int) -> int:
        """벡터 차원이 양수이고 합리적인 범위인지 유효성 검사."""
        return validate_dimension(v)

    @field_validator("timeout")
    @classmethod
    def validate_timeout_field(cls, v: float) -> float:
        """타임아웃이 합리적인 범위인지 유효성 검사."""
        return validate_timeout(v)

    @field_validator("max_connections")
    @classmethod
    def validate_max_connections(cls, v: int) -> int:
        """최대 연결 수가 합리적인 범위인지 유효성 검사."""
        return validate_positive_integer(v, "Maximum connections", 1000)

    @field_validator("backup_interval")
    @classmethod
    def validate_backup_interval(cls, v: int) -> int:
        """백업 간격이 합리적한 범위인지 유효성 검사."""
        if v < 60:
            raise ValueError("Backup interval must be at least 60 seconds")
        return validate_positive_integer(v, "Backup interval", 604800)  # 7일

    @field_validator("backup_path")
    @classmethod
    def validate_backup_path(cls, v: Optional[str]) -> Optional[str]:
        """제공된 경우 백업 경로 형식 유효성 검사."""
        if v is not None:
            return validate_file_path(v, "Backup path")
        return v

    @property
    def db_directory(self) -> Path:
        """필요한 경우 데이터베이스 디렉토리 경로를 가져오고 생성합니다."""
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.parent

    @property
    def backup_directory(self) -> Optional[Path]:
        """
        필요한 경우 백업 디렉토리 경로를 가져오고 생성합니다.
        """
        if self.backup_path is None:
            return None
        path = Path(self.backup_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = {
        "env_prefix": "DB_",
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }
