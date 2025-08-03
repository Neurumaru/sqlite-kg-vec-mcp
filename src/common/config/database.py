"""
데이터베이스 구성 설정.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


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

    backup_path: str | None = Field(default=None, description="데이터베이스 백업 경로")

    @field_validator("db_path")
    @classmethod
    def validate_db_path(cls, v: str) -> str:
        """데이터베이스 경로 형식 유효성 검사."""
        # 유효성 검사기에서 디렉토리를 생성하지 않고 형식만 검증
        if not isinstance(v, str) or not v.strip():
            raise ValueError("데이터베이스 경로는 비어 있지 않은 문자열이어야 합니다")
        return v

    @field_validator("vector_dimension")
    @classmethod
    def validate_vector_dimension(cls, v: int) -> int:
        """벡터 차원이 양수인지 유효성 검사."""
        if v <= 0:
            raise ValueError("벡터 차원은 양수여야 합니다")
        return v

    @field_validator("backup_path")
    @classmethod
    def validate_backup_path(cls, v: str | None) -> str | None:
        """제공된 경우 백업 경로 형식 유효성 검사."""
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError("백업 경로는 비어 있지 않은 문자열이어야 합니다")
        return v

    @property
    def db_directory(self) -> Path:
        """필요한 경우 데이터베이스 디렉토리 경로를 가져오고 생성합니다."""
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.parent

    @property
    def backup_directory(self) -> Path | None:
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
