"""
MCP (모델 컨텍스트 프로토콜) 서버 구성 설정.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class MCPConfig(BaseSettings):
    """
    MCP 서버 구성 설정.

    FastMCP 기반 지식 그래프 서버의 구성.
    """

    # 서버 설정
    server_name: str = Field(default="Knowledge Graph Server", description="MCP 서버 이름")

    server_instructions: str = Field(
        default="벡터 검색 기능을 갖춘 SQLite 기반 지식 그래프",
        description="MCP 서버에 대한 지침/설명",
    )

    host: str = Field(default="localhost", description="서버 호스트 주소")

    port: int = Field(default=8000, description="서버 포트 번호")

    # 벡터 설정
    vector_index_dir: str | None = Field(
        default=None, description="벡터 인덱스 파일을 저장할 디렉토리"
    )

    embedding_dim: int = Field(default=384, description="임베딩 벡터의 차원")

    vector_similarity: str = Field(
        default="cosine", description="벡터 유사도 측정 기준 (cosine, euclidean, dot)"
    )

    # 임베딩 설정
    embedder_type: str = Field(
        default="sentence-transformers", description="사용할 텍스트 임베더 유형"
    )

    embedder_model: str = Field(default="all-MiniLM-L6-v2", description="임베딩을 위한 모델 이름")

    # 성능 설정
    max_connections: int = Field(default=100, description="최대 동시 연결 수")

    timeout: float = Field(default=30.0, description="초 단위 요청 타임아웃")

    # 검색 설정
    max_search_results: int = Field(default=50, description="반환할 최대 검색 결과 수")

    search_threshold: float = Field(default=0.7, description="검색 결과의 최소 유사도 임계값")

    # 로깅
    log_level: str = Field(default="INFO", description="MCP 서버의 로깅 레벨")

    enable_debug: bool = Field(default=False, description="디버그 모드 활성화")

    # CORS 설정
    enable_cors: bool = Field(default=True, description="CORS 지원 활성화")

    cors_origins: list[str] = Field(default=["*"], description="허용된 CORS 원본")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """포트 번호 유효성 검사."""
        if not 1 <= v <= 65535:
            raise ValueError("포트는 1에서 65535 사이여야 합니다")
        return v

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        """임베딩 차원 유효성 검사."""
        if v <= 0:
            raise ValueError("임베딩 차원은 양수여야 합니다")
        return v

    @field_validator("vector_similarity")
    @classmethod
    def validate_vector_similarity(cls, v: str) -> str:
        """벡터 유사도 측정 기준 유효성 검사."""
        valid_metrics = {"cosine", "euclidean", "dot", "l2"}
        if v not in valid_metrics:
            raise ValueError(f"벡터 유사도 측정 기준은 {valid_metrics} 중 하나여야 합니다")
        return v

    @field_validator("search_threshold")
    @classmethod
    def validate_search_threshold(cls, v: float) -> float:
        """검색 임계값 유효성 검사."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("검색 임계값은 0.0에서 1.0 사이여야 합니다")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """로그 레벨 유효성 검사."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"로그 레벨은 {valid_levels} 중 하나여야 합니다")
        return v.upper()

    @field_validator("vector_index_dir")
    @classmethod
    def validate_vector_index_dir(cls, v: str | None) -> str | None:
        """벡터 인덱스 디렉토리 형식 유효성 검사."""
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError("벡터 인덱스 디렉토리는 비어 있지 않은 문자열이어야 합니다")
        return v

    @property
    def vector_index_directory(self) -> Path | None:
        """필요한 경우 벡터 인덱스 디렉토리 경로를 가져오고 생성합니다."""
        if self.vector_index_dir is None:
            return None
        path = Path(self.vector_index_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = {
        "env_prefix": "MCP_",
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }
