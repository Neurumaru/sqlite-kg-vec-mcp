"""
FastMCP 어댑터 설정.
"""

from dataclasses import dataclass

from src.config.search_config import SearchConfig


@dataclass
class FastMCPConfig:
    """FastMCP 어댑터 설정."""

    # 서버 설정
    host: str = "127.0.0.1"
    port: int = 3001
    debug: bool = False

    # API 기본 설정
    max_results: int = 100
    similarity_threshold: float = SearchConfig().node_similarity_threshold

    # 응답 포맷 설정
    content_summary_length: int = 200

    # 관계 기본 설정
    default_relationship_weight: float = 1.0

    # 로깅 설정
    log_level: str = "INFO"

    def __post_init__(self):
        """초기화 후 설정 값 검증."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535: {self.port}")

        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(
                f"Similarity threshold must be between 0 and 1: {self.similarity_threshold}"
            )

        if self.max_results < 1:
            raise ValueError(f"Max results must be positive: {self.max_results}")

        if self.content_summary_length < 1:
            raise ValueError(
                f"Content summary length must be positive: {self.content_summary_length}"
            )

        if self.default_relationship_weight < 0:
            raise ValueError(
                f"Default relationship weight must be non-negative: {self.default_relationship_weight}"
            )
