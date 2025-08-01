"""
Configuration for FastMCP adapter.
"""

from dataclasses import dataclass


@dataclass
class FastMCPConfig:
    """Configuration for FastMCP adapter."""

    # 서버 설정
    host: str = "127.0.0.1"
    port: int = 3001
    debug: bool = False

    # API 기본 설정
    max_results: int = 100
    similarity_threshold: float = 0.7

    # 로깅 설정
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535: {self.port}")

        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError(
                f"Similarity threshold must be between 0 and 1: {self.similarity_threshold}"
            )

        if self.max_results < 1:
            raise ValueError(f"Max results must be positive: {self.max_results}")
