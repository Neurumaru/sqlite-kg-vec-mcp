"""
텍스트 임베딩 포트.
"""

from abc import ABC, abstractmethod

from src.dto import EmbeddingResult


class TextEmbedder(ABC):
    """
    텍스트 임베딩 포트.

    텍스트를 벡터로 변환하는 핵심 기능을 제공합니다.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """
        단일 텍스트를 임베딩합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 결과
        """

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        여러 텍스트를 일괄 임베딩합니다.

        Args:
            texts: 임베딩할 텍스트들

        Returns:
            임베딩 결과들
        """

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        임베딩 벡터의 차원을 반환합니다.

        Returns:
            벡터 차원
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """
        임베딩 서비스가 사용 가능한지 확인합니다.

        Returns:
            사용 가능 여부
        """
