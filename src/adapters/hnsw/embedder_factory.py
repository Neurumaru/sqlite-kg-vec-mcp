"""
텍스트 임베더 생성을 위한 팩토리 모듈.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.common.observability.logger import ObservableLogger


class VectorTextEmbedder(ABC):
    """HNSW 검색을 위한 동기 텍스트 임베더 인터페이스."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩합니다."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트를 임베딩합니다."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """임베딩 차원을 가져옵니다."""


class SyncRandomTextEmbedder(VectorTextEmbedder):
    """테스트 목적의 랜덤 텍스트 임베더."""

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        """랜덤 임베딩을 생성합니다."""
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self._dimension)
        return embedding.tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """여러 텍스트에 대한 랜덤 임베딩을 생성합니다."""
        return [self.embed(text) for text in texts]

    @property
    def dimension(self) -> int:
        """임베딩 차원을 가져옵니다."""
        return self._dimension


def create_embedder(
    embedder_type: str, logger: Optional[ObservableLogger] = None, **kwargs
) -> VectorTextEmbedder:
    """
    텍스트 임베더를 생성하는 팩토리 함수.

    Args:
        embedder_type: 생성할 임베더 유형
        logger: 선택적 로거 인스턴스
        **kwargs: 임베더 생성을 위한 추가 인수

    Returns:
        VectorTextEmbedder 인스턴스
    """
    from src.common.observability.logger import get_logger

    if logger is None:
        logger = get_logger("embedder_factory", "adapter")

    if embedder_type == "random":
        dimension = kwargs.get("dimension", 128)
        logger.info("embedder_created", type="random", dimension=dimension)
        return SyncRandomTextEmbedder(dimension=dimension)
    if embedder_type == "sentence-transformers":
        # 지금은 랜덤으로 대체
        dimension = kwargs.get("dimension", 384)
        logger.info("embedder_created", type="sentence-transformers", dimension=dimension)
        return SyncRandomTextEmbedder(dimension=dimension)

    logger.error("unknown_embedder_type", embedder_type=embedder_type)
    raise ValueError(f"알 수 없는 임베더 유형: {embedder_type}")
