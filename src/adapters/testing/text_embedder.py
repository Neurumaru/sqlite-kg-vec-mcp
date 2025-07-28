"""
테스트용 텍스트 임베딩 어댑터.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.domain.value_objects.vector import Vector
from src.ports.text_embedder import TextEmbedder


class RandomTextEmbedder(TextEmbedder):
    """테스트용 랜덤 벡터 생성 어댑터."""

    def __init__(self, dimension: int = 128):
        """
        지정된 차원으로 초기화합니다.

        Args:
            dimension: 벡터 차원
        """
        self._dimension = dimension
        self.model_name = f"random-{dimension}d"

    async def embed_text(self, text: str) -> Vector:
        """단일 텍스트를 임베딩합니다."""
        # 텍스트를 기반으로 결정론적 랜덤 벡터 생성
        seed = self._text_to_seed(text)
        np.random.seed(seed)
        embedding = np.random.randn(self._dimension).astype(np.float32)
        # 코사인 유사도를 위해 정규화
        embedding = embedding / np.linalg.norm(embedding)
        return Vector(values=embedding.tolist())

    async def embed_texts(self, texts: List[str]) -> List[Vector]:
        """여러 텍스트를 일괄 임베딩합니다."""
        return [await self.embed_text(text) for text in texts]

    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원을 반환합니다."""
        return self._dimension

    async def is_available(self) -> bool:
        """임베딩 서비스가 사용 가능한지 확인합니다."""
        return True  # 항상 사용 가능

    def _text_to_seed(self, text: str) -> int:
        """텍스트를 시드로 변환합니다."""
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**32)
