"""
테스트용 텍스트 임베딩 어댑터.
"""

import asyncio
import hashlib
import time
from typing import List

import numpy as np

from src.dto import EmbeddingResult
from src.ports.text_embedder import TextEmbedder

# 상수 정의
_MAX_SEED_VALUE = 2**32


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

    async def embed_text(self, text: str) -> EmbeddingResult:
        """
        단일 텍스트를 임베딩합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            EmbeddingResult: 텍스트의 임베딩 결과
        """
        start_time = time.time()

        # 텍스트를 기반으로 결정론적 랜덤 벡터 생성
        seed = self._text_to_seed(text)
        np.random.seed(seed)
        embedding = np.random.randn(self._dimension).astype(np.float32)
        # 코사인 유사도를 위해 정규화
        embedding = embedding / np.linalg.norm(embedding)

        processing_time = (time.time() - start_time) * 1000  # ms 변환

        return EmbeddingResult(
            text=text,
            embedding=embedding.tolist(),
            model_name=self.model_name,
            dimension=self._dimension,
            metadata={"seed": seed, "normalized": True},
            processing_time_ms=processing_time,
        )

    async def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        여러 텍스트를 일괄 임베딩합니다.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            List[EmbeddingResult]: 텍스트들의 임베딩 결과 리스트
        """
        # 비동기 처리 최적화를 위해 asyncio.gather 사용
        return await asyncio.gather(*[self.embed_text(text) for text in texts])

    def get_embedding_dimension(self) -> int:
        """
        임베딩 벡터의 차원을 반환합니다.

        Returns:
            int: 임베딩 벡터의 차원
        """
        return self._dimension

    async def is_available(self) -> bool:
        """
        임베딩 서비스가 사용 가능한지 확인합니다.

        Returns:
            bool: 항상 True (테스트 환경에서는 항상 사용 가능)
        """
        return True  # 테스트 환경에서는 항상 사용 가능

    def _text_to_seed(self, text: str) -> int:
        """
        텍스트를 결정론적 시드로 변환합니다.

        Args:
            text: 시드로 변환할 텍스트

        Returns:
            int: MD5 해시를 기반으로 생성된 시드값
        """
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % _MAX_SEED_VALUE
