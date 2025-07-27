"""
테스트용 텍스트 임베딩 어댑터.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.domain.value_objects.vector import Vector
from src.ports.text_embedder import EmbeddingConfig, EmbeddingResult, TextEmbedder


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

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """메타데이터와 함께 텍스트를 임베딩합니다."""
        start_time = time.time()

        vector = await self.embed_text(text)
        processing_time = (time.time() - start_time) * 1000

        return EmbeddingResult(
            text=text,
            vector=vector,
            model_name=self.model_name,
            token_count=len(text.split()),  # 단순히 단어 수로 추정
            processing_time_ms=processing_time,
        )

    async def batch_embed_with_metadata(
        self, texts: List[str]
    ) -> List[EmbeddingResult]:
        """메타데이터와 함께 여러 텍스트를 일괄 임베딩합니다."""
        start_time = time.time()

        vectors = await self.embed_texts(texts)
        processing_time = (time.time() - start_time) * 1000

        return [
            EmbeddingResult(
                text=text,
                vector=vector,
                model_name=self.model_name,
                token_count=len(text.split()),
                processing_time_ms=processing_time / len(texts),
            )
            for text, vector in zip(texts, vectors)
        ]

    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원을 반환합니다."""
        return self._dimension

    def get_model_name(self) -> str:
        """사용 중인 모델명을 반환합니다."""
        return self.model_name

    def get_max_token_length(self) -> Optional[int]:
        """모델의 최대 토큰 길이를 반환합니다."""
        return None  # 랜덤 임베딩은 길이 제한 없음

    async def is_available(self) -> bool:
        """임베딩 서비스가 사용 가능한지 확인합니다."""
        return True  # 항상 사용 가능

    def truncate_text(self, text: str) -> str:
        """모델의 최대 길이에 맞게 텍스트를 자릅니다."""
        return text  # 길이 제한 없음

    async def compute_similarity(self, vector1: Vector, vector2: Vector) -> float:
        """두 벡터 간의 유사도를 계산합니다."""
        v1 = np.array(vector1.values)
        v2 = np.array(vector2.values)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    async def find_most_similar(
        self, query_vector: Vector, candidate_vectors: List[Vector], top_k: int = 10
    ) -> List[tuple[int, float]]:
        """가장 유사한 벡터들을 찾습니다."""
        query = np.array(query_vector.values)
        similarities = []

        for i, candidate in enumerate(candidate_vectors):
            candidate_array = np.array(candidate.values)
            similarity = np.dot(query, candidate_array) / (
                np.linalg.norm(query) * np.linalg.norm(candidate_array)
            )
            similarities.append((i, float(similarity)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def preprocess_text(self, text: str) -> str:
        """임베딩 전 텍스트를 전처리합니다."""
        return text.strip()

    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """임베딩 처리 통계를 반환합니다."""
        return {
            "model": self.model_name,
            "dimension": self._dimension,
            "provider": "Testing",
            "max_tokens": None,
        }

    async def validate_embedding(self, vector: Vector) -> bool:
        """임베딩 벡터가 유효한지 검증합니다."""
        if len(vector.values) != self._dimension:
            return False

        # NaN이나 무한값 확인
        values = np.array(vector.values)
        return not (np.isnan(values).any() or np.isinf(values).any())

    async def warm_up(self) -> bool:
        """모델을 워밍업합니다."""
        return True  # 즉시 사용 가능

    def _text_to_seed(self, text: str) -> int:
        """텍스트를 시드로 변환합니다."""
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % (2**32)
