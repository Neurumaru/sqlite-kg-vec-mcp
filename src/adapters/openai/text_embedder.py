"""
OpenAI 텍스트 임베딩 어댑터.
"""

from typing import Any, Dict, List, Optional

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import numpy as np

from src.common.config.llm import OpenAIConfig
from src.domain.value_objects.vector import Vector
from src.ports.text_embedder import EmbeddingConfig, EmbeddingResult, TextEmbedder


class OpenAITextEmbedder(TextEmbedder):
    """OpenAI API를 사용한 텍스트 임베딩 어댑터."""

    def __init__(
        self,
        config: Optional[OpenAIConfig] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        """
        OpenAI 임베딩 어댑터를 초기화합니다.

        Args:
            config: OpenAI 설정 객체
            api_key: OpenAI API 키 (deprecated, config 사용 권장)
            model: 임베딩 모델명 (deprecated, config 사용 권장)
            dimension: 임베딩 차원 (deprecated, config 사용 권장)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai가 설치되지 않았습니다. 'pip install openai'로 설치해주세요."
            )

        # Use config if provided, otherwise fall back to individual parameters
        if config is None:
            config = OpenAIConfig()
        
        # Override config with individual parameters if provided (for backward compatibility)
        self.api_key = api_key or config.api_key
        if not self.api_key:
            raise ValueError(
                "OpenAI API 키가 제공되지 않았고 OPENAI_API_KEY 환경변수도 없습니다."
            )

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model or config.embedding_model
        self.custom_dimension = dimension or config.embedding_dimension
        self.timeout = config.timeout

        # 알려진 모델의 기본 차원
        self._default_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        self._dimension = dimension or self._default_dimensions.get(model, 1536)

    async def embed_text(self, text: str) -> Vector:
        """단일 텍스트를 임베딩합니다."""
        kwargs = {"input": text, "model": self.model}
        if self.custom_dimension is not None:
            kwargs["dimensions"] = self.custom_dimension

        response = self.client.embeddings.create(**kwargs)
        embedding_data = np.array(response.data[0].embedding, dtype=np.float32)
        return Vector(values=embedding_data.tolist())

    async def embed_texts(self, texts: List[str]) -> List[Vector]:
        """여러 텍스트를 일괄 임베딩합니다."""
        kwargs = {"input": texts, "model": self.model}
        if self.custom_dimension is not None:
            kwargs["dimensions"] = self.custom_dimension

        response = self.client.embeddings.create(**kwargs)
        return [
            Vector(values=np.array(data.embedding, dtype=np.float32).tolist())
            for data in response.data
        ]

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """메타데이터와 함께 텍스트를 임베딩합니다."""
        import time

        start_time = time.time()

        vector = await self.embed_text(text)
        processing_time = (time.time() - start_time) * 1000

        return EmbeddingResult(
            text=text,
            vector=vector,
            model_name=self.model,
            token_count=None,  # OpenAI API는 토큰 수를 직접 제공하지 않음
            processing_time_ms=processing_time,
        )

    async def batch_embed_with_metadata(
        self, texts: List[str]
    ) -> List[EmbeddingResult]:
        """메타데이터와 함께 여러 텍스트를 일괄 임베딩합니다."""
        import time

        start_time = time.time()

        vectors = await self.embed_texts(texts)
        processing_time = (time.time() - start_time) * 1000

        return [
            EmbeddingResult(
                text=text,
                vector=vector,
                model_name=self.model,
                token_count=None,
                processing_time_ms=processing_time / len(texts),
            )
            for text, vector in zip(texts, vectors)
        ]

    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원을 반환합니다."""
        return self._dimension

    def get_model_name(self) -> str:
        """사용 중인 모델명을 반환합니다."""
        return self.model

    def get_max_token_length(self) -> Optional[int]:
        """모델의 최대 토큰 길이를 반환합니다."""
        # OpenAI 임베딩 모델의 일반적인 토큰 제한
        return 8192

    async def is_available(self) -> bool:
        """임베딩 서비스가 사용 가능한지 확인합니다."""
        try:
            await self.embed_text("test")
            return True
        except Exception:
            return False

    def truncate_text(self, text: str) -> str:
        """모델의 최대 길이에 맞게 텍스트를 자릅니다."""
        max_length = self.get_max_token_length()
        if max_length and len(text) > max_length * 4:  # 대략적인 토큰-문자 비율
            return text[: max_length * 4]
        return text

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
        # 기본적인 정리
        text = text.strip()
        # 여러 연속 공백을 하나로
        import re

        text = re.sub(r"\s+", " ", text)
        return text

    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """임베딩 처리 통계를 반환합니다."""
        return {
            "model": self.model,
            "dimension": self._dimension,
            "provider": "OpenAI",
            "max_tokens": self.get_max_token_length(),
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
        try:
            await self.embed_text("warm up test")
            return True
        except Exception:
            return False
