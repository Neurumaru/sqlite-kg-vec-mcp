"""
HuggingFace SentenceTransformers 텍스트 임베딩 어댑터.
"""

import time
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import numpy as np

from src.domain.value_objects.vector import Vector
from src.ports.text_embedder import EmbeddingConfig, EmbeddingResult, TextEmbedder


class HuggingFaceTextEmbedder(TextEmbedder):
    """HuggingFace SentenceTransformers를 사용한 텍스트 임베딩 어댑터."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        SentenceTransformer 모델로 초기화합니다.

        Args:
            model_name: 사용할 모델명. 일반적인 옵션들:
                - "all-MiniLM-L6-v2": 빠름, 384차원
                - "all-mpnet-base-v2": 높은 품질, 768차원
                - "paraphrase-multilingual-MiniLM-L12-v2": 다국어, 384차원
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers가 설치되지 않았습니다. "
                "'pip install sentence-transformers'로 설치해주세요."
            )

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._dimension = self.model.get_sentence_embedding_dimension()

    async def embed_text(self, text: str) -> Vector:
        """단일 텍스트를 임베딩합니다."""
        embedding = self.model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return Vector(values=embedding.tolist())

    async def embed_texts(self, texts: List[str]) -> List[Vector]:
        """여러 텍스트를 일괄 임베딩합니다."""
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

        # 2D 배열인 경우 각 행을 Vector로 변환
        if embeddings.ndim == 2:
            return [
                Vector(values=embeddings[i].tolist()) for i in range(len(embeddings))
            ]
        else:
            return [Vector(values=embeddings.tolist())]

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """메타데이터와 함께 텍스트를 임베딩합니다."""
        start_time = time.time()

        vector = await self.embed_text(text)
        processing_time = (time.time() - start_time) * 1000

        return EmbeddingResult(
            text=text,
            vector=vector,
            model_name=self.model_name,
            token_count=self._estimate_token_count(text),
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
                token_count=self._estimate_token_count(text),
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
        # SentenceTransformer 모델의 일반적인 제한
        return (
            self.model.max_seq_length if hasattr(self.model, "max_seq_length") else 512
        )

    async def is_available(self) -> bool:
        """임베딩 서비스가 사용 가능한지 확인합니다."""
        try:
            await self.embed_text("test")
            return True
        except Exception:
            return False

    def truncate_text(self, text: str) -> str:
        """모델의 최대 길이에 맞게 텍스트를 자릅니다."""
        max_tokens = self.get_max_token_length()
        if max_tokens and len(text) > max_tokens * 4:  # 대략적인 토큰-문자 비율
            return text[: max_tokens * 4]
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
            "model": self.model_name,
            "dimension": self._dimension,
            "provider": "HuggingFace",
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

    def _estimate_token_count(self, text: str) -> int:
        """텍스트의 대략적인 토큰 수를 추정합니다."""
        # 대략적인 추정: 단어 수 * 1.3
        word_count = len(text.split())
        return int(word_count * 1.3)
