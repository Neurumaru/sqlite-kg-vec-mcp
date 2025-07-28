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
from src.ports.text_embedder import TextEmbedder


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

    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원을 반환합니다."""
        return self._dimension

    async def is_available(self) -> bool:
        """임베딩 서비스가 사용 가능한지 확인합니다."""
        try:
            await self.embed_text("test")
            return True
        except Exception:
            return False
