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
from src.ports.text_embedder import TextEmbedder


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
