"""
OpenAI 텍스트 임베딩 어댑터.
"""

import time
from typing import Any

try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import numpy as np

from src.common.config.llm import OpenAIConfig
from src.dto.embedding import EmbeddingResult
from src.ports.text_embedder import TextEmbedder


class OpenAITextEmbedder(TextEmbedder):
    """OpenAI API를 사용한 텍스트 임베딩 어댑터."""

    def __init__(
        self,
        config: OpenAIConfig | None = None,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
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
            raise ImportError("openai가 설치되지 않았습니다. 'pip install openai'로 설치해주세요.")

        # Use config if provided, otherwise fall back to individual parameters
        if config is None:
            config = OpenAIConfig()

        # Override config with individual parameters if provided (for backward compatibility)
        self.api_key = api_key or config.api_key
        if not self.api_key:
            raise ValueError("OpenAI API 키가 제공되지 않았고 OPENAI_API_KEY 환경변수도 없습니다.")

        self.client = AsyncOpenAI(api_key=self.api_key, timeout=config.timeout)
        self.model = model or config.embedding_model
        self.custom_dimension = dimension or config.embedding_dimension
        self.timeout = config.timeout

        # 알려진 모델의 기본 차원
        self._default_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        # 실제 차원 계산 - custom_dimension이 우선, 없으면 모델 기본값
        actual_model = self.model or "text-embedding-3-small"
        self._dimension = self.custom_dimension or self._default_dimensions.get(actual_model, 1536)

    async def embed_text(self, text: str) -> EmbeddingResult:
        """단일 텍스트를 임베딩합니다."""
        if not text or not text.strip():
            raise ValueError("텍스트가 비어있습니다.")

        start_time = time.time()

        try:
            kwargs: dict[str, Any] = {"input": text, "model": self.model}
            if self.custom_dimension is not None and isinstance(self.custom_dimension, int):
                kwargs["dimensions"] = self.custom_dimension

            response = await self.client.embeddings.create(**kwargs)

            if not response.data:
                raise ValueError("OpenAI API에서 임베딩 데이터를 받지 못했습니다.")

            embedding_data = np.array(response.data[0].embedding, dtype=np.float32)
            processing_time = (time.time() - start_time) * 1000  # milliseconds

            return EmbeddingResult(
                text=text,
                embedding=embedding_data.tolist(),
                model_name=self.model,
                dimension=len(embedding_data),
                processing_time_ms=processing_time,
                metadata={
                    "usage": response.usage.model_dump() if response.usage else {},
                    "model": response.model if hasattr(response, "model") else self.model,
                },
            )

        except openai.AuthenticationError as exception:
            raise ValueError(f"OpenAI API 인증 실패: {str(exception)}") from exception
        except openai.RateLimitError as exception:
            raise ValueError(f"OpenAI API 요청 한도 초과: {str(exception)}") from exception
        except openai.APIConnectionError as exception:
            raise ConnectionError(f"OpenAI API 연결 실패: {str(exception)}") from exception
        except openai.APIError as exception:
            raise RuntimeError(f"OpenAI API 오류: {str(exception)}") from exception
        except Exception as exception:
            raise RuntimeError(f"임베딩 생성 중 예상치 못한 오류: {str(exception)}") from exception

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """여러 텍스트를 일괄 임베딩합니다."""
        if not texts:
            return []

        # 빈 텍스트 검증
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"인덱스 {i}의 텍스트가 비어있습니다.")

        start_time = time.time()

        try:
            kwargs: dict[str, Any] = {"input": texts, "model": self.model}
            if self.custom_dimension is not None and isinstance(self.custom_dimension, int):
                kwargs["dimensions"] = self.custom_dimension

            response = await self.client.embeddings.create(**kwargs)

            if not response.data or len(response.data) != len(texts):
                raise ValueError("OpenAI API에서 예상한 수만큼의 임베딩 데이터를 받지 못했습니다.")

            processing_time = (time.time() - start_time) * 1000  # milliseconds

            results = []
            for i, (text, data) in enumerate(zip(texts, response.data, strict=False)):
                embedding_data = np.array(data.embedding, dtype=np.float32)
                results.append(
                    EmbeddingResult(
                        text=text,
                        embedding=embedding_data.tolist(),
                        model_name=self.model,
                        dimension=len(embedding_data),
                        processing_time_ms=processing_time / len(texts),  # 평균 처리 시간
                        metadata={
                            "batch_index": i,
                            "batch_size": len(texts),
                            "usage": response.usage.model_dump() if response.usage else {},
                            "model": response.model if hasattr(response, "model") else self.model,
                        },
                    )
                )

            return results

        except openai.AuthenticationError as exception:
            raise ValueError(f"OpenAI API 인증 실패: {str(exception)}") from exception
        except openai.RateLimitError as exception:
            raise ValueError(f"OpenAI API 요청 한도 초과: {str(exception)}") from exception
        except openai.APIConnectionError as exception:
            raise ConnectionError(f"OpenAI API 연결 실패: {str(exception)}") from exception
        except openai.APIError as exception:
            raise RuntimeError(f"OpenAI API 오류: {str(exception)}") from exception
        except Exception as exception:
            raise RuntimeError(
                f"일괄 임베딩 생성 중 예상치 못한 오류: {str(exception)}"
            ) from exception

    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원을 반환합니다."""
        return self._dimension

    async def is_available(self) -> bool:
        """임베딩 서비스가 사용 가능한지 확인합니다."""
        try:
            # 간단한 테스트 텍스트로 가용성 확인
            test_result = await self.embed_text("test")
            return test_result is not None and len(test_result.embedding) > 0
        except (ValueError, ConnectionError, RuntimeError):
            # 예상되는 오류들은 서비스 사용 불가로 간주
            return False
        except Exception:
            # 예상치 못한 오류도 사용 불가로 간주
            return False
