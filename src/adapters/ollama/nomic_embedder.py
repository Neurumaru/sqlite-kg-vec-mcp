"""
벡터 임베딩을 위한 Nomic Embed Text 통합.
"""

import logging
from typing import Optional

from ollama import AsyncClient

from src.adapters.ollama.exceptions import OllamaModelException
from src.common.config.llm import OllamaConfig
from src.dto.embedding import EmbeddingResult
from src.ports.text_embedder import TextEmbedder


class NomicEmbedder(TextEmbedder):
    """
    Ollama의 Nomic 임베딩 모델을 사용하는 텍스트 임베더 구현체.
    """

    def __init__(
        self,
        client: Optional[AsyncClient] = None,
        model_name: str = "nomic-embed-text",
        config: Optional[OllamaConfig] = None,
    ):
        if config is None:
            config = OllamaConfig()

        self.client = client
        self.model_name = model_name
        self.dimension = config.embedding_dimension

    async def embed_text(self, text: str) -> EmbeddingResult:
        """
        주어진 텍스트에 대한 임베딩을 생성합니다.

        인자:
            text (str): 임베딩할 텍스트.

        반환:
            EmbeddingResult: 임베딩된 벡터와 관련 메타데이터를 포함하는 객체.

        예외:
            OllamaException: Ollama 서비스 호출 중 오류가 발생한 경우.
        """
        if self.client is None:
            raise OllamaModelException(
                model_name=self.model_name,
                operation="embedding",
                message="Ollama 클라이언트가 초기화되지 않았습니다",
            )
        try:
            response = await self.client.embed(model=self.model_name, input=text)
            embedding = response["embedding"]
            if len(embedding) != self.dimension:
                raise OllamaModelException(
                    model_name=self.model_name,
                    operation="embedding",
                    message=f"예상 임베딩 차원 {self.dimension}과 다릅니다: {len(embedding)}",
                )
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self.model_name,
                dimension=self.dimension,
            )
        except (ConnectionError, TimeoutError, ValueError, KeyError) as exception:
            raise OllamaModelException(
                model_name=self.model_name,
                operation="embedding",
                message=f"텍스트 임베딩 실패: {exception}",
                original_error=exception,
            ) from exception

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        주어진 텍스트 목록에 대한 임베딩을 생성합니다.

        인자:
            texts (list[str]): 임베딩할 텍스트 목록.

        반환:
            list[EmbeddingResult]: 각 텍스트에 대한 임베딩 결과 목록.

        예외:
            OllamaException: Ollama 서비스 호출 중 오류가 발생한 경우.
        """
        results = []
        for text in texts:
            results.append(await self.embed_text(text))
        return results

    async def is_available(self) -> bool:
        """
        임베딩 서비스가 사용 가능한지 확인합니다.

        Returns:
            사용 가능 여부
        """
        if self.client is None:
            return False
        try:
            await self.client.list()
            return True
        except (ConnectionError, TimeoutError) as exception:
            logging.error("Ollama 서비스 가용성 확인 실패: %s", exception)
            return False

    def get_embedding_dimension(self) -> int:
        """
        임베딩 벡터의 차원을 반환합니다.

        Returns:
            벡터 차원
        """
        return self.dimension
