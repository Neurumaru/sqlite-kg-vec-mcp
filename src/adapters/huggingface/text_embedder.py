"""
HuggingFace SentenceTransformers 텍스트 임베딩 어댑터.
"""

import time
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


from src.common.observability.logger import get_observable_logger
from src.dto import EmbeddingResult
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

        Raises:
            ImportError: sentence-transformers가 설치되지 않은 경우
            ValueError: 유효하지 않은 모델명인 경우
        """
        if not model_name or not model_name.strip():
            raise ValueError("모델명은 비어있을 수 없습니다.")

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers가 설치되지 않았습니다. "
                "'pip install sentence-transformers'로 설치해주세요."
            )

        self.logger = get_observable_logger(component="huggingface_text_embedder", layer="adapter")

        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            dimension: Optional[int] = self.model.get_sentence_embedding_dimension()
            if dimension is None:
                raise ValueError(f"Unable to determine embedding dimension for model {model_name}")
            self._dimension: int = dimension

            self.logger.info(
                "huggingface_embedder_initialized", model_name=model_name, dimension=self._dimension
            )
        except Exception as exception:
            self.logger.error(
                "huggingface_embedder_initialization_failed",
                model_name=model_name,
                error=str(exception),
            )
            raise

    async def embed_text(self, text: str) -> EmbeddingResult:
        """단일 텍스트를 임베딩합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 결과

        Raises:
            ValueError: 텍스트가 비어있거나 None인 경우
            RuntimeError: 임베딩 생성 중 오류가 발생한 경우
        """
        if not text or not text.strip():
            raise ValueError("텍스트가 비어있거나 None입니다.")

        start_time = time.time()

        try:
            self.logger.debug(
                "embedding_text_started", text_length=len(text), model_name=self.model_name
            )

            embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            processing_time_ms = (time.time() - start_time) * 1000

            result = EmbeddingResult(
                text=text,
                embedding=embedding.tolist(),
                model_name=self.model_name,
                dimension=self._dimension,
                processing_time_ms=processing_time_ms,
            )

            self.logger.debug(
                "embedding_text_completed",
                text_length=len(text),
                processing_time_ms=processing_time_ms,
            )

            return result

        except Exception as exception:
            self.logger.error(
                "embedding_text_failed",
                text_length=len(text),
                error=str(exception),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            raise RuntimeError(
                f"텍스트 임베딩 중 오류가 발생했습니다: {str(exception)}"
            ) from exception

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """여러 텍스트를 일괄 임베딩합니다.

        Args:
            texts: 임베딩할 텍스트들

        Returns:
            임베딩 결과들

        Raises:
            ValueError: 텍스트 리스트가 비어있거나 None인 경우
            RuntimeError: 임베딩 생성 중 오류가 발생한 경우
        """
        if not texts:
            raise ValueError("텍스트 리스트가 비어있습니다.")

        # 빈 문자열이나 None 값 검증
        invalid_texts = [i for i, text in enumerate(texts) if not text or not text.strip()]
        if invalid_texts:
            raise ValueError(f"인덱스 {invalid_texts}의 텍스트가 비어있거나 None입니다.")

        start_time = time.time()

        try:
            self.logger.debug(
                "embedding_texts_started", text_count=len(texts), model_name=self.model_name
            )

            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            processing_time_ms = (time.time() - start_time) * 1000

            results = []

            # 2D 배열인 경우 각 행을 EmbeddingResult로 변환
            if embeddings.ndim == 2:
                for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False)):
                    results.append(
                        EmbeddingResult(
                            text=text,
                            embedding=embedding.tolist(),
                            model_name=self.model_name,
                            dimension=self._dimension,
                            processing_time_ms=processing_time_ms / len(texts),  # 평균 처리 시간
                            metadata={"batch_index": i, "batch_size": len(texts)},
                        )
                    )
            else:
                # 1D 배열인 경우 (단일 텍스트)
                results.append(
                    EmbeddingResult(
                        text=texts[0],
                        embedding=embeddings.tolist(),
                        model_name=self.model_name,
                        dimension=self._dimension,
                        processing_time_ms=processing_time_ms,
                    )
                )

            self.logger.debug(
                "embedding_texts_completed",
                text_count=len(texts),
                processing_time_ms=processing_time_ms,
            )

            return results

        except Exception as exception:
            self.logger.error(
                "embedding_texts_failed",
                text_count=len(texts),
                error=str(exception),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            raise RuntimeError(
                f"텍스트 일괄 임베딩 중 오류가 발생했습니다: {str(exception)}"
            ) from exception

    def get_embedding_dimension(self) -> int:
        """임베딩 벡터의 차원을 반환합니다."""
        return self._dimension

    async def is_available(self) -> bool:
        """임베딩 서비스가 사용 가능한지 확인합니다.

        Returns:
            서비스 사용 가능 여부
        """
        try:
            self.logger.debug("checking_service_availability")
            await self.embed_text("test")
            self.logger.debug("service_availability_check_passed")
            return True
        except Exception as exception:
            self.logger.warning("service_availability_check_failed", error=str(exception))
            return False
