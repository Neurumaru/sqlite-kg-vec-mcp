"""
벡터 임베딩을 위한 Nomic Embed Text 통합.
"""

import json
import logging

import numpy as np
import requests

from src.dto.embedding import EmbeddingResult
from src.ports.text_embedder import TextEmbedder


class NomicEmbedder(TextEmbedder):
    """Ollama를 사용하는 Nomic Embed Text 임베더."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",
        timeout: int = 60,
    ):
        """
        Nomic 임베더를 초기화합니다.

        Args:
            base_url: Ollama 서버 URL
            model_name: Nomic 모델 이름
            timeout: 초 단위 요청 시간 초과
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()

        # 연결을 테스트하고 모델이 사용 가능한지 확인합니다.
        self._ensure_model_available()

    def _ensure_model_available(self) -> bool:
        """Nomic 모델이 사용 가능한지 확인합니다."""
        try:
            # 모델이 이미 사용 가능한지 확인
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            available_models = [model["name"] for model in response.json().get("models", [])]

            if self.model_name in available_models:
                logging.info("Nomic 모델 %s를 사용할 수 있습니다.", self.model_name)
                return True

            # 사용 가능하지 않은 경우 모델을 가져오려고 시도
            logging.info("Nomic 모델 %s를 가져오는 중...", self.model_name)
            pull_response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,  # 모델 다운로드를 위한 5분
            )

            if pull_response.status_code == 200:
                logging.info("%s를 성공적으로 가져왔습니다.", self.model_name)
                return True
            logging.error("%s를 가져오는 데 실패했습니다: %s", self.model_name, pull_response.text)
            return False

        except (requests.ConnectionError, requests.Timeout) as exception:
            logging.error("Nomic 모델 가용성 확인 중 연결 오류: %s", exception)
            return False
        except requests.HTTPError as exception:
            logging.error("Nomic 모델 가용성 확인 중 HTTP 오류: %s", exception)
            return False
        except (ValueError, KeyError) as exception:
            logging.error("Nomic 모델 가용성 확인 중 잘못된 응답 형식: %s", exception)
            return False

    def embed(self, text: str | list[str]) -> np.ndarray:
        """
        Nomic Embed를 사용하여 텍스트에 대한 임베딩을 생성합니다.

        Args:
            text: 단일 텍스트 문자열 또는 텍스트 목록

        Returns:
            임베딩의 numpy 배열
        """
        # 단일 텍스트 입력 처리
        if isinstance(text, str):
            texts = [text]
            return_single = True
        else:
            texts = text
            return_single = False

        embeddings = []

        for text_item in texts:
            try:
                # Ollama 임베딩 엔드포인트에 대한 요청 준비
                data = {"model": self.model_name, "prompt": text_item}

                response = self.session.post(
                    f"{self.base_url}/api/embeddings", json=data, timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()
                embedding = np.array(result["embedding"], dtype=np.float32)
                embeddings.append(embedding)

            except requests.ConnectionError as exception:
                logging.error("텍스트에 대한 Nomic 임베딩 생성 중 연결 오류: %s", exception)
                embeddings.append(np.zeros(768, dtype=np.float32))  # Nomic 기본 차원
            except requests.HTTPError as exception:
                logging.error("텍스트에 대한 Nomic 임베딩 생성 중 HTTP 오류: %s", exception)
                embeddings.append(np.zeros(768, dtype=np.float32))  # Nomic 기본 차원
            except (ValueError, KeyError, json.JSONDecodeError) as exception:
                logging.error("텍스트에 대한 Nomic 임베딩 생성 중 응답 파싱 오류: %s", exception)
                embeddings.append(np.zeros(768, dtype=np.float32))  # Nomic 기본 차원

        embeddings_array = np.array(embeddings)

        if return_single:
            return embeddings_array[0]  # type: ignore[no-any-return]
        return embeddings_array

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """텍스트 배치를 임베딩 벡터로 변환합니다."""
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
        return embeddings

    @property
    def dimension(self) -> int:
        """임베딩의 차원을 반환합니다."""
        # 실제 차원을 얻기 위해 작은 텍스트로 테스트
        try:
            test_embedding = self.embed("test")
            return len(test_embedding)
        except (
            requests.ConnectionError,
            requests.HTTPError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exception:
            logging.warning("임베딩 차원을 가져오는 중 오류 발생: %s", exception)
            # nomic-embed-text의 기본 차원
            return 768

    def get_embedding_dimension(self) -> int:
        """Nomic Embed Text의 임베딩 차원을 가져옵니다."""
        return self.dimension

    # TextEmbedder 포트 구현 (비동기 메서드)

    async def embed_text(self, text: str) -> EmbeddingResult:
        """
        단일 텍스트를 임베딩합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 결과
        """
        embedding = self.embed(text)
        return EmbeddingResult(
            text=text,
            embedding=embedding.tolist(),
            model_name=self.model_name,
            dimension=len(embedding),
        )

    async def embed_texts(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        여러 텍스트를 일괄 임베딩합니다.

        Args:
            texts: 임베딩할 텍스트들

        Returns:
            임베딩 결과들
        """
        embeddings = self.embed(texts)
        results = []
        for i, embedding in enumerate(embeddings):
            results.append(
                EmbeddingResult(
                    text=texts[i],
                    embedding=embedding.tolist(),
                    model_name=self.model_name,
                    dimension=len(embedding),
                )
            )
        return results

    async def is_available(self) -> bool:
        """
        임베딩 서비스가 사용 가능한지 확인합니다.

        Returns:
            사용 가능 여부
        """
        return self._ensure_model_available()

    def batch_embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        텍스트 배치에 대한 임베딩을 생성합니다.

        Args:
            texts: 텍스트 문자열 목록
            batch_size: 한 번에 처리할 텍스트 수

        Returns:
            임베딩의 numpy 배열
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # 일관된 모양을 보장하기 위해 각 텍스트를 개별적으로 처리
            for text in batch:
                embedding = self.embed(text)  # 단일 텍스트에 대해 1D 배열 반환
                all_embeddings.append(embedding)

            # 큰 배치에 대한 진행 상황 기록
            if len(texts) > 100:
                logging.info("%s/%s 텍스트 처리됨", min(i + batch_size, len(texts)), len(texts))

        return np.array(all_embeddings)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 임베딩 간의 코사인 유사도를 계산합니다.

        Args:
            embedding1: 첫 번째 임베딩 벡터
            embedding2: 두 번째 임베딩 벡터

        Returns:
            코사인 유사도 점수
        """
        # 벡터 정규화
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 코사인 유사도 계산
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def most_similar_texts(
        self, query_text: str, candidate_texts: list[str], top_k: int = 5
    ) -> list[tuple]:
        """
        쿼리와 가장 유사한 텍스트를 찾습니다.

        Args:
            query_text: 쿼리 텍스트
            candidate_texts: 후보 텍스트 목록
            top_k: 반환할 상위 결과 수

        Returns:
            (text, similarity_score) 튜플 목록
        """
        # 임베딩 생성
        query_embedding = self.embed(query_text)
        candidate_embeddings = self.batch_embed(candidate_texts)

        # 유사도 계산
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            sim_score = self.similarity(query_embedding, candidate_embedding)
            similarities.append((candidate_texts[i], sim_score))

        # 유사도로 정렬하고 상위 k개 반환
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_nomic_embedder(
    base_url: str = "http://localhost:11434", model_name: str = "nomic-embed-text"
) -> NomicEmbedder:
    """
    Nomic 임베더를 생성하는 팩토리 함수.

    Args:
        base_url: Ollama 서버 URL
        model_name: Nomic 모델 이름

    Returns:
        설정된 NomicEmbedder 인스턴스
    """
    return NomicEmbedder(base_url=base_url, model_name=model_name)
