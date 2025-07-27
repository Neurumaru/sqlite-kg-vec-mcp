"""
텍스트 임베딩 포트.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.domain.value_objects.vector import Vector


@dataclass
class EmbeddingConfig:
    """임베딩 설정."""

    model_name: str
    dimension: int
    max_tokens: Optional[int] = None
    batch_size: int = 32
    normalize: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmbeddingResult:
    """임베딩 결과."""

    text: str
    vector: Vector
    model_name: str
    token_count: Optional[int] = None
    processing_time_ms: Optional[float] = None


class TextEmbedder(ABC):
    """
    텍스트 임베딩 포트.

    텍스트를 벡터로 변환하는 기능을 제공합니다.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> Vector:
        """
        단일 텍스트를 임베딩합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터
        """
        pass

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[Vector]:
        """
        여러 텍스트를 일괄 임베딩합니다.

        Args:
            texts: 임베딩할 텍스트들

        Returns:
            임베딩 벡터들
        """
        pass

    @abstractmethod
    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """
        메타데이터와 함께 텍스트를 임베딩합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 결과 (벡터 + 메타데이터)
        """
        pass

    @abstractmethod
    async def batch_embed_with_metadata(
        self, texts: List[str]
    ) -> List[EmbeddingResult]:
        """
        메타데이터와 함께 여러 텍스트를 일괄 임베딩합니다.

        Args:
            texts: 임베딩할 텍스트들

        Returns:
            임베딩 결과들
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        임베딩 벡터의 차원을 반환합니다.

        Returns:
            벡터 차원
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        사용 중인 모델명을 반환합니다.

        Returns:
            모델명
        """
        pass

    @abstractmethod
    def get_max_token_length(self) -> Optional[int]:
        """
        모델의 최대 토큰 길이를 반환합니다.

        Returns:
            최대 토큰 길이 또는 None
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """
        임베딩 서비스가 사용 가능한지 확인합니다.

        Returns:
            사용 가능 여부
        """
        pass

    @abstractmethod
    def truncate_text(self, text: str) -> str:
        """
        모델의 최대 길이에 맞게 텍스트를 자릅니다.

        Args:
            text: 원본 텍스트

        Returns:
            잘린 텍스트
        """
        pass

    @abstractmethod
    async def compute_similarity(self, vector1: Vector, vector2: Vector) -> float:
        """
        두 벡터 간의 유사도를 계산합니다.

        Args:
            vector1: 첫 번째 벡터
            vector2: 두 번째 벡터

        Returns:
            유사도 점수 (0.0 ~ 1.0)
        """
        pass

    @abstractmethod
    async def find_most_similar(
        self, query_vector: Vector, candidate_vectors: List[Vector], top_k: int = 10
    ) -> List[tuple[int, float]]:
        """
        가장 유사한 벡터들을 찾습니다.

        Args:
            query_vector: 쿼리 벡터
            candidate_vectors: 후보 벡터들
            top_k: 반환할 최대 개수

        Returns:
            (인덱스, 유사도) 튜플들
        """
        pass

    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """
        임베딩 전 텍스트를 전처리합니다.

        Args:
            text: 원본 텍스트

        Returns:
            전처리된 텍스트
        """
        pass

    @abstractmethod
    async def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        임베딩 처리 통계를 반환합니다.

        Returns:
            통계 정보
        """
        pass

    @abstractmethod
    async def validate_embedding(self, vector: Vector) -> bool:
        """
        임베딩 벡터가 유효한지 검증합니다.

        Args:
            vector: 검증할 벡터

        Returns:
            유효성 여부
        """
        pass

    @abstractmethod
    async def warm_up(self) -> bool:
        """
        모델을 워밍업합니다.

        Returns:
            워밍업 성공 여부
        """
        pass
