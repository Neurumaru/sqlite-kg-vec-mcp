"""
임베딩 관련 DTO 정의.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EmbeddingResult:
    """
    임베딩 처리 결과를 나타내는 DTO 클래스.

    Attributes:
        text: 임베딩된 원본 텍스트
        embedding: 임베딩 벡터 값들
        model_name: 사용된 임베딩 모델명
        dimension: 임베딩 벡터의 차원 수
        metadata: 임베딩 과정에서 생성된 추가 메타데이터
        processing_time_ms: 임베딩 처리에 소요된 시간 (밀리초)
    """

    text: str  # 임베딩된 원본 텍스트
    embedding: list[float]  # 임베딩 벡터 값들
    model_name: str  # 사용된 임베딩 모델명
    dimension: int  # 임베딩 벡터의 차원 수
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[float] = None  # 처리 시간 (밀리초)

    def __post_init__(self) -> None:
        """
        객체 생성 후 데이터 검증을 수행합니다.

        Raises:
            ValueError: 잘못된 임베딩 데이터가 제공된 경우
            TypeError: 잘못된 타입이 제공된 경우
        """
        # text 검증
        if not isinstance(self.text, str):
            raise TypeError("text는 문자열이어야 합니다")

        if not self.text.strip():
            raise ValueError("text는 공백이 아닌 문자를 포함해야 합니다")

        # embedding 검증
        if not isinstance(self.embedding, list):
            raise TypeError("embedding은 리스트여야 합니다")

        if not self.embedding:
            raise ValueError("embedding 벡터가 비어있을 수 없습니다")

        # embedding 값들 검증
        for i, value in enumerate(self.embedding):
            if not isinstance(value, int | float):
                raise TypeError(
                    f"embedding 인덱스 {i}의 값은 숫자여야 합니다. 받은 타입: {type(value)}"
                )

            # NaN이나 무한대 값 검증
            if math.isnan(value):  # NaN 체크
                raise ValueError(f"embedding 인덱스 {i}에 NaN 값이 포함되어 있습니다")

            if abs(value) == float("inf"):
                raise ValueError(f"embedding 인덱스 {i}에 무한대 값이 포함되어 있습니다")

        # model_name 검증
        if not isinstance(self.model_name, str):
            raise TypeError("model_name은 문자열이어야 합니다")

        if not self.model_name.strip():
            raise ValueError("model_name은 공백이 아닌 문자를 포함해야 합니다")

        # dimension 검증
        if not isinstance(self.dimension, int):
            raise TypeError("dimension은 정수여야 합니다")

        if self.dimension <= 0:
            raise ValueError("dimension은 양수여야 합니다")

        # embedding 차원과 dimension 일치 검증
        if len(self.embedding) != self.dimension:
            raise ValueError(
                f"embedding 차원({len(self.embedding)})과 지정된 dimension({self.dimension})이 일치하지 않습니다"
            )

        # metadata 검증
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata는 딕셔너리여야 합니다")

        # processing_time_ms 검증
        if self.processing_time_ms is not None:
            if not isinstance(self.processing_time_ms, int | float):
                raise TypeError("processing_time_ms는 숫자여야 합니다")

            if self.processing_time_ms < 0:
                raise ValueError("processing_time_ms는 음수일 수 없습니다")

            # NaN이나 무한대 값 검증
            if self.processing_time_ms != self.processing_time_ms:  # NaN 체크
                raise ValueError("processing_time_ms에 NaN 값이 포함되어 있습니다")

            if abs(self.processing_time_ms) == float("inf"):
                raise ValueError("processing_time_ms에 무한대 값이 포함되어 있습니다")
