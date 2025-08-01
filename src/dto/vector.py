"""
벡터 관련 DTO 정의.
"""

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorData:
    """
    벡터 데이터를 나타내는 DTO 클래스.

    Attributes:
        values: 벡터 값들의 리스트 (부동소수점 숫자)
        dimension: 벡터의 차원 수
        metadata: 벡터와 관련된 추가 메타데이터
    """

    values: list[float]  # 벡터의 실제 값들
    dimension: int  # 벡터의 차원 수
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        객체 생성 후 데이터 검증을 수행합니다.

        Raises:
            ValueError: 잘못된 벡터 데이터가 제공된 경우
            TypeError: 잘못된 타입이 제공된 경우
        """
        # values 타입 검증
        if not isinstance(self.values, list):
            raise TypeError("values는 list 타입이어야 합니다")

        # 빈 벡터 검증
        if not self.values:
            raise ValueError("벡터 값이 비어있을 수 없습니다")

        # 모든 값이 숫자인지 검증
        for i, value in enumerate(self.values):
            if not isinstance(value, int | float):
                raise TypeError(f"인덱스 {i}의 값은 숫자여야 합니다. 받은 타입: {type(value)}")
            # NaN이나 무한대 값 검증
            if math.isnan(value):  # NaN 체크
                raise ValueError(f"인덱스 {i}에 NaN 값이 포함되어 있습니다")
            if abs(value) == float("inf"):
                raise ValueError(f"인덱스 {i}에 무한대 값이 포함되어 있습니다")

        # dimension 타입 및 값 검증
        if not isinstance(self.dimension, int):
            raise TypeError("dimension은 정수여야 합니다")

        if self.dimension <= 0:
            raise ValueError("dimension은 양수여야 합니다")

        # 벡터 차원과 실제 값 개수 일치 검증
        if len(self.values) != self.dimension:
            raise ValueError(
                f"벡터 차원({self.dimension})과 실제 값 개수({len(self.values)})가 일치하지 않습니다"
            )

        # metadata 타입 검증
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata는 딕셔너리여야 합니다")

    def __len__(self) -> int:
        """벡터의 차원 수를 반환합니다."""
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        """
        지정된 인덱스의 벡터 값을 반환합니다.

        Args:
            index: 접근할 인덱스

        Returns:
            해당 인덱스의 벡터 값

        Raises:
            TypeError: 인덱스가 정수가 아닌 경우
            IndexError: 인덱스가 범위를 벗어난 경우
        """
        if not isinstance(index, int):
            raise TypeError("인덱스는 정수여야 합니다")

        if index < 0:
            index = len(self.values) + index

        if index < 0 or index >= len(self.values):
            raise IndexError(f"인덱스 {index}가 벡터 범위(0-{len(self.values)-1})를 벗어났습니다")

        return self.values[index]
