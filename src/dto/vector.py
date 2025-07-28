"""
벡터 관련 DTO 정의.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class VectorData:
    """벡터 데이터."""

    values: List[float]
    dimension: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> float:
        return self.values[index]
