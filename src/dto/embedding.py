"""
임베딩 관련 DTO 정의.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EmbeddingResult:
    """임베딩 결과 데이터."""

    text: str
    embedding: List[float]
    model_name: str
    dimension: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[float] = None
