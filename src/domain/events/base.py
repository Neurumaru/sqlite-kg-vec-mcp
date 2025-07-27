"""
도메인 이벤트 기본 클래스.
"""

import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class DomainEvent(ABC):
    """
    도메인 이벤트 기본 클래스.

    도메인에서 발생하는 중요한 사건들을 나타냅니다.
    """

    event_id: str
    occurred_at: datetime
    event_type: str
    aggregate_id: str
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # metadata는 이제 field(default_factory=dict)로 처리됨
        pass

    @classmethod
    def create(cls, aggregate_id: str, **kwargs) -> "DomainEvent":
        """새로운 도메인 이벤트 생성."""
        event_id = str(uuid.uuid4())
        occurred_at = datetime.now()
        event_type = cls.__name__

        return cls(
            event_id=event_id,
            occurred_at=occurred_at,
            event_type=event_type,
            aggregate_id=aggregate_id,
            **kwargs,
        )
