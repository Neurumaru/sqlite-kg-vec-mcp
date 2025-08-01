"""
Domain event base class.
"""

import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar

T = TypeVar("T", bound="DomainEvent")


@dataclass
class DomainEvent(ABC):
    """
    Domain event base class.

    Represents important events that occur in the domain.
    """

    event_id: str
    occurred_at: datetime
    event_type: str
    aggregate_id: str
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls: type[T], aggregate_id: str, **kwargs: Any) -> T:
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
