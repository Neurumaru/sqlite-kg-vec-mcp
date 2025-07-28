"""
이벤트 관련 DTO 정의.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class EventData:
    """도메인 이벤트 데이터."""

    event_type: str
    entity_id: str
    entity_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    version: int = 1

    def __post_init__(self):
        if self.correlation_id is None:
            self.correlation_id = self.entity_id
