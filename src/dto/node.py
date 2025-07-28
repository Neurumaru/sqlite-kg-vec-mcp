"""
노드 관련 DTO 정의.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(Enum):
    """노드 타입."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    OBJECT = "object"


@dataclass
class NodeData:
    """노드 데이터."""

    id: str
    name: str
    node_type: NodeType
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    source_documents: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
