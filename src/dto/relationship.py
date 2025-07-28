"""
관계 관련 DTO 정의.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RelationshipType(Enum):
    """관계 타입."""

    RELATES_TO = "relates_to"
    IS_PART_OF = "is_part_of"
    CONTAINS = "contains"
    CAUSED_BY = "caused_by"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class RelationshipData:
    """관계 데이터."""

    id: str
    source_node_id: str
    target_node_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    source_documents: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
