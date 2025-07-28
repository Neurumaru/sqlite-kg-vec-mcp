"""
문서 관련 DTO 정의.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DocumentStatus(Enum):
    """문서 상태."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(Enum):
    """문서 타입."""

    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class DocumentData:
    """문서 데이터."""

    id: str
    title: str
    content: str
    doc_type: DocumentType
    status: DocumentStatus
    metadata: Dict[str, Any]
    version: int
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    connected_nodes: List[str] = field(default_factory=list)
    connected_relationships: List[str] = field(default_factory=list)
