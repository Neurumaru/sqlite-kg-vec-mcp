"""
노드 생성 이벤트.
"""

from dataclasses import dataclass
from typing import Optional

from .base import DomainEvent


@dataclass 
class NodeCreated(DomainEvent):
    """
    새로운 노드가 생성되었을 때 발생하는 이벤트.
    """
    
    node_id: str
    node_name: str
    node_type: str
    source_document_id: Optional[str] = None
    
    @classmethod
    def create(cls, node_id: str, node_name: str, node_type: str, 
               source_document_id: Optional[str] = None) -> "NodeCreated":
        """노드 생성 이벤트 생성."""
        return super().create(
            aggregate_id=node_id,
            node_id=node_id,
            node_name=node_name,
            node_type=node_type,
            source_document_id=source_document_id
        )