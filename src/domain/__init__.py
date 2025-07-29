"""
새로운 문서 기반 지식 그래프 도메인.

이 패키지는 문서를 기반으로 노드와 관계를 생성하고,
문서와 노드/관계 간의 연결을 관리하는 도메인 로직을 포함합니다.
"""

# Entities
from .entities.document import Document
from .entities.node import Node
from .entities.relationship import Relationship

# Domain Events
from .events.base import DomainEvent
from .events.document_processed import DocumentProcessed
from .events.node_created import NodeCreated
from .events.relationship_created import RelationshipCreated

# Domain Exceptions
from .exceptions.base import DomainException
from .exceptions.document_exceptions import (
    ConcurrentModificationError,
    DocumentAlreadyExistsException,
    DocumentNotFoundException,
    InvalidDocumentException,
)
from .exceptions.node_exceptions import (
    InvalidNodeException,
    NodeNotFoundException,
)
from .exceptions.relationship_exceptions import (
    InvalidRelationshipException,
    RelationshipNotFoundException,
)

# Value Objects
from .value_objects.document_id import DocumentId
from .value_objects.node_id import NodeId
from .value_objects.relationship_id import RelationshipId
from .value_objects.vector import Vector

__all__ = [
    # Value Objects
    "DocumentId",
    "NodeId",
    "RelationshipId",
    "Vector",
    # Entities
    "Document",
    "Node",
    "Relationship",
    # Events
    "DomainEvent",
    "DocumentProcessed",
    "NodeCreated",
    "RelationshipCreated",
    # Exceptions
    "DomainException",
    "DocumentNotFoundException",
    "DocumentAlreadyExistsException",
    "InvalidDocumentException",
    "ConcurrentModificationError",
    "NodeNotFoundException",
    "InvalidNodeException",
    "RelationshipNotFoundException",
    "InvalidRelationshipException",
]
