"""
DTO 매퍼 구현체.
"""

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.dto import (
    DocumentData,
)
from src.dto import DocumentStatus as DTODocumentStatus
from src.dto import DocumentType as DTODocumentType
from src.dto import (
    NodeData,
)
from src.dto import NodeType as DTONodeType
from src.dto import (
    RelationshipData,
)
from src.dto import RelationshipType as DTORelationshipType
from src.ports.mappers import DocumentMapper, NodeMapper, RelationshipMapper


class DocumentDTOMapper(DocumentMapper):
    """문서 엔티티와 DTO 간 매퍼 구현체."""

    def to_data(self, document: Document) -> DocumentData:
        """도메인 엔티티를 DTO로 변환."""
        return DocumentData(
            id=str(document.id.value),
            title=document.title,
            content=document.content,
            doc_type=DTODocumentType(document.doc_type.value),
            metadata=document.metadata,
            status=self._status_to_dto(document.status),
            created_at=document.created_at,
            updated_at=document.updated_at,
            version=document.version,
        )

    def from_data(self, data: DocumentData) -> Document:
        """DTO를 도메인 엔티티로 변환."""
        return Document(
            id=DocumentId(data.id),
            title=data.title,
            content=data.content,
            doc_type=DocumentType(data.doc_type.value),
            metadata=data.metadata,
            status=self._status_from_dto(data.status),
            created_at=data.created_at,
            updated_at=data.updated_at,
            version=data.version,
        )

    def _status_to_dto(self, status: DocumentStatus) -> DTODocumentStatus:
        """도메인 상태를 DTO 상태로 변환."""
        mapping = {
            DocumentStatus.PENDING: DTODocumentStatus.PENDING,
            DocumentStatus.PROCESSING: DTODocumentStatus.PROCESSING,
            DocumentStatus.PROCESSED: DTODocumentStatus.COMPLETED,
            DocumentStatus.FAILED: DTODocumentStatus.FAILED,
        }
        return mapping[status]

    def _status_from_dto(self, status: DTODocumentStatus) -> DocumentStatus:
        """DTO 상태를 도메인 상태로 변환."""
        mapping = {
            DTODocumentStatus.PENDING: DocumentStatus.PENDING,
            DTODocumentStatus.PROCESSING: DocumentStatus.PROCESSING,
            DTODocumentStatus.COMPLETED: DocumentStatus.PROCESSED,
            DTODocumentStatus.FAILED: DocumentStatus.FAILED,
        }
        return mapping[status]


class NodeDTOMapper(NodeMapper):
    """노드 엔티티와 DTO 간 매퍼 구현체."""

    def to_data(self, node: Node) -> NodeData:
        """도메인 엔티티를 DTO로 변환."""
        return NodeData(
            id=str(node.id.value),
            name=node.name,
            node_type=DTONodeType(node.node_type.value),
            properties=node.properties,
        )

    def from_data(self, data: NodeData) -> Node:
        """DTO를 도메인 엔티티로 변환."""
        return Node(
            id=NodeId(data.id),
            name=data.name,
            node_type=NodeType(data.node_type.value),
            properties=data.properties,
        )


class RelationshipDTOMapper(RelationshipMapper):
    """관계 엔티티와 DTO 간 매퍼 구현체."""

    def to_data(self, relationship: Relationship) -> RelationshipData:
        """도메인 엔티티를 DTO로 변환."""
        return RelationshipData(
            id=str(relationship.id.value),
            source_node_id=str(relationship.source_node_id.value),
            target_node_id=str(relationship.target_node_id.value),
            relationship_type=DTORelationshipType(relationship.relationship_type.value),
            properties=relationship.properties,
        )

    def from_data(self, data: RelationshipData) -> Relationship:
        """DTO를 도메인 엔티티로 변환."""
        return Relationship(
            id=RelationshipId(data.id),
            source_node_id=NodeId(data.source_node_id),
            target_node_id=NodeId(data.target_node_id),
            relationship_type=RelationshipType(data.relationship_type.value),
            label=f"{data.relationship_type.value}",  # DTO에서 레이블 생성
            properties=data.properties,
        )
