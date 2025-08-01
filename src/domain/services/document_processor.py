"""
문서 처리 도메인 서비스.
"""

import logging
from datetime import datetime
from typing import Any

from src.domain.entities.document import Document, DocumentStatus
from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.dto import (
    DocumentData,
)
from src.dto import DocumentStatus as DTODocumentStatus
from src.dto import (
    DocumentType,
    NodeData,
    RelationshipData,
)
from src.ports.knowledge_extractor import KnowledgeExtractor
from src.ports.repositories.document import DocumentRepository


class KnowledgeExtractionResult:
    """지식 추출 결과."""

    def __init__(self, nodes: list[Node], relationships: list[Relationship]):
        self.nodes = nodes
        self.relationships = relationships
        self.extracted_at = datetime.now()

    def is_empty(self) -> bool:
        """추출된 지식이 없는지 확인."""
        return len(self.nodes) == 0 and len(self.relationships) == 0

    def get_node_count(self) -> int:
        """추출된 노드 수."""
        return len(self.nodes)

    def get_relationship_count(self) -> int:
        """추출된 관계 수."""
        return len(self.relationships)


class DocumentProcessor:
    """
    문서 처리 도메인 서비스.

    문서로부터 노드와 관계를 추출하고,
    문서와 추출된 지식 요소들 간의 연결을 관리합니다.
    """

    def __init__(
        self,
        knowledge_extractor: KnowledgeExtractor,
        document_repository: DocumentRepository | None = None,
        logger: logging.Logger | None = None,
    ):
        self.knowledge_extractor = knowledge_extractor
        self.document_repository = document_repository
        self.logger = logger or logging.getLogger(__name__)

    async def process_document(self, document: Document) -> KnowledgeExtractionResult:
        """
        문서를 처리하여 노드와 관계를 추출합니다.

        Args:
            document: 처리할 문서

        Returns:
            추출된 노드와 관계

        Note:
            Repository가 제공된 경우 트랜잭션 기반 처리를 수행합니다.
        """
        if self.document_repository:
            return await self._process_document_with_persistence(document)
        return await self._process_document_in_memory(document)

    async def _process_document_with_persistence(
        self, document: Document
    ) -> KnowledgeExtractionResult:
        """
        영속성 저장소를 사용한 트랜잭션 기반 문서 처리.

        Args:
            document: 처리할 문서

        Returns:
            추출된 노드와 관계
        """
        self.logger.info("Processing document with persistence: %s", document.id)

        try:
            # 1. 상태를 PROCESSING으로 변경하고 저장
            document.mark_as_processing()

            # Repository가 있는 경우에만 저장 처리
            if self.document_repository is not None:
                # Document 엔티티를 DocumentData DTO로 변환
                document_data = self._document_to_data(document)

                # 문서가 이미 존재하는지 확인 후 적절한 메서드 사용
                if await self.document_repository.exists(str(document.id)):
                    await self.document_repository.update(document_data)
                else:
                    await self.document_repository.save(document_data)

            # 2. 지식 추출
            extraction_result = await self._extract_knowledge(document)

            # 3. 트랜잭션으로 문서 상태와 연결 정보 업데이트
            document.mark_as_processed()
            self._link_document_to_knowledge(document, extraction_result)

            node_ids = [node.id for node in extraction_result.nodes]
            relationship_ids = [rel.id for rel in extraction_result.relationships]

            if self.document_repository is not None:
                document_data = self._document_to_data(document)
                node_id_strings = [str(node_id) for node_id in node_ids]
                relationship_id_strings = [str(rel_id) for rel_id in relationship_ids]

                await self.document_repository.update_with_knowledge(
                    document_data, node_id_strings, relationship_id_strings
                )

            self.logger.info(
                "Successfully processed document %s: %s nodes, %s relationships",
                document.id,
                extraction_result.get_node_count(),
                extraction_result.get_relationship_count(),
            )

            return extraction_result

        except Exception as exception:
            self.logger.error("Failed to process document %s: %s", document.id, exception)
            # 실패 시 상태 업데이트
            document.mark_as_failed(str(exception))
            try:
                if self.document_repository is not None:
                    document_data = self._document_to_data(document)
                    await self.document_repository.update(document_data)
            except Exception as update_error:
                self.logger.error("Failed to update document status: %s", update_error)
            raise

    async def _process_document_in_memory(self, document: Document) -> KnowledgeExtractionResult:
        """
        메모리에서만 문서 처리 (기존 동작).

        Args:
            document: 처리할 문서

        Returns:
            추출된 노드와 관계
        """
        self.logger.info("Processing document in memory: %s", document.id)

        try:
            # 문서 상태를 처리 중으로 변경
            document.mark_as_processing()

            # 지식 추출 (지식 추출 포트 사용)
            extraction_result = await self._extract_knowledge(document)

            # 문서와 추출된 요소들 간의 연결 설정
            self._link_document_to_knowledge(document, extraction_result)

            # 문서 상태를 처리 완료로 변경
            document.mark_as_processed()

            self.logger.info(
                "Successfully processed document %s: %s nodes, %s relationships",
                document.id,
                extraction_result.get_node_count(),
                extraction_result.get_relationship_count(),
            )

            return extraction_result

        except Exception as exception:
            self.logger.error("Failed to process document %s: %s", document.id, exception)
            document.mark_as_failed(str(exception))
            raise

    async def _extract_knowledge(self, document: Document) -> KnowledgeExtractionResult:
        """
        문서로부터 지식을 추출합니다.

        지식 추출 포트를 사용하여 문서에서 개체와 관계를 추출합니다.
        """
        # Document 엔티티를 DocumentData DTO로 변환
        document_data = self._document_to_data(document)
        node_data_list, relationship_data_list = await self.knowledge_extractor.extract(
            document_data
        )

        # DTO를 도메인 엔티티로 변환
        nodes = [self._node_data_to_entity(node_data) for node_data in node_data_list]
        relationships = [
            self._relationship_data_to_entity(rel_data) for rel_data in relationship_data_list
        ]

        return KnowledgeExtractionResult(nodes, relationships)

    def _link_document_to_knowledge(
        self, document: Document, extraction_result: KnowledgeExtractionResult
    ) -> None:
        """문서와 추출된 지식 요소들 간의 연결을 설정합니다."""
        # 문서에 연결된 노드들 추가
        for node in extraction_result.nodes:
            document.add_connected_node(node.id)

        # 문서에 연결된 관계들 추가
        for relationship in extraction_result.relationships:
            document.add_connected_relationship(relationship.id)

    def update_document_links(
        self,
        document: Document,
        added_nodes: list[NodeId] = None,
        removed_nodes: list[NodeId] = None,
        added_relationships: list[RelationshipId] = None,
        removed_relationships: list[RelationshipId] = None,
    ) -> None:
        """문서의 지식 요소 연결을 업데이트합니다."""

        # 노드 연결 추가
        if added_nodes:
            for node_id in added_nodes:
                document.add_connected_node(node_id)

        # 노드 연결 제거
        if removed_nodes:
            for node_id in removed_nodes:
                document.remove_connected_node(node_id)

        # 관계 연결 추가
        if added_relationships:
            for rel_id in added_relationships:
                document.add_connected_relationship(rel_id)

        # 관계 연결 제거
        if removed_relationships:
            for rel_id in removed_relationships:
                document.remove_connected_relationship(rel_id)

    def validate_document_for_processing(self, document: Document) -> bool:
        """문서가 처리 가능한 상태인지 검증합니다."""
        if document.status == DocumentStatus.PROCESSING:
            self.logger.warning("Document %s is already being processed", document.id)
            return False

        if document.status == DocumentStatus.PROCESSED:
            self.logger.info("Document %s has already been processed", document.id)
            return False

        if not document.content.strip():
            self.logger.error("Document %s has no content", document.id)
            return False

        return True

    async def reprocess_document(self, document: Document) -> KnowledgeExtractionResult:
        """문서를 재처리합니다."""
        self.logger.info("Reprocessing document: %s", document.id)

        # 기존 연결 정보 초기화
        document.connected_nodes.clear()
        document.connected_relationships.clear()

        # 상태를 대기 중으로 재설정
        document.status = DocumentStatus.PENDING
        document.processed_at = None

        # Repository가 있는 경우 상태 업데이트 저장
        if self.document_repository:
            document_data = self._document_to_data(document)
            await self.document_repository.update(document_data)

        # 재처리 실행
        return await self.process_document(document)

    def _document_to_data(self, document: Document) -> DocumentData:
        """Document 엔티티를 DocumentData DTO로 변환합니다."""
        # DocumentStatus를 DTO 버전으로 변환
        dto_status_map = {
            DocumentStatus.PENDING: DTODocumentStatus.PENDING,
            DocumentStatus.PROCESSING: DTODocumentStatus.PROCESSING,
            DocumentStatus.PROCESSED: DTODocumentStatus.COMPLETED,
            DocumentStatus.FAILED: DTODocumentStatus.FAILED,
        }

        # DocumentType 변환 (엔티티와 DTO에서 동일한 이름 사용)
        dto_type = DocumentType(document.doc_type.value)

        return DocumentData(
            id=str(document.id),
            title=document.title,
            content=document.content,
            doc_type=dto_type,
            status=dto_status_map[document.status],
            metadata=document.metadata,
            version=document.version,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processed_at=document.processed_at,
            connected_nodes=[str(node_id) for node_id in document.connected_nodes],
            connected_relationships=[str(rel_id) for rel_id in document.connected_relationships],
        )

    def _node_data_to_entity(self, node_data: NodeData) -> Node:
        """NodeData DTO를 Node 엔티티로 변환합니다."""

        # NodeType 변환
        entity_type = NodeType(node_data.node_type.value)

        return Node(
            id=NodeId.from_string(node_data.id),
            name=node_data.name,
            node_type=entity_type,
            properties=node_data.properties,
            source_documents=[
                DocumentId.from_string(doc_id) for doc_id in node_data.source_documents
            ],
            created_at=node_data.created_at or datetime.now(),
            updated_at=node_data.updated_at or datetime.now(),
        )

    def _relationship_data_to_entity(self, rel_data: RelationshipData) -> Relationship:
        """RelationshipData DTO를 Relationship 엔티티로 변환합니다."""

        try:
            entity_type = RelationshipType(rel_data.relationship_type.value)
        except ValueError as exception:
            entity_type = RelationshipType.OTHER
            self.logger.warning(
                "Unknown relationship type %s: %s", rel_data.relationship_type.value, exception
            )

        return Relationship(
            id=RelationshipId.from_string(rel_data.id),
            source_node_id=NodeId.from_string(rel_data.source_node_id),
            target_node_id=NodeId.from_string(rel_data.target_node_id),
            relationship_type=entity_type,
            label=rel_data.relationship_type.value,
            confidence=rel_data.confidence_score or 1.0,
            properties=rel_data.properties,
            source_documents=[
                DocumentId.from_string(doc_id) for doc_id in rel_data.source_documents
            ],
            created_at=rel_data.created_at or datetime.now(),
            updated_at=rel_data.updated_at or datetime.now(),
        )

    def get_processing_statistics(self, documents: list[Document]) -> dict[str, Any]:
        """문서 처리 통계 정보를 반환합니다."""
        total = len(documents)
        processed = sum(1 for doc in documents if doc.status == DocumentStatus.PROCESSED)
        processing = sum(1 for doc in documents if doc.status == DocumentStatus.PROCESSING)
        failed = sum(1 for doc in documents if doc.status == DocumentStatus.FAILED)
        pending = sum(1 for doc in documents if doc.status == DocumentStatus.PENDING)

        total_nodes = sum(len(doc.connected_nodes) for doc in documents)
        total_relationships = sum(len(doc.connected_relationships) for doc in documents)

        return {
            "total_documents": total,
            "processed": processed,
            "processing": processing,
            "failed": failed,
            "pending": pending,
            "processing_rate": processed / total if total > 0 else 0,
            "total_extracted_nodes": total_nodes,
            "total_extracted_relationships": total_relationships,
            "avg_nodes_per_document": total_nodes / processed if processed > 0 else 0,
            "avg_relationships_per_document": (
                total_relationships / processed if processed > 0 else 0
            ),
        }
