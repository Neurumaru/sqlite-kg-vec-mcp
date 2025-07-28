"""
문서 처리 도메인 서비스.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.domain.entities.document import Document, DocumentStatus
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.ports.knowledge_extractor import KnowledgeExtractor
from src.ports.repositories.document import DocumentRepository


class KnowledgeExtractionResult:
    """지식 추출 결과."""

    def __init__(self, nodes: List[Node], relationships: List[Relationship]):
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
        document_repository: Optional[DocumentRepository] = None,
        logger: Optional[logging.Logger] = None,
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
        else:
            return await self._process_document_in_memory(document)

    async def _process_document_with_persistence(self, document: Document) -> KnowledgeExtractionResult:
        """
        영속성 저장소를 사용한 트랜잭션 기반 문서 처리.

        Args:
            document: 처리할 문서

        Returns:
            추출된 노드와 관계
        """
        # TODO: 모니터링 강화 - 트랜잭션 성능 메트릭 추가
        # 처리 시작 시간, 각 단계별 소요 시간, 메모리 사용량 등을 측정하여
        # 성능 모니터링 시스템에 전송하는 기능 구현 필요
        # 예: start_time = time.time(), transaction_metrics.record_duration()
        
        self.logger.info(f"Processing document with persistence: {document.id}")

        try:
            # 1. 상태를 PROCESSING으로 변경하고 저장
            document.mark_as_processing()
            
            # 문서가 이미 존재하는지 확인 후 적절한 메서드 사용
            if await self.document_repository.exists(document.id):
                await self.document_repository.update(document)
            else:
                await self.document_repository.save(document)

            # 2. 지식 추출
            extraction_result = await self._extract_knowledge(document)

            # 3. 트랜잭션으로 문서 상태와 연결 정보 업데이트
            document.mark_as_processed()
            self._link_document_to_knowledge(document, extraction_result)

            node_ids = [node.id for node in extraction_result.nodes]
            relationship_ids = [rel.id for rel in extraction_result.relationships]

            await self.document_repository.update_with_knowledge(
                document, node_ids, relationship_ids
            )

            self.logger.info(
                f"Successfully processed document {document.id}: "
                f"{extraction_result.get_node_count()} nodes, "
                f"{extraction_result.get_relationship_count()} relationships"
            )

            return extraction_result

        except Exception as e:
            self.logger.error(f"Failed to process document {document.id}: {e}")
            # 실패 시 상태 업데이트
            document.mark_as_failed(str(e))
            try:
                await self.document_repository.update(document)
            except Exception as update_error:
                self.logger.error(f"Failed to update document status: {update_error}")
            raise

    async def _process_document_in_memory(self, document: Document) -> KnowledgeExtractionResult:
        """
        메모리에서만 문서 처리 (기존 동작).

        Args:
            document: 처리할 문서

        Returns:
            추출된 노드와 관계
        """
        self.logger.info(f"Processing document in memory: {document.id}")

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
                f"Successfully processed document {document.id}: "
                f"{extraction_result.get_node_count()} nodes, "
                f"{extraction_result.get_relationship_count()} relationships"
            )

            return extraction_result

        except Exception as e:
            self.logger.error(f"Failed to process document {document.id}: {e}")
            document.mark_as_failed(str(e))
            raise

    async def _extract_knowledge(self, document: Document) -> KnowledgeExtractionResult:
        """
        문서로부터 지식을 추출합니다.

        지식 추출 포트를 사용하여 문서에서 개체와 관계를 추출합니다.
        """
        nodes, relationships = await self.knowledge_extractor.extract_knowledge(
            document
        )
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
        added_nodes: List[NodeId] = None,
        removed_nodes: List[NodeId] = None,
        added_relationships: List[RelationshipId] = None,
        removed_relationships: List[RelationshipId] = None,
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
            self.logger.warning(f"Document {document.id} is already being processed")
            return False

        if document.status == DocumentStatus.PROCESSED:
            self.logger.info(f"Document {document.id} has already been processed")
            return False

        if not document.content.strip():
            self.logger.error(f"Document {document.id} has no content")
            return False

        return True

    async def reprocess_document(self, document: Document) -> KnowledgeExtractionResult:
        """문서를 재처리합니다."""
        self.logger.info(f"Reprocessing document: {document.id}")

        # 기존 연결 정보 초기화
        document.connected_nodes.clear()
        document.connected_relationships.clear()

        # 상태를 대기 중으로 재설정
        document.status = DocumentStatus.PENDING
        document.processed_at = None

        # Repository가 있는 경우 상태 업데이트 저장
        if self.document_repository:
            await self.document_repository.update(document)

        # 재처리 실행
        return await self.process_document(document)

    def get_processing_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """문서 처리 통계 정보를 반환합니다."""
        total = len(documents)
        processed = sum(
            1 for doc in documents if doc.status == DocumentStatus.PROCESSED
        )
        processing = sum(
            1 for doc in documents if doc.status == DocumentStatus.PROCESSING
        )
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
