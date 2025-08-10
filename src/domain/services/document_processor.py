"""
문서 처리 도메인 서비스.
"""

from typing import Any, Optional

from src.common.observability.logger import ObservableLogger
from src.domain.entities.document import Document, DocumentStatus
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship
from src.domain.services.document_persistence import DocumentPersistenceService
from src.domain.services.document_statistics import DocumentStatisticsService
from src.domain.services.document_validation import DocumentValidationService
from src.domain.value_objects.knowledge_extraction_result import KnowledgeExtractionResult
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.ports.knowledge_extractor import KnowledgeExtractor
from src.ports.mappers import DocumentMapper, NodeMapper, RelationshipMapper
from src.ports.repositories.document import DocumentRepository


class DocumentProcessor:
    """
    문서 처리 도메인 서비스.

    문서로부터 노드와 관계를 추출하고,
    문서와 추출된 지식 요소들 간의 연결을 관리합니다.
    """

    def __init__(
        self,
        knowledge_extractor: KnowledgeExtractor,
        document_mapper: DocumentMapper,
        node_mapper: NodeMapper,
        relationship_mapper: RelationshipMapper,
        document_validation_service: Optional[DocumentValidationService] = None,
        document_persistence_service: Optional[DocumentPersistenceService] = None,
        document_statistics_service: Optional[DocumentStatisticsService] = None,
        document_repository: Optional[DocumentRepository] = None,
        logger: Optional[ObservableLogger] = None,
    ):
        self.knowledge_extractor = knowledge_extractor
        self.document_mapper = document_mapper
        self.node_mapper = node_mapper
        self.relationship_mapper = relationship_mapper
        self.document_validation_service = (
            document_validation_service or DocumentValidationService()
        )
        # 영속성 서비스 초기화 (repository가 있는 경우만)
        self.document_persistence_service: Optional[DocumentPersistenceService]
        if document_repository and not document_persistence_service:
            self.document_persistence_service = DocumentPersistenceService(
                document_repository, document_mapper, logger
            )
        else:
            self.document_persistence_service = document_persistence_service
        self.document_statistics_service = (
            document_statistics_service or DocumentStatisticsService()
        )
        self.document_repository = document_repository
        from src.common.observability.logger import get_logger

        self.logger = logger or get_logger("document_processor", "domain")

    async def process(self, document: Document) -> KnowledgeExtractionResult:
        """
        문서를 처리하여 노드와 관계를 추출합니다.

        Args:
            document: 처리할 문서

        Returns:
            추출된 노드와 관계

        Note:
            영속성 서비스가 제공된 경우 트랜잭션 기반 처리를 수행합니다.
        """
        if self.document_persistence_service:
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
        self.logger.info(
            "document_processing_started", document_id=str(document.id), mode="with_persistence"
        )

        try:
            # 1. 상태를 PROCESSING으로 변경하고 저장
            document.mark_as_processing()
            if self.document_persistence_service:
                await self.document_persistence_service.save_or_update_document(document)

            # 2. 지식 추출
            extraction_result = await self._extract_knowledge(document)

            # 3. 트랜잭션으로 문서 상태와 연결 정보 업데이트
            document.mark_as_processed()
            self._link_document_to_knowledge(document, extraction_result)

            node_id_strings = [str(node.id) for node in extraction_result.nodes]
            relationship_id_strings = [str(rel.id) for rel in extraction_result.relationships]

            if self.document_persistence_service:
                await self.document_persistence_service.update_document_with_knowledge(
                    document, node_id_strings, relationship_id_strings
                )

            self.logger.info(
                "document_processing_completed",
                document_id=str(document.id),
                node_count=extraction_result.get_node_count(),
                relationship_count=extraction_result.get_relationship_count(),
                mode="with_persistence",
            )

            return extraction_result

        except Exception as exception:
            self.logger.error(
                "document_processing_failed",
                document_id=str(document.id),
                error=str(exception),
                mode="with_persistence",
            )
            # 실패 시 상태 업데이트
            document.mark_as_failed(str(exception))
            try:
                if self.document_persistence_service:
                    await self.document_persistence_service.update_document_status(document)
            except Exception as update_error:
                self.logger.error(
                    "document_status_update_failed",
                    document_id=str(document.id),
                    error=str(update_error),
                )
            raise

    async def _process_document_in_memory(self, document: Document) -> KnowledgeExtractionResult:
        """
        메모리에서만 문서 처리 (기존 동작).

        Args:
            document: 처리할 문서

        Returns:
            추출된 노드와 관계
        """
        self.logger.info(
            "document_processing_started", document_id=str(document.id), mode="in_memory"
        )

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
                "document_processing_completed",
                document_id=str(document.id),
                node_count=extraction_result.get_node_count(),
                relationship_count=extraction_result.get_relationship_count(),
                mode="in_memory",
            )

            return extraction_result

        except Exception as exception:
            self.logger.error(
                "document_processing_failed",
                document_id=str(document.id),
                error=str(exception),
                mode="in_memory",
            )
            document.mark_as_failed(str(exception))
            raise

    async def _extract_knowledge(self, document: Document) -> KnowledgeExtractionResult:
        """
        문서로부터 지식을 추출합니다.

        지식 추출 포트를 사용하여 문서에서 개체와 관계를 추출합니다.
        """
        # Document 엔티티를 DTO로 변환
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
        added_nodes: Optional[list[NodeId]] = None,
        removed_nodes: Optional[list[NodeId]] = None,
        added_relationships: Optional[list[RelationshipId]] = None,
        removed_relationships: Optional[list[RelationshipId]] = None,
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
        validation_result = self.document_validation_service.validate_for_processing(document)
        self.document_validation_service.log_validation_result(document, validation_result)
        return validation_result.is_valid

    async def reprocess_document(self, document: Document) -> KnowledgeExtractionResult:
        """문서를 재처리합니다."""
        self.logger.info("document_reprocessing_started", document_id=str(document.id))

        # 기존 연결 정보 초기화
        document.connected_nodes.clear()
        document.connected_relationships.clear()

        # 상태를 대기 중으로 재설정
        document.status = DocumentStatus.PENDING
        document.processed_at = None

        # 상태 업데이트 저장
        if self.document_persistence_service:
            await self.document_persistence_service.update_document_status(document)

        # 재처리 실행
        return await self.process(document)

    def _document_to_data(self, document: Document) -> Any:
        """Document 엔티티를 DTO로 변환합니다."""
        return self.document_mapper.to_data(document)

    def _node_data_to_entity(self, node_data: Any) -> Node:
        """DTO를 Node 엔티티로 변환합니다."""
        return self.node_mapper.from_data(node_data)

    def _relationship_data_to_entity(self, rel_data: Any) -> Relationship:
        """DTO를 Relationship 엔티티로 변환합니다."""
        return self.relationship_mapper.from_data(rel_data)

    def get_processing_statistics(self, documents: list[Document]) -> dict[str, Any]:
        """문서 처리 통계 정보를 반환합니다."""
        return self.document_statistics_service.get_processing_statistics(documents)
