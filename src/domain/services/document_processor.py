"""
문서 처리 도메인 서비스.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.domain.entities.document import Document, DocumentStatus
from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId


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

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def process_document(self, document: Document) -> KnowledgeExtractionResult:
        """
        문서를 처리하여 노드와 관계를 추출합니다.

        Args:
            document: 처리할 문서

        Returns:
            추출된 노드와 관계
        """
        self.logger.info(f"Processing document: {document.id}")

        try:
            # 문서 상태를 처리 중으로 변경
            document.mark_as_processing()

            # 지식 추출 (실제 구현에서는 LLM이나 NLP 파이프라인 사용)
            extraction_result = self._extract_knowledge(document)

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

    def _extract_knowledge(self, document: Document) -> KnowledgeExtractionResult:
        """
        문서로부터 지식을 추출합니다.

        실제 구현에서는 이 메서드가 LLM이나 NLP 파이프라인을 사용하여
        문서에서 개체와 관계를 추출합니다.
        """
        # 예시 구현 - 실제로는 더 정교한 추출 로직이 필요
        nodes = self._extract_entities(document)
        relationships = self._extract_relationships(document, nodes)

        return KnowledgeExtractionResult(nodes, relationships)

    def _extract_entities(self, document: Document) -> List[Node]:
        """
        문서에서 개체(노드)를 추출합니다.

        실제 구현에서는 Named Entity Recognition (NER)이나
        LLM을 사용하여 개체를 추출합니다.
        """
        self.logger.debug(f"Extracting entities from document {document.id}")

        # 예시 구현 - 단순한 키워드 기반 추출
        entities = []

        # 문서 내용을 분석하여 개체 추출
        # 실제로는 더 정교한 NLP 파이프라인이 필요
        content_lower = document.content.lower()

        # 간단한 패턴 매칭 예시
        if "openai" in content_lower:
            node = Node(
                id=NodeId.generate(),
                name="OpenAI",
                node_type=NodeType.ORGANIZATION,
                description="AI research company",
            )
            node.add_source_document(document.id, "Referenced in document")
            entities.append(node)

        if "gpt" in content_lower:
            node = Node(
                id=NodeId.generate(),
                name="GPT",
                node_type=NodeType.TECHNOLOGY,
                description="Generative Pre-trained Transformer",
            )
            node.add_source_document(document.id, "Mentioned as technology")
            entities.append(node)

        return entities

    def _extract_relationships(
        self, document: Document, nodes: List[Node]
    ) -> List[Relationship]:
        """
        문서에서 관계를 추출합니다.

        실제 구현에서는 Relation Extraction 모델이나
        LLM을 사용하여 개체 간 관계를 추출합니다.
        """
        self.logger.debug(f"Extracting relationships from document {document.id}")

        relationships = []

        # 간단한 관계 추출 예시
        # 실제로는 더 정교한 관계 추출 로직이 필요
        for i, source_node in enumerate(nodes):
            for target_node in nodes[i + 1 :]:
                # 예시: 조직과 기술 간의 관계
                if (
                    source_node.node_type == NodeType.ORGANIZATION
                    and target_node.node_type == NodeType.TECHNOLOGY
                ):

                    relationship = Relationship(
                        id=RelationshipId.generate(),
                        source_node_id=source_node.id,
                        target_node_id=target_node.id,
                        relationship_type=RelationshipType.CREATES,
                        label="creates",
                        confidence=0.8,
                    )
                    relationship.add_source_document(
                        document.id,
                        context="Inferred from document content",
                        sentence="Organization and technology mentioned together",
                    )
                    relationships.append(relationship)

        return relationships

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

    def reprocess_document(self, document: Document) -> KnowledgeExtractionResult:
        """문서를 재처리합니다."""
        self.logger.info(f"Reprocessing document: {document.id}")

        # 기존 연결 정보 초기화
        document.connected_nodes.clear()
        document.connected_relationships.clear()

        # 상태를 대기 중으로 재설정
        document.status = DocumentStatus.PENDING
        document.processed_at = None

        # 재처리 실행
        return self.process_document(document)

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
