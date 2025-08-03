"""
문서 통계 도메인 서비스.
"""

from typing import Any

from src.domain.entities.document import Document, DocumentStatus


class DocumentStatisticsService:
    """
    문서 통계 생성 도메인 서비스.

    문서 처리 관련 통계 정보를 생성합니다.
    """

    def get_processing_statistics(self, documents: list[Document]) -> dict[str, Any]:
        """문서 처리 통계 정보를 반환합니다."""
        total = len(documents)
        processed = sum(1 for doc in documents if doc.status == DocumentStatus.PROCESSED)
        processing = sum(1 for doc in documents if doc.status == DocumentStatus.PROCESSING)
        pending = sum(1 for doc in documents if doc.status == DocumentStatus.PENDING)
        failed = sum(1 for doc in documents if doc.status == DocumentStatus.FAILED)

        processing_rate = (processed / total * 100) if total > 0 else 0

        return {
            "total_documents": total,
            "processed": processed,
            "processing": processing,
            "pending": pending,
            "failed": failed,
            "processing_rate": round(processing_rate, 2),
            "status_distribution": {
                "processed": processed,
                "processing": processing,
                "pending": pending,
                "failed": failed,
            },
        }

    def get_document_metrics(self, document: Document) -> dict[str, Any]:
        """개별 문서의 메트릭 정보를 반환합니다."""
        return {
            "id": str(document.id),
            "status": document.status.value,
            "word_count": document.get_word_count(),
            "char_count": document.get_char_count(),
            "connected_nodes_count": len(document.connected_nodes),
            "connected_relationships_count": len(document.connected_relationships),
            "has_connected_elements": document.has_connected_elements(),
            "version": document.version,
            "is_processed": document.is_processed(),
        }

    def get_knowledge_extraction_metrics(self, documents: list[Document]) -> dict[str, Any]:
        """지식 추출 관련 메트릭을 반환합니다."""
        total_nodes = sum(len(doc.connected_nodes) for doc in documents)
        total_relationships = sum(len(doc.connected_relationships) for doc in documents)

        docs_with_knowledge = sum(1 for doc in documents if doc.has_connected_elements())

        avg_nodes_per_doc = (total_nodes / len(documents)) if documents else 0
        avg_relationships_per_doc = (total_relationships / len(documents)) if documents else 0

        knowledge_extraction_rate = (docs_with_knowledge / len(documents) * 100) if documents else 0

        return {
            "total_nodes_extracted": total_nodes,
            "total_relationships_extracted": total_relationships,
            "documents_with_knowledge": docs_with_knowledge,
            "average_nodes_per_document": round(avg_nodes_per_doc, 2),
            "average_relationships_per_document": round(avg_relationships_per_doc, 2),
            "knowledge_extraction_rate": round(knowledge_extraction_rate, 2),
        }
