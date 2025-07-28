"""
SQLite implementation of the DocumentRepository port.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.exceptions.document_exceptions import (
    ConcurrentModificationError,
    DocumentAlreadyExistsException,
    DocumentNotFoundException,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.ports.database import Database
from src.ports.repositories.document import DocumentRepository


class SQLiteDocumentRepository(DocumentRepository):
    """
    SQLite implementation of the DocumentRepository port.

    This adapter provides concrete implementation of document persistence
    using SQLite as the underlying storage engine.
    """

    def __init__(self, database: Database, logger: Optional[logging.Logger] = None):
        """
        Initialize SQLite document repository adapter.

        Args:
            database: Database connection interface
            logger: Optional logger instance
        """
        self.database = database
        self.logger = logger or logging.getLogger(__name__)

    async def save(self, document: Document) -> Document:
        """
        문서를 저장합니다.

        Args:
            document: 저장할 문서

        Returns:
            저장된 문서

        Raises:
            DocumentAlreadyExistsException: 문서가 이미 존재하는 경우
        """
        # 문서가 이미 존재하는지 확인
        existing = await self._get_document_by_id(document.id)
        if existing:
            raise DocumentAlreadyExistsException(str(document.id))

        await self._insert_document(document)
        self.logger.info(f"Document saved: {document.id}")
        return document

    async def find_by_id(self, document_id: DocumentId) -> Optional[Document]:
        """
        ID로 문서를 찾습니다.

        Args:
            document_id: 문서 ID

        Returns:
            찾은 문서 또는 None
        """
        return await self._get_document_by_id(document_id)

    async def find_by_title(self, title: str) -> List[Document]:
        """
        제목으로 문서를 찾습니다.

        Args:
            title: 문서 제목

        Returns:
            매칭되는 문서들
        """
        query = """
            SELECT * FROM documents 
            WHERE title LIKE ? 
            ORDER BY created_at DESC
        """
        parameters = {"title": f"%{title}%"}

        rows = await self.database.execute_query(query, parameters)
        return [self._row_to_document(row) for row in rows]

    async def find_by_status(self, status: DocumentStatus) -> List[Document]:
        """
        상태로 문서를 찾습니다.

        Args:
            status: 문서 상태

        Returns:
            해당 상태의 문서들
        """
        query = """
            SELECT * FROM documents 
            WHERE status = ? 
            ORDER BY created_at DESC
        """
        parameters = {"status": status.value}

        rows = await self.database.execute_query(query, parameters)
        return [self._row_to_document(row) for row in rows]

    async def find_by_type(self, doc_type: DocumentType) -> List[Document]:
        """
        타입으로 문서를 찾습니다.

        Args:
            doc_type: 문서 타입

        Returns:
            해당 타입의 문서들
        """
        query = """
            SELECT * FROM documents 
            WHERE doc_type = ? 
            ORDER BY created_at DESC
        """
        parameters = {"doc_type": doc_type.value}

        rows = await self.database.execute_query(query, parameters)
        return [self._row_to_document(row) for row in rows]

    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Document]:
        """
        모든 문서를 조회합니다.

        Args:
            limit: 최대 반환 개수
            offset: 건너뛸 개수

        Returns:
            문서들
        """
        query = """
            SELECT * FROM documents 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        """
        parameters = {"limit": limit, "offset": offset}

        rows = await self.database.execute_query(query, parameters)
        return [self._row_to_document(row) for row in rows]

    async def find_by_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> List[Document]:
        """
        날짜 범위로 문서를 찾습니다.

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            해당 기간의 문서들
        """
        query = """
            SELECT * FROM documents 
            WHERE created_at BETWEEN ? AND ? 
            ORDER BY created_at DESC
        """
        parameters = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

        rows = await self.database.execute_query(query, parameters)
        return [self._row_to_document(row) for row in rows]

    async def search_content(self, query_text: str, limit: int = 10) -> List[Document]:
        """
        문서 내용을 검색합니다.

        Args:
            query_text: 검색 쿼리
            limit: 최대 반환 개수

        Returns:
            검색 결과 문서들
        """
        query = """
            SELECT * FROM documents 
            WHERE content LIKE ? OR title LIKE ?
            ORDER BY 
                CASE 
                    WHEN title LIKE ? THEN 1 
                    ELSE 2 
                END,
                created_at DESC
            LIMIT ?
        """
        search_pattern = f"%{query_text}%"
        parameters = {
            "content": search_pattern,
            "title": search_pattern,
            "title_priority": search_pattern,
            "limit": limit,
        }

        rows = await self.database.execute_query(query, parameters)
        return [self._row_to_document(row) for row in rows]

    async def update(self, document: Document) -> Document:
        """
        문서를 업데이트합니다.

        Args:
            document: 업데이트할 문서

        Returns:
            업데이트된 문서

        Raises:
            DocumentNotFoundException: 문서가 존재하지 않는 경우
            ConcurrentModificationError: 동시 수정 충돌이 발생한 경우
        """
        # 현재 문서 버전 확인
        current = await self._get_document_by_id(document.id)
        if not current:
            raise DocumentNotFoundException(str(document.id))

        if current.version != document.version:
            # TODO: 모니터링 강화 - 동시성 충돌 빈도 모니터링
            # 동시성 충돌 발생 시 메트릭을 수집하여 충돌 패턴 분석
            # 예: concurrency_metrics.record_conflict(document.id, expected_version, actual_version)
            raise ConcurrentModificationError(
                str(document.id), document.version, current.version
            )

        # 버전 증가
        document.increment_version()
        await self._update_document(document)

        self.logger.info(f"Document updated: {document.id}, version: {document.version}")
        return document

    async def update_with_knowledge(
        self,
        document: Document,
        node_ids: List[NodeId],
        relationship_ids: List[RelationshipId],
    ) -> Document:
        """
        문서를 지식 요소들과 함께 업데이트합니다.

        Args:
            document: 업데이트할 문서
            node_ids: 연결된 노드 ID들
            relationship_ids: 연결된 관계 ID들

        Returns:
            업데이트된 문서
        """
        async with self.database.transaction():
            # 동시성 체크
            current = await self._get_document_by_id(document.id)
            if current and current.version != document.version:
                raise ConcurrentModificationError(
                    str(document.id), document.version, current.version
                )

            # 연결된 요소들 업데이트
            document.connected_nodes = node_ids
            document.connected_relationships = relationship_ids

            # 버전 증가 및 업데이트
            document.increment_version()
            await self._update_document(document)

            return document

    async def delete(self, document_id: DocumentId) -> bool:
        """
        문서를 삭제합니다.

        Args:
            document_id: 삭제할 문서 ID

        Returns:
            삭제 성공 여부
        """
        command = "DELETE FROM documents WHERE id = ?"
        parameters = {"id": str(document_id)}

        affected_rows = await self.database.execute_command(command, parameters)
        success = affected_rows > 0

        if success:
            self.logger.info(f"Document deleted: {document_id}")

        return success

    async def exists(self, document_id: DocumentId) -> bool:
        """
        문서가 존재하는지 확인합니다.

        Args:
            document_id: 확인할 문서 ID

        Returns:
            존재 여부
        """
        query = "SELECT 1 FROM documents WHERE id = ? LIMIT 1"
        parameters = {"id": str(document_id)}

        rows = await self.database.execute_query(query, parameters)
        return len(rows) > 0

    async def count_by_status(self, status: DocumentStatus) -> int:
        """
        상태별 문서 개수를 반환합니다.

        Args:
            status: 문서 상태

        Returns:
            해당 상태의 문서 개수
        """
        query = "SELECT COUNT(*) as count FROM documents WHERE status = ?"
        parameters = {"status": status.value}

        rows = await self.database.execute_query(query, parameters)
        return rows[0]["count"] if rows else 0

    async def count_total(self) -> int:
        """
        전체 문서 개수를 반환합니다.

        Returns:
            전체 문서 개수
        """
        query = "SELECT COUNT(*) as count FROM documents"
        rows = await self.database.execute_query(query)
        return rows[0]["count"] if rows else 0

    async def find_with_connected_elements(self) -> List[Document]:
        """
        연결된 노드나 관계가 있는 문서들을 찾습니다.

        Returns:
            연결된 요소가 있는 문서들
        """
        query = """
            SELECT * FROM documents 
            WHERE 
                (connected_nodes != '[]' AND connected_nodes IS NOT NULL) OR
                (connected_relationships != '[]' AND connected_relationships IS NOT NULL)
            ORDER BY created_at DESC
        """

        rows = await self.database.execute_query(query)
        return [self._row_to_document(row) for row in rows]

    async def find_unprocessed(self, limit: int = 100) -> List[Document]:
        """
        처리되지 않은 문서들을 찾습니다.

        Args:
            limit: 최대 반환 개수

        Returns:
            미처리 문서들
        """
        query = """
            SELECT * FROM documents 
            WHERE status = ? 
            ORDER BY created_at ASC 
            LIMIT ?
        """
        parameters = {"status": DocumentStatus.PENDING.value, "limit": limit}

        rows = await self.database.execute_query(query, parameters)
        return [self._row_to_document(row) for row in rows]

    async def bulk_update_status(
        self, document_ids: List[DocumentId], status: DocumentStatus
    ) -> int:
        """
        여러 문서의 상태를 일괄 업데이트합니다.

        Args:
            document_ids: 업데이트할 문서 ID 목록
            status: 새로운 상태

        Returns:
            업데이트된 문서 개수
        """
        if not document_ids:
            return 0

        # TODO: 성능 최적화 - 대용량 배치 처리를 위한 청크 단위 처리 구현
        # 현재는 모든 ID를 한 번에 처리하지만, 대용량 데이터 처리 시 메모리 효율성을 위해
        # 청크 단위로 나누어 처리하는 방식으로 개선 필요
        # 예: batch_size = 1000으로 설정하여 청크별 처리
        
        placeholders = ", ".join("?" * len(document_ids))
        command = f"""
            UPDATE documents 
            SET status = ?, updated_at = datetime('now'), version = version + 1
            WHERE id IN ({placeholders})
        """

        parameters = [status.value] + [str(doc_id) for doc_id in document_ids]
        affected_rows = await self.database.execute_command(command, parameters)

        self.logger.info(
            f"Bulk updated {affected_rows} documents to status {status.value}"
        )
        return affected_rows

    # Private helper methods
    async def _get_document_by_id(self, document_id: DocumentId) -> Optional[Document]:
        """문서 ID로 문서를 조회하는 내부 메서드."""
        query = "SELECT * FROM documents WHERE id = ?"
        parameters = {"id": str(document_id)}

        rows = await self.database.execute_query(query, parameters)
        return self._row_to_document(rows[0]) if rows else None

    async def _insert_document(self, document: Document) -> None:
        """문서를 삽입하는 내부 메서드."""
        command = """
            INSERT INTO documents (
                id, title, content, doc_type, status, metadata, version,
                created_at, updated_at, processed_at, 
                connected_nodes, connected_relationships
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        parameters = {
            "id": str(document.id),
            "title": document.title,
            "content": document.content,
            "doc_type": document.doc_type.value,
            "status": document.status.value,
            "metadata": json.dumps(document.metadata) if document.metadata else None,
            "version": document.version,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
            "processed_at": document.processed_at.isoformat() if document.processed_at else None,
            "connected_nodes": json.dumps([str(nid) for nid in document.connected_nodes]),
            "connected_relationships": json.dumps([str(rid) for rid in document.connected_relationships]),
        }

        await self.database.execute_command(command, parameters)

    async def _update_document(self, document: Document) -> None:
        """문서를 업데이트하는 내부 메서드."""
        command = """
            UPDATE documents SET 
                title = ?, content = ?, doc_type = ?, status = ?, 
                metadata = ?, version = ?, updated_at = ?, processed_at = ?,
                connected_nodes = ?, connected_relationships = ?
            WHERE id = ?
        """

        parameters = {
            "title": document.title,
            "content": document.content,
            "doc_type": document.doc_type.value,
            "status": document.status.value,
            "metadata": json.dumps(document.metadata) if document.metadata else None,
            "version": document.version,
            "updated_at": document.updated_at.isoformat(),
            "processed_at": document.processed_at.isoformat() if document.processed_at else None,
            "connected_nodes": json.dumps([str(nid) for nid in document.connected_nodes]),
            "connected_relationships": json.dumps([str(rid) for rid in document.connected_relationships]),
            "id": str(document.id),
        }

        await self.database.execute_command(command, parameters)

    def _row_to_document(self, row: Dict[str, Any]) -> Document:
        """데이터베이스 행을 Document 엔티티로 변환하는 내부 메서드."""
        # JSON 필드 파싱
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        connected_nodes = [
            NodeId(nid) for nid in json.loads(row["connected_nodes"] or "[]")
        ]
        connected_relationships = [
            RelationshipId(rid) for rid in json.loads(row["connected_relationships"] or "[]")
        ]

        # 날짜 파싱
        created_at = datetime.fromisoformat(row["created_at"])
        updated_at = datetime.fromisoformat(row["updated_at"])
        processed_at = (
            datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None
        )

        return Document(
            id=DocumentId(row["id"]),
            title=row["title"],
            content=row["content"],
            doc_type=DocumentType(row["doc_type"]),
            status=DocumentStatus(row["status"]),
            metadata=metadata,
            version=row["version"],
            created_at=created_at,
            updated_at=updated_at,
            processed_at=processed_at,
            connected_nodes=connected_nodes,
            connected_relationships=connected_relationships,
        )