"""
DocumentRepository 포트의 SQLite 구현.
"""

import json
import logging
from datetime import datetime
from logging import Logger
from typing import Any

from src.domain.entities.document import Document, DocumentStatus, DocumentType
from src.domain.exceptions.document_exceptions import (
    ConcurrentModificationError,
    DocumentAlreadyExistsException,
    DocumentNotFoundException,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.dto import (
    DocumentData,
)
from src.dto import DocumentStatus as DTODocumentStatus
from src.dto import DocumentType as DTODocumentType
from src.ports.database import Database
from src.ports.repositories.document import DocumentRepository


class SQLiteDocumentRepository(DocumentRepository):
    """
    DocumentRepository 포트의 SQLite 구현.
    이 어댑터는 SQLite를 기본 스토리지 엔진으로 사용하여
    문서 지속성의 구체적인 구현을 제공합니다.
    """

    def __init__(self, database: Database, logger: Logger | None = None):
        """
        SQLite 문서 리포지토리 어댑터를 초기화합니다.
        Args:
            database: 데이터베이스 연결 인터페이스
            logger: 선택적 로거 인스턴스
        """
        self.database = database
        self.logger = logger or logging.getLogger(__name__)

    def _document_to_data(self, document: Document) -> DocumentData:
        """Document 엔티티를 DocumentData DTO로 변환합니다."""
        # 도메인 상태를 DTO 상태로 매핑
        status_mapping = {
            DocumentStatus.PENDING: DTODocumentStatus.PENDING,
            DocumentStatus.PROCESSING: DTODocumentStatus.PROCESSING,
            DocumentStatus.PROCESSED: DTODocumentStatus.COMPLETED,
            DocumentStatus.FAILED: DTODocumentStatus.FAILED,
        }

        return DocumentData(
            id=str(document.id),
            title=document.title,
            content=document.content,
            doc_type=DTODocumentType(document.doc_type.value),
            status=status_mapping[document.status],
            metadata=document.metadata,
            version=document.version,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processed_at=document.processed_at,
            connected_nodes=[str(node_id) for node_id in document.connected_nodes],
            connected_relationships=[str(rel_id) for rel_id in document.connected_relationships],
        )

    def _data_to_document(self, data: DocumentData) -> Document:
        """DocumentData DTO를 Document 엔티티로 변환합니다."""
        # DTO 상태를 도메인 상태로 매핑
        status_mapping = {
            DTODocumentStatus.PENDING: DocumentStatus.PENDING,
            DTODocumentStatus.PROCESSING: DocumentStatus.PROCESSING,
            DTODocumentStatus.COMPLETED: DocumentStatus.PROCESSED,
            DTODocumentStatus.FAILED: DocumentStatus.FAILED,
        }

        return Document(
            id=DocumentId(value=data.id),
            title=data.title,
            content=data.content,
            doc_type=DocumentType(data.doc_type.value),
            status=status_mapping[data.status],
            metadata=data.metadata,
            version=data.version,
            created_at=data.created_at,
            updated_at=data.updated_at,
            processed_at=data.processed_at,
            connected_nodes=[NodeId(value=node_id) for node_id in data.connected_nodes],
            connected_relationships=[
                RelationshipId(value=rel_id) for rel_id in data.connected_relationships
            ],
        )

    async def save(self, document: DocumentData) -> DocumentData:
        """
        문서를 저장합니다.
        Args:
            document: 저장할 문서
        Returns:
            저장된 문서
        Raises:
            DocumentAlreadyExistsException: 문서가 이미 존재하는 경우
        """
        # DTO를 엔티티로 변환
        doc_entity = self._data_to_document(document)
        # 문서가 이미 존재하는지 확인
        existing = await self._get_document_by_id(doc_entity.id)
        if existing:
            raise DocumentAlreadyExistsException(str(doc_entity.id))
        await self._insert_document(doc_entity)
        self.logger.info("문서 저장됨: %s", doc_entity.id)
        return document

    async def find_by_id(self, document_id: str) -> DocumentData | None:
        """
        ID로 문서를 찾습니다.
        Args:
            document_id: 문서 ID
        Returns:
            찾은 문서 또는 None
        """
        doc_entity = await self._get_document_by_id(DocumentId(value=document_id))
        return self._document_to_data(doc_entity) if doc_entity else None

    async def find_by_title(self, title: str) -> list[Document]:
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

    async def find_by_status(self, status: DTODocumentStatus) -> list[DocumentData]:
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
        doc_entities = [self._row_to_document(row) for row in rows]
        return [self._document_to_data(doc) for doc in doc_entities]

    async def find_by_type(self, doc_type: DocumentType) -> list[Document]:
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

    async def find_all(self, limit: int = 100, offset: int = 0) -> list[Document]:
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

    async def find_by_date_range(self, start_date: datetime, end_date: datetime) -> list[Document]:
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

    async def search_content(self, query_text: str, limit: int = 10) -> list[Document]:
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

    async def update(self, document: DocumentData) -> DocumentData:
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
        # DTO를 엔티티로 변환
        doc_entity = self._data_to_document(document)
        # 현재 문서 버전 확인
        current = await self._get_document_by_id(doc_entity.id)
        if not current:
            raise DocumentNotFoundException(str(doc_entity.id))
        if current.version != doc_entity.version:
            # TODO: 모니터링 강화 - 동시성 충돌 빈도 모니터링
            # 동시성 충돌 발생 시 메트릭을 수집하여 충돌 패턴 분석
            # 예: concurrency_metrics.record_conflict(document.id, expected_version, actual_version)
            raise ConcurrentModificationError(
                str(doc_entity.id), doc_entity.version, current.version
            )
        # 버전 증가
        doc_entity.increment_version()
        await self._update_document(doc_entity)
        self.logger.info("문서 업데이트됨: %s, 버전: %s", doc_entity.id, doc_entity.version)
        # 업데이트된 DTO 반환
        return self._document_to_data(doc_entity)

    async def update_with_knowledge(
        self,
        document: DocumentData,
        node_ids: list[str],
        relationship_ids: list[str],
    ) -> DocumentData:
        """
        문서를 지식 요소들과 함께 업데이트합니다.
        Args:
            document: 업데이트할 문서
            node_ids: 연결된 노드 ID들
            relationship_ids: 연결된 관계 ID들
        Returns:
            업데이트된 문서
        """
        # DTO를 엔티티로 변환
        doc_entity = self._data_to_document(document)
        async with self.database.transaction():
            # 동시성 체크
            current = await self._get_document_by_id(doc_entity.id)
            if current and current.version != doc_entity.version:
                raise ConcurrentModificationError(
                    str(doc_entity.id), doc_entity.version, current.version
                )
            # 연결된 요소들 업데이트
            doc_entity.connected_nodes = [NodeId(value=node_id) for node_id in node_ids]
            doc_entity.connected_relationships = [
                RelationshipId(value=rel_id) for rel_id in relationship_ids
            ]
            # 버전 증가 및 업데이트
            doc_entity.increment_version()
            await self._update_document(doc_entity)
            return self._document_to_data(doc_entity)

    async def delete(self, document_id: str) -> bool:
        """
        문서를 삭제합니다.
        Args:
            document_id: 삭제할 문서 ID
        Returns:
            삭제 성공 여부
        """
        command = "DELETE FROM documents WHERE id = ?"
        parameters = {"id": document_id}
        affected_rows = await self.database.execute_command(command, parameters)
        success = affected_rows > 0
        if success:
            self.logger.info("문서 삭제됨: %s", document_id)
        return success

    async def exists(self, document_id: str) -> bool:
        """
        문서가 존재하는지 확인합니다.
        Args:
            document_id: 확인할 문서 ID
        Returns:
            존재 여부
        """
        query = "SELECT 1 FROM documents WHERE id = ? LIMIT 1"
        parameters = {"id": document_id}
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

    async def find_with_connected_elements(self) -> list[Document]:
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

    async def find_unprocessed(self, limit: int = 100) -> list[DocumentData]:
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
        doc_entities = [self._row_to_document(row) for row in rows]
        return [self._document_to_data(doc) for doc in doc_entities]

    async def bulk_update_status(
        self, document_ids: list[DocumentId], status: DocumentStatus
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
        # 위치 매개변수 대신 명명된 매개변수 사용
        id_conditions = " OR ".join(f"id = :id_{i}" for i in range(len(document_ids)))
        command = f"""
            UPDATE documents
            SET status = :status, updated_at = datetime('now'), version = version + 1
            WHERE {id_conditions}
        """
        parameters = {"status": status.value}
        for i, doc_id in enumerate(document_ids):
            parameters[f"id_{i}"] = str(doc_id)
        affected_rows = await self.database.execute_command(command, parameters)
        self.logger.info("%s개 문서 상태를 %s로 일괄 업데이트함", affected_rows, status.value)
        return affected_rows

    # 비공개 헬퍼 메서드
    async def _get_document_by_id(self, document_id: DocumentId) -> Document | None:
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
            "connected_relationships": json.dumps(
                [str(rid) for rid in document.connected_relationships]
            ),
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
            "connected_relationships": json.dumps(
                [str(rid) for rid in document.connected_relationships]
            ),
            "id": str(document.id),
        }
        await self.database.execute_command(command, parameters)

    def _row_to_document(self, row: dict[str, Any]) -> Document:
        """데이터베이스 행을 Document 엔티티로 변환하는 내부 메서드."""
        # JSON 필드 파싱
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        connected_nodes = [NodeId(value=nid) for nid in json.loads(row["connected_nodes"] or "[]")]
        connected_relationships = [
            RelationshipId(value=rid) for rid in json.loads(row["connected_relationships"] or "[]")
        ]
        # 날짜 파싱
        created_at = datetime.fromisoformat(row["created_at"])
        updated_at = datetime.fromisoformat(row["updated_at"])
        processed_at = datetime.fromisoformat(row["processed_at"]) if row["processed_at"] else None
        return Document(
            id=DocumentId(value=row["id"]),
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
