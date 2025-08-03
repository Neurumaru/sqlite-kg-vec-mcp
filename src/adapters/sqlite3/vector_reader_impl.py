"""
SQLite 벡터 저장소의 읽기 작업 구현체.
"""

from typing import Any, Optional

from src.domain.value_objects.document_metadata import DocumentMetadata
from src.domain.value_objects.search_result import VectorSearchResult, VectorSearchResultCollection
from src.domain.value_objects.vector import Vector
from src.ports.vector_reader import VectorReader

from .vector_store_base import SQLiteVectorStoreBase


class SQLiteVectorReader(SQLiteVectorStoreBase, VectorReader):
    """
    SQLite를 사용한 VectorReader 포트의 구현체.

    벡터와 문서의 조회, 검색 작업을 담당합니다.
    """

    async def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """
        ID로 문서를 조회합니다.

        Args:
            document_id: 조회할 문서 ID

        Returns:
            DocumentMetadata 객체 또는 None
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT metadata FROM {self.table_name} WHERE id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            cursor.close()

            if not row or not row[0]:
                return None

            metadata = self._deserialize_metadata(row[0])

            return DocumentMetadata(
                source=metadata.get("source", ""),
                content=metadata.get("content", ""),
                metadata={
                    k: v
                    for k, v in metadata.items()
                    if k not in ["source", "content", "created_at", "updated_at"]
                },
            )
        except Exception:
            return None

    async def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        ID로 벡터를 조회합니다.

        Args:
            vector_id: 조회할 벡터 ID

        Returns:
            Vector 객체 또는 None
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT vector FROM {self.table_name} WHERE id = ?",
                (vector_id,),
            )
            row = cursor.fetchone()
            cursor.close()

            if not row or not row[0]:
                return None

            return self._deserialize_vector(row[0])
        except Exception:
            return None

    async def list_documents(
        self, limit: Optional[int] = None, offset: Optional[int] = None, **kwargs: Any
    ) -> list[DocumentMetadata]:
        """
        저장된 문서 목록을 조회합니다.

        Args:
            limit: 반환할 문서 수 제한
            offset: 시작 오프셋
            **kwargs: 추가 필터링 옵션

        Returns:
            DocumentMetadata 객체 목록
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            # 기본 쿼리
            query = f"SELECT id, metadata FROM {self.table_name}"
            params = []

            # 필터링 조건 추가
            if kwargs:
                conditions = []
                for key, value in kwargs.items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(value)
                if conditions:
                    query += f" WHERE {' AND '.join(conditions)}"

            # 정렬
            query += " ORDER BY created_at DESC"

            # 페이징
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
                if offset is not None:
                    query += " OFFSET ?"
                    params.append(offset)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()

            documents = []
            for _document_id, metadata_json in rows:
                if metadata_json:
                    metadata = self._deserialize_metadata(metadata_json)
                    doc = DocumentMetadata(
                        source=metadata.get("source", ""),
                        content=metadata.get("content", ""),
                        metadata={
                            k: v
                            for k, v in metadata.items()
                            if k not in ["source", "content", "created_at", "updated_at"]
                        },
                    )
                    documents.append(doc)

            return documents
        except Exception:
            return []

    async def count_documents(self, **kwargs: Any) -> int:
        """
        저장된 문서 수를 반환합니다.

        Args:
            **kwargs: 필터링 옵션

        Returns:
            문서 수
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            # 기본 쿼리
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            params = []

            # 필터링 조건 추가
            if kwargs:
                conditions = []
                for key, value in kwargs.items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(value)
                if conditions:
                    query += f" WHERE {' AND '.join(conditions)}"

            cursor.execute(query, params)
            result = cursor.fetchone()
            cursor.close()

            return int(result[0]) if result else 0
        except Exception:
            return 0

    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        텍스트 쿼리로 유사도 검색을 수행합니다.

        Args:
            query: 검색 쿼리 문자열
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션 (필터 등)

        Returns:
            VectorSearchResultCollection 객체
        """
        # 텍스트를 벡터로 변환해야 하는데, 여기서는 기본 구현을 제공
        # 실제로는 임베딩 서비스를 주입받아 사용해야 함

        # 임시로 키워드 기반 검색 수행
        return await self._keyword_search(query, k, **kwargs)

    async def similarity_search_by_vector(
        self,
        vector: Vector,
        k: int = 4,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        벡터로 유사도 검색을 수행합니다.

        Args:
            vector: 쿼리 Vector 객체
            k: 반환할 문서 수
            **kwargs: 추가 검색 옵션

        Returns:
            VectorSearchResultCollection 객체
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            # 필터 조건 처리
            filter_criteria = kwargs.get("filter", {})
            where_clause = ""
            params = []

            if filter_criteria:
                conditions = []
                for key, value in filter_criteria.items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(value)
                if conditions:
                    where_clause = f"WHERE {' AND '.join(conditions)}"

            cursor.execute(
                f"SELECT id, vector, metadata FROM {self.table_name} {where_clause}",
                params,
            )
            rows = cursor.fetchall()
            cursor.close()

            # 유사도 계산
            similarities = []
            for document_id, vector_blob, metadata_json in rows:
                stored_vector = self._deserialize_vector(vector_blob)
                similarity = self._cosine_similarity(vector.values, stored_vector.values)

                metadata = self._deserialize_metadata(metadata_json) if metadata_json else {}

                document = DocumentMetadata(
                    source=metadata.get("source", ""),
                    content=metadata.get("content", ""),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ["source", "content", "created_at", "updated_at"]
                    },
                )
                result = VectorSearchResult(
                    document=document,
                    score=similarity,
                    id=document_id,
                )
                similarities.append((similarity, result))

            # 유사도에 따라 정렬하고 상위 k개 선택
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_results = [result for _, result in similarities[:k]]

            return VectorSearchResultCollection(
                results=top_results,
                total_count=len(top_results),
                query="vector_search",
            )
        except Exception:
            return VectorSearchResultCollection(
                results=[],
                total_count=0,
                query="error",
            )

    async def _keyword_search(
        self, query: str, k: int, **kwargs: Any
    ) -> VectorSearchResultCollection:
        """
        키워드 기반 검색 (벡터 임베딩이 없을 때의 대안).

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            **kwargs: 추가 옵션

        Returns:
            VectorSearchResultCollection 객체
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            # 단순한 LIKE 검색 (실제로는 FTS를 사용하는 것이 좋음)
            filter_criteria = kwargs.get("filter", {})
            where_conditions = []
            params = []

            # 텍스트 검색 조건
            where_conditions.append(
                "(json_extract(metadata, '$.content') LIKE ? OR json_extract(metadata, '$.source') LIKE ?)"
            )
            query_pattern = f"%{query}%"
            params.extend([query_pattern, query_pattern])

            # 추가 필터 조건
            for key, value in filter_criteria.items():
                where_conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)

            where_clause = f"WHERE {' AND '.join(where_conditions)}"

            cursor.execute(
                f"""
                SELECT id, metadata FROM {self.table_name} {where_clause}
                ORDER BY created_at DESC LIMIT ?
            """,
                params + [k],
            )
            rows = cursor.fetchall()
            cursor.close()

            results = []
            for document_id, metadata_json in rows:
                metadata = self._deserialize_metadata(metadata_json) if metadata_json else {}

                # 간단한 관련성 점수 계산 (실제로는 더 정교한 알고리즘 필요)
                content = metadata.get("content", "").lower()
                score = (
                    query.lower().count(" ".join(content.split()[:10])) / 10.0 if content else 0.1
                )

                document = DocumentMetadata(
                    source=metadata.get("source", ""),
                    content=metadata.get("content", ""),
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ["source", "content", "created_at", "updated_at"]
                    },
                )
                result = VectorSearchResult(
                    document=document,
                    score=score,
                    id=document_id,
                )
                results.append(result)

            return VectorSearchResultCollection(
                results=results,
                total_count=len(results),
                query=query,
            )
        except Exception:
            return VectorSearchResultCollection(
                results=[],
                total_count=0,
                query="error",
            )

    # 유틸리티 메서드
    async def vector_exists(self, vector_id: str) -> bool:
        """
        저장소에 벡터가 존재하는지 확인합니다.

        Args:
            vector_id: 벡터 식별자

        Returns:
            벡터가 존재하면 True
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT 1 FROM {self.table_name} WHERE id = ? LIMIT 1",
                (vector_id,),
            )
            exists = cursor.fetchone() is not None
            cursor.close()

            return exists
        except Exception:
            return False

    async def get_metadata(self, vector_id: str) -> Optional[dict[str, Any]]:
        """
        벡터에 대한 메타데이터를 가져옵니다.

        Args:
            vector_id: 벡터 식별자

        Returns:
            찾은 경우 메타데이터 사전, 그렇지 않으면 None
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT metadata FROM {self.table_name} WHERE id = ?",
                (vector_id,),
            )
            row = cursor.fetchone()
            cursor.close()

            if row and row[0]:
                return self._deserialize_metadata(row[0])
            return None
        except Exception:
            return None

    async def search_by_metadata(
        self, filter_criteria: dict[str, Any], limit: int = 100
    ) -> list[str]:
        """
        메타데이터 기준으로 벡터를 검색합니다.

        Args:
            filter_criteria: 메타데이터 필터 기준
            limit: 최대 결과 수

        Returns:
            기준과 일치하는 벡터 ID 목록
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            # WHERE 절 빌드
            conditions = []
            params = []
            for key, value in filter_criteria.items():
                conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(
                f"SELECT id FROM {self.table_name} {where_clause} LIMIT ?",
                params + [limit],
            )
            rows = cursor.fetchall()
            cursor.close()

            return [row[0] for row in rows]
        except Exception:
            return []
