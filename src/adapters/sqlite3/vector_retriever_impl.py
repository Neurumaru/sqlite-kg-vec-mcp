"""
SQLite 벡터 저장소의 검색 작업 구현체.
"""

from typing import Any, Optional

from src.config.search_config import SearchConfig
from src.domain.value_objects.document_metadata import DocumentMetadata
from src.domain.value_objects.search_result import VectorSearchResult, VectorSearchResultCollection
from src.domain.value_objects.vector import Vector
from src.ports.vector_retriever import VectorRetriever

from .vector_store_base import SQLiteVectorStoreBase


class SQLiteVectorRetriever(SQLiteVectorStoreBase, VectorRetriever):
    """
    SQLite를 사용한 VectorRetriever 포트의 구현체.

    고급 벡터 검색 및 리트리벌 작업을 담당합니다.
    """

    def __init__(
        self,
        db_path: str,
        table_name: str = "vectors",
        optimize: bool = True,
        search_config: Optional[SearchConfig] = None,
    ):
        super().__init__(db_path, table_name, optimize)
        self.search_config = search_config or SearchConfig()

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity",
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        다양한 검색 타입으로 문서를 검색합니다.

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            search_type: 검색 타입 ("similarity", "mmr", "similarity_score_threshold")
            **kwargs: 검색 타입별 추가 옵션

        Returns:
            VectorSearchResultCollection 객체
        """
        if search_type == "similarity":
            return await self._similarity_retrieve(query, k, **kwargs)
        if search_type == "mmr":
            return await self.retrieve_mmr(
                query,
                k=k,
                fetch_k=kwargs.get("fetch_k", 20),
                lambda_mult=kwargs.get("lambda_mult", self.search_config.mmr_lambda),
                **kwargs,
            )
        if search_type == "similarity_score_threshold":
            return await self._similarity_score_threshold_retrieve(
                query,
                k=k,
                score_threshold=kwargs.get("score_threshold", self.search_config.score_threshold),
                **kwargs,
            )
        # 기본값으로 similarity 검색 수행
        return await self._similarity_retrieve(query, k, **kwargs)

    async def retrieve_with_filter(
        self,
        query: str,
        filter_criteria: dict[str, Any],
        k: int = 4,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        필터 조건과 함께 문서를 검색합니다.

        Args:
            query: 검색 쿼리
            filter_criteria: 필터 조건
            k: 반환할 결과 수
            **kwargs: 추가 검색 옵션

        Returns:
            VectorSearchResultCollection 객체
        """
        kwargs["filter"] = filter_criteria
        return await self._similarity_retrieve(query, k, **kwargs)

    async def retrieve_mmr(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: Optional[float] = None,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        MMR(Maximal Marginal Relevance) 알고리즘으로 문서를 검색합니다.

        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            fetch_k: MMR 계산을 위해 먼저 가져올 문서 수
            lambda_mult: 관련성과 다양성 간의 균형 (0~1)
            **kwargs: 추가 검색 옵션

        Returns:
            VectorSearchResultCollection 객체
        """
        try:
            # lambda_mult 기본값 설정
            if lambda_mult is None:
                lambda_mult = self.search_config.mmr_lambda

            # 먼저 많은 후보를 가져옴
            initial_results = await self._similarity_retrieve(query, fetch_k, **kwargs)

            if not initial_results.results:
                return VectorSearchResultCollection(
                    results=[],
                    total_count=0,
                    query=query,
                )

            # 쿼리 벡터 생성 (실제로는 임베딩 서비스 필요)
            query_vector = await self._text_to_vector(query)
            if not query_vector:
                # 벡터 변환 실패시 기본 유사도 검색 결과 반환
                return VectorSearchResultCollection(
                    results=initial_results.results[:k],
                    total_count=len(initial_results.results[:k]),
                    query=query,
                )

            # MMR 알고리즘 적용
            selected_results = []
            candidate_results = initial_results.results.copy()

            # 첫 번째 결과는 가장 관련성이 높은 것
            if candidate_results:
                selected_results.append(candidate_results.pop(0))

            # 나머지 결과들에 대해 MMR 계산
            while len(selected_results) < k and candidate_results:
                best_result = None
                best_mmr_score = -1.0
                best_idx = -1

                for idx, candidate in enumerate(candidate_results):
                    # 후보 벡터 가져오기
                    candidate_vector = await self._get_document_vector(candidate.id or "")
                    if not candidate_vector:
                        continue

                    # 쿼리와의 관련성 계산
                    relevance_score = self._cosine_similarity(
                        query_vector.values, candidate_vector.values
                    )

                    # 이미 선택된 문서들과의 최대 유사도 계산
                    max_similarity = 0.0
                    for selected in selected_results:
                        selected_vector = await self._get_document_vector(selected.id or "")
                        if selected_vector:
                            similarity = self._cosine_similarity(
                                candidate_vector.values, selected_vector.values
                            )
                            max_similarity = max(max_similarity, similarity)

                    # MMR 점수 계산
                    mmr_score = lambda_mult * relevance_score - (1 - lambda_mult) * max_similarity

                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_result = candidate
                        best_idx = idx

                if best_result:
                    selected_results.append(best_result)
                    candidate_results.pop(best_idx)
                else:
                    break

            return VectorSearchResultCollection(
                results=selected_results,
                total_count=len(selected_results),
                query=query,
            )
        except Exception:
            # 오류 발생시 기본 유사도 검색으로 폴백
            return await self._similarity_retrieve(query, k, **kwargs)

    async def get_relevant_documents(
        self,
        query: str,
        **kwargs: Any,
    ) -> VectorSearchResultCollection:
        """
        쿼리와 관련된 문서를 검색합니다 (LangChain 호환).

        Args:
            query: 검색 쿼리
            **kwargs: 검색 옵션

        Returns:
            VectorSearchResultCollection 객체
        """
        k = kwargs.get("k", 4)
        return await self.retrieve(query, k=k, search_type="similarity", **kwargs)

    # 내부 구현 메서드들
    async def _similarity_retrieve(
        self, query: str, k: int, **kwargs: Any
    ) -> VectorSearchResultCollection:
        """기본 유사도 검색 구현."""
        try:
            # 텍스트를 벡터로 변환 (실제로는 임베딩 서비스 필요)
            query_vector = await self._text_to_vector(query)
            if not query_vector:
                # 벡터 변환 실패시 키워드 검색으로 폴백
                return await self._keyword_search(query, k, **kwargs)

            return await self._vector_similarity_search(query_vector, k, **kwargs)
        except Exception:
            return VectorSearchResultCollection(
                results=[],
                total_count=0,
                query="",
            )

    async def _similarity_score_threshold_retrieve(
        self, query: str, k: int, score_threshold: float, **kwargs: Any
    ) -> VectorSearchResultCollection:
        """점수 임계값을 적용한 유사도 검색."""
        results = await self._similarity_retrieve(
            query, k * 2, **kwargs
        )  # 더 많이 가져온 후 필터링

        # 임계값 이상의 결과만 필터링
        filtered_results = [
            result for result in results.results if result.score >= score_threshold
        ][:k]

        return VectorSearchResultCollection(
            results=filtered_results,
            total_count=len(filtered_results),
            query=query,
        )

    async def _vector_similarity_search(
        self, query_vector: Vector, k: int, **kwargs: Any
    ) -> VectorSearchResultCollection:
        """벡터 기반 유사도 검색."""
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
                similarity = self._cosine_similarity(query_vector.values, stored_vector.values)

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
                query="",
            )
        except Exception:
            return VectorSearchResultCollection(
                results=[],
                total_count=0,
                query="",
            )

    async def _keyword_search(
        self, query: str, k: int, **kwargs: Any
    ) -> VectorSearchResultCollection:
        """키워드 기반 검색 (벡터 임베딩이 실패했을 때의 대안)."""
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            # 단순한 LIKE 검색
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
                f"SELECT id, metadata FROM {self.table_name} {where_clause} ORDER BY created_at DESC LIMIT ?",
                params + [k],
            )
            rows = cursor.fetchall()
            cursor.close()

            results = []
            for document_id, metadata_json in rows:
                metadata = self._deserialize_metadata(metadata_json) if metadata_json else {}

                # 간단한 관련성 점수 계산
                content = metadata.get("content", "").lower()
                score = content.count(query.lower()) / len(content.split()) if content else 0.1
                score = min(score, 1.0)  # 최대 1.0으로 제한

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
                query="",
            )

    async def _text_to_vector(self, text: str) -> Optional[Vector]:
        """
        텍스트를 벡터로 변환합니다.

        실제 구현에서는 임베딩 서비스를 주입받아 사용해야 합니다.
        현재는 기본 구현으로 None을 반환합니다.
        """
        # TODO: 임베딩 서비스 인터페이스 추가 필요
        return None

    async def _get_document_vector(self, document_id: str) -> Optional[Vector]:
        """문서 ID로 해당 벡터를 가져옵니다."""
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT vector FROM {self.table_name} WHERE id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            cursor.close()

            if row and row[0]:
                return self._deserialize_vector(row[0])
            return None
        except Exception:
            return None

    # 성능 분석 및 진단 메서드
    async def get_store_stats(self) -> dict[str, Any]:
        """
        저장소 통계 정보를 반환합니다.

        Returns:
            통계 정보 딕셔너리
        """
        try:
            conn = self._ensure_connected()
            cursor = conn.cursor()

            stats = {}

            # 전체 문서 수
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            stats["total_documents"] = cursor.fetchone()[0]

            # 데이터베이스 크기 (페이지 수 * 페이지 크기)
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats["database_size_bytes"] = page_count * page_size

            # 인덱스 정보
            cursor.execute(f"PRAGMA index_list({self.table_name})")
            indexes = cursor.fetchall()
            stats["indexes"] = len(indexes)

            # 테이블 정보
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            table_info = cursor.fetchall()
            stats["columns"] = len(table_info)

            cursor.close()
            return stats
        except Exception:
            return {"error": "통계 정보를 가져오는 데 실패했습니다"}
