"""
벡터 유사도 검색 기능.
"""

import sqlite3

# from .relationships import Relationship, RelationshipManager  # TODO: relationships 모듈 구현
from dataclasses import dataclass
from typing import Any

import numpy as np

# from .entities import Entity, EntityManager  # TODO: entities 모듈 구현
# from .hnsw import HNSWIndex  # 순환 종속성 방지를 위해 동적으로 가져오기
from .embedder_factory import VectorTextEmbedder, create_embedder


@dataclass
class SearchResult:
    """벡터 검색 결과를 나타냅니다."""

    entity_type: str
    entity_id: int
    distance: float
    entity: Any] = (
        None  # TODO: Entity/Relationship 클래스가 사용 가능할 때 적절하게 타입 지정
    )

    def to_dict(self) -> dict[str, Any]:
        """사전 표현으로 변환합니다."""
        result = {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "distance": self.distance,
        }

        if self.entity:
            # TODO: Entity/Relationship 클래스가 사용 가능할 때 적절한 엔티티 직렬화 구현
            if hasattr(self.entity, "id"):
                result["entity"] = {
                    "id": getattr(self.entity, "id", None),
                    "name": getattr(self.entity, "name", None),
                    "type": getattr(self.entity, "type", None),
                    "properties": getattr(self.entity, "properties", None),
                }

        return result


class VectorSearch:
    """
    HNSW 인덱스를 사용한 벡터 유사도 검색 기능.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        index_dir: Optional[str] = None,
        embedding_dim: int = 128,
        space: str = "cosine",
        text_embedder: Optional[VectorTextEmbedder] = None,
        embedder_type: str = "sentence-transformers",
        embedder_kwargs: dict[str, Any]] = None,
    ):
        """
        벡터 검색 기능을 초기화합니다.

        Args:
            connection: SQLite 데이터베이스 연결
            index_dir: HNSW 인덱스 파일을 저장할 디렉토리
            embedding_dim: 임베딩 벡터의 차원
            space: 거리 측정 기준 ('cosine', 'ip', or 'l2')
            text_embedder: 텍스트-벡터 변환을 위한 VectorTextEmbedder 인스턴스
            embedder_type: text_embedder가 None인 경우 생성할 임베더 유형
            embedder_kwargs: 임베더 생성을 위한 인수
        """
        self.connection = connection
        # 순환 가져오기를 피하기 위해 EmbeddingManager를 동적으로 가져오기
        from .embeddings import EmbeddingManager  # pylint: disable=import-outside-toplevel

        self.embedding_manager = EmbeddingManager(connection)
        # TODO: 클래스가 사용 가능할 때 관리자 초기화
        # self.entity_manager = EntityManager(connection)
        # self.relationship_manager = RelationshipManager(connection)

        # 동적 가져오기로 인덱스 초기화
        from .hnsw import HNSWIndex  # pylint: disable=import-outside-toplevel

        self.index = HNSWIndex(
            space=space,
            dim=embedding_dim,
            ef_construction=200,
            m_parameter=16,
            index_dir=index_dir,
        )

        # 인덱스가 로드되었는지 추적하는 플래그
        self.index_loaded = False

        # 텍스트 임베더 초기화
        if text_embedder is not None:
            self.text_embedder = text_embedder
        else:
            embedder_kwargs = embedder_kwargs or {}
            # 랜덤 임베더의 경우 인덱스 차원 사용
            if embedder_type == "random":
                embedder_kwargs.setdefault("dimension", embedding_dim)

            self.text_embedder = create_embedder(embedder_type, **embedder_kwargs)

            # 임베더 차원이 인덱스 차원과 일치하는지 확인
            if self.text_embedder.dimension != embedding_dim:
                raise ValueError(
                    f"임베더 차원({self.text_embedder.dimension})이 "
                    f"인덱스 차원({embedding_dim})과 일치하지 않습니다. embedding_dim을 조정하거나 "
                    f"다른 모델을 사용하는 것을 고려하십시오."
                )

    def ensure_index_loaded(self, force_rebuild: bool = False):
        """
        인덱스가 로드되었는지 확인하고, 필요한 경우 처음부터 빌드합니다.

        Args:
            force_rebuild: 이미 로드된 경우에도 강제로 인덱스를 다시 빌드
        """
        if self.index_loaded and not force_rebuild:
            return

        try:
            # 기존 인덱스 로드 시도
            if not force_rebuild:
                self.index.load_index()
                self.index_loaded = True
                return
        except (FileNotFoundError, RuntimeError):
            # 로드에 실패하면 처음부터 빌드
            pass

        # 데이터베이스의 모든 임베딩에서 인덱스 빌드
        self.index.build_from_embeddings(self.embedding_manager)

        # 빌드된 인덱스 저장
        self.index.save_index()
        self.index_loaded = True

    def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        entity_types: list[str]] = None,
        ef_search: Optional[int] = None,
        include_entities: bool = True,
    ) -> list[SearchResult]:
        """
        쿼리 벡터와 유사한 엔티티를 검색합니다.

        Args:
            query_vector: 쿼리 임베딩 벡터
            k: 반환할 결과 수
            entity_types: 포함할 엔티티 유형 목록 또는 모두에 대해 None
            ef_search: 검색 품질을 제어하는 런타임 매개변수
            include_entities: 전체 엔티티 세부 정보를 포함할지 여부

        Returns:
            SearchResult 객체 목록
        """
        # 인덱스가 로드되었는지 확인
        self.ensure_index_loaded()

        # 검색 수행
        search_results = self.index.search(
            query_vector=query_vector,
            k=k,
            ef_search=ef_search,
            filter_entity_types=entity_types,
        )

        # SearchResult 객체로 변환
        results = []

        for entity_type, entity_id, distance in search_results:
            result = SearchResult(entity_type=entity_type, entity_id=entity_id, distance=distance)

            # TODO: 관리자가 사용 가능할 때 엔티티 세부 정보 포함
            # if include_entities:
            #     if entity_type == "node":
            #         result.entity = self.entity_manager.get_entity(entity_id)
            #     elif entity_type == "edge":
            #         result.entity = self.relationship_manager.get_relationship(
            #             entity_id, include_entities=True
            #         )
            #     # 참고: 하이퍼엣지 처리는 여기에 추가될 것입니다.

            results.append(result)

        return results

    def search_similar_to_entity(
        self,
        entity_type: str,
        entity_id: int,
        k: int = 10,
        result_entity_types: list[str]] = None,
        include_entities: bool = True,
    ) -> list[SearchResult]:
        """
        주어진 엔티티와 유사한 엔티티를 검색합니다.

        Args:
            entity_type: 엔티티 유형 ('node', 'edge', or 'hyperedge')
            entity_id: 엔티티 ID
            k: 반환할 결과 수
            result_entity_types: 결과에 포함할 엔티티 유형
            include_entities: 전체 엔티티 세부 정보를 포함할지 여부

        Returns:
            SearchResult 객체 목록
        """
        # 엔티티의 임베딩 가져오기
        embedding = self.embedding_manager.get_embedding(entity_type, entity_id)

        if not embedding:
            raise ValueError(f"{entity_type} {entity_id}에 대한 임베딩을 찾을 수 없습니다.")

        # 엔티티의 임베딩을 사용하여 유사도 검색 수행
        return self.search_similar(
            query_vector=embedding.embedding,
            k=k + 1,  # +1은 엔티티 자체가 결과에 포함되기 때문입니다.
            entity_types=result_entity_types,
            include_entities=include_entities,
        )[
            1:
        ]  # 첫 번째 결과(엔티티 자체) 제외

    def build_text_embedding(self, text: str) -> np.ndarray:
        """
        텍스트 쿼리에 대한 임베딩 벡터를 빌드합니다.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터
        """
        embedding = self.text_embedder.embed(text)
        return np.asarray(embedding, dtype=np.float32)

    def search_by_text(
        self,
        query_text: str,
        k: int = 10,
        entity_types: list[str]] = None,
        include_entities: bool = True,
    ) -> list[SearchResult]:
        """
        텍스트 쿼리와 유사한 엔티티를 검색합니다.

        Args:
            query_text: 텍스트 쿼리
            k: 반환할 결과 수
            entity_types: 포함할 엔티티 유형 목록 또는 모두에 대해 None
            include_entities: 전체 엔티티 세부 정보를 포함할지 여부

        Returns:
            SearchResult 객체 목록
        """
        # 텍스트 쿼리에 대한 임베딩 빌드
        query_embedding = self.build_text_embedding(query_text)

        # 유사도 검색 수행
        return self.search_similar(
            query_vector=query_embedding,
            k=k,
            entity_types=entity_types,
            include_entities=include_entities,
        )

    def update_index(self, batch_size: int = 100):
        """
        아웃박스의 보류 중인 변경 사항으로 인덱스를 업데이트합니다.

        Args:
            batch_size: 한 번에 처리할 작업 수

        Returns:
            처리된 작업 수
        """
        # 인덱스가 로드되었는지 확인
        self.ensure_index_loaded()

        # 보류 중인 작업 처리
        count = self.index.sync_with_outbox(self.embedding_manager, batch_size)

        # 변경 사항이 있는 경우 업데이트된 인덱스 저장
        if count > 0:
            self.index.save_index()

        return count
