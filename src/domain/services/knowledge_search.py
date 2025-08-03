"""
지식 검색 도메인 서비스.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.config.search_config import DEFAULT_SIMILARITY_THRESHOLD, SearchConfig
from src.domain.entities.document import Document
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship
from src.ports.text_embedder import TextEmbedder

# 로거 설정
logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """검색 전략을 정의합니다."""

    KEYWORD = "keyword"  # 키워드 기반 검색 (단순 텍스트 매칭)
    RELATIONSHIP = "relationship"  # 관계 기반 검색
    HYBRID = "hybrid"  # 복합 검색
    SEMANTIC = "semantic"  # 의미적 검색 (임베딩 기반)


@dataclass
class SearchCriteria:
    """검색 조건.

    속성:
        query (str): 검색 쿼리 문자열.
        strategy (SearchStrategy): 사용할 검색 전략. 기본값은 HYBRID.
        limit (int): 반환할 최대 결과 수. 기본값은 10.
        similarity_threshold (float): 의미적 검색에 사용될 유사도 임계값. 기본값은 0.5.
        include_documents (bool): 결과에 문서를 포함할지 여부. 기본값은 True.
        include_nodes (bool): 결과에 노드를 포함할지 여부. 기본값은 True.
        include_relationships (bool): 결과에 관계를 포함할지 여부. 기본값은 True.
        node_types (Optional[list[str]]): 검색을 제한할 노드 타입 목록. 기본값은 None (모든 타입).
        relationship_types (Optional[list[str]]): 검색을 제한할 관계 타입 목록. 기본값은 None (모든 타입).
    """

    query: str
    strategy: SearchStrategy = SearchStrategy.HYBRID
    limit: int = 10
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    include_documents: bool = True
    include_nodes: bool = True
    include_relationships: bool = True
    node_types: Optional[list[str]] = None
    relationship_types: Optional[list[str]] = None


@dataclass
class SearchResult:
    """검색 결과 항목.

    속성:
        score (float): 검색 결과의 관련성 점수.
        document (Optional[Document]): 결과 문서 객체. 문서 검색 시 제공됩니다.
        node (Optional[Node]): 결과 노드 객체. 노드 검색 시 제공됩니다.
        relationship (Optional[Relationship]): 결과 관계 객체. 관계 검색 시 제공됩니다.
        explanation (Optional[str]): 검색 결과에 대한 추가 설명.
    """

    score: float
    document: Optional[Document] = None
    node: Optional[Node] = None
    relationship: Optional[Relationship] = None
    explanation: Optional[str] = None


@dataclass
class SearchResultCollection:
    """
    다양한 검색 결과들을 포함하는 컬렉션.

    속성:
        results (list[SearchResult]): 검색 결과 항목들의 목록.
        total_count (int): 전체 결과 수.
        query (str): 실행된 검색 쿼리.
        strategy (SearchStrategy): 사용된 검색 전략.
        execution_time_ms (Optional[float]): 검색 실행 시간(밀리초).
    """

    results: list[SearchResult]
    total_count: int
    query: str
    strategy: SearchStrategy
    execution_time_ms: Optional[float] = None


class KnowledgeSearchService:
    """
    지식 검색 도메인 서비스.

    문서, 노드, 관계를 통합적으로 검색하고
    관련 정보를 연결하여 제공합니다.
    """

    def __init__(self, text_embedder: TextEmbedder, search_config: Optional[SearchConfig] = None):
        self.text_embedder = text_embedder
        self.search_config = search_config or SearchConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def search(
        self,
        criteria: SearchCriteria,
        documents: list[Document],
        nodes: list[Node],
        relationships: list[Relationship],
    ) -> SearchResultCollection:
        """
        주어진 검색 조건에 따라 지식 그래프를 검색합니다.

        인자:
            criteria (SearchCriteria): 검색 조건 객체.
            documents (list[Document]): 검색에 사용할 문서 목록.
            nodes (list[Node]): 검색에 사용할 노드 목록.
            relationships (list[Relationship]): 검색에 사용할 관계 목록.

        반환:
            SearchResultCollection: 검색 결과 컬렉션.
        """
        start_time = time.time()
        results: list[SearchResult] = []

        if criteria.strategy == SearchStrategy.KEYWORD:
            # 간단한 키워드 검색 (문서, 노드, 관계 전체에 대해)
            results.extend(self._search_documents(criteria, documents))
            results.extend(self._search_nodes(criteria, nodes))
            results.extend(self._search_relationships(criteria, relationships))
        elif criteria.strategy == SearchStrategy.SEMANTIC:
            results.extend(self._semantic_search(criteria, documents, nodes, relationships))
        elif criteria.strategy == SearchStrategy.HYBRID:
            # 키워드 및 의미론적 검색을 모두 수행하고 결과를 병합
            keyword_results = []
            if criteria.include_documents:
                keyword_results.extend(self._search_documents(criteria, documents))
            if criteria.include_nodes:
                keyword_results.extend(self._search_nodes(criteria, nodes))
            if criteria.include_relationships:
                keyword_results.extend(self._search_relationships(criteria, relationships))

            semantic_results = []
            if criteria.include_documents:
                # 문서에 대한 의미적 검색 (현재는 문서 임베딩이 없으므로 키워드로 대체)
                semantic_results.extend(self._search_documents(criteria, documents))
            if criteria.include_nodes:
                # 노드 의미적 검색
                semantic_results.extend(self._semantic_search_nodes(criteria, nodes))
            if criteria.include_relationships:
                # 관계 의미적 검색
                semantic_results.extend(
                    self._semantic_search_relationships(criteria, relationships)
                )

            # 결과 병합 및 중복 제거
            combined_results = {
                (
                    r.document.id.value if r.document else None,
                    r.node.id.value if r.node else None,
                    r.relationship.id.value if r.relationship else None,
                ): r
                for r in keyword_results + semantic_results
            }
            results = list(combined_results.values())
        else:
            raise ValueError(f"알 수 없는 검색 전략: {criteria.strategy}")

        # 점수 기준 내림차순 정렬
        results.sort(key=lambda x: x.score, reverse=True)

        # limit 적용
        limited_results = results[: criteria.limit]

        execution_time = (time.time() - start_time) * 1000  # ms

        result_collection = SearchResultCollection(
            results=limited_results,
            total_count=len(limited_results),
            query=criteria.query,
            strategy=criteria.strategy,
            execution_time_ms=execution_time,
        )

        self.logger.info(
            "Search completed: %d results in %.2fms", len(limited_results), execution_time
        )

        return result_collection

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 문자열 간의 유사도를 계산합니다.
        현재는 간단한 교집합/합집합 기반의 유사도를 사용합니다.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union

    def _search_documents(
        self, criteria: SearchCriteria, documents: list[Document]
    ) -> list[SearchResult]:
        """문서 내용 기반 검색."""
        results = []
        query_lower = criteria.query.lower()

        for doc in documents:
            # 단순 텍스트 매칭 검색 (실제로는 더 정교한 텍스트 검색 필요)
            title_score = self._calculate_text_similarity(query_lower, doc.title.lower())
            content_score = self._calculate_text_similarity(query_lower, doc.content.lower())

            # 제목과 내용에 설정된 가중치 적용
            combined_score = (
                title_score * self.search_config.document_title_weight
                + content_score * self.search_config.document_content_weight
            )

            if combined_score >= criteria.similarity_threshold:
                result = SearchResult(
                    score=combined_score,
                    document=doc,
                )
                results.append(result)

        return results

    def _search_nodes(self, criteria: SearchCriteria, nodes: list[Node]) -> list[SearchResult]:
        """노드 기반 검색."""
        results = []
        query_lower = criteria.query.lower()

        for node in nodes:
            # 노드 타입 필터링
            if criteria.node_types and node.node_type.value not in criteria.node_types:
                continue

            # 노드 이름 및 설명 매칭
            name_score = self._calculate_text_similarity(query_lower, node.name.lower())
            desc_score = 0.0
            if node.description:
                desc_score = self._calculate_text_similarity(query_lower, node.description.lower())

            # 임베딩 기반 유사도 (있는 경우)
            embedding_score = 0.0
            if node.has_embedding():
                # 실제로는 쿼리도 임베딩으로 변환하여 비교
                # 여기서는 간단히 처리
                embedding_score = self.search_config.similarity_threshold  # 예시 점수

            # 가중치 기반 점수와 순수 임베딩 점수 중 최대값 사용
            weighted_score = (
                name_score * self.search_config.node_name_weight
                + desc_score * self.search_config.node_description_weight
                + embedding_score * self.search_config.node_embedding_weight
            )
            combined_score = max(weighted_score, embedding_score)

            if combined_score >= criteria.similarity_threshold:
                result = SearchResult(
                    score=combined_score,
                    node=node,
                    explanation=f"Node match: name({name_score:.3f}), desc({desc_score:.3f}), embedding({embedding_score:.3f})",
                )
                results.append(result)

        return results

    def _search_relationships(
        self, criteria: SearchCriteria, relationships: list[Relationship]
    ) -> list[SearchResult]:
        """관계 기반 검색."""
        results = []
        query_lower = criteria.query.lower()

        for rel in relationships:
            # 관계 타입 필터링
            if (
                criteria.relationship_types
                and rel.relationship_type.value not in criteria.relationship_types
            ):
                continue

            # 관계 레이블 매칭
            label_score = self._calculate_text_similarity(query_lower, rel.label.lower())

            # 임베딩 기반 유사도 (있는 경우)
            embedding_score = 0.0
            if rel.has_embedding():
                # 실제로는 쿼리도 임베딩으로 변환하여 비교
                embedding_score = self.search_config.similarity_threshold  # 예시 점수

            combined_score = max(label_score, embedding_score)

            if combined_score >= criteria.similarity_threshold:
                result = SearchResult(
                    score=combined_score,
                    relationship=rel,
                    explanation=f"Relationship match: label({label_score:.3f}), embedding({embedding_score:.3f})",
                )
                results.append(result)

        return results

    def _semantic_search(
        self,
        criteria: SearchCriteria,
        documents: list[Document],
        nodes: list[Node],
        relationships: list[Relationship],
    ) -> list[SearchResult]:
        """임베딩 기반 의미적 검색."""
        results = []

        # 실제 구현에서는 쿼리를 임베딩으로 변환
        # query_embedding = self.text_embedder.embed(criteria.query)

        # 노드와 관계에 대한 시맨틱 검색 수행
        if criteria.include_nodes:
            semantic_node_results = self._semantic_search_nodes(criteria, nodes)
            results.extend(semantic_node_results)

        if criteria.include_relationships:
            semantic_relationship_results = self._semantic_search_relationships(
                criteria, relationships
            )
            results.extend(semantic_relationship_results)

        return results

    def _semantic_search_nodes(
        self, criteria: SearchCriteria, nodes: list[Node]
    ) -> list[SearchResult]:
        """노드에 대한 의미적 검색을 수행합니다."""
        semantic_results = []
        for node in nodes:
            if node.has_embedding():
                # similarity = query_embedding.cosine_similarity(node.embedding)
                similarity = self.search_config.similarity_threshold  # 예시 점수

                if similarity >= criteria.similarity_threshold:
                    result = SearchResult(
                        score=similarity,
                        node=node,
                        explanation=f"Semantic similarity: {similarity:.3f}",
                    )
                    semantic_results.append(result)
        return semantic_results

    def _semantic_search_relationships(
        self, criteria: SearchCriteria, relationships: list[Relationship]
    ) -> list[SearchResult]:
        """관계에 대한 의미적 검색을 수행합니다."""
        semantic_results = []
        for rel in relationships:
            if rel.has_embedding():
                # similarity = query_embedding.cosine_similarity(rel.embedding)
                similarity = self.search_config.similarity_threshold  # 예시 점수

                if similarity >= criteria.similarity_threshold:
                    result = SearchResult(
                        score=similarity,
                        relationship=rel,
                        explanation=f"Semantic similarity: {similarity:.3f}",
                    )
                    semantic_results.append(result)
        return semantic_results

    def _hybrid_search(
        self,
        criteria: SearchCriteria,
        documents: list[Document],
        nodes: list[Node],
        relationships: list[Relationship],
    ) -> list[SearchResult]:
        """복합 검색을 수행합니다."""
        # 이 부분은 _semantic_search 와 _search_ documents/nodes/relationships를
        # 적절히 조합하여 구현해야 합니다.
        # 현재는 단순히 semantic_search를 호출하도록 단순화되어 있습니다.
        return self._semantic_search(criteria, documents, nodes, relationships)

    def find_related_documents(self, node: Node, all_documents: list[Document]) -> list[Document]:
        """노드와 관련된 문서들을 찾습니다."""
        related_docs = []

        for doc in all_documents:
            if node.is_from_document(doc.id):
                related_docs.append(doc)

        return related_docs

    def find_connected_nodes(self, document: Document, all_nodes: list[Node]) -> list[Node]:
        """문서와 연결된 노드들을 찾습니다."""
        connected_nodes = []

        for node in all_nodes:
            if document.id in node.source_documents:
                connected_nodes.append(node)

        return connected_nodes

    def find_node_relationships(
        self, node: Node, all_relationships: list[Relationship]
    ) -> list[Relationship]:
        """노드와 연결된 관계들을 찾습니다."""
        node_relationships = []

        for rel in all_relationships:
            if rel.involves_node(node.id):
                node_relationships.append(rel)

        return node_relationships

    def get_search_suggestions(
        self, partial_query: str, documents: list[Document], nodes: list[Node]
    ) -> list[str]:
        """검색 자동완성 제안."""
        suggestions = set()
        partial_lower = partial_query.lower()

        # 문서 제목에서 제안
        for doc in documents:
            if partial_lower in doc.title.lower():
                suggestions.add(doc.title)

        # 노드 이름에서 제안
        for node in nodes:
            if partial_lower in node.name.lower():
                suggestions.add(node.name)

        return sorted(suggestions)[:10]  # 최대 10개 제안
