"""
지식 검색 도메인 서비스.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from src.domain.entities.document import Document
from src.domain.entities.node import Node
from src.domain.entities.relationship import Relationship
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.vector import Vector


class SearchStrategy(Enum):
    """검색 전략."""
    DOCUMENT_ONLY = "document_only"  # 문서 내용 기반 검색
    NODE_ONLY = "node_only"          # 노드 기반 검색
    RELATIONSHIP_ONLY = "relationship_only"  # 관계 기반 검색
    HYBRID = "hybrid"                # 복합 검색
    SEMANTIC = "semantic"            # 의미적 검색 (임베딩 기반)


@dataclass
class SearchCriteria:
    """검색 조건."""
    query: str
    strategy: SearchStrategy = SearchStrategy.HYBRID
    limit: int = 10
    similarity_threshold: float = 0.5
    include_documents: bool = True
    include_nodes: bool = True
    include_relationships: bool = True
    node_types: Optional[List[str]] = None
    relationship_types: Optional[List[str]] = None


@dataclass
class SearchResult:
    """검색 결과 항목."""
    score: float
    document: Optional[Document] = None
    node: Optional[Node] = None
    relationship: Optional[Relationship] = None
    explanation: Optional[str] = None


@dataclass
class SearchResultCollection:
    """검색 결과 컬렉션."""
    results: List[SearchResult]
    total_count: int
    query: str
    strategy: SearchStrategy
    execution_time_ms: float = 0.0
    
    def get_documents(self) -> List[Document]:
        """검색 결과에서 문서들만 추출."""
        return [r.document for r in self.results if r.document is not None]
    
    def get_nodes(self) -> List[Node]:
        """검색 결과에서 노드들만 추출."""
        return [r.node for r in self.results if r.node is not None]
    
    def get_relationships(self) -> List[Relationship]:
        """검색 결과에서 관계들만 추출."""
        return [r.relationship for r in self.results if r.relationship is not None]


class KnowledgeSearchService:
    """
    지식 검색 도메인 서비스.
    
    문서, 노드, 관계를 통합적으로 검색하고
    관련 정보를 연결하여 제공합니다.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def search(self, criteria: SearchCriteria,
               documents: List[Document],
               nodes: List[Node], 
               relationships: List[Relationship]) -> SearchResultCollection:
        """
        통합 검색을 수행합니다.
        
        Args:
            criteria: 검색 조건
            documents: 검색 대상 문서들
            nodes: 검색 대상 노드들
            relationships: 검색 대상 관계들
            
        Returns:
            검색 결과 컬렉션
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Performing search with strategy: {criteria.strategy.value}, query: '{criteria.query}'")
        
        search_results = []
        
        if criteria.strategy == SearchStrategy.DOCUMENT_ONLY:
            search_results = self._search_documents(criteria, documents)
        elif criteria.strategy == SearchStrategy.NODE_ONLY:
            search_results = self._search_nodes(criteria, nodes)
        elif criteria.strategy == SearchStrategy.RELATIONSHIP_ONLY:
            search_results = self._search_relationships(criteria, relationships)
        elif criteria.strategy == SearchStrategy.SEMANTIC:
            search_results = self._semantic_search(criteria, documents, nodes, relationships)
        else:  # HYBRID
            search_results = self._hybrid_search(criteria, documents, nodes, relationships)
        
        # 점수순으로 정렬하고 제한
        search_results.sort(key=lambda x: x.score, reverse=True)
        limited_results = search_results[:criteria.limit]
        
        execution_time = (time.time() - start_time) * 1000  # ms
        
        result_collection = SearchResultCollection(
            results=limited_results,
            total_count=len(limited_results),
            query=criteria.query,
            strategy=criteria.strategy,
            execution_time_ms=execution_time
        )
        
        self.logger.info(f"Search completed: {len(limited_results)} results in {execution_time:.2f}ms")
        
        return result_collection
    
    def _search_documents(self, criteria: SearchCriteria, 
                         documents: List[Document]) -> List[SearchResult]:
        """문서 내용 기반 검색."""
        results = []
        query_lower = criteria.query.lower()
        
        for doc in documents:
            # 단순 텍스트 매칭 검색 (실제로는 더 정교한 텍스트 검색 필요)
            title_score = self._calculate_text_similarity(query_lower, doc.title.lower())
            content_score = self._calculate_text_similarity(query_lower, doc.content.lower())
            
            # 제목에 더 높은 가중치
            combined_score = title_score * 0.7 + content_score * 0.3
            
            if combined_score >= criteria.similarity_threshold:
                result = SearchResult(
                    score=combined_score,
                    document=doc,
                    explanation=f"Document match: title({title_score:.3f}), content({content_score:.3f})"
                )
                results.append(result)
        
        return results
    
    def _search_nodes(self, criteria: SearchCriteria, 
                     nodes: List[Node]) -> List[SearchResult]:
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
                embedding_score = 0.5  # 예시 점수
            
            combined_score = max(name_score * 0.6 + desc_score * 0.3 + embedding_score * 0.1, 
                               embedding_score)
            
            if combined_score >= criteria.similarity_threshold:
                result = SearchResult(
                    score=combined_score,
                    node=node,
                    explanation=f"Node match: name({name_score:.3f}), desc({desc_score:.3f}), embedding({embedding_score:.3f})"
                )
                results.append(result)
        
        return results
    
    def _search_relationships(self, criteria: SearchCriteria,
                            relationships: List[Relationship]) -> List[SearchResult]:
        """관계 기반 검색."""
        results = []
        query_lower = criteria.query.lower()
        
        for rel in relationships:
            # 관계 타입 필터링
            if criteria.relationship_types and rel.relationship_type.value not in criteria.relationship_types:
                continue
            
            # 관계 레이블 매칭
            label_score = self._calculate_text_similarity(query_lower, rel.label.lower())
            
            # 임베딩 기반 유사도 (있는 경우)
            embedding_score = 0.0
            if rel.has_embedding():
                # 실제로는 쿼리도 임베딩으로 변환하여 비교
                embedding_score = 0.5  # 예시 점수
            
            combined_score = max(label_score, embedding_score)
            
            if combined_score >= criteria.similarity_threshold:
                result = SearchResult(
                    score=combined_score,
                    relationship=rel,
                    explanation=f"Relationship match: label({label_score:.3f}), embedding({embedding_score:.3f})"
                )
                results.append(result)
        
        return results
    
    def _semantic_search(self, criteria: SearchCriteria,
                        documents: List[Document],
                        nodes: List[Node],
                        relationships: List[Relationship]) -> List[SearchResult]:
        """임베딩 기반 의미적 검색."""
        results = []
        
        # 실제 구현에서는 쿼리를 임베딩으로 변환
        # query_embedding = self.text_embedder.embed(criteria.query)
        
        # 노드들과의 의미적 유사도 계산
        if criteria.include_nodes:
            for node in nodes:
                if node.has_embedding():
                    # similarity = query_embedding.cosine_similarity(node.embedding)
                    similarity = 0.7  # 예시 점수
                    
                    if similarity >= criteria.similarity_threshold:
                        result = SearchResult(
                            score=similarity,
                            node=node,
                            explanation=f"Semantic similarity: {similarity:.3f}"
                        )
                        results.append(result)
        
        # 관계들과의 의미적 유사도 계산
        if criteria.include_relationships:
            for rel in relationships:
                if rel.has_embedding():
                    # similarity = query_embedding.cosine_similarity(rel.embedding)
                    similarity = 0.6  # 예시 점수
                    
                    if similarity >= criteria.similarity_threshold:
                        result = SearchResult(
                            score=similarity,
                            relationship=rel,
                            explanation=f"Semantic similarity: {similarity:.3f}"
                        )
                        results.append(result)
        
        return results
    
    def _hybrid_search(self, criteria: SearchCriteria,
                      documents: List[Document],
                      nodes: List[Node],
                      relationships: List[Relationship]) -> List[SearchResult]:
        """복합 검색 (모든 전략 조합)."""
        all_results = []
        
        # 각 검색 전략별로 실행
        if criteria.include_documents:
            doc_results = self._search_documents(criteria, documents)
            all_results.extend(doc_results)
        
        if criteria.include_nodes:
            node_results = self._search_nodes(criteria, nodes)
            all_results.extend(node_results)
        
        if criteria.include_relationships:
            rel_results = self._search_relationships(criteria, relationships)
            all_results.extend(rel_results)
        
        # 의미적 검색 결과도 포함
        semantic_results = self._semantic_search(criteria, documents, nodes, relationships)
        all_results.extend(semantic_results)
        
        return all_results
    
    def find_related_documents(self, node: Node, 
                             all_documents: List[Document]) -> List[Document]:
        """노드와 관련된 문서들을 찾습니다."""
        related_docs = []
        
        for doc in all_documents:
            if node.is_from_document(doc.id):
                related_docs.append(doc)
        
        return related_docs
    
    def find_connected_nodes(self, document: Document,
                           all_nodes: List[Node]) -> List[Node]:
        """문서와 연결된 노드들을 찾습니다."""
        connected_nodes = []
        
        for node in all_nodes:
            if document.id in node.source_documents:
                connected_nodes.append(node)
        
        return connected_nodes
    
    def find_node_relationships(self, node: Node,
                              all_relationships: List[Relationship]) -> List[Relationship]:
        """노드와 연결된 관계들을 찾습니다."""
        node_relationships = []
        
        for rel in all_relationships:
            if rel.involves_node(node.id):
                node_relationships.append(rel)
        
        return node_relationships
    
    def _calculate_text_similarity(self, query: str, text: str) -> float:
        """간단한 텍스트 유사도 계산."""
        if query in text:
            return 1.0
        
        # 단어 단위 겹치는 정도 계산
        query_words = set(query.split())
        text_words = set(text.split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words)
    
    def get_search_suggestions(self, partial_query: str,
                             documents: List[Document],
                             nodes: List[Node]) -> List[str]:
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
        
        return sorted(list(suggestions))[:10]  # 최대 10개 제안