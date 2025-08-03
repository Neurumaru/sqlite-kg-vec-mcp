"""
SQLite 지식 그래프 벡터 MCP 애플리케이션의 메인 진입점.
"""

import argparse
import asyncio
import os

import numpy as np

from src.adapters.fastmcp.config import FastMCPConfig
from src.adapters.fastmcp.server import KnowledgeGraphServer
from src.adapters.hnsw.embeddings import EmbeddingManager
from src.adapters.sqlite3.database import SQLiteDatabase
from src.adapters.sqlite3.document_repository import SQLiteDocumentRepository
from src.adapters.sqlite3.graph.entities import EntityManager
from src.adapters.sqlite3.graph.relationships import RelationshipManager
from src.adapters.sqlite3.schema import SchemaManager
from src.adapters.sqlite3.vector_store import SQLiteVectorStore
from src.adapters.testing.text_embedder import RandomTextEmbedder
from src.config.search_config import SearchConfig
from src.domain.entities.document import Document, DocumentType
from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.services.knowledge_search import (
    KnowledgeSearchService,
    SearchResult,
    SearchResultCollection,
    SearchStrategy,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.domain.value_objects.vector import Vector
from src.ports.text_embedder import TextEmbedder
from src.use_cases.document import DocumentManagementUseCase
from src.use_cases.knowledge_search import KnowledgeSearchUseCase
from src.use_cases.node import NodeManagementUseCase
from src.use_cases.relationship import RelationshipManagementUseCase


# 유스케이스의 구체적인 구현 (main.py용으로 단순화됨)
# 더 큰 애플리케이션에서는 별도의 application/use_cases 디렉토리에 있을 것입니다.
class ConcreteNodeManagementUseCase(NodeManagementUseCase):
    """노드 관리 유스케이스의 구체적 구현체."""

    def __init__(
        self,
        entity_manager: EntityManager,
        embedding_manager: EmbeddingManager,
        vector_store: SQLiteVectorStore,
        text_embedder: TextEmbedder,
    ):
        self.entity_manager = entity_manager
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.text_embedder = text_embedder

    async def create_node(
        self,
        name: str,
        node_type: NodeType,
        description: str | None = None,
        properties: dict | None = None,
        source_documents: list[DocumentId] | None = None,
    ) -> Node:
        # 단순화됨: 실제 구현은 노드를 엔티티로 변환하고 임베딩을 처리합니다.
        entity = self.entity_manager.create_entity(
            entity_type=node_type.value, name=name, properties=properties
        )
        # 임베딩 생성 및 저장
        embedding_text = f"{name} {description or ''} {properties or ''}"
        embedding_vector = await self.text_embedder.embed_text(embedding_text)
        self.embedding_manager.store_embedding(
            "node", entity.id, np.array(embedding_vector.embedding), embedding_vector.model_name
        )
        return Node(
            id=NodeId(str(entity.id)),
            name=entity.name or "",
            node_type=NodeType(entity.type),
            properties=entity.properties or {},
        )

    async def get_node(self, node_id: NodeId) -> Node | None:
        entity = self.entity_manager.get_entity(int(node_id.value))
        return (
            Node(
                id=NodeId(str(entity.id)),
                name=entity.name or "",
                node_type=NodeType(entity.type),
                properties=entity.properties or {},
            )
            if entity
            else None
        )

    async def list_nodes(
        self,
        node_type: NodeType | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Node]:
        entities, _ = self.entity_manager.find_entities(
            entity_type=node_type.value if node_type else None,
            limit=limit or 100,
            offset=offset or 0,
        )
        return [
            Node(
                id=NodeId(str(e.id)),
                name=e.name or "",
                node_type=NodeType(e.type),
                properties=e.properties or {},
            )
            for e in entities
        ]

    async def update_node(
        self,
        node_id: NodeId,
        name: str | None = None,
        description: str | None = None,
        properties: dict | None = None,
    ) -> Node:
        entity = self.entity_manager.update_entity(
            int(node_id.value), name=name, properties=properties
        )
        if not entity:
            raise ValueError("노드를 찾을 수 없습니다")
        # 텍스트 내용이 변경되면 임베딩 업데이트
        if name or description or properties:
            embedding_text = (
                f"{entity.name or ''} {description or ''} {properties or ''}"  # 현재 설명/속성 사용
            )
            embedding_vector = await self.text_embedder.embed_text(embedding_text)
            self.embedding_manager.store_embedding(
                "node", entity.id, np.array(embedding_vector.embedding), embedding_vector.model_name
            )

        return Node(
            id=NodeId(str(entity.id)),
            name=entity.name or "",
            node_type=NodeType(entity.type),
            properties=entity.properties or {},
        )

    async def delete_node(self, node_id: NodeId) -> bool:
        self.embedding_manager.delete_embedding("node", int(node_id.value))
        return self.entity_manager.delete_entity(int(node_id.value))

    async def add_node_to_document(self, node_id: NodeId, document_id: DocumentId) -> bool:
        # 이것은 새로운 리포지토리 메서드 또는 직접적인 DB 작업을 포함합니다.
        return True  # 단순화됨

    async def remove_node_from_document(self, node_id: NodeId, document_id: DocumentId) -> bool:
        # 이것은 새로운 리포지토리 메서드 또는 직접적인 DB 작업을 포함합니다.
        return True  # 단순화됨


class ConcreteRelationshipManagementUseCase(RelationshipManagementUseCase):
    """관계 관리 유스케이스의 구체적 구현체."""

    def __init__(
        self,
        relationship_manager: RelationshipManager,
        embedding_manager: EmbeddingManager,
        vector_store: SQLiteVectorStore,
        text_embedder: TextEmbedder,
    ):
        self.relationship_manager = relationship_manager
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.text_embedder = text_embedder

    async def create_relationship(
        self,
        source_node_id: NodeId,
        target_node_id: NodeId,
        relationship_type: RelationshipType,
        label: str,
        properties: dict | None = None,
        weight: float = 1.0,
    ) -> Relationship:
        rel = self.relationship_manager.create_relationship(
            source_id=int(source_node_id.value),
            target_id=int(target_node_id.value),
            relation_type=relationship_type.value,
            properties=properties,
        )
        # 임베딩 생성 및 저장
        embedding_text = f"{label} {properties or ''}"
        embedding_vector = await self.text_embedder.embed_text(embedding_text)
        self.embedding_manager.store_embedding(
            "edge", rel.id, np.array(embedding_vector.embedding), embedding_vector.model_name
        )
        return Relationship(
            id=RelationshipId(str(rel.id)),
            source_node_id=NodeId(str(rel.source_id)),
            target_node_id=NodeId(str(rel.target_id)),
            relationship_type=RelationshipType(rel.relation_type),
            label=rel.relation_type,
            properties=rel.properties or {},
        )

    async def get_relationship(self, relationship_id: RelationshipId) -> Relationship | None:
        rel = self.relationship_manager.get_relationship(int(relationship_id.value))
        return (
            Relationship(
                id=RelationshipId(str(rel.id)),
                source_node_id=NodeId(str(rel.source_id)),
                target_node_id=NodeId(str(rel.target_id)),
                relationship_type=RelationshipType(rel.relation_type),
                label=rel.relation_type,
                properties=rel.properties or {},
            )
            if rel
            else None
        )

    async def list_relationships(
        self,
        relationship_type: RelationshipType | None = None,
        source_node_id: NodeId | None = None,
        target_node_id: NodeId | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Relationship]:
        rels, _ = self.relationship_manager.find_relationships(
            relation_type=relationship_type.value if relationship_type else None,
            source_id=int(source_node_id.value) if source_node_id else None,
            target_id=int(target_node_id.value) if target_node_id else None,
            limit=limit or 100,
            offset=offset or 0,
        )
        return [
            Relationship(
                id=RelationshipId(str(r.id)),
                source_node_id=NodeId(str(r.source_id)),
                target_node_id=NodeId(str(r.target_id)),
                relationship_type=RelationshipType(r.relation_type),
                label=r.relation_type,
                properties=r.properties or {},
            )
            for r in rels
        ]

    async def update_relationship(
        self,
        relationship_id: RelationshipId,
        label: str | None = None,
        properties: dict | None = None,
        weight: float | None = None,
    ) -> Relationship:
        rel = self.relationship_manager.update_relationship(
            int(relationship_id.value), properties=properties if properties is not None else {}
        )
        if not rel:
            raise ValueError("관계를 찾을 수 없습니다")
        # 텍스트 내용이 변경되면 임베딩 업데이트
        if label or properties:
            embedding_text = f"{label or rel.relation_type} {properties or ''}"
            embedding_vector = await self.text_embedder.embed_text(embedding_text)
            self.embedding_manager.store_embedding(
                "edge", rel.id, np.array(embedding_vector.embedding), embedding_vector.model_name
            )
        return Relationship(
            id=RelationshipId(str(rel.id)),
            source_node_id=NodeId(str(rel.source_id)),
            target_node_id=NodeId(str(rel.target_id)),
            relationship_type=RelationshipType(rel.relation_type),
            label=rel.relation_type,
            properties=rel.properties or {},
        )

    async def delete_relationship(self, relationship_id: RelationshipId) -> bool:
        self.embedding_manager.delete_embedding("edge", int(relationship_id.value))
        return self.relationship_manager.delete_relationship(int(relationship_id.value))

    async def get_node_relationships(
        self, node_id: NodeId, direction: str = "both"
    ) -> list[Relationship]:
        rels, _ = self.relationship_manager.get_entity_relationships(
            int(node_id.value), direction=direction
        )
        return [
            Relationship(
                id=RelationshipId(str(r.id)),
                source_node_id=NodeId(str(r.source_id)),
                target_node_id=NodeId(str(r.target_id)),
                relationship_type=RelationshipType(r.relation_type),
                label=r.relation_type,
                properties=r.properties or {},
            )
            for r in rels
        ]


class ConcreteKnowledgeSearchUseCase(KnowledgeSearchUseCase):
    """지식 검색 유스케이스의 구체적 구현체."""

    def __init__(
        self,
        knowledge_search_service: KnowledgeSearchService,
        node_use_case: NodeManagementUseCase,
        document_use_case: DocumentManagementUseCase,
        relationship_use_case: RelationshipManagementUseCase,
        vector_store: SQLiteVectorStore,
        text_embedder: TextEmbedder,
        search_config: SearchConfig,  # search_config 추가
    ):
        self.knowledge_search_service = knowledge_search_service
        self.node_use_case = node_use_case
        self.document_use_case = document_use_case
        self.relationship_use_case = relationship_use_case
        self.vector_store = vector_store
        self.text_embedder = text_embedder
        self.search_config = search_config

    async def search_knowledge(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        limit: int = 10,
        similarity_threshold: float | None = None,  # 기본값 제거
        include_documents: bool = True,
        include_nodes: bool = True,
        include_relationships: bool = True,
    ):
        # 이것은 단순화된 구현입니다. 실제 구현은 더 복잡한 로직을 포함하고
        # 다른 서비스/유스케이스의 검색 메서드를 잠재적으로 호출할 수 있습니다.
        # 지금은 벡터 저장소를 사용한 시맨틱 검색만 사용합니다.
        query_embedding_result = await self.text_embedder.embed_text(query)
        query_embedding = query_embedding_result.embedding

        # 노드와 관계에 대한 시맨틱 검색 수행
        similar_nodes_with_scores = await self.vector_store.search_similar(
            Vector(query_embedding), k=limit, filter_criteria={"entity_type": "node"}
        )
        similar_relationships_with_scores = await self.vector_store.search_similar(
            Vector(query_embedding), k=limit, filter_criteria={"entity_type": "edge"}
        )

        # 결과를 SearchResultCollection 형식으로 변환
        results = []
        for vector_id, score in similar_nodes_with_scores:
            node = await self.node_use_case.get_node(NodeId(vector_id))
            if node:
                results.append(SearchResult(score=score, node=node))

        for vector_id, score in similar_relationships_with_scores:
            rel = await self.relationship_use_case.get_relationship(RelationshipId(vector_id))
            if rel:
                results.append(SearchResult(score=score, relationship=rel))

        # 점수 기준 내림차순 정렬
        results.sort(key=lambda x: x.score, reverse=True)

        return SearchResultCollection(
            results=results, total_count=len(results), query=query, strategy=strategy
        )

    async def search_documents(
        self, query: str, limit: int = 10, similarity_threshold: float | None = None
    ) -> list[Document]:
        # 단순화됨: 실제 문서 검색 구현
        return []

    async def search_nodes(
        self,
        query: str,
        node_types: list[str] | None = None,
        limit: int = 10,
        similarity_threshold: float | None = None,
    ) -> list[Node]:
        # 단순화됨: 실제 노드 검색 구현
        return []

    async def search_relationships(
        self,
        query: str,
        relationship_types: list[str] | None = None,
        limit: int = 10,
        similarity_threshold: float | None = None,
    ) -> list[Relationship]:
        # 단순화됨: 실제 관계 검색 구현
        return []

    async def semantic_search(
        self, query: str, limit: int = 10, similarity_threshold: float | None = None
    ):
        # 이 메서드는 SEMANTIC 전략으로 search_knowledge에 위임할 수 있습니다.
        return await self.search_knowledge(
            query,
            strategy=SearchStrategy.SEMANTIC,
            limit=limit,
            similarity_threshold=similarity_threshold or self.search_config.similarity_threshold,
        )


# KnowledgeSearchUseCase의 종속성으로 DocumentManagementUseCase의 플레이스홀더
class ConcreteDocumentManagementUseCase(DocumentManagementUseCase):
    """문서 관리 유스케이스의 구체적 구현체."""

    def __init__(self, document_repository: SQLiteDocumentRepository):
        self.document_repository = document_repository

    async def create_document(
        self, title: str, content: str, metadata: dict | None = None
    ) -> Document:
        # 단순화됨
        return Document(
            id=DocumentId("1"),
            title=title,
            content=content,
            doc_type=DocumentType.TEXT,
            metadata=metadata or {},
        )

    async def get_document(self, document_id: DocumentId) -> Document | None:
        # 단순화됨
        return None

    async def list_documents(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Document]:
        # 단순화됨
        return []

    async def update_document(
        self,
        document_id: DocumentId,
        title: str | None = None,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> Document:
        # 단순화됨
        return Document(
            id=document_id,
            title=title or "",
            content=content or "",
            doc_type=DocumentType.TEXT,
            metadata=metadata or {},
        )

    async def delete_document(self, document_id: DocumentId) -> bool:
        # 단순화됨
        return True


async def main():
    """지식 그래프 MCP 서버를 실행합니다."""
    parser = argparse.ArgumentParser(description="지식 그래프 MCP 서버 실행")
    parser.add_argument(
        "--db-path",
        type=str,
        default="knowledge_graph.db",
        help="SQLite 데이터베이스 파일 경로",
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default="vector_indexes",
        help="벡터 인덱스 파일 디렉토리",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="서버를 실행할 호스트")
    parser.add_argument("--port", type=int, default=8080, help="서버를 실행할 포트")
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "stdio", "streamable-http"],
        help="사용할 전송 프로토콜",
    )
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="데이터베이스 스키마가 없는 경우 초기화합니다",
    )
    parser.add_argument("--dimension", type=int, default=128, help="임베딩 벡터의 차원")
    parser.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "inner_product", "l2"],
        help="벡터 검색을 위한 유사도 메트릭",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="Knowledge Graph Server",
        help="MCP 서버 이름",
    )
    parser.add_argument(
        "--server-instructions",
        type=str,
        default="벡터 검색 기능이 있는 SQLite 기반 지식 그래프",
        help="MCP 서버에 대한 지침",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SearchConfig().similarity_threshold,
        help="지식 검색을 위한 유사도 임계값",
    )

    args = parser.parse_args()

    # 벡터 디렉토리가 없으면 생성
    os.makedirs(args.vector_dir, exist_ok=True)

    # 데이터베이스 연결 및 스키마 설정
    # Database 포트를 구현하는 SQLiteDatabase 사용
    db_adapter = SQLiteDatabase(db_path=args.db_path)  # SQLiteDatabase로 변경
    await db_adapter.connect()
    assert db_adapter.connection is not None  # 연결 후 연결이 None이 아닌지 확인

    schema_manager = SchemaManager(args.db_path)  # SchemaManager는 db_path 문자열을 예상합니다
    if args.init_schema:
        print(f"{args.db_path}에서 데이터베이스 스키마 초기화 중")
        schema_manager.initialize_schema()

    # 리포지토리 및 서비스 초기화. Database 포트가 예상되는 곳에 db_adapter 전달.
    # EntityManager 및 RelationshipManager의 경우 sqlite3.Connection을 예상합니다.
    # 이는 SQLiteDatabase의 _connection 속성입니다.
    entity_manager = EntityManager(db_adapter.connection)
    relationship_manager = RelationshipManager(db_adapter.connection)
    document_repository = SQLiteDocumentRepository(
        db_adapter
    )  # db_adapter가 Database 포트를 구현하므로 전달
    embedding_manager = EmbeddingManager(db_adapter.connection)
    vector_store = SQLiteVectorStore(
        args.db_path, table_name="knowledge_vectors"
    )  # 여기에 db_path 사용
    await vector_store.initialize_store(
        dimension=args.dimension, metric=args.similarity
    )  # 비동기적으로 벡터 저장소 초기화
    text_embedder = RandomTextEmbedder(dimension=args.dimension)  # RandomTextEmbedder 직접 사용
    search_config = SearchConfig(
        similarity_threshold=args.similarity_threshold
    )  # SearchConfig 인스턴스 생성
    knowledge_search_service = KnowledgeSearchService(
        text_embedder=text_embedder, search_config=search_config
    )  # search_config 전달

    # 유스케이스 초기화
    node_use_case = ConcreteNodeManagementUseCase(
        entity_manager, embedding_manager, vector_store, text_embedder
    )
    relationship_use_case = ConcreteRelationshipManagementUseCase(
        relationship_manager, embedding_manager, vector_store, text_embedder
    )
    document_use_case = ConcreteDocumentManagementUseCase(document_repository)
    knowledge_search_use_case = ConcreteKnowledgeSearchUseCase(
        knowledge_search_service,
        node_use_case,
        document_use_case,
        relationship_use_case,
        vector_store,
        text_embedder,
        search_config,  # search_config 전달
    )

    # FastMCPConfig 생성
    config = FastMCPConfig(
        host=args.host,
        port=args.port,
        # 필요한 경우 다른 설정 매개변수를 여기에 설정할 수 있습니다.
    )

    # 서버 생성 및 시작
    print(f"{args.host}:{args.port}에서 지식 그래프 MCP 서버 시작 중")
    server = KnowledgeGraphServer(
        node_use_case=node_use_case,
        relationship_use_case=relationship_use_case,
        knowledge_search_use_case=knowledge_search_use_case,
        config=config,
    )

    try:
        server.start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("사용자에 의해 서버가 중지되었습니다")
    finally:
        server.close()
        print("서버 리소스가 정리되었습니다")


if __name__ == "__main__":
    asyncio.run(main())
