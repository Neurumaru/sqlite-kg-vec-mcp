"""
MCP 서버 인터페이스를 갖춘 SQLite 지식 그래프 및 벡터 데이터베이스.

이 패키지는 SQLite 기반 지식 그래프와 벡터 저장소(선택적으로 HNSW 인덱스 사용)를
결합하고 MCP(모델 컨텍스트 프로토콜) 서버를 통해 인터페이스를 제공합니다.
"""

import sqlite3
from pathlib import Path

from .adapters.hnsw.embeddings import EmbeddingManager
from .adapters.huggingface.text_embedder import HuggingFaceTextEmbedder
from .adapters.openai.text_embedder import OpenAITextEmbedder
from .adapters.sqlite3.connection import DatabaseConnection
from .adapters.sqlite3.graph.entities import EntityManager
from .adapters.sqlite3.graph.relationships import RelationshipManager
from .adapters.sqlite3.graph.traversal import GraphTraversal
from .adapters.sqlite3.schema import SchemaManager
from .adapters.testing.text_embedder import RandomTextEmbedder

# .env 파일에서 환경 변수 로드
try:
    from dotenv import load_dotenv

    # 프로젝트 루트에서 .env 파일 찾기
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)

except ImportError:
    # python-dotenv가 설치되지 않은 경우 로딩 건너뛰기
    pass

__version__ = "0.1.0"

# from .adapters.hnsw.search import SearchResult, VectorSearch  # TODO: 종속성 수정
# from .adapters.hnsw.text_embedder import (
#     VectorTextEmbedder, create_embedder
# ) # TODO: text_embedder 구현


# 예제를 위한 임베더 생성 함수
def create_embedder(embedder_type="random", **kwargs):
    """유형에 따라 텍스트 임베더를 생성합니다.

    Args:
        embedder_type: 임베더 유형 ('random', 'openai',
            'sentence-transformers')
        **kwargs: 임베더에 대한 추가 인수

    Returns:
        텍스트 임베더 인스턴스
    """
    if embedder_type == "random":
        return RandomTextEmbedder(dimension=kwargs.get("dimension", 128))
    if embedder_type == "openai":
        return OpenAITextEmbedder(**kwargs)
    if embedder_type == "sentence-transformers":
        return HuggingFaceTextEmbedder(**kwargs)
    raise ValueError(f"알 수 없는 임베더 유형: {embedder_type}")


# MCP 서버 내보내기
try:
    from .adapters.fastmcp.server import KnowledgeGraphServer

    __all__ = ["KnowledgeGraph", "EmbeddingManager", "create_embedder", "KnowledgeGraphServer"]
except ImportError:
    # MCP 종속성을 사용할 수 없는 경우 기본 클래스만 내보냅니다.
    __all__ = ["KnowledgeGraph", "EmbeddingManager", "create_embedder"]

# 기본 클래스 내보내기 - 순환 종속성을 방지하기 위해 직접 가져오기를 피합니다.
# from .adapters.sqlite3.connection import DatabaseConnection
# from .adapters.sqlite3.graph.entities import Entity, EntityManager
# from .adapters.sqlite3.graph.relationships import Relationship, RelationshipManager
# from .adapters.sqlite3.graph.traversal import GraphTraversal, PathNode
# from .adapters.sqlite3.schema import SchemaManager

# 서버 API 조건부 가져오기
# try:
#     from .adapters.fastmcp.server import KnowledgeGraphServer
# except ImportError:
#     # MCP 서버 종속성을 사용할 수 없는 경우 메시지를 제공합니다.
#     class KnowledgeGraphServer:
#         def __init__(self, *args, **kwargs):
#             raise ImportError(
#                 "KnowledgeGraphServer에는 추가 종속성이 필요합니다. "
#                 "MCP 서버 기능을 사용하려면 'fastmcp' 패키지를 설치하십시오."
#             )


# 직접 사용을 위한 편의 클래스
class KnowledgeGraph:
    """
    엔티티, 관계 및 벡터 검색 기능을 결합한
    기본 지식 그래프 인터페이스.
    """

    def __init__(
        self,
        db_path,
        vector_index_dir=None,
        embedding_dim=128,
        text_embedder=None,
        embedder_type="sentence-transformers",
        embedder_kwargs=None,
    ):
        """
        지식 그래프를 초기화합니다.

        Args:
            db_path: SQLite 데이터베이스 파일 경로
            vector_index_dir: 벡터 인덱스 파일을 저장할 디렉토리
            embedding_dim: 임베딩 벡터의 차원
            text_embedder: 텍스트-벡터 변환을 위한 VectorTextEmbedder 인스턴스
            embedder_type: text_embedder가 None인 경우 생성할 임베더 유형
            embedder_kwargs: 임베더 생성을 위한 인수
        """
        # 순환 종속성을 피하기 위해 지연된 가져오기 사용
        # from .adapters.sqlite3.connection import DatabaseConnection
        # from .adapters.sqlite3.graph.entities import EntityManager
        # from .adapters.sqlite3.graph.relationships import RelationshipManager
        # from .adapters.sqlite3.graph.traversal import GraphTraversal
        # from .adapters.sqlite3.schema import SchemaManager

        # 데이터베이스 초기화
        self.db_connection = DatabaseConnection(db_path)
        self.conn = self.db_connection.connect()

        # 스키마 초기화
        schema_manager = SchemaManager(db_path)
        try:
            if schema_manager.get_schema_version() == 0:
                schema_manager.initialize_schema()
        except sqlite3.OperationalError: # exception 변수명으로 변경
            # 스키마가 아직 존재하지 않음
            schema_manager.initialize_schema()

        # 관리자 생성
        self.entity_manager = EntityManager(self.conn)
        self.relationship_manager = RelationshipManager(self.conn)
        self.embedding_manager = EmbeddingManager(self.conn)
        self.graph_traversal = GraphTraversal(self.conn)
        # TODO: VectorSearch 종속성이 수정되면 다시 활성화
        # self.vector_search = VectorSearch(
        #     connection=self.conn,
        #     index_dir=vector_index_dir,
        #     embedding_dim=embedding_dim,
        #     text_embedder=text_embedder,
        #     embedder_type=embedder_type,
        #     embedder_kwargs=embedder_kwargs,
        # )

    # 엔티티 메서드
    def create_node(self, node_type, name=None, properties=None):
        """그래프에 새 노드를 생성합니다."""
        return self.entity_manager.create_entity(node_type, name, properties)

    def get_node(self, node_id):
        """ID로 노드를 가져옵니다."""
        return self.entity_manager.get_entity(node_id)

    def get_node_by_uuid(self, uuid):
        """UUID로 노드를 가져옵니다."""
        return self.entity_manager.get_entity_by_uuid(uuid)

    def update_node(self, node_id, name=None, properties=None):
        """노드의 속성을 업데이트합니다."""
        return self.entity_manager.update_entity(node_id, name, properties)

    def delete_node(self, node_id):
        """노드를 삭제합니다."""
        return self.entity_manager.delete_entity(node_id)

    def find_nodes(self, node_type=None, name_pattern=None, properties=None, limit=100, offset=0):
        """기준과 일치하는 노드를 찾습니다."""
        return self.entity_manager.find_entities(
            entity_type=node_type,
            name_pattern=name_pattern,
            property_filters=properties,
            limit=limit,
            offset=offset,
        )

    # 관계 메서드
    def create_edge(self, source_id, target_id, relation_type, properties=None):
        """노드 사이에 새 엣지를 생성합니다."""
        return self.relationship_manager.create_relationship(
            source_id, target_id, relation_type, properties
        )

    def get_edge(self, edge_id, include_entities=False):
        """ID로 엣지를 가져옵니다."""
        return self.relationship_manager.get_relationship(edge_id, include_entities)

    def update_edge(self, edge_id, properties):
        """엣지의 속성을 업데이트합니다."""
        return self.relationship_manager.update_relationship(edge_id, properties)

    def delete_edge(self, edge_id):
        """엣지를 삭제합니다."""
        return self.relationship_manager.delete_relationship(edge_id)

    def find_edges(
        self,
        source_id=None,
        target_id=None,
        relation_type=None,
        properties=None,
        include_entities=False,
        limit=100,
        offset=0,
    ):
        """기준과 일치하는 엣지를 찾습니다."""
        return self.relationship_manager.find_relationships(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            property_filters=properties,
            include_entities=include_entities,
            limit=limit,
            offset=offset,
        )

    # 그래프 순회 메서드
    def get_neighbors(
        self,
        node_id,
        direction="both",
        relation_types=None,
        entity_types=None,
        limit=100,
    ):
        """이웃 노드를 가져옵니다."""
        return self.graph_traversal.get_neighbors(
            node_id, direction, relation_types, entity_types, limit
        )

    def find_paths(self, start_id, end_id, max_depth=5, relation_types=None, entity_types=None):
        """노드 간의 경로를 찾습니다."""
        return self.graph_traversal.find_shortest_path(
            start_id, end_id, max_depth, relation_types, entity_types
        )

    # TODO: 벡터 검색 메서드 - VectorSearch가 수정되면 다시 활성화
    # def search_similar_nodes(
    #     self,
    #     query_vector=None,
    #     node_id=None,
    #     limit=10,
    #     entity_types=None,
    #     include_entities=True,
    # ):
    #     """유사한 노드를 검색합니다."""
    #     if node_id is not None:
    #         return self.vector_search.search_similar_to_entity(
    #             "node", node_id, limit, entity_types, include_entities
    #         )
    #     elif query_vector is not None:
    #         return self.vector_search.search_similar(
    #             query_vector, limit, entity_types, include_entities
    #         )
    #     else:
    #         raise ValueError("query_vector 또는 node_id를 제공해야 합니다")

    # def search_by_text(
    #     self, query_text, limit=10, entity_types=None, include_entities=True
    # ):
    #     """텍스트 쿼리를 사용하여 검색합니다."""
    #     return self.vector_search.search_by_text(
    #         query_text, limit, entity_types, include_entities
    #     )

    def close(self):
        """데이터베이스 연결을 닫습니다."""
        self.db_connection.close()
