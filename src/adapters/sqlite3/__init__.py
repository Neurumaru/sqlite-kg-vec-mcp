"""
SQLite3 데이터베이스 어댑터.
이 모듈에는 연결 관리, 스키마 작업 및 트랜잭션을 포함한
데이터베이스 작업을 위한 SQLite3 관련 구현이 포함되어 있습니다.
"""

from .connection import DatabaseConnection
from .database import SQLiteDatabase

# 그래프 도메인 클래스
from .graph.entities import Entity, EntityManager
from .graph.interactive_search import InteractiveSearchEngine
from .graph.relationships import Relationship, RelationshipManager
from .graph.traversal import GraphTraversal, PathNode
from .schema import SchemaManager
from .transactions import TransactionManager

# 분리된 벡터 저장소 구현체들 (선택적 사용 가능)
from .vector_reader_impl import SQLiteVectorReader
from .vector_retriever_impl import SQLiteVectorRetriever
from .vector_store import SQLiteVectorStore
from .vector_store_base import SQLiteVectorStoreBase
from .vector_writer_impl import SQLiteVectorWriter

__all__ = [
    "DatabaseConnection",
    "SchemaManager",
    "SQLiteDatabase",
    "SQLiteVectorStore",
    "TransactionManager",
    # 분리된 벡터 저장소 구현체들
    "SQLiteVectorStoreBase",
    "SQLiteVectorWriter",
    "SQLiteVectorReader",
    "SQLiteVectorRetriever",
    # 그래프 클래스
    "Entity",
    "EntityManager",
    "Relationship",
    "RelationshipManager",
    "GraphTraversal",
    "PathNode",
    "InteractiveSearchEngine",
]
