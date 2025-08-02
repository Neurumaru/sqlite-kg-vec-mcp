"""
포트 인터페이스들.

헥사고널 아키텍처의 포트들을 정의합니다.
이 인터페이스들은 도메인과 외부 어댑터 간의 계약을 정의합니다.
"""

# Repository 포트
from .database import Database
from .knowledge_extractor import KnowledgeExtractor
from .repositories.document import DocumentRepository
from .repositories.node import NodeRepository
from .repositories.relationship import RelationshipRepository

# 서비스 포트
from .text_embedder import TextEmbedder
from .vector_store import VectorStore

__all__ = [
    # Repository 포트
    "DocumentRepository",
    "NodeRepository",
    "RelationshipRepository",
    # 서비스 포트
    "Database",
    "KnowledgeExtractor",
    "TextEmbedder",
    "VectorStore",
]
