"""
Use Case 인터페이스들.

애플리케이션 레이어의 Use Case 인터페이스들을 정의합니다.
이들은 비즈니스 로직의 진입점 역할을 하며, 
도메인 서비스들을 조합하여 구체적인 비즈니스 요구사항을 구현합니다.
"""

# Analytics Use Cases
from .analytics_use_cases import (
    KnowledgeGraphAnalyticsUseCase,
    QualityAnalyticsUseCase,
    SearchAnalyticsUseCase,
)

# Document Use Cases
from .document_use_cases import DocumentManagementUseCase, DocumentProcessingUseCase

# Knowledge Search Use Cases
from .knowledge_search_use_cases import KnowledgeNavigationUseCase, KnowledgeSearchUseCase

# Node Use Cases
from .node_use_cases import NodeEmbeddingUseCase, NodeManagementUseCase

# Relationship Use Cases
from .relationship_use_cases import (
    RelationshipAnalysisUseCase,
    RelationshipEmbeddingUseCase,
    RelationshipManagementUseCase,
)

__all__ = [
    # Document Use Cases
    "DocumentManagementUseCase",
    "DocumentProcessingUseCase",
    # Knowledge Search Use Cases
    "KnowledgeSearchUseCase",
    "KnowledgeNavigationUseCase",
    # Node Use Cases
    "NodeManagementUseCase",
    "NodeEmbeddingUseCase",
    # Relationship Use Cases
    "RelationshipManagementUseCase",
    "RelationshipAnalysisUseCase",
    "RelationshipEmbeddingUseCase",
    # Analytics Use Cases
    "KnowledgeGraphAnalyticsUseCase",
    "SearchAnalyticsUseCase",
    "QualityAnalyticsUseCase",
]