"""
포트 인터페이스들.

헥사고널 아키텍처의 포트들을 정의합니다.
이 인터페이스들은 도메인과 외부 어댑터 간의 계약을 정의합니다.
"""

# Repository Ports
from .database import Database
from .document_repository import DocumentRepository
from .knowledge_extractor import KnowledgeExtractor
from .node_repository import NodeRepository
from .relationship_repository import RelationshipRepository
from .vector_store import VectorStore

# Service Ports  
from .llm_service import (
    KnowledgeExtractionPrompt,
    LLMMessage,
    LLMResponse,
    LLMService,
    MessageRole,
)
from .text_embedder import EmbeddingConfig, EmbeddingResult, TextEmbedder

# Use Case Ports
from .analytics_use_cases import (
    KnowledgeGraphAnalyticsUseCase,
    QualityAnalyticsUseCase,
    SearchAnalyticsUseCase,
)
from .document_use_cases import DocumentManagementUseCase, DocumentProcessingUseCase
from .knowledge_search_use_cases import KnowledgeNavigationUseCase, KnowledgeSearchUseCase
from .node_use_cases import NodeEmbeddingUseCase, NodeManagementUseCase
from .relationship_use_cases import (
    RelationshipAnalysisUseCase,
    RelationshipEmbeddingUseCase,
    RelationshipManagementUseCase,
)

__all__ = [
    # Repository Ports
    "DocumentRepository",
    "NodeRepository", 
    "RelationshipRepository",
    # Service Ports
    "Database",
    "KnowledgeExtractor",
    "LLMService",
    "TextEmbedder",
    "VectorStore",
    # Use Case Ports
    "DocumentManagementUseCase",
    "DocumentProcessingUseCase",
    "KnowledgeSearchUseCase",
    "KnowledgeNavigationUseCase",
    "NodeManagementUseCase",
    "NodeEmbeddingUseCase",
    "RelationshipManagementUseCase",
    "RelationshipAnalysisUseCase",
    "RelationshipEmbeddingUseCase",
    "KnowledgeGraphAnalyticsUseCase",
    "SearchAnalyticsUseCase",
    "QualityAnalyticsUseCase",
    # Data Classes
    "EmbeddingConfig",
    "EmbeddingResult",
    "LLMMessage",
    "LLMResponse",
    "KnowledgeExtractionPrompt",
    # Enums
    "MessageRole",
]
