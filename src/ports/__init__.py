"""
포트 인터페이스들.

헥사고널 아키텍처의 포트들을 정의합니다.
이 인터페이스들은 도메인과 외부 어댑터 간의 계약을 정의합니다.
"""

from .document_repository import DocumentRepository
from .llm_service import (
    KnowledgeExtractionPrompt,
    LLMMessage,
    LLMResponse,
    LLMService,
    MessageRole,
)
from .node_repository import NodeRepository
from .relationship_repository import RelationshipRepository
from .text_embedder import EmbeddingConfig, EmbeddingResult, TextEmbedder

__all__ = [
    # Repository Ports
    "DocumentRepository",
    "NodeRepository",
    "RelationshipRepository",
    # Service Ports
    "TextEmbedder",
    "LLMService",
    # Data Classes
    "EmbeddingConfig",
    "EmbeddingResult",
    "LLMMessage",
    "LLMResponse",
    "KnowledgeExtractionPrompt",
    # Enums
    "MessageRole",
]
