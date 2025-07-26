"""
Ports for the knowledge graph system.

This module contains both primary ports (driving ports) that define how 
external systems can interact with the domain, and secondary ports 
(driven ports) that define how the domain interacts with external systems.
"""

from .knowledge_graph_use_cases import KnowledgeGraphUseCases
from .search_use_cases import SearchUseCases
from .admin_use_cases import AdminUseCases
from .llm_service import LLMService
from .knowledge_extractor import KnowledgeExtractor
from .text_embedder import TextEmbedder
from .vector_store import VectorStore
from .database import Database

# Import repository interfaces for convenience
from .repositories import (
    EntityRepository,
    EmbeddingRepository,
    RelationshipRepository,
    VectorIndexRepository
)

__all__ = [
    "KnowledgeGraphUseCases",
    "SearchUseCases", 
    "AdminUseCases",
    "LLMService",
    "KnowledgeExtractor",
    "TextEmbedder",
    "VectorStore",
    "Database",
    "EntityRepository",
    "EmbeddingRepository",
    "RelationshipRepository", 
    "VectorIndexRepository"
]
