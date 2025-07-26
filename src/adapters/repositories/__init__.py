"""
Repository ports for data persistence.

These interfaces define how the domain interacts with data storage systems.
"""

from .entity_repository import EntityRepository
from .relationship_repository import RelationshipRepository
from .embedding_repository import EmbeddingRepository
from .vector_index_repository import VectorIndexRepository

__all__ = [
    "EntityRepository",
    "RelationshipRepository",
    "EmbeddingRepository",
    "VectorIndexRepository",
]
