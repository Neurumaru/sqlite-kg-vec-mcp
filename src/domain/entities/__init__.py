"""
Domain entities for the knowledge graph system.
"""

from .entity import Entity
from .relationship import Relationship
from .embedding import Embedding
from .search_result import SearchResult, SearchResultCollection
from .knowledge_graph import KnowledgeGraph

__all__ = [
    "Entity",
    "Relationship",
    "Embedding",
    "SearchResult",
    "SearchResultCollection",
    "KnowledgeGraph",
]
