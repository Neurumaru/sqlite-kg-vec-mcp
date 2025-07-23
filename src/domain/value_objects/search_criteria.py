"""
SearchCriteria value object for representing search parameters.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from .vector import Vector
from .entity_type import EntityType


@dataclass(frozen=True)
class SearchCriteria:
    """Immutable search criteria for knowledge graph queries."""

    query_text: Optional[str] = None
    query_vector: Optional[Vector] = None
    entity_types: Optional[tuple[EntityType, ...]] = None
    limit: int = 10
    similarity_threshold: float = 0.0
    property_filters: Optional[tuple] = None  # Frozen dict representation

    def __post_init__(self):
        if self.query_text is None and self.query_vector is None:
            raise ValueError("Either query_text or query_vector must be provided")

        if self.limit <= 0:
            raise ValueError("Limit must be positive")

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")

    @classmethod
    def text_search(
        cls,
        query_text: str,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        property_filters: Optional[Dict[str, Any]] = None
    ) -> "SearchCriteria":
        """Create text-based search criteria."""
        return cls(
            query_text=query_text,
            entity_types=tuple(entity_types) if entity_types else None,
            limit=limit,
            property_filters=cls._freeze_dict(property_filters) if property_filters else None
        )

    @classmethod
    def vector_search(
        cls,
        query_vector: Vector,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0,
        property_filters: Optional[Dict[str, Any]] = None
    ) -> "SearchCriteria":
        """Create vector-based search criteria."""
        return cls(
            query_vector=query_vector,
            entity_types=tuple(entity_types) if entity_types else None,
            limit=limit,
            similarity_threshold=similarity_threshold,
            property_filters=cls._freeze_dict(property_filters) if property_filters else None
        )

    @classmethod
    def hybrid_search(
        cls,
        query_text: str,
        query_vector: Vector,
        entity_types: Optional[List[EntityType]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.0,
        property_filters: Optional[Dict[str, Any]] = None
    ) -> "SearchCriteria":
        """Create hybrid search criteria with both text and vector."""
        return cls(
            query_text=query_text,
            query_vector=query_vector,
            entity_types=tuple(entity_types) if entity_types else None,
            limit=limit,
            similarity_threshold=similarity_threshold,
            property_filters=cls._freeze_dict(property_filters) if property_filters else None
        )

    @staticmethod
    def _freeze_dict(d: Dict[str, Any]) -> tuple:
        """Convert dict to frozen tuple representation."""
        return tuple(sorted(d.items()))

    def get_property_filters(self) -> Optional[Dict[str, Any]]:
        """Get property filters as dict."""
        if self.property_filters is None:
            return None
        return dict(self.property_filters)

    def get_entity_types_list(self) -> Optional[List[EntityType]]:
        """Get entity types as list."""
        if self.entity_types is None:
            return None
        return list(self.entity_types)
