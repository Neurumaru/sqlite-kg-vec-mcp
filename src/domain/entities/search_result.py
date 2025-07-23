"""
SearchResult domain model for the knowledge graph.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

from .entity import Entity
from .embedding import Embedding


@dataclass(frozen=True)
class SearchResult:
    """
    Immutable value object representing a search result.

    Contains the found entity and associated metadata like similarity score.
    """

    entity: Entity
    similarity_score: float
    embedding: Optional[Embedding] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate search result."""
        if not self.entity:
            raise ValueError("SearchResult must have an entity")

        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0.0 and 1.0")

    @classmethod
    def create(
        cls,
        entity: Entity,
        similarity_score: float,
        embedding: Optional[Embedding] = None,
        **metadata
    ) -> "SearchResult":
        """
        Factory method to create a search result.

        Args:
            entity: The found entity
            similarity_score: Similarity score (0.0 to 1.0)
            embedding: Optional embedding used for similarity calculation
            **metadata: Additional metadata

        Returns:
            New search result instance
        """
        return cls(
            entity=entity,
            similarity_score=similarity_score,
            embedding=embedding,
            metadata=metadata if metadata else None
        )

    @classmethod
    def from_entity(cls, entity: Entity) -> "SearchResult":
        """
        Create a search result from an entity with perfect similarity.

        Args:
            entity: The entity

        Returns:
            Search result with similarity score 1.0
        """
        return cls(entity=entity, similarity_score=1.0)

    @property
    def entity_id(self):
        """Get the entity ID."""
        return self.entity.id

    @property
    def entity_type(self):
        """Get the entity type."""
        return self.entity.entity_type

    @property
    def entity_name(self):
        """Get the entity name."""
        return self.entity.name

    def has_high_similarity(self, threshold: float = 0.8) -> bool:
        """
        Check if result has high similarity score.

        Args:
            threshold: Similarity threshold

        Returns:
            True if similarity is above threshold
        """
        return self.similarity_score >= threshold

    def get_metadata_value(self, key: str, default=None):
        """
        Get a metadata value.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default
        """
        if not self.metadata:
            return default
        return self.metadata.get(key, default)

    def with_metadata(self, **additional_metadata) -> "SearchResult":
        """
        Create a new search result with additional metadata.

        Args:
            **additional_metadata: Additional metadata to add

        Returns:
            New search result with merged metadata
        """
        new_metadata = (self.metadata or {}).copy()
        new_metadata.update(additional_metadata)

        return SearchResult(
            entity=self.entity,
            similarity_score=self.similarity_score,
            embedding=self.embedding,
            metadata=new_metadata
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert search result to dictionary representation.

        Returns:
            Dictionary representation
        """
        result = {
            "entity": self.entity.to_dict(),
            "similarity_score": self.similarity_score
        }

        if self.embedding:
            result["embedding"] = self.embedding.to_dict()

        if self.metadata:
            result["metadata"] = self.metadata.copy()

        return result

    def __str__(self) -> str:
        return f"SearchResult({self.entity}, score={self.similarity_score:.3f})"


@dataclass(frozen=True)
class SearchResultCollection:
    """
    Collection of search results with metadata about the search operation.
    """

    results: tuple[SearchResult, ...]
    total_count: int
    query_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate collection."""
        if not isinstance(self.results, tuple):
            object.__setattr__(self, 'results', tuple(self.results))

        if self.total_count < 0:
            raise ValueError("Total count cannot be negative")

        if len(self.results) > self.total_count:
            raise ValueError("Results count cannot exceed total count")

    @classmethod
    def create(
        cls,
        results: list[SearchResult],
        total_count: Optional[int] = None,
        **query_metadata
    ) -> "SearchResultCollection":
        """
        Factory method to create a result collection.

        Args:
            results: List of search results
            total_count: Total number of matching results (for pagination)
            **query_metadata: Metadata about the query

        Returns:
            New result collection
        """
        return cls(
            results=tuple(results),
            total_count=total_count if total_count is not None else len(results),
            query_metadata=query_metadata if query_metadata else None
        )

    @classmethod
    def empty(cls, **query_metadata) -> "SearchResultCollection":
        """
        Create an empty result collection.

        Args:
            **query_metadata: Optional query metadata

        Returns:
            Empty result collection
        """
        return cls(
            results=(),
            total_count=0,
            query_metadata=query_metadata if query_metadata else None
        )

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index):
        return self.results[index]

    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return len(self.results) == 0

    def get_entities(self) -> list[Entity]:
        """Get all entities from results."""
        return [result.entity for result in self.results]

    def filter_by_type(self, entity_type) -> "SearchResultCollection":
        """
        Filter results by entity type.

        Args:
            entity_type: Entity type to filter by

        Returns:
            New filtered collection
        """
        filtered_results = [
            result for result in self.results
            if result.entity.entity_type == entity_type
        ]

        return SearchResultCollection.create(
            results=filtered_results,
            total_count=len(filtered_results),
            **(self.query_metadata or {})
        )

    def filter_by_similarity(self, threshold: float) -> "SearchResultCollection":
        """
        Filter results by similarity threshold.

        Args:
            threshold: Minimum similarity score

        Returns:
            New filtered collection
        """
        filtered_results = [
            result for result in self.results
            if result.similarity_score >= threshold
        ]

        return SearchResultCollection.create(
            results=filtered_results,
            total_count=len(filtered_results),
            **(self.query_metadata or {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert collection to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "results": [result.to_dict() for result in self.results],
            "total_count": self.total_count,
            "returned_count": len(self.results),
            "query_metadata": self.query_metadata
        }
