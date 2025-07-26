"""
Embedding domain model for the knowledge graph.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.vector import Vector


@dataclass
class Embedding:
    """
    Domain entity representing a vector embedding for an entity or relationship.

    This encapsulates the vector representation and associated metadata.
    """

    id: NodeId
    entity_id: NodeId
    vector: Vector
    model_name: str
    model_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate embedding invariants."""
        if not self.id:
            raise ValueError("Embedding must have an ID")

        if not self.entity_id:
            raise ValueError("Embedding must be associated with an entity")

        if not self.vector:
            raise ValueError("Embedding must have a vector")

        if not self.model_name:
            raise ValueError("Embedding must specify the model name")

    @classmethod
    def create(
        cls,
        entity_id: NodeId,
        vector: Vector,
        model_name: str,
        model_version: Optional[str] = None
    ) -> "Embedding":
        """
        Factory method to create a new embedding with generated ID.

        Args:
            entity_id: ID of the associated entity
            vector: The embedding vector
            model_name: Name of the model used to generate the embedding
            model_version: Optional version of the model

        Returns:
            New embedding instance
        """
        return cls(
            id=NodeId.generate(),
            entity_id=entity_id,
            vector=vector,
            model_name=model_name,
            model_version=model_version
        )

    @classmethod
    def restore(
        cls,
        id: NodeId,
        entity_id: NodeId,
        vector: Vector,
        model_name: str,
        model_version: Optional[str] = None,
        created_at: Optional[datetime] = None
    ) -> "Embedding":
        """
        Restore an embedding from persistence.

        Args:
            id: Embedding ID
            entity_id: Associated entity ID
            vector: The embedding vector
            model_name: Model name
            model_version: Model version
            created_at: Creation timestamp

        Returns:
            Restored embedding instance
        """
        return cls(
            id=id,
            entity_id=entity_id,
            vector=vector,
            model_name=model_name,
            model_version=model_version,
            created_at=created_at or datetime.now()
        )

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vector."""
        return self.vector.dimension

    def similarity_to(self, other: "Embedding") -> float:
        """
        Calculate cosine similarity to another embedding.

        Args:
            other: Other embedding to compare with

        Returns:
            Cosine similarity score

        Raises:
            ValueError: If embeddings have different dimensions
        """
        if self.dimension != other.dimension:
            raise ValueError("Cannot compare embeddings with different dimensions")

        return self.vector.cosine_similarity(other.vector)

    def is_compatible_with(self, other: "Embedding") -> bool:
        """
        Check if this embedding is compatible with another for similarity calculation.

        Args:
            other: Other embedding to check compatibility with

        Returns:
            True if embeddings are compatible
        """
        return (
            self.dimension == other.dimension
            and self.model_name == other.model_name
            and (self.model_version == other.model_version
                 or self.model_version is None
                 or other.model_version is None)
        )

    def is_from_same_model(self, other: "Embedding") -> bool:
        """
        Check if this embedding is from the same model as another.

        Args:
            other: Other embedding to compare

        Returns:
            True if from same model
        """
        return (
            self.model_name == other.model_name
            and self.model_version == other.model_version
        )

    def to_dict(self) -> dict:
        """
        Convert embedding to dictionary representation.

        Returns:
            Dictionary representation
        """
        return {
            "id": str(self.id),
            "entity_id": str(self.entity_id),
            "vector": self.vector.to_list(),
            "dimension": self.dimension,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat()
        }

    def __str__(self) -> str:
        return f"Embedding({self.id}, entity={self.entity_id}, dim={self.dimension}, model={self.model_name})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Embedding):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
