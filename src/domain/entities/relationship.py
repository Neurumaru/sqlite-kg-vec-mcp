"""
Relationship domain entity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.domain.value_objects.vector import Vector


class RelationshipType(Enum):
    """Relationship type."""

    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    COLLABORATES_WITH = "collaborates_with"
    PART_OF = "part_of"
    LEADS = "leads"
    CREATES = "creates"
    USES = "uses"
    SIMILAR_TO = "similar_to"
    CAUSED_BY = "caused_by"
    INFLUENCES = "influences"
    CONTAINS = "contains"
    OTHER = "other"


@dataclass
class Relationship:
    """
    Relationship domain entity.

    Represents a relationship in the knowledge graph.
    Expresses relationships between two nodes and
    maintains connection information with source documents.
    """

    id: RelationshipId
    source_node_id: NodeId
    target_node_id: NodeId
    relationship_type: RelationshipType
    label: str  # Natural language label describing the relationship
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Relationship extraction confidence (0.0 ~ 1.0)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Document connection information
    source_documents: list[DocumentId] = field(default_factory=list)

    # Embedding information
    embedding: Vector | None = None
    embedding_model: str | None = None
    embedding_created_at: datetime | None = None

    # Extraction information (context in document, sentences, etc.)
    extraction_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.label.strip():
            raise ValueError("Relationship label cannot be empty")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.source_node_id == self.target_node_id:
            raise ValueError("Source and target nodes cannot be the same")

    def add_source_document(
        self,
        document_id: DocumentId,
        context: str | None = None,
        sentence: str | None = None,
    ) -> None:
        """Add source document."""
        if document_id not in self.source_documents:
            self.source_documents.append(document_id)
            if context or sentence:
                self.add_extraction_context(document_id, context, sentence)
            self.updated_at = datetime.now()

    def remove_source_document(self, document_id: DocumentId) -> None:
        """Remove source document."""
        if document_id in self.source_documents:
            self.source_documents.remove(document_id)
            # Also remove related extraction metadata
            key_to_remove = f"context_{document_id}"
            if key_to_remove in self.extraction_metadata:
                del self.extraction_metadata[key_to_remove]
            self.updated_at = datetime.now()

    def add_extraction_context(
        self,
        document_id: DocumentId,
        context: str | None = None,
        sentence: str | None = None,
    ) -> None:
        """Add extraction context within document."""
        self.extraction_metadata[f"context_{document_id}"] = {
            "context": context,
            "sentence": sentence,
            "extracted_at": datetime.now().isoformat(),
        }
        self.updated_at = datetime.now()

    def get_extraction_context(self, document_id: DocumentId) -> dict[str, str] | None:
        """Query extraction context from specific document."""
        return self.extraction_metadata.get(f"context_{document_id}")

    def set_embedding(self, embedding: Vector, model_name: str) -> None:
        """Set relationship embedding."""
        self.embedding = embedding
        self.embedding_model = model_name
        self.embedding_created_at = datetime.now()
        self.updated_at = datetime.now()

    def has_embedding(self) -> bool:
        """Check if embedding is set."""
        return self.embedding is not None

    def calculate_similarity(self, other: "Relationship") -> float:
        """Calculate similarity with another relationship."""
        if not self.has_embedding() or not other.has_embedding():
            raise ValueError("Both relationships must have embeddings for similarity calculation")

        # mypy: assert used since we confirmed embedding is not None
        assert self.embedding is not None and other.embedding is not None
        return self.embedding.cosine_similarity(other.embedding)

    def update_property(self, key: str, value: Any) -> None:
        """Update property."""
        self.properties[key] = value
        self.updated_at = datetime.now()

    def remove_property(self, key: str) -> None:
        """Remove property."""
        if key in self.properties:
            del self.properties[key]
            self.updated_at = datetime.now()

    def update_confidence(self, confidence: float) -> None:
        """Update confidence."""
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        self.confidence = confidence
        self.updated_at = datetime.now()

    def get_all_contexts(self) -> dict[str, dict[str, str]]:
        """Query extraction contexts from all documents."""
        contexts = {}
        for key, value in self.extraction_metadata.items():
            if key.startswith("context_"):
                doc_id = key.replace("context_", "")
                contexts[doc_id] = value
        return contexts

    def is_from_document(self, document_id: DocumentId) -> bool:
        """Check if relationship was extracted from specific document."""
        return document_id in self.source_documents

    def get_document_count(self) -> int:
        """Return number of connected documents."""
        return len(self.source_documents)

    def involves_node(self, node_id: NodeId) -> bool:
        """Check if specific node is involved in this relationship."""
        return node_id in (self.source_node_id, self.target_node_id)

    def get_other_node_id(self, node_id: NodeId) -> NodeId | None:
        """Return the other node ID for given node."""
        if self.source_node_id == node_id:
            return self.target_node_id
        if self.target_node_id == node_id:
            return self.source_node_id
        return None

    def reverse(self) -> "Relationship":
        """Create new relationship with reversed direction."""
        reversed_rel = Relationship(
            id=RelationshipId.generate(),
            source_node_id=self.target_node_id,
            target_node_id=self.source_node_id,
            relationship_type=self.relationship_type,
            label=f"reverse_of_{self.label}",
            properties=self.properties.copy(),
            confidence=self.confidence,
            source_documents=self.source_documents.copy(),
            embedding=self.embedding,
            embedding_model=self.embedding_model,
            embedding_created_at=self.embedding_created_at,
            extraction_metadata=self.extraction_metadata.copy(),
        )
        return reversed_rel

    def merge_with(self, other: "Relationship") -> None:
        """Merge with another relationship (remove duplicates)."""
        # Check if relationship is between same nodes
        if not (
            (
                self.source_node_id == other.source_node_id
                and self.target_node_id == other.target_node_id
            )
            or (
                self.source_node_id == other.target_node_id
                and self.target_node_id == other.source_node_id
            )
        ):
            raise ValueError("Cannot merge relationships between different nodes")

        # Merge source documents
        for doc_id in other.source_documents:
            if doc_id not in self.source_documents:
                self.source_documents.append(doc_id)

        # Merge properties (existing properties take priority)
        for key, value in other.properties.items():
            if key not in self.properties:
                self.properties[key] = value

        # Merge extraction metadata
        for key, value in other.extraction_metadata.items():
            if key not in self.extraction_metadata:
                self.extraction_metadata[key] = value

        # Use higher confidence
        self.confidence = max(self.confidence, other.confidence)

        # Update if there's a better embedding
        if not self.has_embedding() and other.has_embedding():
            self.embedding = other.embedding
            self.embedding_model = other.embedding_model
            self.embedding_created_at = other.embedding_created_at

        self.updated_at = datetime.now()

    def get_textual_representation(self) -> str:
        """Generate textual representation of relationship (for embedding)."""
        return f"{self.label} (type: {self.relationship_type.value}, confidence: {self.confidence})"

    def __str__(self) -> str:
        return f"Relationship(id={self.id}, {self.source_node_id} --[{self.label}]--> {self.target_node_id})"

    def __repr__(self) -> str:
        return (
            f"Relationship(id={self.id!r}, source={self.source_node_id!r}, "
            f"target={self.target_node_id!r}, type={self.relationship_type.value}, "
            f"label={self.label!r}, confidence={self.confidence})"
        )
