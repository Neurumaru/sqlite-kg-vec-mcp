"""
Node domain entity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.vector import Vector


class NodeType(Enum):
    """Node type."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    OTHER = "other"


@dataclass
class Node:
    """
    Node domain entity.

    Represents a node in the knowledge graph.
    Expresses entities extracted from documents and
    maintains connection information with source documents.
    """

    id: NodeId
    name: str
    node_type: NodeType
    description: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Document connection information
    source_documents: list[DocumentId] = field(default_factory=list)

    # Embedding information
    embedding: Vector | None = None
    embedding_model: str | None = None
    embedding_created_at: datetime | None = None

    # Extraction information (position in document, context, etc.)
    extraction_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Node name cannot be empty")

    def add_source_document(self, document_id: DocumentId, context: str | None = None) -> None:
        """Add source document."""
        if document_id not in self.source_documents:
            self.source_documents.append(document_id)
            if context:
                self.add_extraction_context(document_id, context)
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

    def add_extraction_context(self, document_id: DocumentId, context: str) -> None:
        """Add extraction context within document."""
        self.extraction_metadata[f"context_{document_id}"] = {
            "context": context,
            "extracted_at": datetime.now().isoformat(),
        }
        self.updated_at = datetime.now()

    def get_extraction_context(self, document_id: DocumentId) -> str | None:
        """Query extraction context from specific document."""
        context_data = self.extraction_metadata.get(f"context_{document_id}")
        return context_data.get("context") if context_data else None

    def set_embedding(self, embedding: Vector, model_name: str) -> None:
        """Set node embedding."""
        self.embedding = embedding
        self.embedding_model = model_name
        self.embedding_created_at = datetime.now()
        self.updated_at = datetime.now()

    def has_embedding(self) -> bool:
        """Check if embedding is set."""
        return self.embedding is not None

    def calculate_similarity(self, other: "Node") -> float:
        """Calculate similarity with another node."""
        if not self.has_embedding() or not other.has_embedding():
            raise ValueError("Both nodes must have embeddings for similarity calculation")

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

    def get_all_contexts(self) -> dict[str, str]:
        """Query extraction contexts from all documents."""
        contexts = {}
        for key, value in self.extraction_metadata.items():
            if key.startswith("context_"):
                doc_id = key.replace("context_", "")
                contexts[doc_id] = value.get("context", "")
        return contexts

    def is_from_document(self, document_id: DocumentId) -> bool:
        """Check if node was extracted from specific document."""
        return document_id in self.source_documents

    def get_document_count(self) -> int:
        """Return number of connected documents."""
        return len(self.source_documents)

    def merge_with(self, other: "Node") -> None:
        """Merge with another node (remove duplicates)."""
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

        # Update if there's a better embedding
        if not self.has_embedding() and other.has_embedding():
            self.embedding = other.embedding
            self.embedding_model = other.embedding_model
            self.embedding_created_at = other.embedding_created_at

        self.updated_at = datetime.now()

    def __str__(self) -> str:
        return f"Node(id={self.id}, name='{self.name}', type={self.node_type.value})"

    def __repr__(self) -> str:
        return (
            f"Node(id={self.id!r}, name={self.name!r}, "
            f"node_type={self.node_type.value}, "
            f"source_docs={len(self.source_documents)})"
        )
