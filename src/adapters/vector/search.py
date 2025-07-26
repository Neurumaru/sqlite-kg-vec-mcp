"""
Vector similarity search functionality.
"""

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .entities import Entity, EntityManager
from .relationships import Relationship, RelationshipManager
from .embeddings import Embedding, EmbeddingManager
from .hnsw import HNSWIndex
from .text_embedder import VectorTextEmbedder, create_embedder


@dataclass
class SearchResult:
    """Represents a vector search result."""

    entity_type: str
    entity_id: int
    distance: float
    entity: Optional[Union[Entity, Relationship]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "distance": self.distance,
        }

        if self.entity:
            if isinstance(self.entity, Entity):
                result["entity"] = {
                    "id": self.entity.id,
                    "uuid": self.entity.uuid,
                    "name": self.entity.name,
                    "type": self.entity.type,
                    "properties": self.entity.properties,
                }
            elif isinstance(self.entity, Relationship):
                result["entity"] = {
                    "id": self.entity.id,
                    "source_id": self.entity.source_id,
                    "target_id": self.entity.target_id,
                    "relation_type": self.entity.relation_type,
                    "properties": self.entity.properties,
                }

                # Include source and target if loaded
                if self.entity.source:
                    result["source"] = {
                        "id": self.entity.source.id,
                        "name": self.entity.source.name,
                        "type": self.entity.source.type,
                    }
                if self.entity.target:
                    result["target"] = {
                        "id": self.entity.target.id,
                        "name": self.entity.target.name,
                        "type": self.entity.target.type,
                    }

        return result


class VectorSearch:
    """
    Vector similarity search functionality using HNSW indexes.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        index_dir: Optional[str] = None,
        embedding_dim: int = 128,
        space: str = "cosine",
        text_embedder: Optional[VectorTextEmbedder] = None,
        embedder_type: str = "sentence-transformers",
        embedder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the vector search functionality.

        Args:
            connection: SQLite database connection
            index_dir: Directory to store HNSW index files
            embedding_dim: Dimension of the embedding vectors
            space: Distance metric ('cosine', 'ip', or 'l2')
            text_embedder: VectorTextEmbedder instance for text-to-vector conversion
            embedder_type: Type of embedder to create if text_embedder is None
            embedder_kwargs: Arguments for embedder creation
        """
        self.connection = connection
        self.embedding_manager = EmbeddingManager(connection)
        self.entity_manager = EntityManager(connection)
        self.relationship_manager = RelationshipManager(connection)

        # Initialize the index
        self.index = HNSWIndex(
            space=space,
            dim=embedding_dim,
            ef_construction=200,
            M=16,
            index_dir=index_dir,
        )

        # Flag to track if index is loaded
        self.index_loaded = False

        # Initialize text embedder
        if text_embedder is not None:
            self.text_embedder = text_embedder
        else:
            embedder_kwargs = embedder_kwargs or {}
            # For random embedder, use the index dimension
            if embedder_type == "random":
                embedder_kwargs.setdefault("dimension", embedding_dim)

            self.text_embedder = create_embedder(embedder_type, **embedder_kwargs)

            # Verify embedder dimension matches index dimension
            if self.text_embedder.dimension != embedding_dim:
                raise ValueError(
                    f"Embedder dimension ({self.text_embedder.dimension}) does not match "
                    f"index dimension ({embedding_dim}). Consider adjusting embedding_dim "
                    f"or using a different model."
                )

    def ensure_index_loaded(self, force_rebuild: bool = False):
        """
        Ensure the index is loaded, building it from scratch if necessary.

        Args:
            force_rebuild: Force rebuilding the index even if already loaded
        """
        if self.index_loaded and not force_rebuild:
            return

        try:
            # Try to load existing index
            if not force_rebuild:
                self.index.load_index()
                self.index_loaded = True
                return
        except (FileNotFoundError, RuntimeError):
            # If loading fails, build from scratch
            pass

        # Build index from all embeddings in the database
        self.index.build_from_embeddings(self.embedding_manager)

        # Save the built index
        self.index.save_index()
        self.index_loaded = True

    def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        entity_types: Optional[List[str]] = None,
        ef_search: Optional[int] = None,
        include_entities: bool = True,
    ) -> List[SearchResult]:
        """
        Search for entities similar to the query vector.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            entity_types: List of entity types to include or None for all
            ef_search: Runtime parameter controlling search quality
            include_entities: Whether to include full entity details

        Returns:
            List of SearchResult objects
        """
        # Ensure index is loaded
        self.ensure_index_loaded()

        # Perform search
        search_results = self.index.search(
            query_vector=query_vector,
            k=k,
            ef_search=ef_search,
            filter_entity_types=entity_types,
        )

        # Convert to SearchResult objects
        results = []

        for entity_type, entity_id, distance in search_results:
            result = SearchResult(
                entity_type=entity_type, entity_id=entity_id, distance=distance
            )

            # Include entity details if requested
            if include_entities:
                if entity_type == "node":
                    result.entity = self.entity_manager.get_entity(entity_id)
                elif entity_type == "edge":
                    result.entity = self.relationship_manager.get_relationship(
                        entity_id, include_entities=True
                    )
                # Note: hyperedge handling would be added here

            results.append(result)

        return results

    def search_similar_to_entity(
        self,
        entity_type: str,
        entity_id: int,
        k: int = 10,
        result_entity_types: Optional[List[str]] = None,
        include_entities: bool = True,
    ) -> List[SearchResult]:
        """
        Search for entities similar to a given entity.

        Args:
            entity_type: Type of the entity ('node', 'edge', or 'hyperedge')
            entity_id: ID of the entity
            k: Number of results to return
            result_entity_types: Types of entities to include in results
            include_entities: Whether to include full entity details

        Returns:
            List of SearchResult objects
        """
        # Get the entity's embedding
        embedding = self.embedding_manager.get_embedding(entity_type, entity_id)

        if not embedding:
            raise ValueError(f"No embedding found for {entity_type} {entity_id}")

        # Perform similarity search using the entity's embedding
        return self.search_similar(
            query_vector=embedding.embedding,
            k=k + 1,  # +1 because the entity itself will be in results
            entity_types=result_entity_types,
            include_entities=include_entities,
        )[
            1:
        ]  # Exclude the first result (the entity itself)

    def build_text_embedding(self, text: str) -> np.ndarray:
        """
        Build an embedding vector for a text query.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.text_embedder.embed(text)

    def search_by_text(
        self,
        query_text: str,
        k: int = 10,
        entity_types: Optional[List[str]] = None,
        include_entities: bool = True,
    ) -> List[SearchResult]:
        """
        Search for entities similar to a text query.

        Args:
            query_text: Text query
            k: Number of results to return
            entity_types: List of entity types to include or None for all
            include_entities: Whether to include full entity details

        Returns:
            List of SearchResult objects
        """
        # Build embedding for the text query
        query_embedding = self.build_text_embedding(query_text)

        # Perform similarity search
        return self.search_similar(
            query_vector=query_embedding,
            k=k,
            entity_types=entity_types,
            include_entities=include_entities,
        )

    def update_index(self, batch_size: int = 100):
        """
        Update the index with any pending changes from the outbox.

        Args:
            batch_size: Number of operations to process at once

        Returns:
            Number of operations processed
        """
        # Ensure index is loaded
        self.ensure_index_loaded()

        # Process pending operations
        count = self.index.sync_with_outbox(self.embedding_manager, batch_size)

        # Save the updated index if changes were made
        if count > 0:
            self.index.save_index()

        return count
