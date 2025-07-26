"""
Repository port for embedding persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ....domain import NodeId, Embedding, Vector


class EmbeddingRepository(ABC):
    """
    Secondary port for embedding persistence operations.

    This interface defines how the domain interacts with embedding storage.
    """

    @abstractmethod
    async def save(self, embedding: Embedding) -> Embedding:
        """
        Save an embedding to persistent storage.

        Args:
            embedding: Embedding to save

        Returns:
            Saved embedding with updated metadata
        """
        pass

    @abstractmethod
    async def find_by_id(self, embedding_id: NodeId) -> Optional[Embedding]:
        """
        Find an embedding by its ID.

        Args:
            embedding_id: ID of the embedding to find

        Returns:
            Embedding if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_entity_id(self, entity_id: NodeId) -> Optional[Embedding]:
        """
        Find an embedding by entity ID.

        Args:
            entity_id: ID of the entity

        Returns:
            Embedding if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_content_hash(self, content_hash: str) -> Optional[Embedding]:
        """
        Find an embedding by content hash.

        Args:
            content_hash: Hash of the content

        Returns:
            Embedding if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_similar(
        self,
        query_vector: Vector,
        limit: int = 10,
        similarity_threshold: float = 0.0,
        entity_ids: Optional[List[NodeId]] = None
    ) -> List[Embedding]:
        """
        Find similar embeddings by vector similarity.

        Args:
            query_vector: Vector to search for
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            entity_ids: Optional filter by entity IDs

        Returns:
            List of similar embeddings ordered by similarity
        """
        pass

    @abstractmethod
    async def find_by_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Embedding]:
        """
        Find embeddings by model information.

        Args:
            model_name: Name of the embedding model
            model_version: Optional model version
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of embeddings
        """
        pass

    @abstractmethod
    async def update(self, embedding: Embedding) -> Embedding:
        """
        Update an existing embedding.

        Args:
            embedding: Embedding with updated data

        Returns:
            Updated embedding
        """
        pass

    @abstractmethod
    async def delete(self, embedding_id: NodeId) -> bool:
        """
        Delete an embedding from storage.

        Args:
            embedding_id: ID of the embedding to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def delete_by_entity_id(self, entity_id: NodeId) -> bool:
        """
        Delete embedding by entity ID.

        Args:
            entity_id: ID of the entity

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def exists(self, embedding_id: NodeId) -> bool:
        """
        Check if an embedding exists.

        Args:
            embedding_id: ID of the embedding to check

        Returns:
            True if embedding exists
        """
        pass

    @abstractmethod
    async def exists_for_entity(self, entity_id: NodeId) -> bool:
        """
        Check if an embedding exists for an entity.

        Args:
            entity_id: ID of the entity

        Returns:
            True if embedding exists for the entity
        """
        pass

    @abstractmethod
    async def count(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> int:
        """
        Count embeddings matching criteria.

        Args:
            model_name: Optional filter by model name
            model_version: Optional filter by model version

        Returns:
            Number of matching embeddings
        """
        pass

    @abstractmethod
    async def bulk_save(self, embeddings: List[Embedding]) -> List[Embedding]:
        """
        Save multiple embeddings in batch.

        Args:
            embeddings: List of embeddings to save

        Returns:
            List of saved embeddings
        """
        pass

    @abstractmethod
    async def bulk_delete(self, embedding_ids: List[NodeId]) -> int:
        """
        Delete multiple embeddings in batch.

        Args:
            embedding_ids: List of embedding IDs to delete

        Returns:
            Number of embeddings deleted
        """
        pass

    @abstractmethod
    async def get_outdated_embeddings(
        self,
        current_model: str,
        current_version: str,
        limit: int = 100
    ) -> List[Embedding]:
        """
        Get embeddings that need to be updated due to model changes.

        Args:
            current_model: Current model name
            current_version: Current model version
            limit: Maximum number of results

        Returns:
            List of outdated embeddings
        """
        pass
