"""
Repository port for vector index operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from ....domain import NodeId, Vector


class VectorIndexRepository(ABC):
    """
    Secondary port for vector index operations.

    This interface defines how the domain interacts with vector indexing systems.
    """

    @abstractmethod
    async def build_index(
        self,
        vectors: Dict[NodeId, Vector],
        index_type: str = "hnsw",
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build a vector index from vectors.

        Args:
            vectors: Dictionary mapping entity IDs to vectors
            index_type: Type of index to build (hnsw, flat, etc.)
            parameters: Optional index parameters

        Returns:
            Index build results and metadata
        """
        pass

    @abstractmethod
    async def add_vector(self, entity_id: NodeId, vector: Vector) -> bool:
        """
        Add a vector to the index.

        Args:
            entity_id: ID of the entity
            vector: Vector to add

        Returns:
            True if addition was successful
        """
        pass

    @abstractmethod
    async def update_vector(self, entity_id: NodeId, vector: Vector) -> bool:
        """
        Update a vector in the index.

        Args:
            entity_id: ID of the entity
            vector: Updated vector

        Returns:
            True if update was successful
        """
        pass

    @abstractmethod
    async def remove_vector(self, entity_id: NodeId) -> bool:
        """
        Remove a vector from the index.

        Args:
            entity_id: ID of the entity

        Returns:
            True if removal was successful
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: Vector,
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[Tuple[NodeId, float]]:
        """
        Search for similar vectors in the index.

        Args:
            query_vector: Vector to search for
            k: Number of nearest neighbors to return
            ef: Search parameter for HNSW (exploration factor)

        Returns:
            List of (entity_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    async def batch_search(
        self,
        query_vectors: List[Vector],
        k: int = 10,
        ef: Optional[int] = None
    ) -> List[List[Tuple[NodeId, float]]]:
        """
        Perform batch search for multiple query vectors.

        Args:
            query_vectors: List of vectors to search for
            k: Number of nearest neighbors per query
            ef: Search parameter for HNSW

        Returns:
            List of search results for each query
        """
        pass

    @abstractmethod
    async def search_with_filter(
        self,
        query_vector: Vector,
        entity_ids: List[NodeId],
        k: int = 10
    ) -> List[Tuple[NodeId, float]]:
        """
        Search with entity ID filtering.

        Args:
            query_vector: Vector to search for
            entity_ids: List of entity IDs to search within
            k: Number of results to return

        Returns:
            List of (entity_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    async def get_vector(self, entity_id: NodeId) -> Optional[Vector]:
        """
        Get a vector from the index by entity ID.

        Args:
            entity_id: ID of the entity

        Returns:
            Vector if found, None otherwise
        """
        pass

    @abstractmethod
    async def contains(self, entity_id: NodeId) -> bool:
        """
        Check if the index contains a vector for the entity.

        Args:
            entity_id: ID of the entity

        Returns:
            True if vector exists in index
        """
        pass

    @abstractmethod
    async def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.

        Returns:
            Index information and statistics
        """
        pass

    @abstractmethod
    async def get_index_size(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            Number of vectors in the index
        """
        pass

    @abstractmethod
    async def rebuild_index(
        self,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Rebuild the entire index.

        Args:
            parameters: Optional rebuild parameters

        Returns:
            Rebuild results and statistics
        """
        pass

    @abstractmethod
    async def optimize_index(self) -> Dict[str, Any]:
        """
        Optimize the index for better performance.

        Returns:
            Optimization results
        """
        pass

    @abstractmethod
    async def save_index(self, file_path: str) -> bool:
        """
        Save the index to disk.

        Args:
            file_path: Path to save the index

        Returns:
            True if save was successful
        """
        pass

    @abstractmethod
    async def load_index(self, file_path: str) -> bool:
        """
        Load an index from disk.

        Args:
            file_path: Path to load the index from

        Returns:
            True if load was successful
        """
        pass

    @abstractmethod
    async def clear_index(self) -> bool:
        """
        Clear all vectors from the index.

        Returns:
            True if clear was successful
        """
        pass

    @abstractmethod
    async def validate_index(self) -> Dict[str, Any]:
        """
        Validate index integrity and consistency.

        Returns:
            Validation results
        """
        pass
