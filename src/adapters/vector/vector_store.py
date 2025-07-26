"""
Vector store infrastructure port for vector operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

from ...domain import Vector


class VectorStore(ABC):
    """
    Secondary port for vector store infrastructure operations.

    This interface defines how the domain interacts with vector storage systems
    for high-performance vector operations and similarity search.
    """

    # Store management
    @abstractmethod
    async def initialize_store(
        self,
        dimension: int,
        metric: str = "cosine",
        parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Initialize the vector store.

        Args:
            dimension: Vector dimension
            metric: Distance metric ("cosine", "euclidean", "dot_product")
            parameters: Optional store parameters

        Returns:
            True if initialization was successful
        """
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the vector store.

        Returns:
            True if connection was successful
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the vector store.

        Returns:
            True if disconnection was successful
        """
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """
        Check if connected to the vector store.

        Returns:
            True if connected
        """
        pass

    # Vector operations
    @abstractmethod
    async def add_vector(
        self,
        vector_id: str,
        vector: Vector,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a vector to the store.

        Args:
            vector_id: Unique identifier for the vector
            vector: Vector data
            metadata: Optional metadata

        Returns:
            True if addition was successful
        """
        pass

    @abstractmethod
    async def add_vectors(
        self,
        vectors: Dict[str, Vector],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> bool:
        """
        Add multiple vectors to the store in batch.

        Args:
            vectors: Dictionary mapping vector IDs to vectors
            metadata: Optional metadata for each vector

        Returns:
            True if batch addition was successful
        """
        pass

    @abstractmethod
    async def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        Retrieve a vector by ID.

        Args:
            vector_id: Vector identifier

        Returns:
            Vector if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_vectors(self, vector_ids: List[str]) -> Dict[str, Optional[Vector]]:
        """
        Retrieve multiple vectors by IDs.

        Args:
            vector_ids: List of vector identifiers

        Returns:
            Dictionary mapping vector IDs to vectors (None if not found)
        """
        pass

    @abstractmethod
    async def update_vector(
        self,
        vector_id: str,
        vector: Vector,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing vector.

        Args:
            vector_id: Vector identifier
            vector: New vector data
            metadata: Optional new metadata

        Returns:
            True if update was successful
        """
        pass

    @abstractmethod
    async def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.

        Args:
            vector_id: Vector identifier

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    async def delete_vectors(self, vector_ids: List[str]) -> int:
        """
        Delete multiple vectors from the store.

        Args:
            vector_ids: List of vector identifiers

        Returns:
            Number of vectors successfully deleted
        """
        pass

    @abstractmethod
    async def vector_exists(self, vector_id: str) -> bool:
        """
        Check if a vector exists in the store.

        Args:
            vector_id: Vector identifier

        Returns:
            True if vector exists
        """
        pass

    # Search operations
    @abstractmethod
    async def search_similar(
        self,
        query_vector: Vector,
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_criteria: Optional filter criteria

        Returns:
            List of (vector_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    async def search_similar_with_vectors(
        self,
        query_vector: Vector,
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, Vector, float]]:
        """
        Search for similar vectors and return the vectors themselves.

        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_criteria: Optional filter criteria

        Returns:
            List of (vector_id, vector, similarity_score) tuples
        """
        pass

    @abstractmethod
    async def search_by_ids(
        self,
        query_vector: Vector,
        candidate_ids: List[str],
        k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Search within a specific set of vector IDs.

        Args:
            query_vector: Query vector
            candidate_ids: List of candidate vector IDs
            k: Optional limit on results (defaults to all candidates)

        Returns:
            List of (vector_id, similarity_score) tuples
        """
        pass

    @abstractmethod
    async def batch_search(
        self,
        query_vectors: List[Vector],
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[List[Tuple[str, float]]]:
        """
        Perform batch search for multiple query vectors.

        Args:
            query_vectors: List of query vectors
            k: Number of results per query
            filter_criteria: Optional filter criteria

        Returns:
            List of search results for each query
        """
        pass

    # Metadata operations
    @abstractmethod
    async def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a vector.

        Args:
            vector_id: Vector identifier

        Returns:
            Metadata dictionary if found, None otherwise
        """
        pass

    @abstractmethod
    async def update_metadata(
        self,
        vector_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a vector.

        Args:
            vector_id: Vector identifier
            metadata: New metadata

        Returns:
            True if update was successful
        """
        pass

    @abstractmethod
    async def search_by_metadata(
        self,
        filter_criteria: Dict[str, Any],
        limit: int = 100
    ) -> List[str]:
        """
        Search vectors by metadata criteria.

        Args:
            filter_criteria: Metadata filter criteria
            limit: Maximum number of results

        Returns:
            List of vector IDs matching the criteria
        """
        pass

    # Store information and maintenance
    @abstractmethod
    async def get_store_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store.

        Returns:
            Store information including size, dimension, etc.
        """
        pass

    @abstractmethod
    async def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the store.

        Returns:
            Number of vectors
        """
        pass

    @abstractmethod
    async def get_dimension(self) -> int:
        """
        Get the vector dimension of the store.

        Returns:
            Vector dimension
        """
        pass

    @abstractmethod
    async def optimize_store(self) -> Dict[str, Any]:
        """
        Optimize the vector store for better performance.

        Returns:
            Optimization results
        """
        pass

    @abstractmethod
    async def rebuild_index(self, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Rebuild the vector index.

        Args:
            parameters: Optional rebuild parameters

        Returns:
            True if rebuild was successful
        """
        pass

    @abstractmethod
    async def clear_store(self) -> bool:
        """
        Clear all vectors from the store.

        Returns:
            True if clearing was successful
        """
        pass

    # Backup and recovery
    @abstractmethod
    async def create_snapshot(self, snapshot_path: str) -> bool:
        """
        Create a snapshot of the vector store.

        Args:
            snapshot_path: Path to save the snapshot

        Returns:
            True if snapshot creation was successful
        """
        pass

    @abstractmethod
    async def restore_snapshot(self, snapshot_path: str) -> bool:
        """
        Restore the vector store from a snapshot.

        Args:
            snapshot_path: Path to the snapshot file

        Returns:
            True if restoration was successful
        """
        pass

    # Health and diagnostics
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector store.

        Returns:
            Health status information
        """
        pass

    @abstractmethod
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the vector store.

        Returns:
            Performance statistics
        """
        pass
