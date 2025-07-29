"""
HNSW (Hierarchical Navigable Small World) index for vector similarity search.
Uses hnswlib backend for fast approximate nearest neighbor search.
"""

import pickle
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import hnswlib
import numpy as np

from src.common.observability import get_observable_logger

from .embeddings import EmbeddingManager


class HNSWBackend(Enum):
    """Available HNSW backends."""

    HNSWLIB = "hnswlib"


class HNSWIndex:
    """
    HNSW index for fast approximate nearest neighbor search.
    Uses hnswlib backend for efficient vector similarity search.
    """

    def __init__(
        self,
        space: str = "cosine",
        dim: int = 128,
        ef_construction: int = 200,
        m_parameter: int = 16,
        index_dir: Optional[Union[str, Path]] = None,
        backend: Union[str, HNSWBackend] = HNSWBackend.HNSWLIB,
    ):
        """
        Initialize the HNSW index.

        Args:
            space: Distance metric ('cosine', 'ip' for inner product, or 'l2')
            dim: Vector dimension
            ef_construction: Controls quality/speed trade-off at index construction
            m_parameter: Parameter controlling index graph connectivity
            index_dir: Directory to save/load index files (None for memory-only)
            backend: Backend to use (only 'hnswlib' supported)
        """
        self.space = space
        self.dim = dim
        self.ef_construction = ef_construction
        self.m_parameter = m_parameter
        self.index_dir = Path(index_dir) if index_dir else None

        # Handle backend selection
        if isinstance(backend, str):
            backend = HNSWBackend(backend.lower())
        self.backend = backend

        # Initialize the backend-specific index
        self.index: hnswlib.Index
        self._init_backend()

        # Maps from SQLite ID to index ID
        self.id_to_idx: Dict[Tuple[str, int], int] = {}
        self.idx_to_id: Dict[int, Tuple[str, int]] = {}

        # Track the current size
        self.current_size = 0
        self.current_capacity = 0
        self.is_initialized = False

    def _init_backend(self):
        """Initialize the backend-specific index."""
        self.index = hnswlib.Index(space=self.space, dim=self.dim)

    def init_index(self, max_elements: int = 1000) -> None:
        """
        Initialize a new index with the specified capacity.

        Args:
            max_elements: Maximum number of elements in the index
        """
        if self.index is not None:
            self.index.init_index(
                max_elements=max_elements,
                ef_construction=self.ef_construction,
                M=self.m_parameter,
            )
            # Set the search parameter
            self.index.set_ef(max(self.ef_construction, 100))

        self.current_capacity = max_elements
        self.current_size = 0
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.is_initialized = True

    def load_index(self, filename: Optional[str] = None) -> None:
        """
        Load a previously saved index and mappings.

        Args:
            filename: Base filename without extension (uses index_dir if provided)
        """
        if self.index_dir is None and filename is None:
            raise ValueError("Either index_dir or filename must be provided")

        if filename is None:
            # Use default filenames based on parameters
            base_name = f"hnsw_{self.backend.value}_{self.space}_{self.dim}_{self.m_parameter}"
        else:
            base_name = filename

        # Determine file paths
        if self.index_dir:
            index_path = self.index_dir / f"{base_name}.bin"
            mapping_path = self.index_dir / f"{base_name}_mapping.pkl"
        else:
            index_path = Path(f"{base_name}.bin")
            mapping_path = Path(f"{base_name}_mapping.pkl")

        # Load the index
        if index_path.exists() and self.index is not None:
            self.index.load_index(str(index_path))
            self.current_capacity = self.index.get_max_elements()
            self.current_size = self.index.get_current_count()
            self.is_initialized = True

            # Load ID mappings
            if mapping_path.exists():
                with open(mapping_path, "rb") as file_handle:
                    mappings = pickle.load(file_handle)
                    self.id_to_idx = mappings["id_to_idx"]
                    self.idx_to_id = mappings["idx_to_id"]
            else:
                raise FileNotFoundError(f"Index mappings file not found: {mapping_path}")
        else:
            raise FileNotFoundError(f"Index file not found: {index_path}")

    def save_index(self, filename: Optional[str] = None) -> None:
        """
        Save the index and ID mappings to disk.

        Args:
            filename: Base filename without extension (uses index_dir if provided)
        """
        if not self.is_initialized:
            raise RuntimeError("Index is not initialized")

        if self.index_dir is None and filename is None:
            raise ValueError("Either index_dir or filename must be provided")

        if filename is None:
            # Use default filenames based on parameters
            base_name = f"hnsw_{self.backend.value}_{self.space}_{self.dim}_{self.m_parameter}"
        else:
            base_name = filename

        # Determine file paths
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            index_path = self.index_dir / f"{base_name}.bin"
            mapping_path = self.index_dir / f"{base_name}_mapping.pkl"
        else:
            index_path = Path(f"{base_name}.bin")
            mapping_path = Path(f"{base_name}_mapping.pkl")

        # Save the index
        if self.index is not None:
            self.index.save_index(str(index_path))

        # Save ID mappings
        mappings = {"id_to_idx": self.id_to_idx, "idx_to_id": self.idx_to_id}
        with open(mapping_path, "wb") as file_handle:
            pickle.dump(mappings, file_handle)

    def resize_index(self, new_size: int) -> None:
        """
        Resize the index to accommodate more elements.

        Args:
            new_size: New maximum capacity
        """
        if new_size <= self.current_capacity:
            return

        if self.index is not None:
            self.index.resize_index(new_size)
        self.current_capacity = new_size

    def add_item(
        self,
        entity_type: str,
        entity_id: int,
        vector: np.ndarray,
        replace_existing: bool = True,
    ) -> int:
        """
        Add an item to the index.

        Args:
            entity_type: Type of entity ('node', 'edge', or 'hyperedge')
            entity_id: ID of the entity
            vector: Embedding vector
            replace_existing: Whether to replace if the item already exists

        Returns:
            Index ID of the added item
        """
        if not self.is_initialized:
            raise RuntimeError("Index is not initialized")

        item_key = (entity_type, entity_id)

        # Check if item already exists
        if item_key in self.id_to_idx:
            if replace_existing:
                # Remove old item first
                self.remove_item(entity_type, entity_id)
            else:
                # Return existing index
                return self.id_to_idx[item_key]

        # Check if we need to resize
        if self.current_size >= self.current_capacity:
            # Resize to double the capacity
            new_capacity = max(1000, self.current_capacity * 2)
            self.resize_index(new_capacity)

        # Prepare vector
        vector = vector.astype(np.float32)  # Ensure correct data type

        idx = self.current_size
        if self.index is not None:
            self.index.add_items(vector, [idx])

        # Update mappings
        self.id_to_idx[item_key] = idx
        self.idx_to_id[idx] = item_key
        self.current_size += 1

        return idx

    def add_items_batch(
        self,
        entity_types: List[str],
        entity_ids: List[int],
        vectors: np.ndarray,
        replace_existing: bool = True,
    ) -> List[int]:
        """
        Add multiple items to the index efficiently using batch operations.

        Args:
            entity_types: List of entity types
            entity_ids: List of entity IDs
            vectors: 2D numpy array of vectors (n_vectors x dimension)
            replace_existing: Whether to replace if items already exist

        Returns:
            List of index IDs of the added items
        """
        if not self.is_initialized:
            raise RuntimeError("Index is not initialized")

        if len(entity_types) != len(entity_ids) or len(entity_types) != len(vectors):
            raise ValueError("entity_types, entity_ids, and vectors must have the same length")

        if len(vectors) == 0:
            return []

        # Prepare data
        vectors = vectors.astype(np.float32)
        item_keys = [
            (entity_type, entity_id) for entity_type, entity_id in zip(entity_types, entity_ids)
        ]

        # Filter out existing items if not replacing
        if not replace_existing:
            new_indices = []
            new_vectors_list = []
            new_item_keys = []

            for i, item_key in enumerate(item_keys):
                if item_key not in self.id_to_idx:
                    new_indices.append(i)
                    new_vectors_list.append(vectors[i])
                    new_item_keys.append(item_key)

            if not new_vectors_list:
                # All items already exist
                return [self.id_to_idx[key] for key in item_keys]

            vectors = np.array(new_vectors_list)
            item_keys = new_item_keys
        else:
            # Remove existing items first
            for item_key in item_keys:
                if item_key in self.id_to_idx:
                    entity_type, entity_id = item_key
                    self.remove_item(entity_type, entity_id)

        n_items = len(vectors)

        # Check if we need to resize
        if self.current_size + n_items > self.current_capacity:
            new_capacity = max(self.current_capacity * 2, self.current_size + n_items + 1000)
            self.resize_index(new_capacity)

        # Generate new indices
        start_idx = self.current_size
        indices = list(range(start_idx, start_idx + n_items))

        # Batch add to index
        if self.index is not None:
            self.index.add_items(vectors, indices)

        # Batch update mappings
        for i, (item_key, idx) in enumerate(zip(item_keys, indices)):
            self.id_to_idx[item_key] = idx
            self.idx_to_id[idx] = item_key

        self.current_size += n_items
        return indices

    def remove_item(self, entity_type: str, entity_id: int) -> bool:
        """
        Remove an item from the index.

        Args:
            entity_type: Type of entity
            entity_id: ID of the entity

        Returns:
            True if item was removed, False if it wasn't found
        """
        if not self.is_initialized:
            raise RuntimeError("Index is not initialized")

        item_key = (entity_type, entity_id)

        if item_key in self.id_to_idx:
            idx = self.id_to_idx[item_key]

            if self.index is not None:
                self.index.mark_deleted(idx)

            # Update mappings
            del self.id_to_idx[item_key]
            del self.idx_to_id[idx]

            return True

        return False

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
        filter_entity_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, int, float]]:
        """
        Search for the nearest vectors to the query vector.

        Args:
            query_vector: Query embedding vector
            k: Number of nearest neighbors to retrieve
            ef_search: Runtime parameter controlling query accuracy/speed trade-off
            filter_entity_types: List of entity types to include in results

        Returns:
            List of (entity_type, entity_id, distance) tuples
        """
        if not self.is_initialized:
            raise RuntimeError("Index is not initialized")

        # Handle empty index
        if self.current_size == 0:
            return []

        # Set search parameter if provided
        if ef_search is not None and self.index is not None:
            self.index.set_ef(ef_search)

        # Convert to correct type
        query_vector = query_vector.astype(np.float32)

        # Adjust k to not exceed index size
        adjusted_k = min(k, self.current_size)
        if adjusted_k <= 0:
            return []

        # Search the index
        if self.index is not None:
            indices, distances = self.index.knn_query(query_vector, k=adjusted_k)
        else:
            return []
        indices, distances = indices[0], distances[0]

        # Process results (optimized with list comprehension and fewer lookups)
        if filter_entity_types:
            # Convert to set for O(1) lookup instead of O(n) list lookup
            filter_set = set(filter_entity_types)
            results = [
                (entity_type, entity_id, float(dist))
                for idx, dist in zip(indices, distances)
                if idx in self.idx_to_id
                and (entity_type := self.idx_to_id[idx][0]) in filter_set
                and (entity_id := self.idx_to_id[idx][1]) is not None
            ]
        else:
            results = [
                (entity_type, entity_id, float(dist))
                for idx, dist in zip(indices, distances)
                if idx in self.idx_to_id
                and (entity_type := self.idx_to_id[idx][0]) is not None
                and (entity_id := self.idx_to_id[idx][1]) is not None
            ]

        return results

    def build_from_embeddings(
        self,
        embedding_manager: EmbeddingManager,
        entity_types: Optional[List[str]] = None,
        model_info: Optional[str] = None,
        batch_size: int = 1000,
    ) -> int:
        """
        Build the index from embeddings in the database.

        Args:
            embedding_manager: EmbeddingManager instance
            entity_types: List of entity types to include, or None for all
            model_info: Filter by model info, or None for all models
            batch_size: Batch size for processing embeddings

        Returns:
            Number of embeddings added to the index
        """
        entity_types = entity_types or ["node", "edge", "hyperedge"]
        total_embeddings = 0

        # Get total embedding count to determine initial capacity
        count_query = """
        SELECT COUNT(*)
        FROM (
            SELECT 1 FROM node_embeddings
            WHERE {model_clause}
            UNION ALL
            SELECT 1 FROM edge_embeddings
            WHERE {model_clause}
            UNION ALL
            SELECT 1 FROM hyperedge_embeddings
            WHERE {model_clause}
        )
        """

        model_clause = "1=1" if model_info is None else "model_info = ?"
        params = [] if model_info is None else [model_info, model_info, model_info]

        cursor = embedding_manager.connection.cursor()
        cursor.execute(count_query.format(model_clause=model_clause), params)
        total_count = cursor.fetchone()[0]

        # Initialize with slightly larger capacity
        init_capacity = max(1000, int(total_count * 1.2))
        self.init_index(max_elements=init_capacity)

        # Process each entity type
        for entity_type in entity_types:
            # Fetch embeddings in batches
            offset = 0

            while True:
                embeddings = embedding_manager.get_all_embeddings(
                    entity_type=entity_type,
                    model_info=model_info,
                    batch_size=batch_size,
                    offset=offset,
                )

                if not embeddings:
                    break

                # Prepare batch data for efficient insertion
                entity_types_batch = [entity_type] * len(embeddings)
                entity_ids_batch = [emb.entity_id for emb in embeddings]

                # Optimized vector batch creation - stack directly instead of list->array conversion
                if embeddings:
                    vectors_batch = np.stack([emb.embedding for emb in embeddings]).astype(
                        np.float32
                    )
                else:
                    vectors_batch = np.array([], dtype=np.float32)

                # Use batch insertion for better performance
                self.add_items_batch(
                    entity_types=entity_types_batch,
                    entity_ids=entity_ids_batch,
                    vectors=vectors_batch,
                    replace_existing=False,  # Avoid checking for replacements during initial build
                )

                total_embeddings += len(embeddings)
                offset += batch_size

                # If we got fewer than batch_size, we're done
                if len(embeddings) < batch_size:
                    break

        return total_embeddings

    def sync_with_outbox(self, embedding_manager: EmbeddingManager, batch_size: int = 100) -> int:
        """
        Process vector operations from the outbox and update the index.

        Args:
            embedding_manager: EmbeddingManager instance
            batch_size: Number of operations to process in one batch

        Returns:
            Number of operations processed
        """
        # Process the outbox first to ensure embeddings are up to date
        embedding_manager.process_outbox(batch_size)

        # Now sync the index with the processed embeddings
        cursor = embedding_manager.connection.cursor()

        # Get completed operations
        cursor.execute(
            """
            SELECT operation_type, entity_type, entity_id
            FROM vector_outbox
            WHERE status = 'completed'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (batch_size,),
        )

        operations = cursor.fetchall()
        sync_count = 0

        for operation in operations:
            operation_type = operation["operation_type"]
            entity_type = operation["entity_type"]
            entity_id = operation["entity_id"]

            try:
                if operation_type == "delete":
                    # Remove from index
                    self.remove_item(entity_type, entity_id)

                elif operation_type in ("insert", "update"):
                    # Get embedding from database
                    embedding = embedding_manager.get_embedding(entity_type, entity_id)

                    if embedding:
                        # Add or update in index
                        self.add_item(
                            entity_type=entity_type,
                            entity_id=entity_id,
                            vector=embedding.embedding,
                            replace_existing=True,
                        )

                sync_count += 1

            except Exception as exception:
                # Log error but continue with other operations
                # Use structured logging instead of print
                logger = get_observable_logger("hnsw_index", "adapter")
                logger.error(
                    "entity_sync_failed",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    error_type=type(exception).__name__,
                    error_message=str(exception),
                )

        return sync_count
