"""
HNSW (Hierarchical Navigable Small World) index for vector similarity search.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import hnswlib
import numpy as np

from .embeddings import Embedding, EmbeddingManager


class HNSWIndex:
    """
    HNSW index for fast approximate nearest neighbor search.
    """

    def __init__(
        self,
        space: str = "cosine",
        dim: int = 128,
        ef_construction: int = 200,
        M: int = 16,
        index_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the HNSW index.

        Args:
            space: Distance metric ('cosine', 'ip' for inner product, or 'l2')
            dim: Vector dimension
            ef_construction: Controls quality/speed trade-off at index construction
            M: Parameter controlling index graph connectivity
            index_dir: Directory to save/load index files (None for memory-only)
        """
        self.space = space
        self.dim = dim
        self.ef_construction = ef_construction
        self.M = M
        self.index_dir = Path(index_dir) if index_dir else None

        # Initialize the index with parameters
        self.index = hnswlib.Index(space=space, dim=dim)

        # Maps from SQLite ID to index ID
        self.id_to_idx: Dict[Tuple[str, int], int] = {}
        self.idx_to_id: Dict[int, Tuple[str, int]] = {}

        # Track the current size
        self.current_size = 0
        self.current_capacity = 0
        self.is_initialized = False

    def init_index(self, max_elements: int = 1000) -> None:
        """
        Initialize a new index with the specified capacity.

        Args:
            max_elements: Maximum number of elements in the index
        """
        self.index.init_index(
            max_elements=max_elements, ef_construction=self.ef_construction, M=self.M
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
            base_name = f"hnsw_{self.space}_{self.dim}_{self.M}"
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
        if index_path.exists():
            self.index.load_index(str(index_path))
            self.current_capacity = self.index.get_max_elements()
            self.current_size = self.index.get_current_count()
            self.is_initialized = True

            # Load ID mappings
            if mapping_path.exists():
                with open(mapping_path, "rb") as f:
                    mappings = pickle.load(f)
                    self.id_to_idx = mappings["id_to_idx"]
                    self.idx_to_id = mappings["idx_to_id"]
            else:
                raise FileNotFoundError(
                    f"Index mappings file not found: {mapping_path}"
                )
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
            base_name = f"hnsw_{self.space}_{self.dim}_{self.M}"
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
        self.index.save_index(str(index_path))

        # Save ID mappings
        mappings = {"id_to_idx": self.id_to_idx, "idx_to_id": self.idx_to_id}
        with open(mapping_path, "wb") as f:
            pickle.dump(mappings, f)

    def resize_index(self, new_size: int) -> None:
        """
        Resize the index to accommodate more elements.

        Args:
            new_size: New maximum capacity
        """
        if new_size <= self.current_capacity:
            return

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
                # Replace existing item
                idx = self.id_to_idx[item_key]
                self.index.mark_deleted(idx)  # Mark old vector as deleted
                vector = vector.astype(np.float32)  # Ensure correct data type
                self.index.add_items(vector, [idx])  # Replace with new vector
                return idx
            else:
                # Return existing index
                return self.id_to_idx[item_key]

        # Check if we need to resize
        if self.current_size >= self.current_capacity:
            # Resize to double the capacity
            new_capacity = max(1000, self.current_capacity * 2)
            self.resize_index(new_capacity)

        # Add new item
        vector = vector.astype(np.float32)  # Ensure correct data type
        idx = self.current_size
        self.index.add_items(vector, [idx])

        # Update mappings
        self.id_to_idx[item_key] = idx
        self.idx_to_id[idx] = item_key
        self.current_size += 1

        return idx

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
        if ef_search is not None:
            self.index.set_ef(ef_search)

        # Convert to correct type
        query_vector = query_vector.astype(np.float32)

        # Adjust k to not exceed index size
        adjusted_k = min(k, self.current_size)
        if adjusted_k <= 0:
            return []

        # Search the index
        indices, distances = self.index.knn_query(query_vector, k=adjusted_k)

        # Process results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx in self.idx_to_id:
                entity_type, entity_id = self.idx_to_id[idx]

                # Apply entity type filter if provided
                if filter_entity_types and entity_type not in filter_entity_types:
                    continue

                results.append((entity_type, entity_id, float(dist)))

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
                )

                if not embeddings:
                    break

                # Add each embedding to the index
                for emb in embeddings:
                    self.add_item(
                        entity_type=entity_type,
                        entity_id=emb.entity_id,
                        vector=emb.embedding,
                    )

                total_embeddings += len(embeddings)
                offset += batch_size

                # If we got fewer than batch_size, we're done
                if len(embeddings) < batch_size:
                    break

        return total_embeddings

    def sync_with_outbox(
        self, embedding_manager: EmbeddingManager, batch_size: int = 100
    ) -> int:
        """
        Process vector operations from the outbox and update the index.

        Args:
            embedding_manager: EmbeddingManager instance
            batch_size: Number of operations to process in one batch

        Returns:
            Number of operations processed
        """
        # Process the outbox first to ensure embeddings are up to date
        processed_count = embedding_manager.process_outbox(batch_size)

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

        for op in operations:
            operation_type = op["operation_type"]
            entity_type = op["entity_type"]
            entity_id = op["entity_id"]

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

            except Exception as e:
                # Log error but continue with other operations
                print(f"Error syncing {entity_type} {entity_id}: {e}")

        return sync_count
