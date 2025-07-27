"""
SQLite implementation of the VectorStore port using sqlite-vec extension.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.domain import Vector
from src.ports.vector_store import VectorStore

from .connection import DatabaseConnection


class SQLiteVectorStore(VectorStore):
    """
    SQLite implementation of the VectorStore port.

    This adapter provides concrete implementation of vector operations
    using SQLite with the sqlite-vec extension for vector storage and search.
    """

    def __init__(
        self, db_path: str, table_name: str = "vectors", optimize: bool = True
    ):
        """
        Initialize SQLite vector store adapter.

        Args:
            db_path: Path to the SQLite database file
            table_name: Name of the table to store vectors
            optimize: Whether to apply optimization PRAGMAs
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.optimize = optimize
        self._connection_manager = DatabaseConnection(db_path, optimize)
        self._connection: Optional[sqlite3.Connection] = None
        self._dimension: Optional[int] = None
        self._metric: str = "cosine"

    # Store management
    async def initialize_store(
        self,
        dimension: int,
        metric: str = "cosine",
        parameters: Optional[Dict[str, Any]] = None,
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
        try:
            if not self._connection:
                await self.connect()

            self._dimension = dimension
            self._metric = metric

            # Create vectors table if it doesn't exist
            cursor = self._connection.cursor()

            # Create the main vectors table
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create metadata index
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata 
                ON {self.table_name} (metadata)
            """
            )

            # Store configuration in a metadata table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_store_config (
                    table_name TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    parameters TEXT
                )
            """
            )

            # Insert or update configuration
            config_data = json.dumps(parameters) if parameters else None
            cursor.execute(
                """
                INSERT OR REPLACE INTO vector_store_config 
                (table_name, dimension, metric, parameters)
                VALUES (?, ?, ?, ?)
            """,
                (self.table_name, dimension, metric, config_data),
            )

            self._connection.commit()
            cursor.close()

            return True
        except Exception:
            return False

    async def connect(self) -> bool:
        """
        Connect to the vector store.

        Returns:
            True if connection was successful
        """
        try:
            self._connection = self._connection_manager.connect()

            # Load configuration if exists
            await self._load_config()

            return True
        except Exception:
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from the vector store.

        Returns:
            True if disconnection was successful
        """
        try:
            self._connection_manager.close()
            self._connection = None
            return True
        except Exception:
            return False

    async def is_connected(self) -> bool:
        """
        Check if connected to the vector store.

        Returns:
            True if connected
        """
        if not self._connection:
            return False

        try:
            self._connection.execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    # Vector operations
    async def add_vector(
        self, vector_id: str, vector: Vector, metadata: Optional[Dict[str, Any]] = None
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
        try:
            if not self._connection:
                return False

            cursor = self._connection.cursor()

            # Convert vector to blob
            vector_blob = self._vector_to_blob(vector)
            metadata_json = json.dumps(metadata) if metadata else None

            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {self.table_name} 
                (id, vector, metadata, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (vector_id, vector_blob, metadata_json),
            )

            self._connection.commit()
            cursor.close()

            return True
        except Exception:
            return False

    async def add_vectors(
        self,
        vectors: Dict[str, Vector],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> bool:
        """
        Add multiple vectors to the store in batch.

        Args:
            vectors: Dictionary mapping vector IDs to vectors
            metadata: Optional metadata for each vector

        Returns:
            True if batch addition was successful
        """
        try:
            if not self._connection:
                return False

            cursor = self._connection.cursor()

            # Prepare batch data
            batch_data = []
            for vector_id, vector in vectors.items():
                vector_blob = self._vector_to_blob(vector)
                vector_metadata = metadata.get(vector_id) if metadata else None
                metadata_json = json.dumps(vector_metadata) if vector_metadata else None
                batch_data.append((vector_id, vector_blob, metadata_json))

            cursor.executemany(
                f"""
                INSERT OR REPLACE INTO {self.table_name} 
                (id, vector, metadata, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
                batch_data,
            )

            self._connection.commit()
            cursor.close()

            return True
        except Exception:
            return False

    async def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        Retrieve a vector by ID.

        Args:
            vector_id: Vector identifier

        Returns:
            Vector if found, None otherwise
        """
        try:
            if not self._connection:
                return None

            cursor = self._connection.cursor()
            cursor.execute(
                f"""
                SELECT vector FROM {self.table_name} WHERE id = ?
            """,
                (vector_id,),
            )

            row = cursor.fetchone()
            cursor.close()

            if row:
                return self._blob_to_vector(row[0])
            return None
        except Exception:
            return None

    async def get_vectors(self, vector_ids: List[str]) -> Dict[str, Optional[Vector]]:
        """
        Retrieve multiple vectors by IDs.

        Args:
            vector_ids: List of vector identifiers

        Returns:
            Dictionary mapping vector IDs to vectors (None if not found)
        """
        try:
            if not self._connection:
                return {vid: None for vid in vector_ids}

            cursor = self._connection.cursor()

            # Create placeholders for IN clause
            placeholders = ", ".join("?" * len(vector_ids))
            cursor.execute(
                f"""
                SELECT id, vector FROM {self.table_name} 
                WHERE id IN ({placeholders})
            """,
                vector_ids,
            )

            rows = cursor.fetchall()
            cursor.close()

            # Build result dictionary
            result = {vid: None for vid in vector_ids}
            for row in rows:
                vector_id, vector_blob = row
                result[vector_id] = self._blob_to_vector(vector_blob)

            return result
        except Exception:
            return {vid: None for vid in vector_ids}

    async def update_vector(
        self, vector_id: str, vector: Vector, metadata: Optional[Dict[str, Any]] = None
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
        # For SQLite, update is the same as add with REPLACE
        return await self.add_vector(vector_id, vector, metadata)

    async def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from the store.

        Args:
            vector_id: Vector identifier

        Returns:
            True if deletion was successful
        """
        try:
            if not self._connection:
                return False

            cursor = self._connection.cursor()
            cursor.execute(
                f"""
                DELETE FROM {self.table_name} WHERE id = ?
            """,
                (vector_id,),
            )

            success = cursor.rowcount > 0
            self._connection.commit()
            cursor.close()

            return success
        except Exception:
            return False

    async def delete_vectors(self, vector_ids: List[str]) -> int:
        """
        Delete multiple vectors from the store.

        Args:
            vector_ids: List of vector identifiers

        Returns:
            Number of vectors successfully deleted
        """
        try:
            if not self._connection:
                return 0

            cursor = self._connection.cursor()

            placeholders = ", ".join("?" * len(vector_ids))
            cursor.execute(
                f"""
                DELETE FROM {self.table_name} WHERE id IN ({placeholders})
            """,
                vector_ids,
            )

            deleted_count = cursor.rowcount
            self._connection.commit()
            cursor.close()

            return deleted_count
        except Exception:
            return 0

    async def vector_exists(self, vector_id: str) -> bool:
        """
        Check if a vector exists in the store.

        Args:
            vector_id: Vector identifier

        Returns:
            True if vector exists
        """
        try:
            if not self._connection:
                return False

            cursor = self._connection.cursor()
            cursor.execute(
                f"""
                SELECT 1 FROM {self.table_name} WHERE id = ? LIMIT 1
            """,
                (vector_id,),
            )

            exists = cursor.fetchone() is not None
            cursor.close()

            return exists
        except Exception:
            return False

    # Search operations (basic implementations without sqlite-vec)
    async def search_similar(
        self,
        query_vector: Vector,
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors using basic similarity calculation.

        Note: This is a basic implementation without sqlite-vec extension.
        For production use, consider using sqlite-vec for better performance.

        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_criteria: Optional filter criteria

        Returns:
            List of (vector_id, similarity_score) tuples
        """
        try:
            if not self._connection:
                return []

            cursor = self._connection.cursor()

            # Build WHERE clause for filters
            where_clause = ""
            params = []
            if filter_criteria:
                conditions = []
                for key, value in filter_criteria.items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(value)
                if conditions:
                    where_clause = f"WHERE {' AND '.join(conditions)}"

            cursor.execute(
                f"""
                SELECT id, vector FROM {self.table_name} {where_clause}
            """,
                params,
            )

            rows = cursor.fetchall()
            cursor.close()

            # Calculate similarities
            similarities = []
            for row in rows:
                vector_id, vector_blob = row
                stored_vector = self._blob_to_vector(vector_blob)
                if stored_vector:
                    similarity = self._calculate_similarity(query_vector, stored_vector)
                    similarities.append((vector_id, similarity))

            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
        except Exception:
            return []

    async def search_similar_with_vectors(
        self,
        query_vector: Vector,
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None,
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
        try:
            # Get similar vector IDs and scores
            similar = await self.search_similar(query_vector, k, filter_criteria)

            # Fetch the actual vectors
            vector_ids = [item[0] for item in similar]
            vectors = await self.get_vectors(vector_ids)

            # Combine results
            result = []
            for vector_id, score in similar:
                vector = vectors.get(vector_id)
                if vector:
                    result.append((vector_id, vector, score))

            return result
        except Exception:
            return []

    async def search_by_ids(
        self, query_vector: Vector, candidate_ids: List[str], k: Optional[int] = None
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
        try:
            # Get vectors for candidate IDs
            vectors = await self.get_vectors(candidate_ids)

            # Calculate similarities
            similarities = []
            for vector_id, vector in vectors.items():
                if vector:
                    similarity = self._calculate_similarity(query_vector, vector)
                    similarities.append((vector_id, similarity))

            # Sort and limit
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k] if k else similarities
        except Exception:
            return []

    async def batch_search(
        self,
        query_vectors: List[Vector],
        k: int = 10,
        filter_criteria: Optional[Dict[str, Any]] = None,
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
        try:
            results = []
            for query_vector in query_vectors:
                result = await self.search_similar(query_vector, k, filter_criteria)
                results.append(result)
            return results
        except Exception:
            return [[] for _ in query_vectors]

    # Metadata operations
    async def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a vector.

        Args:
            vector_id: Vector identifier

        Returns:
            Metadata dictionary if found, None otherwise
        """
        try:
            if not self._connection:
                return None

            cursor = self._connection.cursor()
            cursor.execute(
                f"""
                SELECT metadata FROM {self.table_name} WHERE id = ?
            """,
                (vector_id,),
            )

            row = cursor.fetchone()
            cursor.close()

            if row and row[0]:
                return json.loads(row[0])
            return None
        except Exception:
            return None

    async def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a vector.

        Args:
            vector_id: Vector identifier
            metadata: New metadata

        Returns:
            True if update was successful
        """
        try:
            if not self._connection:
                return False

            cursor = self._connection.cursor()
            metadata_json = json.dumps(metadata)

            cursor.execute(
                f"""
                UPDATE {self.table_name} 
                SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (metadata_json, vector_id),
            )

            success = cursor.rowcount > 0
            self._connection.commit()
            cursor.close()

            return success
        except Exception:
            return False

    async def search_by_metadata(
        self, filter_criteria: Dict[str, Any], limit: int = 100
    ) -> List[str]:
        """
        Search vectors by metadata criteria.

        Args:
            filter_criteria: Metadata filter criteria
            limit: Maximum number of results

        Returns:
            List of vector IDs matching the criteria
        """
        try:
            if not self._connection:
                return []

            cursor = self._connection.cursor()

            # Build WHERE clause
            conditions = []
            params = []
            for key, value in filter_criteria.items():
                conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            cursor.execute(
                f"""
                SELECT id FROM {self.table_name} {where_clause} LIMIT ?
            """,
                params + [limit],
            )

            rows = cursor.fetchall()
            cursor.close()

            return [row[0] for row in rows]
        except Exception:
            return []

    # Store information and maintenance
    async def get_store_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store.

        Returns:
            Store information including size, dimension, etc.
        """
        try:
            info = {
                "table_name": self.table_name,
                "dimension": self._dimension,
                "metric": self._metric,
                "db_path": str(self.db_path),
            }

            if self._connection:
                cursor = self._connection.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                row = cursor.fetchone()
                info["vector_count"] = row[0] if row else 0
                cursor.close()

            return info
        except Exception:
            return {"error": "Failed to get store info"}

    async def get_vector_count(self) -> int:
        """
        Get the total number of vectors in the store.

        Returns:
            Number of vectors
        """
        try:
            if not self._connection:
                return 0

            cursor = self._connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            row = cursor.fetchone()
            cursor.close()

            return row[0] if row else 0
        except Exception:
            return 0

    async def get_dimension(self) -> int:
        """
        Get the vector dimension of the store.

        Returns:
            Vector dimension
        """
        return self._dimension or 0

    async def optimize_store(self) -> Dict[str, Any]:
        """
        Optimize the vector store for better performance.

        Returns:
            Optimization results
        """
        try:
            if not self._connection:
                return {"error": "Not connected"}

            cursor = self._connection.cursor()

            # Run VACUUM to reclaim space
            cursor.execute("VACUUM")

            # Analyze tables for better query planning
            cursor.execute(f"ANALYZE {self.table_name}")

            cursor.close()

            return {"status": "optimized", "operations": ["vacuum", "analyze"]}
        except Exception as e:
            return {"error": f"Optimization failed: {str(e)}"}

    async def rebuild_index(self, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Rebuild the vector index.

        Args:
            parameters: Optional rebuild parameters

        Returns:
            True if rebuild was successful
        """
        try:
            if not self._connection:
                return False

            cursor = self._connection.cursor()

            # Drop and recreate metadata index
            cursor.execute(f"DROP INDEX IF EXISTS idx_{self.table_name}_metadata")
            cursor.execute(
                f"""
                CREATE INDEX idx_{self.table_name}_metadata 
                ON {self.table_name} (metadata)
            """
            )

            self._connection.commit()
            cursor.close()

            return True
        except Exception:
            return False

    async def clear_store(self) -> bool:
        """
        Clear all vectors from the store.

        Returns:
            True if clearing was successful
        """
        try:
            if not self._connection:
                return False

            cursor = self._connection.cursor()
            cursor.execute(f"DELETE FROM {self.table_name}")

            self._connection.commit()
            cursor.close()

            return True
        except Exception:
            return False

    # Backup and recovery
    async def create_snapshot(self, snapshot_path: str) -> bool:
        """
        Create a snapshot of the vector store.

        Args:
            snapshot_path: Path to save the snapshot

        Returns:
            True if snapshot creation was successful
        """
        try:
            import shutil

            if not self._connection:
                return False

            # Close connection temporarily
            self._connection.close()

            # Copy database file
            shutil.copy2(self.db_path, snapshot_path)

            # Reconnect
            await self.connect()

            return True
        except Exception:
            return False

    async def restore_snapshot(self, snapshot_path: str) -> bool:
        """
        Restore the vector store from a snapshot.

        Args:
            snapshot_path: Path to the snapshot file

        Returns:
            True if restoration was successful
        """
        try:
            import shutil

            if not Path(snapshot_path).exists():
                return False

            # Close connection
            if self._connection:
                self._connection.close()

            # Copy snapshot to database location
            shutil.copy2(snapshot_path, self.db_path)

            # Reconnect and reload config
            await self.connect()

            return True
        except Exception:
            return False

    # Health and diagnostics
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector store.

        Returns:
            Health status information
        """
        health = {
            "connected": await self.is_connected(),
            "table_exists": False,
            "config_loaded": self._dimension is not None,
        }

        if health["connected"]:
            try:
                cursor = self._connection.cursor()
                cursor.execute(
                    f"""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='{self.table_name}'
                """
                )
                health["table_exists"] = cursor.fetchone() is not None
                cursor.close()
            except Exception:
                pass

        health["status"] = (
            "healthy"
            if all(
                [health["connected"], health["table_exists"], health["config_loaded"]]
            )
            else "unhealthy"
        )

        return health

    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the vector store.

        Returns:
            Performance statistics
        """
        try:
            stats = await self.get_store_info()

            if self._connection:
                cursor = self._connection.cursor()

                # Get table size info
                cursor.execute(f"PRAGMA table_info({self.table_name})")
                table_info = cursor.fetchall()
                stats["columns"] = len(table_info)

                cursor.close()

            return stats
        except Exception:
            return {"error": "Failed to get performance stats"}

    # Helper methods
    def _vector_to_blob(self, vector: Vector) -> bytes:
        """Convert Vector to bytes for storage."""
        import struct

        return struct.pack(f"{len(vector)}f", *vector)

    def _blob_to_vector(self, blob: bytes) -> Vector:
        """Convert bytes back to Vector."""
        import struct

        return list(struct.unpack(f"{len(blob)//4}f", blob))

    def _calculate_similarity(self, vector1: Vector, vector2: Vector) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            import math

            # Ensure vectors have same dimension
            if len(vector1) != len(vector2):
                return 0.0

            # Calculate dot product and magnitudes
            dot_product = sum(a * b for a, b in zip(vector1, vector2))
            magnitude1 = math.sqrt(sum(a * a for a in vector1))
            magnitude2 = math.sqrt(sum(a * a for a in vector2))

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0

    async def _load_config(self) -> None:
        """Load configuration from database."""
        try:
            if not self._connection:
                return

            cursor = self._connection.cursor()
            cursor.execute(
                """
                SELECT dimension, metric, parameters 
                FROM vector_store_config 
                WHERE table_name = ?
            """,
                (self.table_name,),
            )

            row = cursor.fetchone()
            if row:
                self._dimension = row[0]
                self._metric = row[1]
                # Parameters are stored as JSON but not used in this basic implementation

            cursor.close()
        except Exception:
            # Config table might not exist yet
            pass
