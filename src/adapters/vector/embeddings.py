"""
Vector embedding storage and management.
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .transactions import UnitOfWork


@dataclass
class Embedding:
    """Represents a vector embedding for an entity or relationship."""

    entity_id: int
    entity_type: str  # 'node', 'edge', or 'hyperedge'
    embedding: np.ndarray
    dimensions: int
    model_info: str
    embedding_version: int
    created_at: str
    updated_at: str

    # Class-level constant for performance
    _ENTITY_ID_FIELDS = {
        "node": "node_id",
        "edge": "edge_id", 
        "hyperedge": "hyperedge_id"
    }

    @classmethod
    def from_row(cls, row: sqlite3.Row, entity_type: str) -> "Embedding":
        """
        Create an Embedding from a database row.

        Args:
            row: SQLite Row object with embedding data
            entity_type: Type of entity ('node', 'edge', or 'hyperedge')

        Returns:
            Embedding object
        """
        # Convert BLOB to numpy array (optimized - avoid intermediate variable)
        embedding = np.frombuffer(row["embedding"], dtype=np.float32)

        # Fast dictionary lookup instead of if-elif chain
        id_field = cls._ENTITY_ID_FIELDS.get(entity_type)
        if not id_field:
            raise ValueError(f"Unsupported entity type: {entity_type}")
        
        entity_id = row[id_field]

        return cls(
            entity_id=entity_id,
            entity_type=entity_type,
            embedding=embedding,
            dimensions=row["dimensions"],
            model_info=row["model_info"],
            embedding_version=row["embedding_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class EmbeddingManager:
    """
    Manages vector embeddings for entities, relationships, and hyperedges.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize the embedding manager.

        Args:
            connection: SQLite database connection
        """
        self.connection = connection
        self.unit_of_work = UnitOfWork(connection)

    def store_embedding(
        self,
        entity_type: str,
        entity_id: int,
        embedding: np.ndarray,
        model_info: str,
        embedding_version: int = 1,
    ) -> bool:
        """
        Store a vector embedding for an entity, edge, or hyperedge.

        Args:
            entity_type: Type of entity ('node', 'edge', or 'hyperedge')
            entity_id: ID of the entity
            embedding: Numpy array of embedding values
            model_info: Information about the embedding model
            embedding_version: Version number for the embedding

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If entity_type is invalid
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("Entity type must be 'node', 'edge', or 'hyperedge'")

        # Get table name and ID column based on entity type
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        # Convert numpy array to bytes for storage
        embedding_blob = embedding.astype(np.float32).tobytes()
        dimensions = len(embedding)

        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()

            # Check if embedding already exists
            cursor.execute(f"SELECT 1 FROM {table} WHERE {id_column} = ?", (entity_id,))

            if cursor.fetchone():
                # Update existing embedding
                cursor.execute(
                    f"""
                    UPDATE {table} 
                    SET embedding = ?, dimensions = ?, model_info = ?, 
                        embedding_version = ?
                    WHERE {id_column} = ?
                    """,
                    (
                        embedding_blob,
                        dimensions,
                        model_info,
                        embedding_version,
                        entity_id,
                    ),
                )
            else:
                # Insert new embedding
                cursor.execute(
                    f"""
                    INSERT INTO {table} 
                    ({id_column}, embedding, dimensions, model_info, embedding_version)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        entity_id,
                        embedding_blob,
                        dimensions,
                        model_info,
                        embedding_version,
                    ),
                )

            return cursor.rowcount > 0

    def get_embedding(self, entity_type: str, entity_id: int) -> Optional[Embedding]:
        """
        Get the embedding for a specific entity.

        Args:
            entity_type: Type of entity ('node', 'edge', or 'hyperedge')
            entity_id: ID of the entity

        Returns:
            Embedding object or None if not found
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("Entity type must be 'node', 'edge', or 'hyperedge'")

        # Get table name and ID column based on entity type
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        cursor = self.connection.cursor()
        cursor.execute(f"SELECT * FROM {table} WHERE {id_column} = ?", (entity_id,))

        row = cursor.fetchone()
        return Embedding.from_row(row, entity_type) if row else None

    def delete_embedding(self, entity_type: str, entity_id: int) -> bool:
        """
        Delete the embedding for a specific entity.

        Args:
            entity_type: Type of entity ('node', 'edge', or 'hyperedge')
            entity_id: ID of the entity

        Returns:
            True if successful, False if not found
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("Entity type must be 'node', 'edge', or 'hyperedge'")

        # Get table name and ID column based on entity type
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        cursor = self.connection.cursor()
        cursor.execute(f"DELETE FROM {table} WHERE {id_column} = ?", (entity_id,))

        return cursor.rowcount > 0

    def get_all_embeddings(
        self, entity_type: str, model_info: Optional[str] = None, batch_size: int = 1000, offset: int = 0
    ) -> List[Embedding]:
        """
        Get all embeddings of a specific type, optionally filtered by model_info.

        Args:
            entity_type: Type of entity ('node', 'edge', or 'hyperedge')
            model_info: Optional filter for specific model
            batch_size: Number of embeddings to fetch per batch

        Returns:
            List of Embedding objects
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("Entity type must be 'node', 'edge', or 'hyperedge'")

        # Get table name and ID column based on entity type
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        # Build query
        query = f"SELECT * FROM {table}"
        params = []

        if model_info:
            query += " WHERE model_info = ?"
            params.append(model_info)

        query += f" ORDER BY {id_column}"

        # Fetch embeddings in batches to avoid memory issues
        cursor = self.connection.cursor()
        offset = 0
        all_embeddings = []

        while True:
            batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
            cursor.execute(batch_query, params)

            batch = cursor.fetchall()
            if not batch:
                break

            # Convert rows to embeddings (optimized with list comprehension)
            batch_embeddings = [Embedding.from_row(row, entity_type) for row in batch]
            all_embeddings.extend(batch_embeddings)

            offset += batch_size

        return all_embeddings

    def get_outdated_embeddings(
        self, entity_type: str, current_version: int, batch_size: int = 1000
    ) -> List[int]:
        """
        Get IDs of entities with outdated embeddings (version < current_version).

        Args:
            entity_type: Type of entity ('node', 'edge', or 'hyperedge')
            current_version: Current embedding version
            batch_size: Number of IDs to fetch per batch

        Returns:
            List of entity IDs with outdated embeddings
        """
        if entity_type not in ("node", "edge", "hyperedge"):
            raise ValueError("Entity type must be 'node', 'edge', or 'hyperedge'")

        # Get table name and ID column based on entity type
        if entity_type == "node":
            table = "node_embeddings"
            id_column = "node_id"
        elif entity_type == "edge":
            table = "edge_embeddings"
            id_column = "edge_id"
        else:  # hyperedge
            table = "hyperedge_embeddings"
            id_column = "hyperedge_id"

        # Query for outdated embeddings
        query = f"""
        SELECT {id_column} FROM {table}
        WHERE embedding_version < ?
        ORDER BY {id_column}
        """

        # Fetch IDs in batches
        cursor = self.connection.cursor()
        offset = 0
        all_ids = []

        while True:
            batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
            cursor.execute(batch_query, (current_version,))

            batch = cursor.fetchall()
            if not batch:
                break

            for row in batch:
                all_ids.append(row[0])

            offset += batch_size

        return all_ids

    def process_outbox(self, batch_size: int = 100) -> int:
        """
        Process pending vector operations from the outbox.

        Args:
            batch_size: Number of operations to process in one batch

        Returns:
            Number of operations processed
        """
        cursor = self.connection.cursor()

        # Get pending operations
        cursor.execute(
            """
            SELECT id, operation_type, entity_type, entity_id, model_info
            FROM vector_outbox
            WHERE status = 'pending'
            ORDER BY created_at
            LIMIT ?
            """,
            (batch_size,),
        )

        operations = cursor.fetchall()
        processed_count = 0

        for op in operations:
            outbox_id = op["id"]
            operation_type = op["operation_type"]
            entity_type = op["entity_type"]
            entity_id = op["entity_id"]
            model_info = op["model_info"]

            try:
                # Mark as processing
                cursor.execute(
                    "UPDATE vector_outbox SET status = 'processing' WHERE id = ?",
                    (outbox_id,),
                )

                if operation_type == "delete":
                    # Handle deletion
                    self.delete_embedding(entity_type, entity_id)

                elif operation_type in ("insert", "update"):
                    # Generate actual embedding based on entity content
                    embedding = self._generate_embedding_for_entity(
                        entity_type, entity_id, model_info
                    )

                    # Store the embedding
                    self.store_embedding(
                        entity_type=entity_type,
                        entity_id=entity_id,
                        embedding=embedding,
                        model_info=model_info or "default_model",
                        embedding_version=1,
                    )

                # Mark as completed
                cursor.execute(
                    "UPDATE vector_outbox SET status = 'completed' WHERE id = ?",
                    (outbox_id,),
                )

                processed_count += 1

            except Exception as e:
                # Log error and mark as failed
                cursor.execute(
                    """
                    UPDATE vector_outbox 
                    SET status = 'failed', 
                        retry_count = retry_count + 1,
                        last_error = ?
                    WHERE id = ?
                    """,
                    (str(e), outbox_id),
                )

                # Record in sync_failures table
                cursor.execute(
                    """
                    INSERT INTO sync_failures
                    (outbox_id, entity_type, entity_id, operation_type, 
                     error_message, retry_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outbox_id,
                        entity_type,
                        entity_id,
                        operation_type,
                        str(e),
                        cursor.execute(
                            "SELECT retry_count FROM vector_outbox WHERE id = ?",
                            (outbox_id,),
                        ).fetchone()[0],
                    ),
                )

        return processed_count

    def _generate_embedding_for_entity(
        self, entity_type: str, entity_id: int, model_info: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate embedding for an entity by extracting its text content.

        Args:
            entity_type: Type of entity ('node' or 'edge')
            entity_id: ID of the entity
            model_info: Model information for embedding generation

        Returns:
            Generated embedding vector
        """
        try:
            # Get entity content for embedding
            text_content = self._extract_entity_text(entity_type, entity_id)

            # Use text embedder if available
            if hasattr(self, "text_embedder") and self.text_embedder is not None:
                result = self.text_embedder.embed(text_content)
                return result

            # Fallback: try to create a default embedder
            from .text_embedder import create_embedder

            # Determine embedding dimension from model_info or use default
            embedding_dim = 384  # Default for sentence-transformers
            if model_info and "dim" in model_info:
                try:
                    embedding_dim = int(model_info.split("dim=")[1].split(",")[0])
                except:
                    pass

            # Create default sentence-transformers embedder
            embedder = create_embedder(
                embedder_type="sentence-transformers", model_name="all-MiniLM-L6-v2"
            )

            return embedder.embed(text_content)

        except Exception as e:
            # Fallback to random embedding with warning
            import warnings

            warnings.warn(
                f"Failed to generate embedding for {entity_type} {entity_id}: {e}. Using random embedding."
            )
            return np.random.rand(384).astype(np.float32)

    def _extract_entity_text(self, entity_type: str, entity_id: int) -> str:
        """
        Extract text content from an entity for embedding generation.

        Args:
            entity_type: Type of entity ('node' or 'edge')
            entity_id: ID of the entity

        Returns:
            Text representation of the entity
        """
        cursor = self.connection.cursor()

        if entity_type == "node":
            # Extract text from entity
            cursor.execute(
                """
                SELECT name, type, properties 
                FROM entities 
                WHERE id = ?
            """,
                (entity_id,),
            )

            result = cursor.fetchone()
            if not result:
                return f"Entity {entity_id} not found"

            name, ent_type, properties = result

            # Combine name, type and relevant properties
            text_parts = []
            if name:
                text_parts.append(f"Name: {name}")
            if ent_type:
                text_parts.append(f"Type: {ent_type}")

            # Extract text from properties JSON
            if properties:
                import json

                try:
                    props = (
                        json.loads(properties)
                        if isinstance(properties, str)
                        else properties
                    )
                    for key, value in props.items():
                        if isinstance(value, (str, int, float)):
                            text_parts.append(f"{key}: {value}")
                except:
                    text_parts.append(f"Properties: {properties}")

            return " | ".join(text_parts)

        elif entity_type == "edge":
            # Extract text from relationship
            cursor.execute(
                """
                SELECT r.relation_type, r.properties,
                       e1.name as source_name, e1.type as source_type,
                       e2.name as target_name, e2.type as target_type
                FROM edges r
                JOIN entities e1 ON r.source_id = e1.id
                JOIN entities e2 ON r.target_id = e2.id
                WHERE r.id = ?
            """,
                (entity_id,),
            )

            result = cursor.fetchone()
            if not result:
                return f"Edge {entity_id} not found"

            relation_type, properties, src_name, src_type, tgt_name, tgt_type = result

            # Create relationship text
            text_parts = [
                f"Relationship: {relation_type}",
                f"From: {src_name or src_type}",
                f"To: {tgt_name or tgt_type}",
            ]

            # Add properties if available
            if properties:
                import json

                try:
                    props = (
                        json.loads(properties)
                        if isinstance(properties, str)
                        else properties
                    )
                    for key, value in props.items():
                        if isinstance(value, (str, int, float)):
                            text_parts.append(f"{key}: {value}")
                except:
                    text_parts.append(f"Properties: {properties}")

            return " | ".join(text_parts)

        else:
            return f"Unknown entity type: {entity_type}"
