"""
Entity (node) management for the knowledge graph.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.adapters.sqlite3.transactions import UnitOfWork


@dataclass
class Entity:
    """Represents a node in the knowledge graph."""

    id: int
    uuid: str
    name: Optional[str]
    type: str
    properties: Dict[str, Any]
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Entity:
        """
        Create an Entity from a database row.
        Args:
            row: SQLite Row object with entity data
        Returns:
            Entity object
        """
        # Parse JSON properties if needed
        properties = row["properties"]
        if isinstance(properties, str):
            properties = json.loads(properties)
        elif properties is None:
            properties = {}
        return cls(
            id=row["id"],
            uuid=row["uuid"],
            name=row["name"],
            type=row["type"],
            properties=properties,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class EntityManager:
    """
    Manages entity (node) operations for the knowledge graph.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize the entity manager.
        Args:
            connection: SQLite database connection
        """
        self.connection = connection
        self.unit_of_work = UnitOfWork(connection)

    def create_entity(
        self,
        type: str,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        custom_uuid: Optional[str] = None,
    ) -> Entity:
        """
        Create a new entity in the knowledge graph.
        Args:
            type: Type of the entity
            name: Optional name of the entity
            properties: Optional properties dictionary
            custom_uuid: Optional custom UUID (generated if not provided)
        Returns:
            The created Entity object
        """
        entity_uuid = custom_uuid or str(uuid.uuid4())
        props = properties or {}
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            # Insert the entity
            cursor.execute(
                """
            INSERT INTO entities (uuid, name, type, properties)
            VALUES (?, ?, ?, ?)
            """,
                (entity_uuid, name, type, json.dumps(props)),
            )
            entity_id = cursor.lastrowid
            if entity_id is None:
                raise RuntimeError("Failed to insert entity")
            # Register for vector processing if needed
            self.unit_of_work.register_vector_operation(
                entity_type="node", entity_id=entity_id, operation_type="insert"
            )
            # Fetch the created entity
            cursor.execute(
                """
            SELECT * FROM entities WHERE id = ?
            """,
                (entity_id,),
            )
            return Entity.from_row(cursor.fetchone())

    def get_entity(self, entity_id: int) -> Optional[Entity]:
        """
        Get an entity by its ID.
        Args:
            entity_id: Entity ID
        Returns:
            Entity object or None if not found
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        return Entity.from_row(row) if row else None

    def get_entity_by_uuid(self, entity_uuid: str) -> Optional[Entity]:
        """
        Get an entity by its UUID.
        Args:
            entity_uuid: Entity UUID
        Returns:
            Entity object or None if not found
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM entities WHERE uuid = ?", (entity_uuid,))
        row = cursor.fetchone()
        return Entity.from_row(row) if row else None

    def update_entity(
        self,
        entity_id: int,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[Entity]:
        """
        Update an entity's properties.
        Args:
            entity_id: Entity ID
            name: New name (unchanged if None)
            properties: New properties (unchanged if None)
        Returns:
            Updated Entity object or None if not found
        """
        # Get current entity to merge properties
        current_entity = self.get_entity(entity_id)
        if not current_entity:
            return None
        # Prepare updates
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if properties is not None:
            # Merge with existing properties if needed
            merged_props = {**current_entity.properties, **properties}
            updates.append("properties = ?")
            params.append(json.dumps(merged_props))
        if not updates:
            return current_entity  # Nothing to update
        # Execute update
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            query = f"UPDATE entities SET {', '.join(updates)} WHERE id = ?"
            params.append(str(entity_id))
            cursor.execute(query, params)
            if cursor.rowcount > 0:
                # Register for vector processing
                self.unit_of_work.register_vector_operation(
                    entity_type="node", entity_id=entity_id, operation_type="update"
                )
                # Fetch updated entity
                cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
                return Entity.from_row(cursor.fetchone())
        return None

    def delete_entity(self, entity_id: int) -> bool:
        """
        Delete an entity from the knowledge graph.
        Args:
            entity_id: Entity ID
        Returns:
            True if deleted, False if not found
        """
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()
            # Register for vector processing before deletion
            self.unit_of_work.register_vector_operation(
                entity_type="node", entity_id=entity_id, operation_type="delete"
            )
            # Delete the entity (cascades to related tables)
            cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
            return cursor.rowcount > 0

    def find_entities(
        self,
        entity_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        property_filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Entity], int]:
        """
        Find entities matching the given criteria.
        Args:
            entity_type: Optional entity type filter
            name_pattern: Optional name pattern (SQL LIKE pattern)
            property_filters: Optional property filters
            limit: Maximum number of results
            offset: Query offset for pagination
        Returns:
            Tuple of (list of Entity objects, total count)
        """
        # Build query conditions
        conditions = []
        params = []
        if entity_type:
            conditions.append("type = ?")
            params.append(entity_type)
        if name_pattern:
            conditions.append("name LIKE ?")
            params.append(name_pattern)
        # Property filters require special handling with JSON
        # This is simplified and may need optimization for production
        property_clauses = []
        if property_filters:
            for key, value in property_filters.items():
                # Using JSON_EXTRACT to query into the JSON properties
                property_clauses.append(f"JSON_EXTRACT(properties, '$.{key}') = ?")
                params.append(value)
        if property_clauses:
            conditions.extend(property_clauses)
        # Build the final query
        query = "SELECT DISTINCT * FROM entities"  # Added DISTINCT to remove duplicates
        count_query = "SELECT COUNT(DISTINCT id) FROM entities"  # Use COUNT DISTINCT
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
            query += where_clause
            count_query += where_clause
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        # Convert to string for SQL parameters
        params.extend([str(limit), str(offset)])
        # Execute queries
        cursor = self.connection.cursor()
        # Get total count
        cursor.execute(count_query, params[:-2] if params else [])
        total_count = cursor.fetchone()[0]
        # Get entities
        cursor.execute(query, params)
        entities = [Entity.from_row(row) for row in cursor.fetchall()]
        return entities, total_count
