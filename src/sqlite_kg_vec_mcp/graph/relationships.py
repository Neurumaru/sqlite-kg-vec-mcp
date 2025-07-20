"""
Relationship (edge) management for the knowledge graph.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from ..db.transactions import UnitOfWork
from .entities import Entity


@dataclass
class Relationship:
    """Represents a binary relationship (edge) in the knowledge graph."""

    id: int
    source_id: int
    target_id: int
    relation_type: str
    properties: Dict[str, Any]
    created_at: str
    updated_at: str

    # These fields are populated when loading details
    source: Optional[Entity] = None
    target: Optional[Entity] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Relationship:
        """
        Create a Relationship from a database row.

        Args:
            row: SQLite Row object with relationship data

        Returns:
            Relationship object
        """
        # Parse JSON properties if needed
        properties = row["properties"]
        if isinstance(properties, str):
            properties = json.loads(properties)
        elif properties is None:
            properties = {}

        return cls(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=row["relation_type"],
            properties=properties,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class RelationshipManager:
    """
    Manages relationship (edge) operations for the knowledge graph.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize the relationship manager.

        Args:
            connection: SQLite database connection
        """
        self.connection = connection
        self.unit_of_work = UnitOfWork(connection)

    def create_relationship(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Relationship:
        """
        Create a new relationship (edge) between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of the relationship
            properties: Optional properties dictionary

        Returns:
            The created Relationship object

        Raises:
            ValueError: If source or target entities don't exist
        """
        # Verify that source and target entities exist
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM entities WHERE id IN (?, ?)", (source_id, target_id)
        )
        if cursor.fetchone()[0] != 2:
            raise ValueError("Source or target entity does not exist")

        props = properties or {}

        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()

            # Insert the relationship
            cursor.execute(
                """
            INSERT INTO edges (source_id, target_id, relation_type, properties)
            VALUES (?, ?, ?, ?)
            """,
                (source_id, target_id, relation_type, json.dumps(props)),
            )

            edge_id = cursor.lastrowid
            if edge_id is None:
                raise RuntimeError("Failed to insert edge")

            # Register for vector processing if needed
            self.unit_of_work.register_vector_operation(
                entity_type="edge", entity_id=edge_id, operation_type="insert"
            )

            # Fetch the created relationship
            cursor.execute(
                """
            SELECT * FROM edges WHERE id = ?
            """,
                (edge_id,),
            )

            return Relationship.from_row(cursor.fetchone())

    def get_relationship(
        self, relationship_id: int, include_entities: bool = False
    ) -> Optional[Relationship]:
        """
        Get a relationship by its ID.

        Args:
            relationship_id: Relationship ID
            include_entities: Whether to include source and target entities

        Returns:
            Relationship object or None if not found
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM edges WHERE id = ?", (relationship_id,))

        row = cursor.fetchone()
        if not row:
            return None

        relationship = Relationship.from_row(row)

        # Include source and target entities if requested
        if include_entities:
            self._load_relationship_entities(relationship)

        return relationship

    def _load_relationship_entities(self, relationship: Relationship) -> None:
        """
        Load the source and target entities for a relationship.

        Args:
            relationship: Relationship object to populate
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT * FROM entities WHERE id IN (?, ?)",
            (relationship.source_id, relationship.target_id),
        )

        entities = {row["id"]: Entity.from_row(row) for row in cursor.fetchall()}

        relationship.source = entities.get(relationship.source_id)
        relationship.target = entities.get(relationship.target_id)

    def update_relationship(
        self, relationship_id: int, properties: Dict[str, Any]
    ) -> Optional[Relationship]:
        """
        Update a relationship's properties.

        Args:
            relationship_id: Relationship ID
            properties: New properties to merge with existing ones

        Returns:
            Updated Relationship object or None if not found
        """
        # Get current relationship to merge properties
        current = self.get_relationship(relationship_id)
        if not current:
            return None

        # Merge with existing properties
        merged_props = {**current.properties, **properties}

        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE edges SET properties = ? WHERE id = ?",
                (json.dumps(merged_props), relationship_id),
            )

            if cursor.rowcount > 0:
                # Register for vector processing
                self.unit_of_work.register_vector_operation(
                    entity_type="edge",
                    entity_id=relationship_id,
                    operation_type="update",
                )

                # Fetch updated relationship
                cursor.execute("SELECT * FROM edges WHERE id = ?", (relationship_id,))
                return Relationship.from_row(cursor.fetchone())

        return None

    def delete_relationship(self, relationship_id: int) -> bool:
        """
        Delete a relationship from the knowledge graph.

        Args:
            relationship_id: Relationship ID

        Returns:
            True if deleted, False if not found
        """
        with self.unit_of_work.begin() as conn:
            cursor = conn.cursor()

            # Register for vector processing before deletion
            self.unit_of_work.register_vector_operation(
                entity_type="edge", entity_id=relationship_id, operation_type="delete"
            )

            # Delete the relationship
            cursor.execute("DELETE FROM edges WHERE id = ?", (relationship_id,))

            return cursor.rowcount > 0

    def find_relationships(
        self,
        source_id: Optional[int] = None,
        target_id: Optional[int] = None,
        relation_type: Optional[str] = None,
        property_filters: Optional[Dict[str, Any]] = None,
        include_entities: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Relationship], int]:
        """
        Find relationships matching the given criteria.

        Args:
            source_id: Optional source entity ID filter
            target_id: Optional target entity ID filter
            relation_type: Optional relationship type filter
            property_filters: Optional property filters
            include_entities: Whether to include source and target entities
            limit: Maximum number of results
            offset: Query offset for pagination

        Returns:
            Tuple of (list of Relationship objects, total count)
        """
        # Build query conditions
        conditions = []
        params = []

        if source_id is not None:
            conditions.append("source_id = ?")
            params.append(source_id)

        if target_id is not None:
            conditions.append("target_id = ?")
            params.append(target_id)

        if relation_type is not None:
            conditions.append("relation_type = ?")
            params.append(relation_type)  # type: ignore

        # Property filters require special handling with JSON
        property_clauses = []
        if property_filters:
            for key, value in property_filters.items():
                property_clauses.append(f"JSON_EXTRACT(properties, '$.{key}') = ?")
                params.append(value)

        if property_clauses:
            conditions.extend(property_clauses)

        # Build the final query
        query = "SELECT * FROM edges"
        count_query = "SELECT COUNT(*) FROM edges"

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
            query += where_clause
            count_query += where_clause

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        # Execute queries
        cursor = self.connection.cursor()

        # Get total count
        cursor.execute(count_query, params[:-2] if params else [])
        total_count = cursor.fetchone()[0]

        # Get relationships
        cursor.execute(query, params)
        relationships = [Relationship.from_row(row) for row in cursor.fetchall()]

        # Load entities if requested
        if include_entities and relationships:
            for relationship in relationships:
                self._load_relationship_entities(relationship)

        return relationships, total_count

    def get_entity_relationships(
        self,
        entity_id: int,
        direction: str = "both",
        relation_types: Optional[List[str]] = None,
        include_entities: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Relationship], int]:
        """
        Get relationships for a specific entity.

        Args:
            entity_id: Entity ID
            direction: 'outgoing', 'incoming', or 'both'
            relation_types: Optional list of relationship types to filter
            include_entities: Whether to include related entities
            limit: Maximum number of results
            offset: Query offset for pagination

        Returns:
            Tuple of (list of Relationship objects, total count)

        Raises:
            ValueError: If direction is invalid
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError("Direction must be 'outgoing', 'incoming', or 'both'")

        # Build conditions based on direction
        conditions = []
        params = []

        if direction == "outgoing":
            conditions.append("source_id = ?")
            params.append(entity_id)
        elif direction == "incoming":
            conditions.append("target_id = ?")
            params.append(entity_id)
        else:  # 'both'
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([entity_id, entity_id])

        # Add relation type filter if provided
        if relation_types:
            placeholders = ", ".join(["?"] * len(relation_types))
            conditions.append(f"relation_type IN ({placeholders})")
            params.extend(relation_types)  # type: ignore

        # Build the final query
        query = "SELECT * FROM edges"
        count_query = "SELECT COUNT(*) FROM edges"

        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
            query += where_clause
            count_query += where_clause

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        # Execute queries
        cursor = self.connection.cursor()

        # Get total count
        cursor.execute(count_query, params[:-2])
        total_count = cursor.fetchone()[0]

        # Get relationships
        cursor.execute(query, params)
        relationships = [Relationship.from_row(row) for row in cursor.fetchall()]

        # Load entities if requested
        if include_entities and relationships:
            for relationship in relationships:
                self._load_relationship_entities(relationship)

        return relationships, total_count
