"""
Graph traversal algorithms for exploring the knowledge graph.
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from src.common.observability import get_observable_logger

from .entities import Entity
from .relationships import Relationship


@dataclass
class PathNode:
    """Represents a node in a path during graph traversal."""

    entity: Entity
    relationship: Relationship | None = None
    parent: Optional["PathNode"] = None
    depth: int = 0

    @property
    def path_to_root(self) -> list["PathNode"]:
        """Get the path from this node back to the root."""
        result = [self]
        current = self
        while current.parent:
            current = current.parent
            result.insert(0, current)
        return result


class GraphTraversal:
    """
    Implements graph traversal algorithms for knowledge graph exploration.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        Initialize the graph traversal utility.
        Args:
            connection: SQLite database connection
        """
        self.connection = connection

    def get_neighbors(
        self,
        entity_id: int,
        direction: str = "both",
        relation_types: list[str] | None = None,
        entity_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[tuple[Entity, Relationship]]:
        """
        Get neighboring entities connected to the given entity.
        Args:
            entity_id: Root entity ID
            direction: 'outgoing', 'incoming', or 'both'
            relation_types: Optional list of relationship types to filter
            entity_types: Optional list of entity types to filter
            limit: Maximum number of results
        Returns:
            List of (entity, relationship) tuples
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError("Direction must be 'outgoing', 'incoming', or 'both'")
        queries = []
        params: list[Any] = []
        # Outgoing relationships (entity_id -> neighbor)
        if direction in ("outgoing", "both"):
            conditions = ["r.source_id = ?"]
            query_params: list[Any] = [entity_id]
            if relation_types:
                placeholders = ", ".join(["?"] * len(relation_types))
                conditions.append(f"r.relation_type IN ({placeholders})")
                query_params.extend(relation_types)
            if entity_types:
                placeholders = ", ".join(["?"] * len(entity_types))
                conditions.append(f"e.type IN ({placeholders})")
                query_params.extend(entity_types)
            query = f"""
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   r.created_at as rel_created_at, r.updated_at as rel_updated_at
            FROM entities e
            JOIN edges r ON e.id = r.target_id
            WHERE {" AND ".join(conditions)}
            """
            queries.append(query)
            params.extend(query_params)
        # Incoming relationships (neighbor -> entity_id)
        if direction in ("incoming", "both"):
            conditions = ["r.target_id = ?"]
            query_params_inc: list[Any] = [entity_id]
            if relation_types:
                placeholders = ", ".join(["?"] * len(relation_types))
                conditions.append(f"r.relation_type IN ({placeholders})")
                query_params_inc.extend(relation_types)
            if entity_types:
                placeholders = ", ".join(["?"] * len(entity_types))
                conditions.append(f"e.type IN ({placeholders})")
                query_params_inc.extend(entity_types)
            query = f"""
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   r.created_at as rel_created_at, r.updated_at as rel_updated_at
            FROM entities e
            JOIN edges r ON e.id = r.source_id
            WHERE {" AND ".join(conditions)}
            """
            queries.append(query)
            params.extend(query_params_inc)
        # Combine queries with UNION if needed
        if len(queries) > 1:
            # SQLite doesn't like parentheses around the entire SELECT statement in UNION
            final_query = f"{queries[0]} UNION {queries[1]} LIMIT ?"
            params.append(limit)
        else:
            final_query = f"{queries[0]} LIMIT ?"
            params.append(limit)
        # Execute query
        cursor = self.connection.cursor()
        cursor.execute(final_query, params)
        # Process results
        results = []
        for row in cursor.fetchall():
            # Extract entity fields
            entity = Entity(
                id=row["id"],
                uuid=row["uuid"],
                name=row["name"],
                type=row["type"],
                properties=json.loads(row["properties"]) if row["properties"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            # Extract relationship fields
            rel_props = row["rel_properties"]
            relationship = Relationship(
                id=row["rel_id"],
                source_id=row["source_id"],
                target_id=row["target_id"],
                relation_type=row["relation_type"],
                properties=json.loads(rel_props) if rel_props else {},
                created_at=row["rel_created_at"],
                updated_at=row["rel_updated_at"],
            )
            results.append((entity, relationship))
        return results

    def breadth_first_search(
        self,
        start_id: int,
        max_depth: int = 5,
        relation_types: list[str] | None = None,
        entity_types: list[str] | None = None,
    ) -> list[PathNode]:
        """BFS to find all nodes reachable from start_id within max_depth."""
        if max_depth < 0:
            return []

        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (start_id,))
        start_row = cursor.fetchone()
        if not start_row:
            return []

        start_entity = Entity(
            id=start_row["id"],
            uuid=start_row["uuid"],
            name=start_row["name"],
            type=start_row["type"],
            properties=(json.loads(start_row["properties"]) if start_row["properties"] else {}),
            created_at=start_row["created_at"],
            updated_at=start_row["updated_at"],
        )

        queue = [PathNode(entity=start_entity, depth=0)]
        visited: dict[int, PathNode] = {start_id: queue[0]}
        results: list[PathNode] = []

        while queue:
            current = queue.pop(0)
            results.append(current)

            if current.depth >= max_depth:
                continue

            try:
                neighbors = self.get_neighbors(
                    current.entity.id,
                    direction="both",
                    relation_types=relation_types,
                    entity_types=entity_types,
                )

                for neighbor_entity, relationship in neighbors:
                    if neighbor_entity.id not in visited:
                        new_node = PathNode(
                            entity=neighbor_entity,
                            relationship=relationship,
                            parent=current,
                            depth=current.depth + 1,
                        )
                        queue.append(new_node)
                        visited[neighbor_entity.id] = new_node
            except Exception as exception:
                logger = get_observable_logger("graph_traversal", "adapter")
                logger.error(
                    "neighbor_query_failed_bfs",
                    entity_id=current.entity.id,
                    error_type=type(exception).__name__,
                    error_message=str(exception),
                )

        return results

    def find_shortest_path(
        self,
        start_id: int,
        end_id: int,
        max_depth: int = 5,
        relation_types: list[str] | None = None,
        entity_types: list[str] | None = None,
    ) -> list[PathNode] | None:
        """Finds the shortest path between two entities using BFS."""
        if start_id == end_id:
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM entities WHERE id = ?", (start_id,))
            start_row = cursor.fetchone()
            if start_row:
                start_entity = Entity(
                    id=start_row["id"],
                    uuid=start_row["uuid"],
                    name=start_row["name"],
                    type=start_row["type"],
                    properties=(
                        json.loads(start_row["properties"]) if start_row["properties"] else {}
                    ),
                    created_at=start_row["created_at"],
                    updated_at=start_row["updated_at"],
                )
                return [PathNode(entity=start_entity)]
            return None

        # Use BFS to find all reachable nodes, and check if end_id is among them
        reachable_nodes = self.breadth_first_search(
            start_id=start_id,
            max_depth=max_depth,
            relation_types=relation_types,
            entity_types=entity_types,
        )

        # Find the PathNode corresponding to the end_id
        end_node = next((node for node in reachable_nodes if node.entity.id == end_id), None)

        if end_node:
            return end_node.path_to_root
        return None

    def recursive_query(
        self,
        start_id: int,
        direction: str = "outgoing",
        relation_types: list[str] | None = None,
        entity_types: list[str] | None = None,
        max_depth: int = 3,
        limit: int = 100,
    ) -> list[dict]:
        """
        Perform a recursive query to find a subgraph using SQLite's recursive CTE.
        Args:
            start_id: Start entity ID
            direction: 'outgoing', 'incoming', or 'both'
            relation_types: Optional list of relationship types to filter
            entity_types: Optional list of entity types to filter
            max_depth: Maximum traversal depth
            limit: Maximum number of results
        Returns:
            List of result rows with entity and relationship information
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise ValueError("Direction must be 'outgoing', 'incoming', or 'both'")
        relation_filter = ""
        entity_filter = ""
        # Build filter clauses and params separately for each query part
        relation_filter_params = []
        entity_filter_params = []
        # Add relation type filter
        if relation_types:
            placeholders = ", ".join(["?"] * len(relation_types))
            relation_filter = f"AND r.relation_type IN ({placeholders})"
            relation_filter_params = list(relation_types)
        # Add entity type filter
        if entity_types:
            placeholders = ", ".join(["?"] * len(entity_types))
            entity_filter = f"AND e.type IN ({placeholders})"
            entity_filter_params = list(entity_types)
        # Build parameters based on direction
        if direction == "both":
            # Both direction needs start_id twice (outgoing and incoming base cases)
            base_params = [start_id] + relation_filter_params + [start_id] + relation_filter_params
            recursive_params = [max_depth] + relation_filter_params
        else:
            # Single direction needs start_id once
            base_params = [start_id] + relation_filter_params
            recursive_params = [max_depth] + relation_filter_params
        final_params = entity_filter_params + [limit]
        # Combine all parameters in order
        params = base_params + recursive_params + final_params
        # Build the recursive CTE query based on direction
        if direction == "outgoing":
            recursive_query = f"""
            WITH RECURSIVE
            graph_path(source_id, target_id, relation_id, depth) AS (
                -- Base case: direct relationships from start_id
                SELECT r.source_id, r.target_id, r.id, 1
                FROM edges r
                WHERE r.source_id = ? {relation_filter}
                UNION ALL
                -- Recursive case: follow outgoing edges
                SELECT r.source_id, r.target_id, r.id, gp.depth + 1
                FROM graph_path gp
                JOIN edges r ON gp.target_id = r.source_id
                WHERE gp.depth < ? {relation_filter}
            )
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   gp.depth, s.id as source_entity_id, s.name as source_name,
                   s.type as source_type
            FROM graph_path gp
            JOIN edges r ON r.id = gp.relation_id
            JOIN entities e ON e.id = gp.target_id
            JOIN entities s ON s.id = gp.source_id
            WHERE 1=1 {entity_filter}
            ORDER BY gp.depth
            LIMIT ?
            """
        elif direction == "incoming":
            recursive_query = f"""
            WITH RECURSIVE
            graph_path(source_id, target_id, relation_id, depth) AS (
                -- Base case: direct relationships to start_id
                SELECT r.source_id, r.target_id, r.id, 1
                FROM edges r
                WHERE r.target_id = ? {relation_filter}
                UNION ALL
                -- Recursive case: follow incoming edges
                SELECT r.source_id, r.target_id, r.id, gp.depth + 1
                FROM graph_path gp
                JOIN edges r ON gp.source_id = r.target_id
                WHERE gp.depth < ? {relation_filter}
            )
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   gp.depth, t.id as target_entity_id, t.name as target_name,
                   t.type as target_type
            FROM graph_path gp
            JOIN edges r ON r.id = gp.relation_id
            JOIN entities e ON e.id = gp.source_id
            JOIN entities t ON t.id = gp.target_id
            WHERE 1=1 {entity_filter}
            ORDER BY gp.depth
            LIMIT ?
            """
        else:  # 'both'
            recursive_query = f"""
            WITH RECURSIVE
            graph_path(entity_id, related_id, relation_id, direction, depth) AS (
                -- Base case: direct outgoing relationships from start_id
                SELECT r.source_id, r.target_id, r.id, 'outgoing', 1
                FROM edges r
                WHERE r.source_id = ? {relation_filter}
                UNION ALL
                -- Base case: direct incoming relationships to start_id
                SELECT r.target_id, r.source_id, r.id, 'incoming', 1
                FROM edges r
                WHERE r.target_id = ? {relation_filter}
                UNION ALL
                -- Recursive case: follow relationships in both directions
                SELECT
                    CASE
                        WHEN gp.direction = 'outgoing' THEN r.source_id
                        ELSE r.target_id
                    END,
                    CASE
                        WHEN gp.direction = 'outgoing' THEN r.target_id
                        ELSE r.source_id
                    END,
                    r.id,
                    gp.direction,
                    gp.depth + 1
                FROM graph_path gp
                JOIN edges r ON
                    (gp.direction = 'outgoing' AND gp.related_id = r.source_id) OR
                    (gp.direction = 'incoming' AND gp.related_id = r.target_id)
                WHERE gp.depth < ? {relation_filter}
            )
            SELECT e.*, r.id as rel_id, r.source_id, r.target_id,
                   r.relation_type, r.properties as rel_properties,
                   gp.depth, gp.direction
            FROM graph_path gp
            JOIN edges r ON r.id = gp.relation_id
            JOIN entities e ON e.id = gp.related_id
            WHERE 1=1 {entity_filter}
            ORDER BY gp.depth
            LIMIT ?
            """
            # For 'both' direction, start_id is used twice
            both_params: list[Any] = [start_id, start_id, max_depth] + params[2:]
            params = both_params
        params.append(limit)
        # Execute the recursive query
        cursor = self.connection.cursor()
        cursor.execute(recursive_query, params)
        # Process results
        results = []
        for row in cursor.fetchall():
            # Convert to dict for easier manipulation
            result_dict = dict(row)
            # Parse JSON properties
            if result_dict.get("properties"):
                result_dict["properties"] = json.loads(result_dict["properties"])
            else:
                result_dict["properties"] = {}
            if result_dict.get("rel_properties"):
                result_dict["rel_properties"] = json.loads(result_dict["rel_properties"])
            else:
                result_dict["rel_properties"] = {}
            results.append(result_dict)
        return results

    def depth_first_search(
        self,
        start_id: int,
        max_depth: int = 5,
        relation_types: list[str] | None = None,
        entity_types: list[str] | None = None,
    ) -> list[PathNode]:
        """DFS to find all nodes reachable from start_id within max_depth."""
        if max_depth < 0:
            return []

        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (start_id,))
        start_row = cursor.fetchone()
        if not start_row:
            return []

        start_entity = Entity(
            id=start_row["id"],
            uuid=start_row["uuid"],
            name=start_row["name"],
            type=start_row["type"],
            properties=(json.loads(start_row["properties"]) if start_row["properties"] else {}),
            created_at=start_row["created_at"],
            updated_at=start_row["updated_at"],
        )

        stack = [PathNode(entity=start_entity, depth=0)]
        visited: dict[int, PathNode] = {start_id: stack[0]}
        results: list[PathNode] = []

        while stack:
            current = stack.pop()
            results.append(current)

            if current.depth >= max_depth:
                continue

            try:
                # Get neighbors in reverse order to simulate DFS (stack behavior)
                neighbors = self.get_neighbors(
                    current.entity.id,
                    direction="both",
                    relation_types=relation_types,
                    entity_types=entity_types,
                )
                # Add neighbors to stack in reverse to process in natural order
                for neighbor_entity, relationship in reversed(neighbors):
                    if neighbor_entity.id not in visited:
                        new_node = PathNode(
                            entity=neighbor_entity,
                            relationship=relationship,
                            parent=current,
                            depth=current.depth + 1,
                        )
                        stack.append(new_node)
                        visited[neighbor_entity.id] = new_node
            except Exception as exception:
                logger = get_observable_logger("graph_traversal", "adapter")
                logger.error(
                    "neighbor_query_failed_dfs",
                    entity_id=current.entity.id,
                    error_type=type(exception).__name__,
                    error_message=str(exception),
                )

        return results

    def get_connected_components(self) -> list[list[int]]:
        """Finds all connected components in the graph (undirected)."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM entities")
        all_entity_ids = [row[0] for row in cursor.fetchall()]

        visited_all = set()
        components = []

        for entity_id in all_entity_ids:
            if entity_id not in visited_all:
                component_nodes = []
                queue = [entity_id]
                visited_component = {entity_id}

                while queue:
                    current_id = queue.pop(0)
                    component_nodes.append(current_id)
                    visited_all.add(current_id)

                    # Get neighbors (undirected for connected components)
                    neighbors = self.get_neighbors(current_id, direction="both")
                    for neighbor_entity, _ in neighbors:
                        if neighbor_entity.id not in visited_component:
                            queue.append(neighbor_entity.id)
                            visited_component.add(neighbor_entity.id)

                components.append(component_nodes)

        return components
