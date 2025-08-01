"""
Relationship management handler for MCP operations.
"""

from typing import Any

from fastmcp import Context

from src.domain.entities.relationship import RelationshipType
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.use_cases.relationship import RelationshipManagementUseCase

from ..exceptions import MCPServerException
from .base import BaseHandler


class RelationshipHandler(BaseHandler):
    """Handler for relationship-related MCP operations."""

    def __init__(self, relationship_use_case: RelationshipManagementUseCase, config):
        """
        Initialize relationship handler.

        Args:
            relationship_use_case: Relationship management use case
            config: FastMCP configuration
        """
        super().__init__(config)
        self.relationship_use_case = relationship_use_case

    async def create_edge(
        self,
        source_node_id: int,
        target_node_id: int,
        relation_type: str,
        label: str,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Create a new relationship between nodes.

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            relation_type: Type of the relationship
            label: Label for the relationship
            properties: Custom properties for the relationship (optional)
            weight: Weight of the relationship (uses config default if not provided)
            ctx: MCP context object

        Returns:
            Created relationship data
        """
        self.logger.info("Creating edge from %s to %s", source_node_id, target_node_id)

        try:
            # Convert to domain objects
            source_id = NodeId(str(source_node_id))
            target_id = NodeId(str(target_node_id))
            domain_relation_type = RelationshipType(relation_type)

            # Use config default weight if not provided
            actual_weight = (
                weight if weight is not None else self.config.default_relationship_weight
            )

            # Call use case
            relationship = await self.relationship_use_case.create_relationship(
                source_node_id=source_id,
                target_node_id=target_id,
                relationship_type=domain_relation_type,
                label=label,
                properties=properties,
                weight=actual_weight,
            )

            # Convert to MCP response format
            return self._relationship_to_mcp_response(relationship)

        except ValueError as e:
            error_msg = f"Invalid relationship parameters: {e}"
            self.logger.error(error_msg)
            return self._create_error_response(error_msg, "INVALID_PARAMETERS")
        except Exception as e:
            self.logger.error("Error creating relationship: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="create_edge",
                message=str(e),
                original_error=e,
            ) from e

    async def get_edge(
        self,
        edge_id: int,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Get a relationship from the knowledge graph.

        Args:
            edge_id: ID of the relationship to retrieve
            ctx: MCP context object

        Returns:
            Relationship data or error
        """
        self.logger.info("Retrieving edge with ID %s", edge_id)

        try:
            domain_relationship_id = RelationshipId(str(edge_id))

            # Call use case
            relationship = await self.relationship_use_case.get_relationship(domain_relationship_id)

            if not relationship:
                error_msg = "Relationship not found"
                self.logger.error(error_msg)
                return self._create_error_response(error_msg, "RELATIONSHIP_NOT_FOUND")

            # Convert to MCP response format
            return self._relationship_to_mcp_response(relationship)

        except Exception as e:
            self.logger.error("Error getting relationship: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="get_edge",
                message=str(e),
                original_error=e,
            ) from e

    async def update_edge(
        self,
        edge_id: int,
        label: str | None = None,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Update a relationship in the knowledge graph.

        Args:
            edge_id: ID of the relationship to update
            label: New label for the relationship (optional)
            properties: New properties for the relationship (optional)
            weight: New weight for the relationship (optional)
            ctx: MCP context object

        Returns:
            Updated relationship data
        """
        self.logger.info("Updating edge with ID %s", edge_id)

        try:
            domain_relationship_id = RelationshipId(str(edge_id))

            # Call use case
            relationship = await self.relationship_use_case.update_relationship(
                relationship_id=domain_relationship_id,
                label=label,
                properties=properties,
                weight=weight,
            )

            # Convert to MCP response format
            return self._relationship_to_mcp_response(relationship)

        except Exception as e:
            self.logger.error("Error updating relationship: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="update_edge",
                message=str(e),
                original_error=e,
            ) from e

    async def delete_edge(
        self,
        edge_id: int,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Delete a relationship from the knowledge graph.

        Args:
            edge_id: ID of the relationship to delete
            ctx: MCP context object

        Returns:
            Success or error message
        """
        self.logger.info("Deleting edge with ID %s", edge_id)

        try:
            domain_relationship_id = RelationshipId(str(edge_id))

            # Call use case
            await self.relationship_use_case.delete_relationship(domain_relationship_id)

            return self._create_success_response(
                {"message": f"Relationship {edge_id} deleted successfully"}
            )

        except Exception as e:
            self.logger.error("Error deleting relationship: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="delete_edge",
                message=str(e),
                original_error=e,
            ) from e

    async def find_edges(
        self,
        relation_type: str | None = None,
        source_node_id: int | None = None,
        target_node_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Find relationships in the knowledge graph.

        Args:
            relation_type: Filter by relationship type (optional)
            source_node_id: Filter by source node ID (optional)
            target_node_id: Filter by target node ID (optional)
            limit: Maximum number of results to return (default 100)
            offset: Number of results to skip (default 0)
            ctx: MCP context object

        Returns:
            List of relationships matching criteria
        """
        self.logger.info(
            "Finding edges with type=%s, source=%s, target=%s",
            relation_type,
            source_node_id,
            target_node_id,
        )

        try:
            # Convert to domain objects
            domain_relation_type = RelationshipType(relation_type) if relation_type else None
            domain_source_id = NodeId(str(source_node_id)) if source_node_id else None
            domain_target_id = NodeId(str(target_node_id)) if target_node_id else None

            # Call use case
            relationships = await self.relationship_use_case.list_relationships(
                relationship_type=domain_relation_type,
                source_node_id=domain_source_id,
                target_node_id=domain_target_id,
                limit=limit,
                offset=offset,
            )

            # Convert to MCP response format
            result_relationships = [
                self._relationship_to_mcp_response(rel) for rel in relationships
            ]

            self.logger.info("Found %s relationships", len(result_relationships))

            return {
                "edges": result_relationships,
                "count": len(result_relationships),
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            self.logger.error("Error finding relationships: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="find_edges",
                message=str(e),
                original_error=e,
            ) from e

    def _relationship_to_mcp_response(self, relationship) -> dict[str, Any]:
        """Convert domain relationship to MCP response format."""
        return {
            "edge_id": str(relationship.id),
            "source_id": str(relationship.source_node_id),
            "target_id": str(relationship.target_node_id),
            "relation_type": relationship.relationship_type.value,
            "label": relationship.label,
            "properties": relationship.properties or {},
            "confidence": relationship.confidence,
            "created_at": (
                relationship.created_at.isoformat() if hasattr(relationship, "created_at") else None
            ),
        }
