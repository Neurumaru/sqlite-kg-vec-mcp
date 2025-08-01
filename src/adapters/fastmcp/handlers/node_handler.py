"""
Node management handler for MCP operations.
"""

from typing import Any

from fastmcp import Context

from src.domain.entities.node import NodeType
from src.domain.value_objects.node_id import NodeId
from src.use_cases.node import NodeManagementUseCase

from ..exceptions import MCPServerException
from .base import BaseHandler


class NodeHandler(BaseHandler):
    """Handler for node-related MCP operations."""

    def __init__(self, node_use_case: NodeManagementUseCase, config):
        """
        Initialize node handler.

        Args:
            node_use_case: Node management use case
            config: FastMCP configuration
        """
        super().__init__(config)
        self.node_use_case = node_use_case

    async def create_node(
        self,
        node_type: str,
        name: str | None = None,
        properties: dict[str, Any] | None = None,
        node_uuid: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Create a new node in the knowledge graph.

        Args:
            node_type: Type of the node to create
            name: Name of the node (optional)
            properties: Custom properties for the node (optional)
            node_uuid: Custom UUID for the node (optional)
            ctx: MCP context object

        Returns:
            Created node data
        """
        self.logger.info("Creating node of type '%s'", node_type)

        try:
            # Convert to domain objects
            domain_node_type = NodeType(node_type)

            # Call use case
            node = await self.node_use_case.create_node(
                name=name or f"Node_{node_type}",
                node_type=domain_node_type,
                properties=properties,
            )

            # Convert to MCP response format
            return self._node_to_mcp_response(node)

        except ValueError as e:
            error_msg = f"Invalid node type or parameters: {e}"
            self.logger.error(error_msg)
            return self._create_error_response(error_msg, "INVALID_PARAMETERS")
        except Exception as e:
            self.logger.error("Error creating node: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="create_node",
                message=str(e),
                original_error=e,
            ) from e

    async def get_node(
        self,
        node_id: int | None = None,
        node_uuid: str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Get a node from the knowledge graph.

        Args:
            node_id: ID of the node to retrieve (optional if uuid is provided)
            node_uuid: UUID of the node to retrieve (optional if id is provided)
            ctx: MCP context object

        Returns:
            Node data or error
        """
        if node_id is None and node_uuid is None:
            error_msg = "Missing required parameter: either id or node_uuid must be provided"
            self.logger.error(error_msg)
            return self._create_error_response(error_msg, "MISSING_PARAMETERS")

        try:
            if node_id is not None:
                domain_node_id = NodeId(str(node_id))
                self.logger.info("Retrieving node with ID %s", node_id)
            else:
                # node_uuid is guaranteed to be str here since we checked both are not None above
                assert node_uuid is not None, "node_uuid must not be None at this point"
                domain_node_id = NodeId(node_uuid)
                self.logger.info("Retrieving node with UUID %s", node_uuid)

            # Call use case
            node = await self.node_use_case.get_node(domain_node_id)

            if not node:
                error_msg = "Node not found"
                self.logger.error(error_msg)
                return self._create_error_response(error_msg, "NODE_NOT_FOUND")

            # Convert to MCP response format
            return self._node_to_mcp_response(node)

        except Exception as e:
            self.logger.error("Error getting node: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="get_node",
                message=str(e),
                original_error=e,
            ) from e

    async def update_node(
        self,
        node_id: int,
        name: str | None = None,
        properties: dict[str, Any] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Update a node in the knowledge graph.

        Args:
            node_id: ID of the node to update
            name: New name for the node (optional)
            properties: New properties for the node (optional)
            ctx: MCP context object

        Returns:
            Updated node data
        """
        self.logger.info("Updating node with ID %s", node_id)

        try:
            domain_node_id = NodeId(str(node_id))

            # Call use case
            node = await self.node_use_case.update_node(
                node_id=domain_node_id,
                name=name,
                properties=properties,
            )

            # Convert to MCP response format
            return self._node_to_mcp_response(node)

        except Exception as e:
            self.logger.error("Error updating node: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="update_node",
                message=str(e),
                original_error=e,
            ) from e

    async def delete_node(
        self,
        node_id: int,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Delete a node from the knowledge graph.

        Args:
            node_id: ID of the node to delete
            ctx: MCP context object

        Returns:
            Success or error message
        """
        self.logger.info("Deleting node with ID %s", node_id)

        try:
            domain_node_id = NodeId(str(node_id))

            # Call use case
            await self.node_use_case.delete_node(domain_node_id)

            return self._create_success_response(
                {"message": f"Node {node_id} deleted successfully"}
            )

        except Exception as e:
            self.logger.error("Error deleting node: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="delete_node",
                message=str(e),
                original_error=e,
            ) from e

    async def find_nodes(
        self,
        node_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Find nodes in the knowledge graph.

        Args:
            node_type: Filter by node type (optional)
            limit: Maximum number of results to return (default 100)
            offset: Number of results to skip (default 0)
            ctx: MCP context object

        Returns:
            List of nodes matching criteria
        """
        self.logger.info(
            "Finding nodes with type=%s, limit=%s, offset=%s", node_type, limit, offset
        )

        try:
            domain_node_type = NodeType(node_type) if node_type else None

            # Call use case
            nodes = await self.node_use_case.list_nodes(
                node_type=domain_node_type,
                limit=limit,
                offset=offset,
            )

            # Convert to MCP response format
            result_nodes = [self._node_to_mcp_response(node) for node in nodes]

            self.logger.info("Found %s nodes", len(result_nodes))

            return {
                "nodes": result_nodes,
                "count": len(result_nodes),
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            self.logger.error("Error finding nodes: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="find_nodes",
                message=str(e),
                original_error=e,
            ) from e

    def _node_to_mcp_response(self, node) -> dict[str, Any]:
        """Convert domain node to MCP response format."""
        return {
            "node_id": str(node.id),
            "uuid": str(node.id),  # Using node.id as UUID for now
            "name": node.name,
            "type": node.node_type.value,
            "properties": node.properties or {},
            "created_at": node.created_at.isoformat() if hasattr(node, "created_at") else None,
        }
