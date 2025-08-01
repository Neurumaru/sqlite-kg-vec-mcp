"""
API endpoints and handlers for the MCP server interface.
"""

import logging
from typing import Any

from fastmcp import Context, FastMCP

from src.domain.entities.node import NodeType
from src.domain.entities.relationship import RelationshipType
from src.domain.services.knowledge_search import SearchStrategy
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.use_cases.knowledge_search import KnowledgeSearchUseCase
from src.use_cases.node import NodeManagementUseCase
from src.use_cases.relationship import RelationshipManagementUseCase

from .config import FastMCPConfig
from .exceptions import MCPServerException


class KnowledgeGraphServer:
    """
    MCP server providing a knowledge graph API.

    This adapter converts MCP protocol messages to domain use case calls
    and formats the responses back to MCP format.
    """

    def __init__(
        self,
        node_use_case: NodeManagementUseCase,
        relationship_use_case: RelationshipManagementUseCase,
        knowledge_search_use_case: KnowledgeSearchUseCase,
        config: FastMCPConfig,
    ):
        """
        Initialize the knowledge graph MCP server.

        Args:
            node_use_case: Node management use case
            relationship_use_case: Relationship management use case
            knowledge_search_use_case: Knowledge search use case
            config: MCP server configuration
        """
        self.node_use_case = node_use_case
        self.relationship_use_case = relationship_use_case
        self.knowledge_search_use_case = knowledge_search_use_case
        self.config = config

        # Setup logging
        self.logger = logging.getLogger("kg_server")
        self.logger.setLevel(getattr(logging, config.log_level))

        # Create MCP server with FastMCP
        self.mcp_server: FastMCP = FastMCP(
            name="Knowledge Graph Server",
            instructions="SQLite-based knowledge graph with vector search capabilities",
        )

        # Register all tools
        self._register_tools()

        self.logger.info("Knowledge Graph Server initialized")

    def _register_tools(self):
        """Register all API endpoint tools."""
        # Entity tools
        self.mcp_server.tool()(self.create_node)
        self.mcp_server.tool()(self.get_node)
        self.mcp_server.tool()(self.update_node)
        self.mcp_server.tool()(self.delete_node)
        self.mcp_server.tool()(self.find_nodes)

        # Relationship tools
        self.mcp_server.tool()(self.create_edge)
        self.mcp_server.tool()(self.get_edge)
        self.mcp_server.tool()(self.update_edge)
        self.mcp_server.tool()(self.delete_edge)
        self.mcp_server.tool()(self.find_edges)

        # Graph traversal tools
        self.mcp_server.tool()(self.get_neighbors)
        self.mcp_server.tool()(self.find_paths)

        # Vector search tools
        self.mcp_server.tool()(self.search_similar_nodes)
        self.mcp_server.tool()(self.search_by_text)

    # === Node Management Methods ===

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
            return {"error": error_msg}
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
            return {"error": error_msg}

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
                if ctx:
                    self.logger.error(error_msg)
                return {"error": error_msg}

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

            return {"success": True, "message": f"Node {node_id} deleted successfully"}

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

            if ctx:
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

    # === Search Methods ===

    async def search_by_text(
        self,
        query: str,
        limit: int = 10,
        include_documents: bool = True,
        include_nodes: bool = True,
        include_relationships: bool = True,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Search knowledge graph by text query.

        Args:
            query: Search query string
            limit: Maximum number of results to return (default 10)
            include_documents: Include document results (default True)
            include_nodes: Include node results (default True)
            include_relationships: Include relationship results (default True)
            ctx: MCP context object

        Returns:
            Search results
        """
        self.logger.info("Searching by text: '%s'", query)

        try:
            # Call use case
            results = await self.knowledge_search_use_case.search_knowledge(
                query=query,
                strategy=SearchStrategy.SEMANTIC,
                limit=limit,
                include_documents=include_documents,
                include_nodes=include_nodes,
                include_relationships=include_relationships,
            )

            # Convert to MCP response format
            result_items = []
            for result in results.results:
                if result.document:
                    result_items.append(
                        {
                            "type": "document",
                            "id": str(result.document.id),
                            "title": result.document.title,
                            "content": (
                                result.document.content[: self.config.content_summary_length]
                                + "..."
                                if len(result.document.content) > self.config.content_summary_length
                                else result.document.content
                            ),
                            "similarity": result.score,
                        }
                    )
                elif result.node:
                    result_items.append(
                        {
                            "type": "node",
                            "id": str(result.node.id),
                            "name": result.node.name,
                            "node_type": result.node.node_type.value,
                            "similarity": result.score,
                        }
                    )
                elif result.relationship:
                    result_items.append(
                        {
                            "type": "relationship",
                            "id": str(result.relationship.id),
                            "label": result.relationship.label,
                            "relationship_type": result.relationship.relationship_type.value,
                            "similarity": result.score,
                        }
                    )

            if ctx:
                self.logger.info("Found %s results", len(result_items))

            return {"results": result_items, "count": len(result_items)}

        except Exception as e:
            self.logger.error("Error searching by text: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="search_by_text",
                message=str(e),
                original_error=e,
            ) from e

    # === Helper Methods ===

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

    # === Placeholder methods for remaining functionality ===
    # These would be implemented following the same pattern

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
            return {"error": error_msg}
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
                if ctx:
                    self.logger.error(error_msg)
                return {"error": error_msg}

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

            return {"success": True, "message": f"Relationship {edge_id} deleted successfully"}

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

            if ctx:
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

    async def get_neighbors(
        self,
        node_id: int,
        depth: int = 1,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Get neighboring nodes in the knowledge graph.

        Args:
            node_id: ID of the central node
            depth: Depth of neighbors to retrieve (default 1)
            ctx: MCP context object

        Returns:
            List of neighboring nodes
        """
        self.logger.info("Getting neighbors for node %s at depth %s", node_id, depth)

        try:
            domain_node_id = NodeId(str(node_id))

            # Get relationships for the node
            relationships = await self.relationship_use_case.get_node_relationships(
                node_id=domain_node_id, direction="both"
            )

            # Extract neighbor node IDs
            neighbor_ids = set()
            for rel in relationships:
                if str(rel.source_node_id) != str(node_id):
                    neighbor_ids.add(rel.source_node_id)
                if str(rel.target_node_id) != str(node_id):
                    neighbor_ids.add(rel.target_node_id)

            # Get neighbor nodes
            neighbors = []
            for neighbor_id in neighbor_ids:
                neighbor = await self.node_use_case.get_node(neighbor_id)
                if neighbor:
                    neighbors.append(self._node_to_mcp_response(neighbor))

            if ctx:
                self.logger.info("Found %s neighbors", len(neighbors))

            return {
                "neighbors": neighbors,
                "count": len(neighbors),
                "depth": depth,
            }

        except Exception as e:
            self.logger.error("Error getting neighbors: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="get_neighbors",
                message=str(e),
                original_error=e,
            ) from e

    async def find_paths(
        self,
        source_node_id: int,
        target_node_id: int,
        max_depth: int = 5,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Find paths between two nodes in the knowledge graph.

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            max_depth: Maximum path depth to search (default 5)
            ctx: MCP context object

        Returns:
            Shortest path between nodes or None if no path exists
        """
        self.logger.info("Finding path from %s to %s", source_node_id, target_node_id)

        try:
            # Check if RelationshipAnalysisUseCase is available
            if not hasattr(self.relationship_use_case, "find_shortest_path"):
                # Fallback to basic implementation using available methods
                return {
                    "path": None,
                    "length": 0,
                    "message": "Path finding requires RelationshipAnalysisUseCase implementation",
                }

            source_id = NodeId(str(source_node_id))
            target_id = NodeId(str(target_node_id))

            # Call use case (if available)
            path_relationships = await self.relationship_use_case.find_shortest_path(
                source_node_id=source_id,
                target_node_id=target_id,
                max_depth=max_depth,
            )

            if not path_relationships:
                if ctx:
                    self.logger.info("No path found")
                return {
                    "path": None,
                    "length": 0,
                    "message": "No path found between the specified nodes",
                }

            # Convert path to MCP response format
            path_edges = [self._relationship_to_mcp_response(rel) for rel in path_relationships]

            if ctx:
                self.logger.info("Found path with %s edges", len(path_edges))

            return {
                "path": path_edges,
                "length": len(path_edges),
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
            }

        except Exception as e:
            self.logger.error("Error finding path: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="find_paths",
                message=str(e),
                original_error=e,
            ) from e

    async def search_similar_nodes(
        self,
        node_id: int,
        limit: int = 10,
        threshold: float | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """
        Search for nodes similar to the given node.

        Args:
            node_id: ID of the reference node
            limit: Maximum number of similar nodes to return (default 10)
            threshold: Similarity threshold (uses config default if not provided)
            ctx: MCP context object

        Returns:
            List of similar nodes with similarity scores
        """
        self.logger.info("Searching for nodes similar to %s", node_id)

        try:
            domain_node_id = NodeId(str(node_id))

            # Use config default threshold if not provided
            actual_threshold = (
                threshold if threshold is not None else self.config.similarity_threshold
            )

            # Check if NodeEmbeddingUseCase is available
            if not hasattr(self.node_use_case, "find_similar_nodes"):
                # Fallback: use knowledge search with node name
                node = await self.node_use_case.get_node(domain_node_id)
                if not node:
                    return {"error": f"Node {node_id} not found"}

                # Use text search as fallback
                results = await self.knowledge_search_use_case.search_knowledge(
                    query=node.name,
                    strategy=SearchStrategy.SEMANTIC,
                    limit=limit,
                    include_documents=False,
                    include_nodes=True,
                    include_relationships=False,
                )

                # Convert to similar nodes format
                similar_nodes = []
                for result in results.results:
                    if result.node and str(result.node.id) != str(node_id):
                        similar_nodes.append(
                            {
                                **self._node_to_mcp_response(result.node),
                                "similarity": result.score,
                            }
                        )

                if ctx:
                    self.logger.info(
                        "Found %s similar nodes (using text search)", len(similar_nodes)
                    )

                return {
                    "similar_nodes": similar_nodes,
                    "count": len(similar_nodes),
                    "reference_node_id": node_id,
                    "method": "text_search_fallback",
                }

            # Use dedicated similarity search if available
            similar_nodes_with_scores = await self.node_use_case.find_similar_nodes(
                node_id=domain_node_id,
                limit=limit,
                threshold=actual_threshold,
            )

            # Convert to MCP response format
            similar_nodes = []
            for node, similarity in similar_nodes_with_scores:
                similar_nodes.append(
                    {
                        **self._node_to_mcp_response(node),
                        "similarity": similarity,
                    }
                )

            if ctx:
                self.logger.info("Found %s similar nodes", len(similar_nodes))

            return {
                "similar_nodes": similar_nodes,
                "count": len(similar_nodes),
                "reference_node_id": node_id,
                "method": "embedding_similarity",
            }

        except Exception as e:
            self.logger.error("Error searching similar nodes: %s", e)
            raise MCPServerException(
                server_state="running",
                operation="search_similar_nodes",
                message=str(e),
                original_error=e,
            ) from e

    # === Server Lifecycle ===

    def start(
        self,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """
        Start the MCP server.

        Args:
            host: Server host (uses config default if not provided)
            port: Server port (uses config default if not provided)
        """
        actual_host = host or self.config.host
        actual_port = port or self.config.port

        self.logger.info("Starting Knowledge Graph MCP server on %s:%s", actual_host, actual_port)

        # Start the server using FastMCP's built-in method
        self.mcp_server.run()

    def close(self) -> None:
        """Close the server."""
        self.logger.info("Closing MCP server")
        # FastMCP handles server lifecycle automatically
