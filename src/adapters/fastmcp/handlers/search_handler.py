"""
Search and traversal handler for MCP operations.
"""

from typing import Any

from fastmcp import Context

from src.domain.services.knowledge_search import SearchStrategy
from src.domain.value_objects.node_id import NodeId
from src.use_cases.knowledge_search import KnowledgeSearchUseCase
from src.use_cases.node import NodeManagementUseCase
from src.use_cases.relationship import RelationshipManagementUseCase

from ..exceptions import MCPServerException
from .base import BaseHandler


class SearchHandler(BaseHandler):
    """Handler for search and graph traversal MCP operations."""

    def __init__(
        self,
        node_use_case: NodeManagementUseCase,
        relationship_use_case: RelationshipManagementUseCase,
        knowledge_search_use_case: KnowledgeSearchUseCase,
        config,
    ):
        """
        Initialize search handler.

        Args:
            node_use_case: Node management use case
            relationship_use_case: Relationship management use case
            knowledge_search_use_case: Knowledge search use case
            config: FastMCP configuration
        """
        super().__init__(config)
        self.node_use_case = node_use_case
        self.relationship_use_case = relationship_use_case
        self.knowledge_search_use_case = knowledge_search_use_case

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
                    return self._create_error_response(
                        f"Node {node_id} not found", "NODE_NOT_FOUND"
                    )

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

                self.logger.info("Found %s similar nodes (using text search)", len(similar_nodes))

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
                self.logger.info("No path found")
                return {
                    "path": None,
                    "length": 0,
                    "message": "No path found between the specified nodes",
                }

            # Convert path to MCP response format
            path_edges = [self._relationship_to_mcp_response(rel) for rel in path_relationships]

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
