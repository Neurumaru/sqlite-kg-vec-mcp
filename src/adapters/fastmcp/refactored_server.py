"""
Refactored API endpoints and handlers for the MCP server interface.

This version splits the large KnowledgeGraphServer class into smaller,
focused handler classes following the Single Responsibility Principle.
"""

import logging

from fastmcp import FastMCP

from src.use_cases.knowledge_search import KnowledgeSearchUseCase
from src.use_cases.node import NodeManagementUseCase
from src.use_cases.relationship import RelationshipManagementUseCase

from .config import FastMCPConfig
from .exceptions import MCPServerException
from .handlers import NodeHandler, RelationshipHandler, SearchHandler


class KnowledgeGraphServer:
    """
    Refactored MCP server providing a knowledge graph API.

    This server orchestrates requests between MCP clients and specialized
    handlers, each responsible for a specific domain of operations.
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
        self.config = config

        # Setup logging
        self.logger = logging.getLogger("kg_server")
        self.logger.setLevel(getattr(logging, config.log_level))

        # Initialize specialized handlers
        self.node_handler = NodeHandler(node_use_case, config)
        self.relationship_handler = RelationshipHandler(relationship_use_case, config)
        self.search_handler = SearchHandler(
            node_use_case, relationship_use_case, knowledge_search_use_case, config
        )

        # Create MCP server with FastMCP
        self.mcp_server: FastMCP = FastMCP(
            name="Knowledge Graph Server",
            instructions="SQLite-based knowledge graph with vector search capabilities",
        )

        # Register all tools
        self._register_tools()

        self.logger.info("Knowledge Graph Server initialized with specialized handlers")

    def _register_tools(self):
        """Register all API endpoint tools with their respective handlers."""
        # Node management tools
        self.mcp_server.tool()(self.node_handler.create_node)
        self.mcp_server.tool()(self.node_handler.get_node)
        self.mcp_server.tool()(self.node_handler.update_node)
        self.mcp_server.tool()(self.node_handler.delete_node)
        self.mcp_server.tool()(self.node_handler.find_nodes)

        # Relationship management tools
        self.mcp_server.tool()(self.relationship_handler.create_edge)
        self.mcp_server.tool()(self.relationship_handler.get_edge)
        self.mcp_server.tool()(self.relationship_handler.update_edge)
        self.mcp_server.tool()(self.relationship_handler.delete_edge)
        self.mcp_server.tool()(self.relationship_handler.find_edges)

        # Search and traversal tools
        self.mcp_server.tool()(self.search_handler.get_neighbors)
        self.mcp_server.tool()(self.search_handler.find_paths)
        self.mcp_server.tool()(self.search_handler.search_similar_nodes)
        self.mcp_server.tool()(self.search_handler.search_by_text)

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

        try:
            # Start the server using FastMCP's built-in method
            self.mcp_server.run()
        except Exception as e:
            self.logger.error("Failed to start MCP server: %s", e)
            raise MCPServerException(
                server_state="starting",
                operation="start",
                message=str(e),
                host=actual_host,
                port=actual_port,
                original_error=e,
            ) from e

    def close(self) -> None:
        """Close the server and cleanup resources."""
        self.logger.info("Closing MCP server and cleaning up resources")
        try:
            # FastMCP handles server lifecycle automatically
            # Additional cleanup can be added here if needed
            self.logger.info("MCP server closed successfully")
        except Exception as e:
            self.logger.error("Error during server shutdown: %s", e)
            raise MCPServerException(
                server_state="stopping",
                operation="close",
                message=str(e),
                original_error=e,
            ) from e
