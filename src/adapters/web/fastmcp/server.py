"""
API endpoints and handlers for the MCP server interface.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from fastmcp import Context, FastMCP

from src.adapters.database.connection import DatabaseConnection
from src.adapters.database.schema import SchemaManager
from src.adapters.graph.entities import Entity, EntityManager
from src.adapters.graph.relationships import Relationship, RelationshipManager
from src.adapters.graph.traversal import GraphTraversal
from src.adapters.vector.embeddings import EmbeddingManager
from src.adapters.vector.search import VectorSearch
from src.adapters.vector.text_embedder import VectorTextEmbedder


class KnowledgeGraphServer:
    """
    MCP server providing a knowledge graph API using SQLite and vector embeddings.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        vector_index_dir: Optional[Union[str, Path]] = None,
        embedding_dim: int = 128,
        vector_similarity: str = "cosine",
        log_level: str = "INFO",
        server_name: str = "Knowledge Graph Server",
        server_instructions: str = "SQLite-based knowledge graph with vector search capabilities",
        text_embedder: Optional[VectorTextEmbedder] = None,
        embedder_type: str = "sentence-transformers",
        embedder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the knowledge graph MCP server.

        Args:
            db_path: Path to SQLite database file
            vector_index_dir: Directory for storing vector index files
            embedding_dim: Dimension of embedding vectors
            vector_similarity: Similarity metric for vector search
            log_level: Logging level
            server_name: Name of the MCP server
            server_instructions: Instructions for the MCP server
            text_embedder: VectorTextEmbedder instance for text-to-vector conversion
            embedder_type: Type of embedder to create if text_embedder is None
            embedder_kwargs: Arguments for embedder creation
        """
        self.db_path = Path(db_path)
        self.vector_index_dir = Path(vector_index_dir) if vector_index_dir else None
        self.embedding_dim = embedding_dim
        self.vector_similarity = vector_similarity

        # Setup logging
        self.logger = logging.getLogger("kg_server")
        self.logger.setLevel(getattr(logging, log_level))

        # Initialize database connection
        self.db_connection = DatabaseConnection(db_path)
        self.conn = self.db_connection.connect()

        # Initialize schema if needed
        schema_manager = SchemaManager(db_path)
        if schema_manager.get_schema_version() == 0:
            self.logger.info("Initializing database schema")
            schema_manager.initialize_schema()

        # Initialize managers
        self.entity_manager = EntityManager(self.conn)
        self.relationship_manager = RelationshipManager(self.conn)
        self.embedding_manager = EmbeddingManager(self.conn)
        self.graph_traversal = GraphTraversal(self.conn)

        # Initialize vector search
        self.vector_search = VectorSearch(
            connection=self.conn,
            index_dir=vector_index_dir,
            embedding_dim=embedding_dim,
            space=vector_similarity,
            text_embedder=text_embedder,
            embedder_type=embedder_type,
            embedder_kwargs=embedder_kwargs,
        )

        # Create MCP server with FastMCP
        self.mcp_server = FastMCP(name=server_name, instructions=server_instructions)

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

    def create_node(
        self,
        type: str,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        uuid: Optional[str] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Create a new node in the knowledge graph.

        Args:
            type: Type of the node to create
            name: Name of the node (optional)
            properties: Custom properties for the node (optional)
            uuid: Custom UUID for the node (optional)
            ctx: MCP context object

        Returns:
            Created node data
        """
        if ctx:
            ctx.info(f"Creating node of type '{type}'")

        properties = properties or {}

        try:
            entity = self.entity_manager.create_entity(
                type=type, name=name, properties=properties, custom_uuid=uuid
            )

            return {
                "node_id": entity.id,
                "uuid": entity.uuid,
                "name": entity.name,
                "type": entity.type,
                "properties": entity.properties,
                "created_at": entity.created_at,
            }
        except Exception as e:
            self.logger.error(f"Error creating node: {e}")
            if ctx:
                ctx.error(f"Failed to create node: {e}")
            return {"error": str(e)}

    def get_node(
        self, id: Optional[int] = None, uuid: Optional[str] = None, ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Get a node from the knowledge graph.

        Args:
            id: ID of the node to retrieve (optional if uuid is provided)
            uuid: UUID of the node to retrieve (optional if id is provided)
            ctx: MCP context object

        Returns:
            Node data or error
        """
        # Check if we have either id or uuid
        if id is None and uuid is None:
            error_msg = "Missing required parameter: either id or uuid must be provided"
            if ctx:
                ctx.error(error_msg)
            return {"error": error_msg}

        try:
            # Get by ID or UUID
            if id is not None:
                entity = self.entity_manager.get_entity(id)
                if ctx:
                    ctx.info(f"Retrieving node with ID {id}")
            else:
                entity = self.entity_manager.get_entity_by_uuid(uuid)
                if ctx:
                    ctx.info(f"Retrieving node with UUID {uuid}")

            if not entity:
                error_msg = "Node not found"
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

            return {
                "node_id": entity.id,
                "uuid": entity.uuid,
                "name": entity.name,
                "type": entity.type,
                "properties": entity.properties,
                "created_at": entity.created_at,
                "updated_at": entity.updated_at,
            }
        except Exception as e:
            self.logger.error(f"Error getting node: {e}")
            if ctx:
                ctx.error(f"Failed to get node: {e}")
            return {"error": str(e)}

    def update_node(
        self,
        id: int,
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Update a node in the knowledge graph.

        Args:
            id: ID of the node to update
            name: New name for the node (optional)
            properties: New or updated properties (optional)
            ctx: MCP context object

        Returns:
            Updated node data or error
        """
        if ctx:
            ctx.info(f"Updating node with ID {id}")

        # At least one of name or properties must be provided
        if name is None and properties is None:
            error_msg = "At least one of name or properties must be provided"
            if ctx:
                ctx.error(error_msg)
            return {"error": error_msg}

        try:
            entity = self.entity_manager.update_entity(
                entity_id=id, name=name, properties=properties
            )

            if not entity:
                error_msg = "Node not found or update failed"
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

            return {
                "node_id": entity.id,
                "uuid": entity.uuid,
                "name": entity.name,
                "type": entity.type,
                "properties": entity.properties,
                "updated_at": entity.updated_at,
            }
        except Exception as e:
            self.logger.error(f"Error updating node: {e}")
            if ctx:
                ctx.error(f"Failed to update node: {e}")
            return {"error": str(e)}

    def delete_node(self, id: int, ctx: Context = None) -> Dict[str, Any]:
        """
        Delete a node from the knowledge graph.

        Args:
            id: ID of the node to delete
            ctx: MCP context object

        Returns:
            Success or error message
        """
        if ctx:
            ctx.info(f"Deleting node with ID {id}")

        try:
            success = self.entity_manager.delete_entity(id)

            if not success:
                error_msg = "Node not found or already deleted"
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

            return {"success": True, "message": f"Node {id} deleted successfully"}
        except Exception as e:
            self.logger.error(f"Error deleting node: {e}")
            if ctx:
                ctx.error(f"Failed to delete node: {e}")
            return {"error": str(e)}

    def find_nodes(
        self,
        type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Find nodes matching specified criteria.

        Args:
            type: Filter by entity type (optional)
            name_pattern: Filter by name pattern (optional)
            properties: Filter by properties (optional)
            limit: Maximum number of results to return (default 100)
            offset: Number of results to skip (default 0)
            ctx: MCP context object

        Returns:
            List of nodes matching the criteria
        """
        if ctx:
            ctx.info(
                f"Finding nodes with filters: type={type}, name_pattern={name_pattern}"
            )

        try:
            entities, total_count = self.entity_manager.find_entities(
                entity_type=type,
                name_pattern=name_pattern,
                property_filters=properties,
                limit=limit,
                offset=offset,
            )

            # Convert to dictionaries
            result_entities = []
            for entity in entities:
                result_entities.append(
                    {
                        "id": entity.id,
                        "uuid": entity.uuid,
                        "name": entity.name,
                        "type": entity.type,
                        "properties": entity.properties,
                        "created_at": entity.created_at,
                        "updated_at": entity.updated_at,
                    }
                )

            if ctx:
                ctx.info(
                    f"Found {total_count} matching nodes, returning {len(result_entities)}"
                )

            return {
                "nodes": result_entities,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
            }
        except Exception as e:
            self.logger.error(f"Error finding nodes: {e}")
            if ctx:
                ctx.error(f"Failed to find nodes: {e}")
            return {"error": str(e)}

    def create_edge(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        properties: Optional[Dict[str, Any]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Create a new edge in the knowledge graph.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            relation_type: Type of the relationship
            properties: Custom properties for the edge (optional)
            ctx: MCP context object

        Returns:
            Created edge data
        """
        if ctx:
            ctx.info(
                f"Creating edge of type '{relation_type}' from {source_id} to {target_id}"
            )

        properties = properties or {}

        try:
            relationship = self.relationship_manager.create_relationship(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                properties=properties,
            )

            return {
                "edge_id": relationship.id,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "relation_type": relationship.relation_type,
                "properties": relationship.properties,
                "created_at": relationship.created_at,
            }
        except Exception as e:
            self.logger.error(f"Error creating edge: {e}")
            if ctx:
                ctx.error(f"Failed to create edge: {e}")
            return {"error": str(e)}

    def get_edge(
        self, id: int, include_entities: bool = False, ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Get an edge from the knowledge graph.

        Args:
            id: ID of the edge to retrieve
            include_entities: Whether to include source and target entity data
            ctx: MCP context object

        Returns:
            Edge data or error
        """
        if ctx:
            ctx.info(f"Retrieving edge with ID {id}")

        try:
            relationship = self.relationship_manager.get_relationship(
                relationship_id=id, include_entities=include_entities
            )

            if not relationship:
                error_msg = "Edge not found"
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

            result = {
                "edge_id": relationship.id,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "relation_type": relationship.relation_type,
                "properties": relationship.properties,
                "created_at": relationship.created_at,
                "updated_at": relationship.updated_at,
            }

            # Include source and target entities if requested and loaded
            if include_entities:
                if relationship.source:
                    result["source"] = {
                        "id": relationship.source.id,
                        "name": relationship.source.name,
                        "type": relationship.source.type,
                    }
                if relationship.target:
                    result["target"] = {
                        "id": relationship.target.id,
                        "name": relationship.target.name,
                        "type": relationship.target.type,
                    }

            return result
        except Exception as e:
            self.logger.error(f"Error getting edge: {e}")
            if ctx:
                ctx.error(f"Failed to get edge: {e}")
            return {"error": str(e)}

    def update_edge(
        self, id: int, properties: Dict[str, Any], ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Update an edge in the knowledge graph.

        Args:
            id: ID of the edge to update
            properties: New or updated properties
            ctx: MCP context object

        Returns:
            Updated edge data or error
        """
        if ctx:
            ctx.info(f"Updating edge with ID {id}")

        try:
            relationship = self.relationship_manager.update_relationship(
                relationship_id=id, properties=properties
            )

            if not relationship:
                error_msg = "Edge not found or update failed"
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

            return {
                "edge_id": relationship.id,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "relation_type": relationship.relation_type,
                "properties": relationship.properties,
                "updated_at": relationship.updated_at,
            }
        except Exception as e:
            self.logger.error(f"Error updating edge: {e}")
            if ctx:
                ctx.error(f"Failed to update edge: {e}")
            return {"error": str(e)}

    def delete_edge(self, id: int, ctx: Context = None) -> Dict[str, Any]:
        """
        Delete an edge from the knowledge graph.

        Args:
            id: ID of the edge to delete
            ctx: MCP context object

        Returns:
            Success or error message
        """
        if ctx:
            ctx.info(f"Deleting edge with ID {id}")

        try:
            success = self.relationship_manager.delete_relationship(id)

            if not success:
                error_msg = "Edge not found or already deleted"
                if ctx:
                    ctx.error(error_msg)
                return {"error": error_msg}

            return {"success": True, "message": f"Edge {id} deleted successfully"}
        except Exception as e:
            self.logger.error(f"Error deleting edge: {e}")
            if ctx:
                ctx.error(f"Failed to delete edge: {e}")
            return {"error": str(e)}

    def find_edges(
        self,
        source_id: Optional[int] = None,
        target_id: Optional[int] = None,
        relation_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        include_entities: bool = False,
        limit: int = 100,
        offset: int = 0,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Find edges matching specified criteria.

        Args:
            source_id: Filter by source node ID (optional)
            target_id: Filter by target node ID (optional)
            relation_type: Filter by relationship type (optional)
            properties: Filter by properties (optional)
            include_entities: Whether to include source and target entity data
            limit: Maximum number of results to return (default 100)
            offset: Number of results to skip (default 0)
            ctx: MCP context object

        Returns:
            List of edges matching the criteria
        """
        if ctx:
            ctx.info(
                f"Finding edges with filters: source={source_id}, target={target_id}, type={relation_type}"
            )

        try:
            relationships, total_count = self.relationship_manager.find_relationships(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                property_filters=properties,
                include_entities=include_entities,
                limit=limit,
                offset=offset,
            )

            # Convert to dictionaries
            result_edges = []
            for rel in relationships:
                edge_dict = {
                    "id": rel.id,
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "relation_type": rel.relation_type,
                    "properties": rel.properties,
                    "created_at": rel.created_at,
                    "updated_at": rel.updated_at,
                }

                if include_entities:
                    if rel.source:
                        edge_dict["source"] = {
                            "id": rel.source.id,
                            "name": rel.source.name,
                            "type": rel.source.type,
                        }
                    if rel.target:
                        edge_dict["target"] = {
                            "id": rel.target.id,
                            "name": rel.target.name,
                            "type": rel.target.type,
                        }

                result_edges.append(edge_dict)

            if ctx:
                ctx.info(
                    f"Found {total_count} matching edges, returning {len(result_edges)}"
                )

            return {
                "edges": result_edges,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
            }
        except Exception as e:
            self.logger.error(f"Error finding edges: {e}")
            if ctx:
                ctx.error(f"Failed to find edges: {e}")
            return {"error": str(e)}

    def get_neighbors(
        self,
        node_id: int,
        direction: str = "both",
        relation_types: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        limit: int = 100,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Get neighboring nodes and relationships for a given node.

        Args:
            node_id: ID of the node to get neighbors for
            direction: Direction of relationships ('incoming', 'outgoing', or 'both')
            relation_types: Filter by relationship types (optional)
            entity_types: Filter by entity types (optional)
            limit: Maximum number of results to return (default 100)
            ctx: MCP context object

        Returns:
            List of neighboring nodes and their relationships
        """
        if ctx:
            ctx.info(f"Getting neighbors for node {node_id} with direction {direction}")

        try:
            neighbors = self.graph_traversal.get_neighbors(
                entity_id=node_id,
                direction=direction,
                relation_types=relation_types,
                entity_types=entity_types,
                limit=limit,
            )

            # Format results
            result_neighbors = []
            for entity, relationship in neighbors:
                neighbor = {
                    "node": {
                        "id": entity.id,
                        "uuid": entity.uuid,
                        "name": entity.name,
                        "type": entity.type,
                        "properties": entity.properties,
                    },
                    "edge": {
                        "id": relationship.id,
                        "relation_type": relationship.relation_type,
                        "source_id": relationship.source_id,
                        "target_id": relationship.target_id,
                        "properties": relationship.properties,
                    },
                    "direction": (
                        "outgoing" if relationship.source_id == node_id else "incoming"
                    ),
                }
                result_neighbors.append(neighbor)

            if ctx:
                ctx.info(f"Found {len(result_neighbors)} neighbors for node {node_id}")

            return {"neighbors": result_neighbors, "count": len(result_neighbors)}
        except Exception as e:
            self.logger.error(f"Error getting neighbors: {e}")
            if ctx:
                ctx.error(f"Failed to get neighbors: {e}")
            return {"error": str(e)}

    def find_paths(
        self,
        start_id: int,
        end_id: int,
        max_depth: int = 5,
        relation_types: Optional[List[str]] = None,
        entity_types: Optional[List[str]] = None,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Find paths between start and end nodes.

        Args:
            start_id: ID of the start node
            end_id: ID of the end node
            max_depth: Maximum path depth to search (default 5)
            relation_types: Filter by relationship types (optional)
            entity_types: Filter by entity types (optional)
            ctx: MCP context object

        Returns:
            List of paths between start and end nodes
        """
        if ctx:
            ctx.info(
                f"Finding paths from node {start_id} to node {end_id} with max depth {max_depth}"
            )

        try:
            paths = self.graph_traversal.find_paths(
                start_id=start_id,
                end_id=end_id,
                max_depth=max_depth,
                relation_types=relation_types,
                entity_types=entity_types,
            )

            # Format results
            result_paths = []
            for path in paths:
                path_nodes = []
                for i, node in enumerate(path):
                    path_node = {
                        "entity": {
                            "id": node.entity.id,
                            "uuid": node.entity.uuid,
                            "name": node.entity.name,
                            "type": node.entity.type,
                        },
                        "depth": node.depth,
                    }

                    # Include relationship (except for the start node)
                    if i > 0 and node.relationship:
                        path_node["relationship"] = {
                            "id": node.relationship.id,
                            "relation_type": node.relationship.relation_type,
                            "source_id": node.relationship.source_id,
                            "target_id": node.relationship.target_id,
                        }

                    path_nodes.append(path_node)

                result_paths.append(path_nodes)

            if ctx:
                ctx.info(
                    f"Found {len(result_paths)} paths from node {start_id} to node {end_id}"
                )

            return {"paths": result_paths, "count": len(result_paths)}
        except Exception as e:
            self.logger.error(f"Error finding paths: {e}")
            if ctx:
                ctx.error(f"Failed to find paths: {e}")
            return {"error": str(e)}

    def search_similar_nodes(
        self,
        node_id: Optional[int] = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 10,
        entity_types: Optional[List[str]] = None,
        include_entities: bool = True,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar nodes using vector similarity.

        Args:
            node_id: ID of the node to find similar nodes to (optional if query_vector is provided)
            query_vector: Query vector to find similar nodes to (optional if node_id is provided)
            limit: Maximum number of results to return (default 10)
            entity_types: Filter by entity types (optional)
            include_entities: Whether to include entity data in results
            ctx: MCP context object

        Returns:
            List of similar nodes
        """
        # Check for required parameters - either node_id or query_vector
        if node_id is None and query_vector is None:
            error_msg = "Missing required parameter: either node_id or query_vector must be provided"
            if ctx:
                ctx.error(error_msg)
            return {"error": error_msg}

        if ctx:
            if node_id is not None:
                ctx.info(f"Searching for nodes similar to node {node_id}")
            else:
                ctx.info(f"Searching for nodes similar to provided vector")

        entity_types = entity_types or ["node"]

        try:
            # Search based on node ID or query vector
            if node_id is not None:
                results = self.vector_search.search_similar_to_entity(
                    entity_type="node",
                    entity_id=node_id,
                    k=limit,
                    result_entity_types=entity_types,
                    include_entities=include_entities,
                )
            else:
                # Convert query vector to numpy array
                vector = np.array(query_vector, dtype=np.float32)
                results = self.vector_search.search_similar(
                    query_vector=vector,
                    k=limit,
                    entity_types=entity_types,
                    include_entities=include_entities,
                )

            # Format results
            result_items = []
            for item in results:
                result_items.append(item.to_dict())

            if ctx:
                ctx.info(f"Found {len(result_items)} similar nodes")

            return {"results": result_items, "count": len(result_items)}
        except Exception as e:
            self.logger.error(f"Error searching similar nodes: {e}")
            if ctx:
                ctx.error(f"Failed to search similar nodes: {e}")
            return {"error": str(e)}

    def search_by_text(
        self,
        query: str,
        limit: int = 10,
        entity_types: Optional[List[str]] = None,
        include_entities: bool = True,
        ctx: Optional[Context] = None,
    ) -> Dict[str, Any]:
        """
        Search for entities matching a text query.

        Args:
            query: Text query to search for
            limit: Maximum number of results to return (default 10)
            entity_types: Filter by entity types (optional)
            include_entities: Whether to include entity data in results
            ctx: MCP context object

        Returns:
            List of entities matching the text query
        """
        if ctx:
            ctx.info(f"Searching for entities matching text query: '{query}'")

        try:
            results = self.vector_search.search_by_text(
                query_text=query,
                k=limit,
                entity_types=entity_types,
                include_entities=include_entities,
            )

            # Format results
            result_items = []
            for item in results:
                result_items.append(item.to_dict())

            if ctx:
                ctx.info(f"Found {len(result_items)} entities matching text query")

            return {"results": result_items, "count": len(result_items)}
        except Exception as e:
            self.logger.error(f"Error searching by text: {e}")
            if ctx:
                ctx.error(f"Failed to search by text: {e}")
            return {"error": str(e)}

    async def maintenance_task(self) -> None:
        """Periodic maintenance task for vector index updates."""
        while True:
            try:
                # Update vector index with pending changes
                count = self.vector_search.update_index(batch_size=100)
                if count > 0:
                    self.logger.info(f"Updated vector index with {count} changes")
            except Exception as e:
                self.logger.error(f"Error in maintenance task: {e}")

            # Sleep for a while
            await asyncio.sleep(60)  # Run every minute

    def start(
        self, host: str = "127.0.0.1", port: int = 8080, transport: str = "sse"
    ) -> None:
        """
        Start the MCP server.

        Args:
            host: Server host
            port: Server port
            transport: Transport protocol (sse, stdio, or streamable-http)
        """
        # Start maintenance task in a separate thread
        asyncio.create_task(self.maintenance_task())

        # Start the server
        self.mcp_server.run(host=host, port=port, transport=transport)

    def close(self) -> None:
        """Close the server and database connections."""
        # Close database connection
        if hasattr(self, "db_connection"):
            self.db_connection.close()

        # Close MCP server
        if hasattr(self, "mcp_server"):
            # No need to stop in new API, handled automatically
            pass
