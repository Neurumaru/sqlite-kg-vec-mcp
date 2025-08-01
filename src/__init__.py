"""
SQLite Knowledge Graph and Vector Database with MCP Server Interface.

This package combines a SQLite-based knowledge graph with vector storage
(optionally using HNSW index) and provides an interface through an MCP
(Model Context Protocol) server.
"""

import sqlite3
from pathlib import Path

from .adapters.hnsw.embeddings import EmbeddingManager
from .adapters.huggingface.text_embedder import HuggingFaceTextEmbedder
from .adapters.openai.text_embedder import OpenAITextEmbedder
from .adapters.sqlite3.connection import DatabaseConnection
from .adapters.sqlite3.graph.entities import EntityManager
from .adapters.sqlite3.graph.relationships import RelationshipManager
from .adapters.sqlite3.graph.traversal import GraphTraversal
from .adapters.sqlite3.schema import SchemaManager
from .adapters.testing.text_embedder import RandomTextEmbedder

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Look for .env file in project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)

except ImportError:
    # python-dotenv not installed, skip loading
    pass

__version__ = "0.1.0"

# from .adapters.hnsw.search import SearchResult, VectorSearch  # TODO: Fix dependencies
# from .adapters.hnsw.text_embedder import (
#     VectorTextEmbedder, create_embedder
# ) # TODO: Implement text_embedder


# Create embedder function for examples
def create_embedder(embedder_type="random", **kwargs):
    """Create a text embedder based on type.

    Args:
        embedder_type: Type of embedder ('random', 'openai',
            'sentence-transformers')
        **kwargs: Additional arguments for embedder

    Returns:
        Text embedder instance
    """
    if embedder_type == "random":
        return RandomTextEmbedder(dimension=kwargs.get("dimension", 128))
    if embedder_type == "openai":
        return OpenAITextEmbedder(**kwargs)
    if embedder_type == "sentence-transformers":
        return HuggingFaceTextEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder type: {embedder_type}")


# MCP Server export
try:
    from .adapters.fastmcp.server import KnowledgeGraphServer

    __all__ = ["KnowledgeGraph", "EmbeddingManager", "create_embedder", "KnowledgeGraphServer"]
except ImportError:
    # If MCP dependencies aren't available, just export the main classes
    __all__ = ["KnowledgeGraph", "EmbeddingManager", "create_embedder"]

# Export main classes - avoid direct imports to prevent circular dependencies
# from .adapters.sqlite3.connection import DatabaseConnection
# from .adapters.sqlite3.graph.entities import Entity, EntityManager
# from .adapters.sqlite3.graph.relationships import Relationship, RelationshipManager
# from .adapters.sqlite3.graph.traversal import GraphTraversal, PathNode
# from .adapters.sqlite3.schema import SchemaManager

# Import server API conditionally
# try:
#     from .adapters.fastmcp.server import KnowledgeGraphServer
# except ImportError:
#     # If MCP server dependencies aren't available, provide a message
#     class KnowledgeGraphServer:
#         def __init__(self, *args, **kwargs):
#             raise ImportError(
#                 "KnowledgeGraphServer requires additional dependencies. "
#                 "Please install 'fastmcp' package to use MCP server functionality."
#             )


# Convenience class for direct usage
class KnowledgeGraph:
    """
    Main knowledge graph interface combining entity, relationship,
    and vector search functionality.
    """

    def __init__(
        self,
        db_path,
        vector_index_dir=None,
        embedding_dim=128,
        text_embedder=None,
        embedder_type="sentence-transformers",
        embedder_kwargs=None,
    ):
        """
        Initialize a knowledge graph.

        Args:
            db_path: Path to SQLite database file
            vector_index_dir: Directory for storing vector index files
            embedding_dim: Dimension of embedding vectors
            text_embedder: VectorTextEmbedder instance for text-to-vector conversion
            embedder_type: Type of embedder to create if text_embedder is None
            embedder_kwargs: Arguments for embedder creation
        """
        # Use delayed imports to avoid circular dependencies
        # from .adapters.sqlite3.connection import DatabaseConnection
        # from .adapters.sqlite3.graph.entities import EntityManager
        # from .adapters.sqlite3.graph.relationships import RelationshipManager
        # from .adapters.sqlite3.graph.traversal import GraphTraversal
        # from .adapters.sqlite3.schema import SchemaManager

        # Initialize database
        self.db_connection = DatabaseConnection(db_path)
        self.conn = self.db_connection.connect()

        # Initialize schema
        schema_manager = SchemaManager(db_path)
        try:
            if schema_manager.get_schema_version() == 0:
                schema_manager.initialize_schema()
        except sqlite3.OperationalError:
            # Schema doesn't exist yet
            schema_manager.initialize_schema()

        # Create managers
        self.entity_manager = EntityManager(self.conn)
        self.relationship_manager = RelationshipManager(self.conn)
        self.embedding_manager = EmbeddingManager(self.conn)
        self.graph_traversal = GraphTraversal(self.conn)
        # TODO: Re-enable when VectorSearch dependencies are fixed
        # self.vector_search = VectorSearch(
        #     connection=self.conn,
        #     index_dir=vector_index_dir,
        #     embedding_dim=embedding_dim,
        #     text_embedder=text_embedder,
        #     embedder_type=embedder_type,
        #     embedder_kwargs=embedder_kwargs,
        # )

    # Entity methods
    def create_node(self, node_type, name=None, properties=None):
        """Create a new node in the graph."""
        return self.entity_manager.create_entity(node_type, name, properties)

    def get_node(self, node_id):
        """Get a node by ID."""
        return self.entity_manager.get_entity(node_id)

    def get_node_by_uuid(self, uuid):
        """Get a node by UUID."""
        return self.entity_manager.get_entity_by_uuid(uuid)

    def update_node(self, node_id, name=None, properties=None):
        """Update a node's properties."""
        return self.entity_manager.update_entity(node_id, name, properties)

    def delete_node(self, node_id):
        """Delete a node."""
        return self.entity_manager.delete_entity(node_id)

    def find_nodes(self, node_type=None, name_pattern=None, properties=None, limit=100, offset=0):
        """Find nodes matching criteria."""
        return self.entity_manager.find_entities(
            entity_type=node_type,
            name_pattern=name_pattern,
            property_filters=properties,
            limit=limit,
            offset=offset,
        )

    # Relationship methods
    def create_edge(self, source_id, target_id, relation_type, properties=None):
        """Create a new edge between nodes."""
        return self.relationship_manager.create_relationship(
            source_id, target_id, relation_type, properties
        )

    def get_edge(self, edge_id, include_entities=False):
        """Get an edge by ID."""
        return self.relationship_manager.get_relationship(edge_id, include_entities)

    def update_edge(self, edge_id, properties):
        """Update an edge's properties."""
        return self.relationship_manager.update_relationship(edge_id, properties)

    def delete_edge(self, edge_id):
        """Delete an edge."""
        return self.relationship_manager.delete_relationship(edge_id)

    def find_edges(
        self,
        source_id=None,
        target_id=None,
        relation_type=None,
        properties=None,
        include_entities=False,
        limit=100,
        offset=0,
    ):
        """Find edges matching criteria."""
        return self.relationship_manager.find_relationships(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            property_filters=properties,
            include_entities=include_entities,
            limit=limit,
            offset=offset,
        )

    # Graph traversal methods
    def get_neighbors(
        self,
        node_id,
        direction="both",
        relation_types=None,
        entity_types=None,
        limit=100,
    ):
        """Get neighboring nodes."""
        return self.graph_traversal.get_neighbors(
            node_id, direction, relation_types, entity_types, limit
        )

    def find_paths(self, start_id, end_id, max_depth=5, relation_types=None, entity_types=None):
        """Find paths between nodes."""
        return self.graph_traversal.find_shortest_path(
            start_id, end_id, max_depth, relation_types, entity_types
        )

    # TODO: Vector search methods - re-enable when VectorSearch is fixed
    # def search_similar_nodes(
    #     self,
    #     query_vector=None,
    #     node_id=None,
    #     limit=10,
    #     entity_types=None,
    #     include_entities=True,
    # ):
    #     """Search for similar nodes."""
    #     if node_id is not None:
    #         return self.vector_search.search_similar_to_entity(
    #             "node", node_id, limit, entity_types, include_entities
    #         )
    #     elif query_vector is not None:
    #         return self.vector_search.search_similar(
    #             query_vector, limit, entity_types, include_entities
    #         )
    #     else:
    #         raise ValueError("Either query_vector or node_id must be provided")

    # def search_by_text(
    #     self, query_text, limit=10, entity_types=None, include_entities=True
    # ):
    #     """Search using text query."""
    #     return self.vector_search.search_by_text(
    #         query_text, limit, entity_types, include_entities
    #     )

    def close(self):
        """Close the database connection."""
        self.db_connection.close()
