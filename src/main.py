"""
Main entry point for the SQLite Knowledge Graph Vector MCP application.
"""

import argparse
import asyncio
import os

import numpy as np

from src.adapters.fastmcp.config import FastMCPConfig
from src.adapters.fastmcp.server import KnowledgeGraphServer
from src.adapters.hnsw.embeddings import EmbeddingManager
from src.adapters.sqlite3.database import SQLiteDatabase
from src.adapters.sqlite3.document_repository import SQLiteDocumentRepository
from src.adapters.sqlite3.graph.entities import EntityManager
from src.adapters.sqlite3.graph.relationships import RelationshipManager
from src.adapters.sqlite3.schema import SchemaManager
from src.adapters.sqlite3.vector_store import SQLiteVectorStore
from src.adapters.testing.text_embedder import RandomTextEmbedder
from src.domain.entities.document import Document, DocumentType
from src.domain.entities.node import Node, NodeType
from src.domain.entities.relationship import Relationship, RelationshipType
from src.domain.services.knowledge_search import (
    KnowledgeSearchService,
    SearchResult,
    SearchResultCollection,
    SearchStrategy,
)
from src.domain.value_objects.document_id import DocumentId
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.relationship_id import RelationshipId
from src.domain.value_objects.vector import Vector
from src.ports.text_embedder import TextEmbedder
from src.use_cases.document import DocumentManagementUseCase
from src.use_cases.knowledge_search import KnowledgeSearchUseCase
from src.use_cases.node import NodeManagementUseCase
from src.use_cases.relationship import RelationshipManagementUseCase


# Concrete implementations for Use Cases (simplified for main.py)
# In a larger application, these would be in a separate application/use_cases directory.
class ConcreteNodeManagementUseCase(NodeManagementUseCase):
    """노드 관리 유스케이스의 구체적 구현체."""

    def __init__(
        self,
        entity_manager: EntityManager,
        embedding_manager: EmbeddingManager,
        vector_store: SQLiteVectorStore,
        text_embedder: TextEmbedder,
    ):
        self.entity_manager = entity_manager
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.text_embedder = text_embedder

    async def create_node(
        self,
        name: str,
        node_type: NodeType,
        description: str | None = None,
        properties: dict | None = None,
        source_documents: list[DocumentId] | None = None,
    ) -> Node:
        # Simplified: A real implementation would convert Node to Entity and handle embedding
        entity = self.entity_manager.create_entity(
            entity_type=node_type.value, name=name, properties=properties
        )
        # Generate and store embedding
        embedding_text = f"{name} {description or ''} {properties or ''}"
        embedding_vector = await self.text_embedder.embed_text(embedding_text)
        self.embedding_manager.store_embedding(
            "node", entity.id, np.array(embedding_vector.embedding), embedding_vector.model_name
        )
        return Node(
            id=NodeId(str(entity.id)),
            name=entity.name or "",
            node_type=NodeType(entity.type),
            properties=entity.properties or {},
        )

    async def get_node(self, node_id: NodeId) -> Node | None:
        entity = self.entity_manager.get_entity(int(node_id.value))
        return (
            Node(
                id=NodeId(str(entity.id)),
                name=entity.name or "",
                node_type=NodeType(entity.type),
                properties=entity.properties or {},
            )
            if entity
            else None
        )

    async def list_nodes(
        self, node_type: NodeType | None = None, limit: int | None = None, offset: int | None = None
    ) -> list[Node]:
        entities, _ = self.entity_manager.find_entities(
            entity_type=node_type.value if node_type else None,
            limit=limit or 100,
            offset=offset or 0,
        )
        return [
            Node(
                id=NodeId(str(e.id)),
                name=e.name or "",
                node_type=NodeType(e.type),
                properties=e.properties or {},
            )
            for e in entities
        ]

    async def update_node(
        self,
        node_id: NodeId,
        name: str | None = None,
        description: str | None = None,
        properties: dict | None = None,
    ) -> Node:
        entity = self.entity_manager.update_entity(
            int(node_id.value), name=name, properties=properties
        )
        if not entity:
            raise ValueError("Node not found")
        # Update embedding if text content changes
        if name or description or properties:
            embedding_text = f"{entity.name or ''} {description or ''} {properties or ''}"  # Use current description/properties
            embedding_vector = await self.text_embedder.embed_text(embedding_text)
            self.embedding_manager.store_embedding(
                "node", entity.id, np.array(embedding_vector.embedding), embedding_vector.model_name
            )

        return Node(
            id=NodeId(str(entity.id)),
            name=entity.name or "",
            node_type=NodeType(entity.type),
            properties=entity.properties or {},
        )

    async def delete_node(self, node_id: NodeId) -> bool:
        self.embedding_manager.delete_embedding("node", int(node_id.value))
        return self.entity_manager.delete_entity(int(node_id.value))

    async def add_node_to_document(self, node_id: NodeId, document_id: DocumentId) -> bool:
        # This would involve a new repository method or direct DB operation
        return True  # Simplified

    async def remove_node_from_document(self, node_id: NodeId, document_id: DocumentId) -> bool:
        # This would involve a new repository method or direct DB operation
        return True  # Simplified


class ConcreteRelationshipManagementUseCase(RelationshipManagementUseCase):
    """관계 관리 유스케이스의 구체적 구현체."""

    def __init__(
        self,
        relationship_manager: RelationshipManager,
        embedding_manager: EmbeddingManager,
        vector_store: SQLiteVectorStore,
        text_embedder: TextEmbedder,
    ):
        self.relationship_manager = relationship_manager
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.text_embedder = text_embedder

    async def create_relationship(
        self,
        source_node_id: NodeId,
        target_node_id: NodeId,
        relationship_type: RelationshipType,
        label: str,
        properties: dict | None = None,
        weight: float = 1.0,
    ) -> Relationship:
        rel = self.relationship_manager.create_relationship(
            source_id=int(source_node_id.value),
            target_id=int(target_node_id.value),
            relation_type=relationship_type.value,
            properties=properties,
        )
        # Generate and store embedding
        embedding_text = f"{label} {properties or ''}"
        embedding_vector = await self.text_embedder.embed_text(embedding_text)
        self.embedding_manager.store_embedding(
            "edge", rel.id, np.array(embedding_vector.embedding), embedding_vector.model_name
        )
        return Relationship(
            id=RelationshipId(str(rel.id)),
            source_node_id=NodeId(str(rel.source_id)),
            target_node_id=NodeId(str(rel.target_id)),
            relationship_type=RelationshipType(rel.relation_type),
            label=rel.relation_type,
            properties=rel.properties or {},
        )

    async def get_relationship(self, relationship_id: RelationshipId) -> Relationship | None:
        rel = self.relationship_manager.get_relationship(int(relationship_id.value))
        return (
            Relationship(
                id=RelationshipId(str(rel.id)),
                source_node_id=NodeId(str(rel.source_id)),
                target_node_id=NodeId(str(rel.target_id)),
                relationship_type=RelationshipType(rel.relation_type),
                label=rel.relation_type,
                properties=rel.properties or {},
            )
            if rel
            else None
        )

    async def list_relationships(
        self,
        relationship_type: RelationshipType | None = None,
        source_node_id: NodeId | None = None,
        target_node_id: NodeId | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Relationship]:
        rels, _ = self.relationship_manager.find_relationships(
            relation_type=relationship_type.value if relationship_type else None,
            source_id=int(source_node_id.value) if source_node_id else None,
            target_id=int(target_node_id.value) if target_node_id else None,
            limit=limit or 100,
            offset=offset or 0,
        )
        return [
            Relationship(
                id=RelationshipId(str(r.id)),
                source_node_id=NodeId(str(r.source_id)),
                target_node_id=NodeId(str(r.target_id)),
                relationship_type=RelationshipType(r.relation_type),
                label=r.relation_type,
                properties=r.properties or {},
            )
            for r in rels
        ]

    async def update_relationship(
        self,
        relationship_id: RelationshipId,
        label: str | None = None,
        properties: dict | None = None,
        weight: float | None = None,
    ) -> Relationship:
        rel = self.relationship_manager.update_relationship(
            int(relationship_id.value), properties=properties if properties is not None else {}
        )
        if not rel:
            raise ValueError("Relationship not found")
        # Update embedding if text content changes
        if label or properties:
            embedding_text = f"{label or rel.relation_type} {properties or ''}"
            embedding_vector = await self.text_embedder.embed_text(embedding_text)
            self.embedding_manager.store_embedding(
                "edge", rel.id, np.array(embedding_vector.embedding), embedding_vector.model_name
            )
        return Relationship(
            id=RelationshipId(str(rel.id)),
            source_node_id=NodeId(str(rel.source_id)),
            target_node_id=NodeId(str(rel.target_id)),
            relationship_type=RelationshipType(rel.relation_type),
            label=rel.relation_type,
            properties=rel.properties or {},
        )

    async def delete_relationship(self, relationship_id: RelationshipId) -> bool:
        self.embedding_manager.delete_embedding("edge", int(relationship_id.value))
        return self.relationship_manager.delete_relationship(int(relationship_id.value))

    async def get_node_relationships(
        self, node_id: NodeId, direction: str = "both"
    ) -> list[Relationship]:
        rels, _ = self.relationship_manager.get_entity_relationships(
            int(node_id.value), direction=direction
        )
        return [
            Relationship(
                id=RelationshipId(str(r.id)),
                source_node_id=NodeId(str(r.source_id)),
                target_node_id=NodeId(str(r.target_id)),
                relationship_type=RelationshipType(r.relation_type),
                label=r.relation_type,
                properties=r.properties or {},
            )
            for r in rels
        ]


class ConcreteKnowledgeSearchUseCase(KnowledgeSearchUseCase):
    """지식 검색 유스케이스의 구체적 구현체."""

    def __init__(
        self,
        knowledge_search_service: KnowledgeSearchService,
        node_use_case: NodeManagementUseCase,
        document_use_case: DocumentManagementUseCase,
        relationship_use_case: RelationshipManagementUseCase,
        vector_store: SQLiteVectorStore,
        text_embedder: TextEmbedder,
    ):
        self.knowledge_search_service = knowledge_search_service
        self.node_use_case = node_use_case
        self.document_use_case = document_use_case
        self.relationship_use_case = relationship_use_case
        self.vector_store = vector_store
        self.text_embedder = text_embedder

    async def search_knowledge(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.HYBRID,
        limit: int = 10,
        similarity_threshold: float = 0.5,
        include_documents: bool = True,
        include_nodes: bool = True,
        include_relationships: bool = True,
    ):
        # This is a simplified implementation. A real one would involve more complex logic
        # and potentially call search methods on other services/use cases.
        # For now, we'll just use semantic search with the vector store.
        query_embedding_result = await self.text_embedder.embed_text(query)
        query_embedding = query_embedding_result.embedding

        # Perform semantic search for nodes and relationships
        similar_nodes_with_scores = await self.vector_store.search_similar(
            Vector(query_embedding), k=limit, filter_criteria={"entity_type": "node"}
        )
        similar_relationships_with_scores = await self.vector_store.search_similar(
            Vector(query_embedding), k=limit, filter_criteria={"entity_type": "edge"}
        )

        # Convert results to SearchResultCollection format
        results = []
        for vector_id, score in similar_nodes_with_scores:
            node = await self.node_use_case.get_node(NodeId(vector_id))
            if node:
                results.append(SearchResult(score=score, node=node))

        for vector_id, score in similar_relationships_with_scores:
            rel = await self.relationship_use_case.get_relationship(RelationshipId(vector_id))
            if rel:
                results.append(SearchResult(score=score, relationship=rel))

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        return SearchResultCollection(
            results=results, total_count=len(results), query=query, strategy=strategy
        )

    async def search_documents(
        self, query: str, limit: int = 10, similarity_threshold: float = 0.5
    ) -> list[Document]:
        # Simplified: implement actual document search
        return []

    async def search_nodes(
        self,
        query: str,
        node_types: list[str] | None = None,
        limit: int = 10,
        similarity_threshold: float = 0.5,
    ) -> list[Node]:
        # Simplified: implement actual node search
        return []

    async def search_relationships(
        self,
        query: str,
        relationship_types: list[str] | None = None,
        limit: int = 10,
        similarity_threshold: float = 0.5,
    ) -> list[Relationship]:
        # Simplified: implement actual relationship search
        return []

    async def semantic_search(self, query: str, limit: int = 10, similarity_threshold: float = 0.7):
        # This method could delegate to search_knowledge with SEMANTIC strategy
        return await self.search_knowledge(
            query,
            strategy=SearchStrategy.SEMANTIC,
            limit=limit,
            similarity_threshold=similarity_threshold,
        )


# Placeholder for DocumentManagementUseCase as it's a dependency for KnowledgeSearchUseCase
class ConcreteDocumentManagementUseCase(DocumentManagementUseCase):
    """문서 관리 유스케이스의 구체적 구현체."""

    def __init__(self, document_repository: SQLiteDocumentRepository):
        self.document_repository = document_repository

    async def create_document(
        self, title: str, content: str, metadata: dict | None = None
    ) -> Document:
        # Simplified
        return Document(
            id=DocumentId("1"),
            title=title,
            content=content,
            doc_type=DocumentType.TEXT,
            metadata=metadata or {},
        )

    async def get_document(self, document_id: DocumentId) -> Document | None:
        # Simplified
        return None

    async def list_documents(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[Document]:
        # Simplified
        return []

    async def update_document(
        self,
        document_id: DocumentId,
        title: str | None = None,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> Document:
        # Simplified
        return Document(
            id=document_id,
            title=title or "",
            content=content or "",
            doc_type=DocumentType.TEXT,
            metadata=metadata or {},
        )

    async def delete_document(self, document_id: DocumentId) -> bool:
        # Simplified
        return True


async def main():
    """Run the Knowledge Graph MCP server."""
    parser = argparse.ArgumentParser(description="Run the Knowledge Graph MCP server")
    parser.add_argument(
        "--db-path",
        type=str,
        default="knowledge_graph.db",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--vector-dir",
        type=str,
        default="vector_indexes",
        help="Directory for vector index files",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "stdio", "streamable-http"],
        help="Transport protocol to use",
    )
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="Initialize the database schema if it doesn't exist",
    )
    parser.add_argument("--dimension", type=int, default=128, help="Dimension of embedding vectors")
    parser.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "inner_product", "l2"],
        help="Similarity metric for vector search",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="Knowledge Graph Server",
        help="Name of the MCP server",
    )
    parser.add_argument(
        "--server-instructions",
        type=str,
        default="SQLite-based knowledge graph with vector search capabilities",
        help="Instructions for the MCP server",
    )

    args = parser.parse_args()

    # Create vector directory if it doesn't exist
    os.makedirs(args.vector_dir, exist_ok=True)

    # Setup database connection and schema
    # Use SQLiteDatabase which implements the Database port
    db_adapter = SQLiteDatabase(db_path=args.db_path)  # Change to SQLiteDatabase
    await db_adapter.connect()
    assert db_adapter.connection is not None  # Ensure connection is not None after connect

    schema_manager = SchemaManager(args.db_path)  # SchemaManager expects db_path string
    if args.init_schema:
        print(f"Initializing database schema at {args.db_path}")
        schema_manager.initialize_schema()

    # Initialize repositories and services. Pass db_adapter where Database port is expected.
    # For EntityManager and RelationshipManager, they expect sqlite3.Connection
    # which is _connection attribute of SQLiteDatabase.
    entity_manager = EntityManager(db_adapter.connection)
    relationship_manager = RelationshipManager(db_adapter.connection)
    document_repository = SQLiteDocumentRepository(
        db_adapter
    )  # Pass db_adapter as it implements Database port
    embedding_manager = EmbeddingManager(db_adapter.connection)
    vector_store = SQLiteVectorStore(
        args.db_path, table_name="knowledge_vectors"
    )  # Use db_path here
    await vector_store.initialize_store(
        dimension=args.dimension, metric=args.similarity
    )  # Initialize vector store asynchronously
    text_embedder = RandomTextEmbedder(dimension=args.dimension)  # Use RandomTextEmbedder directly
    knowledge_search_service = KnowledgeSearchService()

    # Initialize Use Cases
    node_use_case = ConcreteNodeManagementUseCase(
        entity_manager, embedding_manager, vector_store, text_embedder
    )
    relationship_use_case = ConcreteRelationshipManagementUseCase(
        relationship_manager, embedding_manager, vector_store, text_embedder
    )
    document_use_case = ConcreteDocumentManagementUseCase(document_repository)
    knowledge_search_use_case = ConcreteKnowledgeSearchUseCase(
        knowledge_search_service,
        node_use_case,
        document_use_case,
        relationship_use_case,
        vector_store,
        text_embedder,
    )

    # Create FastMCPConfig
    config = FastMCPConfig(
        host=args.host,
        port=args.port,
        # Other config parameters can be set here if needed
    )

    # Create and start the server
    print(f"Starting Knowledge Graph MCP server at {args.host}:{args.port}")
    server = KnowledgeGraphServer(
        node_use_case=node_use_case,
        relationship_use_case=relationship_use_case,
        knowledge_search_use_case=knowledge_search_use_case,
        config=config,
    )

    try:
        server.start(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("Server stopped by user")
    finally:
        server.close()
        print("Server resources cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
