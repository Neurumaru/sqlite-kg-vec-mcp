"""
Comprehensive test of all implemented features.
"""
import os
import tempfile
import unittest
import numpy as np

from src import KnowledgeGraph
from sqlite_kg_vec_mcp.db.connection import DatabaseConnection
from sqlite_kg_vec_mcp.db.schema import SchemaManager
from sqlite_kg_vec_mcp.graph.entities import EntityManager
from sqlite_kg_vec_mcp.graph.relationships import RelationshipManager
from sqlite_kg_vec_mcp.graph.traversal import GraphTraversal
from sqlite_kg_vec_mcp.vector.embeddings import EmbeddingManager
from sqlite_kg_vec_mcp.vector.hnsw import HNSWIndex
from sqlite_kg_vec_mcp.vector.search import VectorSearch, SearchResult


class TestAllFeatures(unittest.TestCase):
    """Test all implemented features."""
    
    def setUp(self):
        """Set up a test database."""
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Initialize database connection
        self.db_connection = DatabaseConnection(self.db_path)
        self.conn = self.db_connection.connect()
        
        # Initialize schema
        schema_manager = SchemaManager(self.db_path)
        schema_manager.initialize_schema()
        
        # Create managers
        self.entity_manager = EntityManager(self.conn)
        self.relationship_manager = RelationshipManager(self.conn)
        self.embedding_manager = EmbeddingManager(self.conn)
        self.graph_traversal = GraphTraversal(self.conn)
        
        # Create test data
        self._create_test_data()
        
        # Create vector data
        self._create_vector_data()
        
        # Initialize search
        self.vector_search = VectorSearch(
            connection=self.conn,
            index_dir=None,  # In-memory only
            embedding_dim=128
        )
    
    def tearDown(self):
        """Clean up the test database."""
        self.db_connection.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def _create_test_data(self):
        """Create test data."""
        # Create category nodes
        self.categories = {}
        for category in ['Technology', 'Finance', 'Healthcare']:
            entity = self.entity_manager.create_entity(
                type="Category",
                name=category
            )
            self.categories[category] = entity
        
        # Create company nodes
        self.companies = {}
        companies_data = [
            ("TechCorp", "Technology", 2010),
            ("FinBank", "Finance", 2005),
            ("HealthPlus", "Healthcare", 2015),
            ("DataSoft", "Technology", 2018)
        ]
        
        for name, industry, founded in companies_data:
            entity = self.entity_manager.create_entity(
                type="Company",
                name=name,
                properties={"industry": industry, "founded": founded}
            )
            self.companies[name] = entity
            
            # Create relationship to category
            self.relationship_manager.create_relationship(
                source_id=entity.id,
                target_id=self.categories[industry].id,
                relation_type="BELONGS_TO"
            )
        
        # Create person nodes
        self.persons = {}
        persons_data = [
            ("Alice", 30, "Data Scientist"),
            ("Bob", 35, "Software Engineer"),
            ("Charlie", 42, "Financial Analyst"),
            ("David", 28, "Healthcare Specialist"),
            ("Eve", 31, "Data Engineer")
        ]
        
        for name, age, occupation in persons_data:
            entity = self.entity_manager.create_entity(
                type="Person",
                name=name,
                properties={"age": age, "occupation": occupation}
            )
            self.persons[name] = entity
        
        # Create employment relationships
        employment_data = [
            ("Alice", "TechCorp", 2019, "Senior Data Scientist"),
            ("Bob", "TechCorp", 2015, "Lead Developer"),
            ("Charlie", "FinBank", 2010, "Senior Analyst"),
            ("David", "HealthPlus", 2018, "Research Specialist"),
            ("Eve", "DataSoft", 2020, "Data Architect")
        ]
        
        for person, company, since, position in employment_data:
            self.relationship_manager.create_relationship(
                source_id=self.persons[person].id,
                target_id=self.companies[company].id,
                relation_type="WORKS_FOR",
                properties={"since": since, "position": position}
            )
        
        # Create personal relationships
        friendship_data = [
            ("Alice", "Bob", "FRIEND", 2018),
            ("Bob", "Charlie", "FRIEND", 2019),
            ("Charlie", "David", "COLLEAGUE", 2015),
            ("David", "Eve", "FRIEND", 2017),
            ("Alice", "Eve", "COLLEAGUE", 2020)
        ]
        
        for person1, person2, rel_type, since in friendship_data:
            self.relationship_manager.create_relationship(
                source_id=self.persons[person1].id,
                target_id=self.persons[person2].id,
                relation_type=rel_type,
                properties={"since": since}
            )
    
    def _create_vector_data(self):
        """Create vector embeddings for test entities."""
        # Create random embeddings for entities
        for entity_type, entities in [
            ("node", list(self.persons.values()) + list(self.companies.values())),
            # We could also add edges here
        ]:
            for entity in entities:
                # Create a random vector (normally you'd use a proper embedding model)
                embedding = np.random.randn(128).astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                
                # Store the embedding
                self.embedding_manager.store_embedding(
                    entity_type=entity_type,
                    entity_id=entity.id,
                    embedding=embedding,
                    model_info="test_model"
                )
    
    def test_1_database_connection(self):
        """Test database connection."""
        # Verify connection is active
        self.assertTrue(self.conn is not None)
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1")
        self.assertEqual(cursor.fetchone()[0], 1)
        
        print("✓ Database connection works")
    
    def test_2_entity_operations(self):
        """Test entity operations."""
        # Create a new entity
        new_entity = self.entity_manager.create_entity(
            type="Product",
            name="TestProduct",
            properties={"price": 99.99, "stock": 100}
        )
        
        # Verify entity was created
        self.assertTrue(new_entity.id is not None)
        self.assertEqual(new_entity.name, "TestProduct")
        self.assertEqual(new_entity.type, "Product")
        self.assertEqual(new_entity.properties.get("price"), 99.99)
        
        # Retrieve the entity
        retrieved = self.entity_manager.get_entity(new_entity.id)
        self.assertEqual(retrieved.name, "TestProduct")
        
        # Update the entity
        updated = self.entity_manager.update_entity(
            new_entity.id,
            properties={"price": 89.99, "on_sale": True}
        )
        self.assertEqual(updated.properties.get("price"), 89.99)
        self.assertEqual(updated.properties.get("on_sale"), True)
        
        # Find entities by type
        products, count = self.entity_manager.find_entities(entity_type="Product")
        self.assertGreaterEqual(count, 1)
        self.assertTrue(any(e.name == "TestProduct" for e in products))
        
        # Delete the entity
        result = self.entity_manager.delete_entity(new_entity.id)
        self.assertTrue(result)
        
        # Verify it's gone
        deleted = self.entity_manager.get_entity(new_entity.id)
        self.assertIsNone(deleted)
        
        print("✓ Entity operations work")
    
    def test_3_relationship_operations(self):
        """Test relationship operations."""
        # Get two entities to create a relationship
        alice = self.persons["Alice"]
        david = self.persons["David"]
        
        # Create a relationship
        new_rel = self.relationship_manager.create_relationship(
            source_id=alice.id,
            target_id=david.id,
            relation_type="KNOWS",
            properties={"since": 2021, "context": "Conference"}
        )
        
        # Verify relationship was created
        self.assertTrue(new_rel.id is not None)
        self.assertEqual(new_rel.relation_type, "KNOWS")
        self.assertEqual(new_rel.properties.get("context"), "Conference")
        
        # Retrieve the relationship
        retrieved = self.relationship_manager.get_relationship(new_rel.id, include_entities=True)
        self.assertEqual(retrieved.relation_type, "KNOWS")
        self.assertEqual(retrieved.source.name, "Alice")
        self.assertEqual(retrieved.target.name, "David")
        
        # Update the relationship
        updated = self.relationship_manager.update_relationship(
            new_rel.id,
            properties={"since": 2021, "context": "Virtual Conference", "rating": 5}
        )
        self.assertEqual(updated.properties.get("context"), "Virtual Conference")
        self.assertEqual(updated.properties.get("rating"), 5)
        
        # Find relationships by type
        knows_rels, count = self.relationship_manager.find_relationships(relation_type="KNOWS")
        self.assertGreaterEqual(count, 1)
        
        # Find relationships by source
        alice_rels, count = self.relationship_manager.find_relationships(source_id=alice.id)
        self.assertGreaterEqual(count, 1)
        
        # Delete the relationship
        result = self.relationship_manager.delete_relationship(new_rel.id)
        self.assertTrue(result)
        
        # Verify it's gone
        deleted = self.relationship_manager.get_relationship(new_rel.id)
        self.assertIsNone(deleted)
        
        print("✓ Relationship operations work")
    
    def test_4_graph_traversal(self):
        """Test graph traversal functionality."""
        # Get neighbors
        alice = self.persons["Alice"]
        neighbors = self.graph_traversal.get_neighbors(alice.id)
        self.assertGreaterEqual(len(neighbors), 2)  # Alice should have at least 2 connections
        
        # Outgoing only
        outgoing = self.graph_traversal.get_neighbors(alice.id, direction="outgoing")
        self.assertGreaterEqual(len(outgoing), 1)
        
        # Filter by relation type
        work_relations = self.graph_traversal.get_neighbors(
            alice.id, 
            relation_types=["WORKS_FOR"]
        )
        self.assertGreaterEqual(len(work_relations), 1)
        
        # Find paths
        bob = self.persons["Bob"]
        charlie = self.persons["Charlie"]
        
        paths = self.graph_traversal.find_paths(alice.id, charlie.id)
        self.assertGreaterEqual(len(paths), 1)  # Should be at least one path
        
        # Test path through bob
        path_through_bob = False
        for path in paths:
            nodes = [node.entity.id for node in path]
            if alice.id in nodes and bob.id in nodes and charlie.id in nodes:
                path_through_bob = True
                break
                
        self.assertTrue(path_through_bob, "Should find a path from Alice to Charlie through Bob")
        
        print("✓ Graph traversal works")
    
    def test_5_vector_operations(self):
        """Test vector embedding operations."""
        # Get some embeddings
        alice = self.persons["Alice"]
        alice_embedding = self.embedding_manager.get_embedding("node", alice.id)
        
        self.assertIsNotNone(alice_embedding)
        self.assertEqual(alice_embedding.entity_id, alice.id)
        self.assertEqual(alice_embedding.entity_type, "node")
        self.assertEqual(alice_embedding.dimensions, 128)
        
        # Create a new embedding
        david = self.persons["David"]
        new_embedding = np.random.randn(128).astype(np.float32)
        new_embedding = new_embedding / np.linalg.norm(new_embedding)
        
        result = self.embedding_manager.store_embedding(
            entity_type="node",
            entity_id=david.id,
            embedding=new_embedding,
            model_info="updated_model",
            embedding_version=2
        )
        
        self.assertTrue(result)
        
        # Get the updated embedding
        david_embedding = self.embedding_manager.get_embedding("node", david.id)
        self.assertEqual(david_embedding.model_info, "updated_model")
        self.assertEqual(david_embedding.embedding_version, 2)
        
        # Get all embeddings
        all_embeddings = self.embedding_manager.get_all_embeddings("node")
        self.assertGreaterEqual(len(all_embeddings), len(self.persons) + len(self.companies))
        
        print("✓ Vector embedding operations work")
    
    def test_6_vector_search(self):
        """Test vector similarity search."""
        # Get a query vector
        alice = self.persons["Alice"]
        alice_embedding = self.embedding_manager.get_embedding("node", alice.id)

        # Create a new index just for testing
        hnsw_index = HNSWIndex(
            space="cosine",
            dim=128,
            ef_construction=200,
            M=16
        )
        hnsw_index.init_index(max_elements=100)

        # Add all test embeddings
        all_embeddings = self.embedding_manager.get_all_embeddings("node")
        for emb in all_embeddings:
            hnsw_index.add_item(
                entity_type=emb.entity_type,
                entity_id=emb.entity_id,
                vector=emb.embedding
            )

        # Replace the index with our test one
        self.vector_search.index = hnsw_index
        self.vector_search.index_loaded = True

        # Search similar nodes
        results = self.vector_search.search_similar(
            query_vector=alice_embedding.embedding,
            k=5,
            entity_types=["node"]
        )
        
        self.assertGreaterEqual(len(results), 1)
        
        # First result should be Alice herself
        self.assertEqual(results[0].entity_id, alice.id if hasattr(results[0], 'entity_id') else results[0].entity.id)
        
        # Search based on entity
        results2 = self.vector_search.search_similar_to_entity(
            entity_type="node",
            entity_id=alice.id,
            k=5
        )
        
        self.assertGreaterEqual(len(results2), 1)
        
        # Text search (simulated)
        results3 = self.vector_search.search_by_text(
            query_text="data science technology",
            k=5
        )
        
        self.assertGreaterEqual(len(results3), 1)
        
        print("✓ Vector similarity search works")
    
    def test_7_hnsw_index(self):
        """Test HNSW index."""
        # Create an in-memory index
        index = HNSWIndex(
            space="cosine",
            dim=128,
            ef_construction=200,
            M=16
        )
        
        # Initialize index
        index.init_index(max_elements=100)
        
        # Get all embeddings
        all_embeddings = self.embedding_manager.get_all_embeddings("node")
        
        # Add to index
        for emb in all_embeddings:
            index.add_item(
                entity_type=emb.entity_type,
                entity_id=emb.entity_id,
                vector=emb.embedding
            )
            
        # Search in index
        alice = self.persons["Alice"]
        alice_embedding = self.embedding_manager.get_embedding("node", alice.id)
        
        results = index.search(
            query_vector=alice_embedding.embedding,
            k=5
        )
        
        self.assertGreaterEqual(len(results), 1)
        
        # First result should be Alice herself
        self.assertEqual(results[0][1], alice.id)
        
        print("✓ HNSW index works")
    
    def test_8_knowledge_graph_class(self):
        """Test the convenience KnowledgeGraph class."""
        # Create a new KnowledgeGraph instance
        kg = KnowledgeGraph(self.db_path)
        
        # Test node operations
        new_node = kg.create_node(
            type="TestNode",
            name="Test",
            properties={"test": True}
        )
        
        self.assertTrue(new_node.id is not None)
        self.assertEqual(new_node.name, "Test")
        
        # Test edge operations
        alice = kg.get_node(self.persons["Alice"].id)
        self.assertEqual(alice.name, "Alice")
        
        new_edge = kg.create_edge(
            source_id=alice.id,
            target_id=new_node.id,
            relation_type="TEST_RELATION",
            properties={"test": True}
        )
        
        self.assertTrue(new_edge.id is not None)
        self.assertEqual(new_edge.relation_type, "TEST_RELATION")
        
        # Test get neighbors
        neighbors = kg.get_neighbors(alice.id)
        self.assertGreaterEqual(len(neighbors), 1)
        
        # Test find paths
        bob = kg.get_node(self.persons["Bob"].id)
        paths = kg.find_paths(alice.id, bob.id)
        self.assertGreaterEqual(len(paths), 1)
        
        # Clean up
        kg.delete_edge(new_edge.id)
        kg.delete_node(new_node.id)
        kg.close()
        
        print("✓ KnowledgeGraph convenience class works")


if __name__ == "__main__":
    unittest.main()