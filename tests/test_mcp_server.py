"""
Tests for the MCP server interface.
"""
import os
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from sqlite_kg_vec_mcp.server.api import KnowledgeGraphServer


class TestMCPServer(unittest.TestCase):
    """Tests for KnowledgeGraphServer."""
    
    def setUp(self):
        """Set up a temporary database for testing."""
        # Create a temporary database file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.vector_dir = Path(self.temp_dir.name) / "vectors"
        os.makedirs(self.vector_dir, exist_ok=True)
        
        # Initialize the schema first
        from sqlite_kg_vec_mcp.db.schema import SchemaManager
        
        # Initialize schema
        schema_manager = SchemaManager(self.db_path)
        schema_manager.initialize_schema()
        
        # Create server instance without starting it
        self.server = KnowledgeGraphServer(
            db_path=self.db_path,
            vector_index_dir=self.vector_dir,
            embedding_dim=8,  # Small dimension for testing
            vector_similarity="cosine"
        )
    
    def tearDown(self):
        """Clean up resources."""
        self.server.close()
        self.temp_dir.cleanup()
    
    def test_create_node(self):
        """Test creating a node."""
        result = self.server.create_node(
            type="Person",
            name="John Doe",
            properties={"age": 30, "occupation": "Developer"}
        )
        
        self.assertIn("node_id", result)
        self.assertIn("uuid", result)
        self.assertEqual(result["name"], "John Doe")
        self.assertEqual(result["type"], "Person")
        self.assertEqual(result["properties"]["age"], 30)
        self.assertEqual(result["properties"]["occupation"], "Developer")
    
    def test_get_node(self):
        """Test retrieving a node."""
        # Create a node first
        create_result = self.server.create_node(
            type="Person",
            name="Jane Smith",
            properties={"age": 28}
        )
        
        # Get the node by ID
        result = self.server.get_node(id=create_result["node_id"])
        
        self.assertIn("node_id", result)
        self.assertEqual(result["name"], "Jane Smith")
        self.assertEqual(result["type"], "Person")
        self.assertEqual(result["properties"]["age"], 28)
        
        # Get the node by UUID
        result_uuid = self.server.get_node(uuid=create_result["uuid"])
        self.assertEqual(result_uuid["node_id"], create_result["node_id"])
    
    def test_update_node(self):
        """Test updating a node."""
        # Create a node first
        create_result = self.server.create_node(
            type="Person",
            name="Bob Johnson",
            properties={"age": 35}
        )
        
        # Update the node
        update_result = self.server.update_node(
            id=create_result["node_id"],
            name="Robert Johnson",
            properties={"age": 36, "occupation": "Manager"}
        )
        
        self.assertEqual(update_result["name"], "Robert Johnson")
        self.assertEqual(update_result["properties"]["age"], 36)
        self.assertEqual(update_result["properties"]["occupation"], "Manager")
        
        # Verify the update was stored
        get_result = self.server.get_node(id=create_result["node_id"])
        self.assertEqual(get_result["name"], "Robert Johnson")
        self.assertEqual(get_result["properties"]["age"], 36)
    
    def test_create_edge(self):
        """Test creating an edge."""
        # Create nodes first
        person = self.server.create_node(
            type="Person",
            name="Alice Cooper"
        )
        
        company = self.server.create_node(
            type="Company",
            name="Acme Inc."
        )
        
        # Create edge
        edge_result = self.server.create_edge(
            source_id=person["node_id"],
            target_id=company["node_id"],
            relation_type="WORKS_AT",
            properties={"since": 2020, "position": "Developer"}
        )
        
        self.assertIn("edge_id", edge_result)
        self.assertEqual(edge_result["source_id"], person["node_id"])
        self.assertEqual(edge_result["target_id"], company["node_id"])
        self.assertEqual(edge_result["relation_type"], "WORKS_AT")
        self.assertEqual(edge_result["properties"]["position"], "Developer")
    
    def test_find_nodes(self):
        """Test finding nodes."""
        # Create multiple nodes
        self.server.create_node(type="Person", name="Person 1", properties={"age": 25})
        self.server.create_node(type="Person", name="Person 2", properties={"age": 30})
        self.server.create_node(type="Person", name="Person 3", properties={"age": 35})
        self.server.create_node(type="Company", name="Company 1")
        
        # Find by type
        result = self.server.find_nodes(type="Person")
        self.assertEqual(result["total_count"], 3)
        
        # Find by property
        result = self.server.find_nodes(type="Person", properties={"age": 30})
        self.assertEqual(result["total_count"], 1)
        self.assertEqual(result["nodes"][0]["name"], "Person 2")
        
        # Find by name pattern
        result = self.server.find_nodes(name_pattern="Company%")
        self.assertEqual(result["total_count"], 1)
        self.assertEqual(result["nodes"][0]["type"], "Company")
    
    def test_get_neighbors(self):
        """Test getting neighbors."""
        # Create nodes
        person1 = self.server.create_node(type="Person", name="Person A")
        person2 = self.server.create_node(type="Person", name="Person B")
        company = self.server.create_node(type="Company", name="Company X")
        
        # Create edges
        self.server.create_edge(
            source_id=person1["node_id"],
            target_id=company["node_id"],
            relation_type="WORKS_AT"
        )
        
        self.server.create_edge(
            source_id=person2["node_id"],
            target_id=company["node_id"],
            relation_type="WORKS_AT"
        )
        
        self.server.create_edge(
            source_id=person1["node_id"],
            target_id=person2["node_id"],
            relation_type="KNOWS"
        )
        
        # Get neighbors of person1
        result = self.server.get_neighbors(node_id=person1["node_id"])
        self.assertEqual(result["count"], 2)
        
        # Get neighbors with specific relation type
        result = self.server.get_neighbors(
            node_id=person1["node_id"],
            relation_types=["KNOWS"]
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["neighbors"][0]["edge"]["relation_type"], "KNOWS")
        
        # Get incoming neighbors of company
        result = self.server.get_neighbors(
            node_id=company["node_id"],
            direction="incoming"
        )
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["neighbors"][0]["edge"]["target_id"], company["node_id"])
    
    def test_server_configuration(self):
        """Test server configuration and properties."""
        # Check that server has proper tools registered
        self.assertTrue(hasattr(self.server, "mcp_server"))
        self.assertTrue(hasattr(self.server, "create_node"))
        self.assertTrue(hasattr(self.server, "get_node"))
        self.assertTrue(hasattr(self.server, "update_node"))
        self.assertTrue(hasattr(self.server, "delete_node"))
        self.assertTrue(hasattr(self.server, "find_nodes"))
        self.assertTrue(hasattr(self.server, "create_edge"))
        self.assertTrue(hasattr(self.server, "get_edge"))

        # Check initialization parameters
        self.assertEqual(self.server.embedding_dim, 8)
        self.assertEqual(self.server.vector_similarity, "cosine")
        self.assertEqual(str(self.server.db_path), str(self.db_path))


if __name__ == "__main__":
    unittest.main()