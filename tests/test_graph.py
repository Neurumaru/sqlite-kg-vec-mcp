"""
Tests for the knowledge graph functionality.
"""
import os
import tempfile
import unittest

import numpy as np

from src import KnowledgeGraph


class TestKnowledgeGraph(unittest.TestCase):
    """Test the KnowledgeGraph class."""
    
    def setUp(self):
        """Set up a test database."""
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Create a test graph
        self.kg = KnowledgeGraph(self.db_path)
        
        # Create test data
        self._create_test_data()
    
    def tearDown(self):
        """Clean up the test database."""
        self.kg.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)
    
    def _create_test_data(self):
        """Create test data in the graph."""
        # Create person nodes
        self.person1 = self.kg.create_node(
            type="Person", name="Person1", properties={"age": 30}
        )
        self.person2 = self.kg.create_node(
            type="Person", name="Person2", properties={"age": 40}
        )
        
        # Create a company node
        self.company = self.kg.create_node(
            type="Company", name="TestCorp", properties={"founded": 2000}
        )
        
        # Create relationships
        self.rel1 = self.kg.create_edge(
            source_id=self.person1.id,
            target_id=self.company.id,
            relation_type="WORKS_FOR",
            properties={"since": 2010}
        )
        
        self.rel2 = self.kg.create_edge(
            source_id=self.person2.id,
            target_id=self.company.id,
            relation_type="WORKS_FOR",
            properties={"since": 2015}
        )
        
        self.rel3 = self.kg.create_edge(
            source_id=self.person1.id,
            target_id=self.person2.id,
            relation_type="KNOWS",
            properties={"since": 2012}
        )
    
    def test_create_and_get_node(self):
        """Test creating and retrieving nodes."""
        # Create a new node
        node = self.kg.create_node(
            type="Test", name="TestNode", properties={"key": "value"}
        )
        
        # Check node properties
        self.assertIsNotNone(node.id)
        self.assertEqual(node.name, "TestNode")
        self.assertEqual(node.type, "Test")
        self.assertEqual(node.properties.get("key"), "value")
        
        # Retrieve the node by ID
        retrieved = self.kg.get_node(node.id)
        
        # Check retrieved node
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, node.id)
        self.assertEqual(retrieved.name, "TestNode")
        self.assertEqual(retrieved.properties.get("key"), "value")
        
        # Retrieve by UUID
        retrieved_by_uuid = self.kg.get_node_by_uuid(node.uuid)
        self.assertEqual(retrieved_by_uuid.id, node.id)
    
    def test_update_node(self):
        """Test updating a node."""
        # Update the node
        updated = self.kg.update_node(
            self.person1.id,
            name="UpdatedName",
            properties={"age": 31, "new_key": "new_value"}
        )
        
        # Check updated properties
        self.assertEqual(updated.name, "UpdatedName")
        self.assertEqual(updated.properties.get("age"), 31)
        self.assertEqual(updated.properties.get("new_key"), "new_value")
        
        # Retrieve the node again to verify persistence
        retrieved = self.kg.get_node(self.person1.id)
        self.assertEqual(retrieved.name, "UpdatedName")
        self.assertEqual(retrieved.properties.get("age"), 31)
    
    def test_delete_node(self):
        """Test deleting a node."""
        # Delete the node
        result = self.kg.delete_node(self.person1.id)
        self.assertTrue(result)
        
        # Try to retrieve the deleted node
        retrieved = self.kg.get_node(self.person1.id)
        self.assertIsNone(retrieved)
        
        # Check that relationships are also deleted
        edges, _ = self.kg.find_edges(source_id=self.person1.id)
        self.assertEqual(len(edges), 0)
    
    def test_find_nodes(self):
        """Test finding nodes by criteria."""
        # Find by type
        persons, count = self.kg.find_nodes(type="Person")
        self.assertEqual(count, 2)
        self.assertEqual(len(persons), 2)
        
        # Find by name pattern
        results, count = self.kg.find_nodes(name_pattern="Person%")
        self.assertEqual(count, 2)
        
        # Find by property
        results, count = self.kg.find_nodes(properties={"age": 30})
        self.assertEqual(count, 1)
        self.assertEqual(results[0].name, "Person1")
    
    def test_create_and_get_edge(self):
        """Test creating and retrieving edges."""
        # Create a new edge
        edge = self.kg.create_edge(
            source_id=self.person1.id,
            target_id=self.person2.id,
            relation_type="TEST_RELATION",
            properties={"key": "value"}
        )
        
        # Check edge properties
        self.assertIsNotNone(edge.id)
        self.assertEqual(edge.source_id, self.person1.id)
        self.assertEqual(edge.target_id, self.person2.id)
        self.assertEqual(edge.relation_type, "TEST_RELATION")
        self.assertEqual(edge.properties.get("key"), "value")
        
        # Retrieve the edge
        retrieved = self.kg.get_edge(edge.id, include_entities=True)
        
        # Check retrieved edge
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, edge.id)
        self.assertEqual(retrieved.relation_type, "TEST_RELATION")
        self.assertEqual(retrieved.properties.get("key"), "value")
        
        # Check that entities are included
        self.assertIsNotNone(retrieved.source)
        self.assertIsNotNone(retrieved.target)
        self.assertEqual(retrieved.source.name, "Person1")
        self.assertEqual(retrieved.target.name, "Person2")
    
    def test_update_edge(self):
        """Test updating an edge."""
        # Update the edge
        updated = self.kg.update_edge(
            self.rel1.id,
            properties={"since": 2011, "position": "Manager"}
        )
        
        # Check updated properties
        self.assertEqual(updated.properties.get("since"), 2011)
        self.assertEqual(updated.properties.get("position"), "Manager")
        
        # Retrieve the edge again to verify persistence
        retrieved = self.kg.get_edge(self.rel1.id)
        self.assertEqual(retrieved.properties.get("since"), 2011)
        self.assertEqual(retrieved.properties.get("position"), "Manager")
    
    def test_delete_edge(self):
        """Test deleting an edge."""
        # Delete the edge
        result = self.kg.delete_edge(self.rel1.id)
        self.assertTrue(result)
        
        # Try to retrieve the deleted edge
        retrieved = self.kg.get_edge(self.rel1.id)
        self.assertIsNone(retrieved)
    
    def test_find_edges(self):
        """Test finding edges by criteria."""
        # Find by relation type
        works_for, count = self.kg.find_edges(relation_type="WORKS_FOR")
        self.assertEqual(count, 2)
        self.assertEqual(len(works_for), 2)
        
        # Find by source
        outgoing, count = self.kg.find_edges(source_id=self.person1.id)
        self.assertEqual(count, 2)
        
        # Find by target
        incoming, count = self.kg.find_edges(target_id=self.company.id)
        self.assertEqual(count, 2)
        
        # Find by property
        results, count = self.kg.find_edges(properties={"since": 2010})
        self.assertEqual(count, 1)
    
    def test_get_neighbors(self):
        """Test getting neighbors of a node."""
        # Get all neighbors of person1
        neighbors = self.kg.get_neighbors(self.person1.id)
        self.assertEqual(len(neighbors), 2)
        
        # Get only companies that person1 works for
        works_for = self.kg.get_neighbors(
            self.person1.id,
            direction="outgoing",
            relation_types=["WORKS_FOR"]
        )
        self.assertEqual(len(works_for), 1)
        self.assertEqual(works_for[0][0].id, self.company.id)
        
        # Get people that person1 knows
        knows = self.kg.get_neighbors(
            self.person1.id,
            direction="outgoing",
            relation_types=["KNOWS"]
        )
        self.assertEqual(len(knows), 1)
        self.assertEqual(knows[0][0].id, self.person2.id)
        
        # Get employees of the company
        employees = self.kg.get_neighbors(
            self.company.id,
            direction="incoming",
            relation_types=["WORKS_FOR"]
        )
        self.assertEqual(len(employees), 2)
    
    def test_find_paths(self):
        """Test finding paths between nodes."""
        # Find path from person1 to company
        paths = self.kg.find_paths(self.person1.id, self.company.id)
        self.assertGreaterEqual(len(paths), 1)  # At least one path should exist

        # Check if any path matches the expected direct path
        direct_path_found = False
        for path in paths:
            if len(path) == 2 and path[0].entity.id == self.person1.id and path[1].entity.id == self.company.id:
                direct_path_found = True
                break

        self.assertTrue(direct_path_found, "Direct path should be found")
        
        # Find path from person1 to person2
        paths = self.kg.find_paths(self.person1.id, self.person2.id)
        self.assertGreaterEqual(len(paths), 1)  # At least one path should exist
        
        # Find indirect paths
        # Person1 -> Company -> Person2
        # We need to delete the direct KNOWS relationship to test indirect paths
        self.kg.delete_edge(self.rel3.id)
        
        paths = self.kg.find_paths(self.person1.id, self.person2.id, max_depth=3)
        self.assertEqual(len(paths), 1)  # Indirect path
        
        # Check the path
        path = paths[0]
        self.assertEqual(len(path), 3)  # Three nodes
        self.assertEqual(path[0].entity.id, self.person1.id)
        self.assertEqual(path[1].entity.id, self.company.id)
        self.assertEqual(path[2].entity.id, self.person2.id)


if __name__ == "__main__":
    unittest.main()