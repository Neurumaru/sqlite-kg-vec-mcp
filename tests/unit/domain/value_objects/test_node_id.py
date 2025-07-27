"""
Unit tests for NodeId value object.
"""

import unittest
import uuid

from src.domain.value_objects.node_id import NodeId


class TestNodeId(unittest.TestCase):
    """Test cases for NodeId value object."""

    def test_create_node_id_with_valid_string(self):
        """Test creating NodeId with valid string value."""
        value = "test-node-id"
        node_id = NodeId(value)
        self.assertEqual(node_id.value, value)
        self.assertEqual(str(node_id), value)

    def test_create_node_id_with_empty_string_raises_error(self):
        """Test that creating NodeId with empty string raises ValueError."""
        with self.assertRaises(ValueError) as context:
            NodeId("")
        self.assertIn("NodeId value cannot be empty", str(context.exception))

    def test_create_node_id_with_non_string_raises_error(self):
        """Test that creating NodeId with non-string raises ValueError."""
        with self.assertRaises(ValueError) as context:
            NodeId(123)
        self.assertIn("NodeId value must be a string", str(context.exception))

    def test_generate_creates_valid_uuid(self):
        """Test that generate() creates a valid UUID-based NodeId."""
        node_id = NodeId.generate()
        self.assertIsInstance(node_id, NodeId)
        self.assertIsInstance(node_id.value, str)

        # Verify it's a valid UUID format
        try:
            uuid.UUID(node_id.value)
        except ValueError:
            self.fail("Generated NodeId should be a valid UUID")

    def test_generate_creates_unique_ids(self):
        """Test that generate() creates unique IDs."""
        id1 = NodeId.generate()
        id2 = NodeId.generate()
        self.assertNotEqual(id1.value, id2.value)


    def test_equality(self):
        """Test NodeId equality comparison."""
        value = "test-id"
        id1 = NodeId(value)
        id2 = NodeId(value)
        id3 = NodeId("different-id")

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)

    def test_hash_consistency(self):
        """Test that NodeId hash is consistent."""
        value = "test-id"
        id1 = NodeId(value)
        id2 = NodeId(value)

        self.assertEqual(hash(id1), hash(id2))

        # Can be used in sets and dicts
        id_set = {id1, id2}
        self.assertEqual(len(id_set), 1)

    def test_immutability(self):
        """Test that NodeId is immutable."""
        node_id = NodeId("test-id")

        # Should not be able to modify value
        with self.assertRaises(AttributeError):
            node_id.value = "new-value"


if __name__ == "__main__":
    unittest.main()
