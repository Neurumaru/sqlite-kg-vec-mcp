"""
Unit tests for Entity domain model.
"""

import unittest
from datetime import datetime
from unittest.mock import patch

from src.domain.entities.entity import Entity
from src.domain.value_objects.node_id import NodeId
from src.domain.value_objects.entity_type import EntityType
from src.domain.exceptions.entity_exceptions import InvalidEntityException


class TestEntity(unittest.TestCase):
    """Test cases for Entity domain model."""

    def setUp(self):
        """Set up test fixtures."""
        self.node_id = NodeId.generate()
        self.entity_type = EntityType.person()
        self.name = "John Doe"
        self.properties = {"age": 30, "city": "Seoul"}

    def test_create_valid_entity(self):
        """Test creating a valid entity."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name,
            properties=self.properties
        )
        
        self.assertEqual(entity.id, self.node_id)
        self.assertEqual(entity.entity_type, self.entity_type)
        self.assertEqual(entity.name, self.name)
        self.assertEqual(entity.properties, self.properties)
        self.assertIsInstance(entity.created_at, datetime)
        self.assertIsInstance(entity.updated_at, datetime)

    def test_create_entity_without_id_raises_error(self):
        """Test that creating entity without ID raises InvalidEntityException."""
        with self.assertRaises(InvalidEntityException) as context:
            Entity(
                id=None,
                entity_type=self.entity_type,
                name=self.name
            )
        self.assertIn("Entity must have an ID", str(context.exception))

    def test_create_entity_without_type_raises_error(self):
        """Test that creating entity without type raises InvalidEntityException."""
        with self.assertRaises(InvalidEntityException) as context:
            Entity(
                id=self.node_id,
                entity_type=None,
                name=self.name
            )
        self.assertIn("Entity must have a type", str(context.exception))

    def test_person_entity_without_name_raises_error(self):
        """Test that Person entity without name raises InvalidEntityException."""
        with self.assertRaises(InvalidEntityException) as context:
            Entity(
                id=self.node_id,
                entity_type=EntityType.person(),
                name=None
            )
        self.assertIn("Person entities must have a name", str(context.exception))

    def test_organization_entity_without_name_raises_error(self):
        """Test that Organization entity without name raises InvalidEntityException."""
        with self.assertRaises(InvalidEntityException) as context:
            Entity(
                id=self.node_id,
                entity_type=EntityType.organization(),
                name=None
            )
        self.assertIn("Organization entities must have a name", str(context.exception))

    def test_product_entity_without_name_raises_error(self):
        """Test that Product entity without name raises InvalidEntityException."""
        with self.assertRaises(InvalidEntityException) as context:
            Entity(
                id=self.node_id,
                entity_type=EntityType.product(),
                name=None
            )
        self.assertIn("Product entities must have a name", str(context.exception))

    def test_concept_entity_can_have_no_name(self):
        """Test that Concept entity can be created without name."""
        entity = Entity(
            id=self.node_id,
            entity_type=EntityType.concept(),
            name=None
        )
        self.assertIsNone(entity.name)

    def test_properties_with_reserved_keys_raises_error(self):
        """Test that properties with reserved keys raise InvalidEntityException."""
        reserved_properties = {"id": "some-id", "type": "some-type"}
        
        with self.assertRaises(InvalidEntityException) as context:
            Entity(
                id=self.node_id,
                entity_type=self.entity_type,
                name=self.name,
                properties=reserved_properties
            )
        self.assertIn("Properties cannot contain reserved keys", str(context.exception))

    def test_create_factory_method(self):
        """Test Entity.create() factory method."""
        entity = Entity.create(
            entity_type=self.entity_type,
            name=self.name,
            properties=self.properties
        )
        
        self.assertIsInstance(entity.id, NodeId)
        self.assertEqual(entity.entity_type, self.entity_type)
        self.assertEqual(entity.name, self.name)
        self.assertEqual(entity.properties, self.properties)

    def test_restore_factory_method(self):
        """Test Entity.restore() factory method."""
        created_at = datetime(2023, 1, 1)
        updated_at = datetime(2023, 1, 2)
        
        entity = Entity.restore(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name,
            properties=self.properties,
            created_at=created_at,
            updated_at=updated_at
        )
        
        self.assertEqual(entity.id, self.node_id)
        self.assertEqual(entity.entity_type, self.entity_type)
        self.assertEqual(entity.name, self.name)
        self.assertEqual(entity.properties, self.properties)
        self.assertEqual(entity.created_at, created_at)
        self.assertEqual(entity.updated_at, updated_at)

    def test_update_name(self):
        """Test updating entity name."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name
        )
        
        old_updated_at = entity.updated_at
        new_name = "Jane Doe"
        
        with patch('src.domain.entities.entity.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 2)
            mock_datetime.now.return_value = mock_now
            
            entity.update_name(new_name)
            
            self.assertEqual(entity.name, new_name)
            self.assertEqual(entity.updated_at, mock_now)

    def test_update_name_to_empty_for_required_type_raises_error(self):
        """Test that updating name to empty for required type raises error."""
        entity = Entity(
            id=self.node_id,
            entity_type=EntityType.person(),
            name=self.name
        )
        
        with self.assertRaises(InvalidEntityException) as context:
            entity.update_name("")
        self.assertIn("Person entities must have a name", str(context.exception))

    def test_update_property(self):
        """Test updating a single property."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name,
            properties={"age": 30}
        )
        
        with patch('src.domain.entities.entity.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 2)
            mock_datetime.now.return_value = mock_now
            
            entity.update_property("age", 31)
            
            self.assertEqual(entity.properties["age"], 31)
            self.assertEqual(entity.updated_at, mock_now)

    def test_update_reserved_property_raises_error(self):
        """Test that updating reserved property raises error."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name
        )
        
        with self.assertRaises(InvalidEntityException) as context:
            entity.update_property("id", "new-id")
        self.assertIn("Cannot update reserved property: id", str(context.exception))

    def test_update_properties(self):
        """Test updating multiple properties."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name,
            properties={"age": 30}
        )
        
        new_properties = {"age": 31, "city": "Busan"}
        
        with patch('src.domain.entities.entity.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 2)
            mock_datetime.now.return_value = mock_now
            
            entity.update_properties(new_properties)
            
            self.assertEqual(entity.properties["age"], 31)
            self.assertEqual(entity.properties["city"], "Busan")
            self.assertEqual(entity.updated_at, mock_now)

    def test_update_properties_with_reserved_keys_raises_error(self):
        """Test that updating properties with reserved keys raises error."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name
        )
        
        with self.assertRaises(InvalidEntityException) as context:
            entity.update_properties({"id": "new-id", "type": "new-type"})
        self.assertIn("Cannot update reserved properties", str(context.exception))

    def test_remove_property(self):
        """Test removing a property."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name,
            properties={"age": 30, "city": "Seoul"}
        )
        
        with patch('src.domain.entities.entity.datetime') as mock_datetime:
            mock_now = datetime(2023, 1, 2)
            mock_datetime.now.return_value = mock_now
            
            entity.remove_property("age")
            
            self.assertNotIn("age", entity.properties)
            self.assertIn("city", entity.properties)
            self.assertEqual(entity.updated_at, mock_now)

    def test_remove_nonexistent_property(self):
        """Test removing a non-existent property does nothing."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name,
            properties={"age": 30}
        )
        
        old_updated_at = entity.updated_at
        entity.remove_property("nonexistent")
        
        # Should not change updated_at if property doesn't exist
        self.assertEqual(entity.updated_at, old_updated_at)

    def test_get_property(self):
        """Test getting property values."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name,
            properties={"age": 30}
        )
        
        self.assertEqual(entity.get_property("age"), 30)
        self.assertEqual(entity.get_property("nonexistent", "default"), "default")
        self.assertIsNone(entity.get_property("nonexistent"))

    def test_has_property(self):
        """Test checking if property exists."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name,
            properties={"age": 30}
        )
        
        self.assertTrue(entity.has_property("age"))
        self.assertFalse(entity.has_property("nonexistent"))

    def test_is_same_type(self):
        """Test checking if entities have same type."""
        entity1 = Entity(
            id=NodeId.generate(),
            entity_type=EntityType.person(),
            name="John"
        )
        
        entity2 = Entity(
            id=NodeId.generate(),
            entity_type=EntityType.person(),
            name="Jane"
        )
        
        entity3 = Entity(
            id=NodeId.generate(),
            entity_type=EntityType.organization(),
            name="ACME Corp"
        )
        
        self.assertTrue(entity1.is_same_type(entity2))
        self.assertFalse(entity1.is_same_type(entity3))

    def test_to_dict(self):
        """Test converting entity to dictionary."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name,
            properties=self.properties
        )
        
        result = entity.to_dict()
        
        self.assertEqual(result["id"], str(self.node_id))
        self.assertEqual(result["type"], str(self.entity_type))
        self.assertEqual(result["name"], self.name)
        self.assertEqual(result["properties"], self.properties)
        self.assertIn("created_at", result)
        self.assertIn("updated_at", result)

    def test_string_representation(self):
        """Test string representation of entity."""
        entity = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name=self.name
        )
        
        expected = f"Person({self.node_id}) '{self.name}'"
        self.assertEqual(str(entity), expected)

    def test_string_representation_without_name(self):
        """Test string representation of entity without name."""
        entity = Entity(
            id=self.node_id,
            entity_type=EntityType.concept()
        )
        
        expected = f"Concept({self.node_id})"
        self.assertEqual(str(entity), expected)

    def test_equality(self):
        """Test entity equality based on ID."""
        entity1 = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name="John"
        )
        
        entity2 = Entity(
            id=self.node_id,
            entity_type=EntityType.organization(),  # Different type
            name="Different Name"  # Different name
        )
        
        entity3 = Entity(
            id=NodeId.generate(),  # Different ID
            entity_type=self.entity_type,
            name="John"
        )
        
        self.assertEqual(entity1, entity2)  # Same ID
        self.assertNotEqual(entity1, entity3)  # Different ID
        self.assertNotEqual(entity1, "not an entity")  # Different type

    def test_hash_consistency(self):
        """Test that entity hash is based on ID."""
        entity1 = Entity(
            id=self.node_id,
            entity_type=self.entity_type,
            name="John"
        )
        
        entity2 = Entity(
            id=self.node_id,
            entity_type=EntityType.organization(),
            name="Different Name"
        )
        
        self.assertEqual(hash(entity1), hash(entity2))
        
        # Can be used in sets and dicts
        entity_set = {entity1, entity2}
        self.assertEqual(len(entity_set), 1)


if __name__ == '__main__':
    unittest.main()