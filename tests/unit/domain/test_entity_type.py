"""
Unit tests for EntityType value object.
"""

import unittest

from src.domain.value_objects.entity_type import EntityType


class TestEntityType(unittest.TestCase):
    """Test cases for EntityType value object."""

    def test_create_entity_type_with_valid_name(self):
        """Test creating EntityType with valid name."""
        name = "TestType"
        entity_type = EntityType(name)
        # EntityType normalizes to title case, so "TestType" becomes "Testtype"
        self.assertEqual(entity_type.name, "Testtype")
        self.assertEqual(str(entity_type), "Testtype")

    def test_create_entity_type_with_empty_name_raises_error(self):
        """Test that creating EntityType with empty name raises ValueError."""
        with self.assertRaises(ValueError) as context:
            EntityType("")
        self.assertIn("EntityType name cannot be empty", str(context.exception))

    def test_create_entity_type_with_non_string_raises_error(self):
        """Test that creating EntityType with non-string raises ValueError."""
        with self.assertRaises(ValueError) as context:
            EntityType(123)
        self.assertIn("EntityType name must be a string", str(context.exception))

    def test_name_normalization(self):
        """Test that entity type names are normalized to title case."""
        test_cases = [
            ("person", "Person"),
            ("ORGANIZATION", "Organization"),
            ("  location  ", "Location"),
            ("concept", "Concept"),
        ]

        for input_name, expected_name in test_cases:
            entity_type = EntityType(input_name)
            self.assertEqual(entity_type.name, expected_name)

    def test_person_factory_method(self):
        """Test person() factory method."""
        entity_type = EntityType.person()
        self.assertEqual(entity_type.name, "Person")

    def test_organization_factory_method(self):
        """Test organization() factory method."""
        entity_type = EntityType.organization()
        self.assertEqual(entity_type.name, "Organization")

    def test_location_factory_method(self):
        """Test location() factory method."""
        entity_type = EntityType.location()
        self.assertEqual(entity_type.name, "Location")

    def test_concept_factory_method(self):
        """Test concept() factory method."""
        entity_type = EntityType.concept()
        self.assertEqual(entity_type.name, "Concept")

    def test_product_factory_method(self):
        """Test product() factory method."""
        entity_type = EntityType.product()
        self.assertEqual(entity_type.name, "Product")

    def test_event_factory_method(self):
        """Test event() factory method."""
        entity_type = EntityType.event()
        self.assertEqual(entity_type.name, "Event")

    def test_equality(self):
        """Test EntityType equality comparison."""
        type1 = EntityType("Person")
        type2 = EntityType("person")  # Should be normalized to same value
        type3 = EntityType("Organization")

        self.assertEqual(type1, type2)
        self.assertNotEqual(type1, type3)

    def test_hash_consistency(self):
        """Test that EntityType hash is consistent."""
        type1 = EntityType("Person")
        type2 = EntityType("person")  # Should be normalized to same value

        self.assertEqual(hash(type1), hash(type2))

        # Can be used in sets and dicts
        type_set = {type1, type2}
        self.assertEqual(len(type_set), 1)

    def test_immutability(self):
        """Test that EntityType is immutable."""
        entity_type = EntityType("Person")

        # Should not be able to modify name
        with self.assertRaises(AttributeError):
            entity_type.name = "NewType"

    def test_factory_methods_return_consistent_instances(self):
        """Test that factory methods return consistent instances."""
        person1 = EntityType.person()
        person2 = EntityType.person()

        self.assertEqual(person1, person2)
        self.assertEqual(hash(person1), hash(person2))


if __name__ == "__main__":
    unittest.main()
