"""
Unit tests for entity exceptions.
"""

import unittest
from src.domain.exceptions.entity_exceptions import InvalidEntityException


class TestEntityExceptions(unittest.TestCase):
    """Test cases for entity exception classes."""

    def test_invalid_entity_exception_creation(self):
        """Test creating InvalidEntityException."""
        message = "Test error message"
        exception = InvalidEntityException(message)
        expected_str = "[INVALID_ENTITY] Invalid entity: Test error message"
        self.assertEqual(str(exception), expected_str)
        self.assertEqual(exception.message, "Invalid entity: Test error message")
        self.assertEqual(exception.error_code, "INVALID_ENTITY")
        self.assertIsInstance(exception, Exception)

    def test_invalid_entity_exception_inheritance(self):
        """Test that InvalidEntityException inherits from appropriate base."""
        from src.domain.exceptions.base import DomainException
        exception = InvalidEntityException("test")
        self.assertIsInstance(exception, DomainException)
        self.assertIsInstance(exception, Exception)


if __name__ == '__main__':
    unittest.main()