"""
Unit tests for DatabaseConnection adapter.
"""

import os
import tempfile
import unittest
from pathlib import Path

from src.adapters.sqlite3.connection import DatabaseConnection


class TestDatabaseConnection(unittest.TestCase):
    """Test cases for DatabaseConnection adapter."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temp directory and files
        import shutil

        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_create_database_connection(self):
        """Test creating a database connection."""
        db_conn = DatabaseConnection(self.db_path)
        self.assertEqual(db_conn.db_path, self.db_path)
        self.assertIsNone(db_conn.connection)

    def test_connect_creates_database_file(self):
        """Test that connect() creates the database file."""
        db_conn = DatabaseConnection(self.db_path)
        connection = db_conn.connect()

        self.assertTrue(self.db_path.exists())
        self.assertIsNotNone(connection)
        self.assertIsNotNone(db_conn.connection)

        db_conn.close()

    def test_connect_creates_parent_directory(self):
        """Test that connect() creates parent directories."""
        nested_path = Path(self.temp_dir) / "nested" / "test.db"
        db_conn = DatabaseConnection(nested_path)

        connection = db_conn.connect()

        self.assertTrue(nested_path.parent.exists())
        self.assertTrue(nested_path.exists())

        db_conn.close()

    def test_context_manager(self):
        """Test using DatabaseConnection as context manager."""
        db_conn = DatabaseConnection(self.db_path)

        with db_conn as connection:
            self.assertIsNotNone(connection)
            # Test basic SQL operation
            result = connection.execute("SELECT 1").fetchone()
            self.assertEqual(result[0], 1)

        # Connection should be closed after context
        self.assertIsNone(db_conn.connection)

    def test_connection_optimizations_applied(self):
        """Test that connection optimizations are applied."""
        db_conn = DatabaseConnection(self.db_path, optimize=True)

        with db_conn as connection:
            # Check WAL mode
            result = connection.execute("PRAGMA journal_mode").fetchone()
            self.assertEqual(result[0].upper(), "WAL")

            # Check foreign keys enabled
            result = connection.execute("PRAGMA foreign_keys").fetchone()
            self.assertEqual(result[0], 1)

    def test_no_optimizations_when_disabled(self):
        """Test that optimizations are not applied when disabled."""
        db_conn = DatabaseConnection(self.db_path, optimize=False)

        with db_conn as connection:
            # Should still be able to connect and execute queries
            result = connection.execute("SELECT 1").fetchone()
            self.assertEqual(result[0], 1)

    def test_close_connection(self):
        """Test closing the connection explicitly."""
        db_conn = DatabaseConnection(self.db_path)
        connection = db_conn.connect()

        self.assertIsNotNone(db_conn.connection)

        db_conn.close()

        self.assertIsNone(db_conn.connection)


if __name__ == "__main__":
    unittest.main()
