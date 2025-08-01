"""
Integration tests for database components.
"""

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.adapters.sqlite3.connection import DatabaseConnection
from src.adapters.sqlite3.schema import SchemaManager


class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.connection = DatabaseConnection(self.db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, "connection") and self.connection.connection:
            self.connection.close()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_database_connection_and_schema_integration(self):
        """Test that connection and schema work together."""
        # Connect to database
        conn = self.connection.connect()
        self.assertIsNotNone(conn)

        # Initialize schema
        schema_manager = SchemaManager(self.db_path)
        schema_manager.initialize_schema()

        # Verify schema was created by checking for expected tables
        cursor = conn.cursor()

        # Check if entities table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='entities'
        """
        )
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "entities")

        # Check if edges table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='edges'
        """
        )
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "edges")

        cursor.close()

    def test_basic_crud_operations(self):
        """Test basic CRUD operations with real database."""
        conn = self.connection.connect()
        schema_manager = SchemaManager(self.db_path)
        schema_manager.initialize_schema()

        cursor = conn.cursor()

        # Insert test data
        cursor.execute(
            """
            INSERT INTO entities (uuid, type, name, properties, created_at, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        """,
            ("test-uuid-1", "Person", "John Doe", "{}"),
        )

        # Read data
        cursor.execute("SELECT uuid, type, name FROM entities WHERE uuid = ?", ("test-uuid-1",))
        result = cursor.fetchone()

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "test-uuid-1")
        self.assertEqual(result[1], "Person")
        self.assertEqual(result[2], "John Doe")

        # Update data
        cursor.execute(
            """
            UPDATE entities SET name = ?, updated_at = datetime('now')
            WHERE uuid = ?
        """,
            ("Jane Doe", "test-uuid-1"),
        )

        cursor.execute("SELECT name FROM entities WHERE uuid = ?", ("test-uuid-1",))
        result = cursor.fetchone()
        self.assertEqual(result[0], "Jane Doe")

        # Delete data
        cursor.execute("DELETE FROM entities WHERE uuid = ?", ("test-uuid-1",))
        cursor.execute("SELECT * FROM entities WHERE uuid = ?", ("test-uuid-1",))
        result = cursor.fetchone()
        self.assertIsNone(result)

        cursor.close()

    def test_foreign_key_constraints(self):
        """Test that foreign key constraints work."""
        conn = self.connection.connect()
        schema_manager = SchemaManager(self.db_path)
        schema_manager.initialize_schema()

        cursor = conn.cursor()

        # Create entities first
        cursor.execute(
            """
            INSERT INTO entities (uuid, type, name, properties, created_at, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        """,
            ("entity-1", "Person", "John", "{}"),
        )

        cursor.execute(
            """
            INSERT INTO entities (uuid, type, name, properties, created_at, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        """,
            ("entity-2", "Person", "Jane", "{}"),
        )

        # Get entity IDs
        cursor.execute("SELECT id FROM entities WHERE uuid = ?", ("entity-1",))
        entity1_id = cursor.fetchone()[0]

        cursor.execute("SELECT id FROM entities WHERE uuid = ?", ("entity-2",))
        entity2_id = cursor.fetchone()[0]

        # Create edge using integer IDs
        cursor.execute(
            """
            INSERT INTO edges (source_id, target_id, relation_type, properties, created_at, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
        """,
            (entity1_id, entity2_id, "knows", "{}"),
        )

        # Verify edge was created
        cursor.execute(
            "SELECT * FROM edges WHERE source_id = ? AND target_id = ?",
            (entity1_id, entity2_id),
        )
        result = cursor.fetchone()
        self.assertIsNotNone(result)

        # Try to create edge with non-existent entity (should fail)
        with self.assertRaises(sqlite3.IntegrityError):  # Should raise foreign key constraint error
            cursor.execute(
                """
                INSERT INTO edges (source_id, target_id, relation_type, properties, created_at, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
                (99999, entity2_id, "knows", "{}"),
            )

        cursor.close()

    def test_transaction_rollback(self):
        """Test transaction rollback functionality."""
        conn = self.connection.connect()
        schema_manager = SchemaManager(self.db_path)
        schema_manager.initialize_schema()

        cursor = conn.cursor()

        try:
            # Start transaction
            cursor.execute("BEGIN")

            # Insert data
            cursor.execute(
                """
                INSERT INTO entities (uuid, type, name, properties, created_at, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
                ("test-transaction", "Person", "Test", "{}"),
            )

            # Verify data exists in transaction
            cursor.execute("SELECT * FROM entities WHERE uuid = ?", ("test-transaction",))
            result = cursor.fetchone()
            self.assertIsNotNone(result)

            # Rollback transaction
            cursor.execute("ROLLBACK")

            # Verify data was rolled back
            cursor.execute("SELECT * FROM entities WHERE uuid = ?", ("test-transaction",))
            result = cursor.fetchone()
            self.assertIsNone(result)

        finally:
            cursor.close()


if __name__ == "__main__":
    unittest.main()
