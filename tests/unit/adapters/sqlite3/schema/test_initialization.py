"""
SchemaManager 초기화 및 스키마 생성 단위 테스트.
"""

# pylint: disable=protected-access

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.sqlite3.connection import DatabaseConnection
from src.adapters.sqlite3.schema import SchemaManager


class TestSchemaManagerInitialization(unittest.TestCase):
    """SchemaManager 초기화 및 스키마 생성 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.schema_manager = SchemaManager(self.db_path)

    def tearDown(self):
        """테스트 정리."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Given: 데이터베이스 경로가 주어질 때
        When: SchemaManager를 초기화하면
        Then: DatabaseConnection이 설정된다
        """
        # Given & When
        manager = SchemaManager(self.db_path)

        # Then
        self.assertEqual(manager.db_connection.db_path, self.db_path)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_initialize_schema_success(self, mock_connection_class):
        """Given: 정상적인 데이터베이스 연결이 있을 때
        When: initialize_schema를 호출하면
        Then: 모든 테이블과 스키마가 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)

        # Create a mock instance of DatabaseConnection that behaves like a context manager
        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)

        # When
        manager.initialize_schema()

        # Then
        # executescript이 여러 번 호출되었는지 확인
        self.assertTrue(mock_connection.executescript.called)
        call_count = mock_connection.executescript.call_count
        self.assertGreaterEqual(call_count, 7)  # 최소 7개의 테이블 그룹

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_backup_schema_success(self, mock_connection_class):
        """Given: 정상적인 데이터베이스가 있을 때
        When: backup_schema를 호출하면
        Then: 백업이 생성되고 True를 반환한다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_backup_connection = Mock()

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)
        backup_path = str(self.db_path.parent / "backup.db")

        with patch("sqlite3.connect", return_value=mock_backup_connection):
            # When
            result = manager.backup_schema(backup_path)

            # Then
            self.assertTrue(result)
            mock_connection.backup.assert_called_once_with(mock_backup_connection)
            mock_backup_connection.close.assert_called_once()

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_validate_schema_valid(self, mock_connection_class):
        """Given: 유효한 스키마가 있을 때
        When: validate_schema를 호출하면
        Then: 유효성 검사 결과를 반환한다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_cursor.fetchall.side_effect = [[], [(0, "ok")], [(0, "idx_test")]]
        mock_cursor.fetchone.return_value = (
            "ok",
        )  # For the integrity check query, it expects a tuple
        mock_connection.cursor.return_value = mock_cursor

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)

        with patch.object(manager, "get_schema_version", return_value=1):
            # When
            result = manager.validate_schema()

            # Then
            self.assertTrue(result["valid"])
            self.assertEqual(result["version"], 1)
            self.assertEqual(len(result["errors"]), 0)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_validate_schema_foreign_key_error(self, mock_connection_class):
        """Given: 외래 키 오류가 있을 때
        When: validate_schema를 호출하면
        Then: 유효하지 않은 결과를 반환한다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_cursor.fetchall.side_effect = [
            [("table", "rowid", "parent_table", "fkid")],  # FK 오류
            [],  # 인덱스 리스트
        ]
        mock_cursor.fetchone.return_value = (
            0,
        )  # For the integrity check query, it expects a tuple
        mock_connection.cursor.return_value = mock_cursor

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)

        with patch.object(manager, "get_schema_version", return_value=1):
            # When
            result = manager.validate_schema()

            # Then
            self.assertFalse(result["valid"])
            self.assertGreater(len(result["errors"]), 0)
            self.assertIn("외래 키 오류", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
