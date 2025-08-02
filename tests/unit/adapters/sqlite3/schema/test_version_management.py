"""
SchemaManager 스키마 버전 관리 단위 테스트.
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


class TestSchemaManagerVersionManagement(unittest.TestCase):
    """SchemaManager 스키마 버전 관리 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.schema_manager = SchemaManager(self.db_path)

    def tearDown(self):
        """테스트 정리."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_update_schema_version(self, mock_connection_class):
        """Given: 데이터베이스 연결이 있을 때
        When: _update_schema_version을 호출하면
        Then: 스키마 버전이 업데이트된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        manager = SchemaManager(self.db_path)
        version = 2

        # When
        manager._update_schema_version(mock_connection, version)

        # Then
        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args[0]
        self.assertIn("INSERT INTO schema_version", call_args[0])
        self.assertEqual(call_args[1], (version,))

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_get_schema_version_exists(self, mock_connection_class):
        """Given: 스키마 버전이 설정되어 있을 때
        When: get_schema_version을 호출하면
        Then: 현재 버전을 반환한다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (3,)
        mock_connection.cursor.return_value = mock_cursor

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)

        # When
        version = manager.get_schema_version()

        # Then
        self.assertEqual(version, 3)
        mock_cursor.execute.assert_called_with("SELECT version FROM schema_version WHERE id = 1")

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_get_schema_version_not_exists(self, mock_connection_class):
        """Given: 스키마 버전이 설정되어 있지 않을 때
        When: get_schema_version을 호출하면
        Then: 0을 반환한다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = None
        mock_connection.cursor.return_value = mock_cursor

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)

        # When
        version = manager.get_schema_version()

        # Then
        self.assertEqual(version, 0)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_get_schema_version_table_not_exists(self, mock_connection_class):
        """Given: schema_version 테이블이 존재하지 않을 때
        When: get_schema_version을 호출하면
        Then: 0을 반환한다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_cursor.execute.side_effect = sqlite3.OperationalError("no such table")
        mock_connection.cursor.return_value = mock_cursor

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)

        # When
        version = manager.get_schema_version()

        # Then
        self.assertEqual(version, 0)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_migrate_schema_same_version(self, mock_connection_class):
        """Given: 현재 버전과 목표 버전이 같을 때
        When: migrate_schema를 호출하면
        Then: True를 반환하고 아무것도 하지 않는다
        """
        # Given
        manager = SchemaManager(self.db_path)
        with patch.object(manager, "get_schema_version", return_value=2):
            # When
            result = manager.migrate_schema(target_version=2)

            # Then
            self.assertTrue(result)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_migrate_schema_downgrade_error(self, mock_connection_class):
        """Given: 현재 버전보다 낮은 버전으로 마이그레이션을 시도할 때
        When: migrate_schema를 호출하면
        Then: ValueError가 발생한다
        """
        # Given
        manager = SchemaManager(self.db_path)
        with patch.object(manager, "get_schema_version", return_value=3):
            # When & Then
            with self.assertRaises(ValueError) as context:
                manager.migrate_schema(target_version=2)

            self.assertIn("다운그레이드", str(context.exception))

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_migrate_schema_invalid_target(self, mock_connection_class):
        """Given: 지원하지 않는 버전으로 마이그레이션을 시도할 때
        When: migrate_schema를 호출하면
        Then: ValueError가 발생한다
        """
        # Given
        manager = SchemaManager(self.db_path)
        with patch.object(manager, "get_schema_version", return_value=1):
            # When & Then
            with self.assertRaises(ValueError) as context:
                manager.migrate_schema(target_version=999)

            self.assertIn("최신 버전", str(context.exception))

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_migrate_schema_success(self, mock_connection_class):
        """Given: 유효한 마이그레이션 요청이 있을 때
        When: migrate_schema를 호출하면
        Then: 마이그레이션이 성공하고 True를 반환한다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = (
            mock_db_conn_instance  # This line ensures the mocked class returns our instance
        )

        manager = SchemaManager(self.db_path)

        with (
            patch.object(manager, "get_schema_version", return_value=1),
            patch.object(manager, "_apply_migration") as mock_apply,
            patch.object(manager, "_update_schema_version") as mock_update,
        ):
            # When
            result = manager.migrate_schema(target_version=2)

            # Then
            self.assertTrue(result)
            mock_apply.assert_called_once_with(mock_connection, 2)
            mock_update.assert_called_once_with(mock_connection, 2)
            mock_connection.execute.assert_any_call("BEGIN TRANSACTION")
            mock_connection.execute.assert_any_call("COMMIT")

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_apply_migration_version_1(self, mock_connection_class):
        """Given: 버전 1 마이그레이션을 적용할 때
        When: _apply_migration을 호출하면
        Then: 기본 스키마가 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        manager = SchemaManager(self.db_path)

        # When
        manager._apply_migration(mock_connection, 1)

        # Then
        # 여러 테이블 생성 메서드가 호출되었는지 확인
        self.assertTrue(mock_connection.executescript.called)

    def test_apply_migration_unknown_version(self):
        """Given: 알 수 없는 버전 번호가 주어질 때
        When: _apply_migration을 호출하면
        Then: ValueError가 발생한다
        """
        # Given
        manager = SchemaManager(self.db_path)
        mock_connection = Mock()

        # When & Then
        with self.assertRaises(ValueError) as context:
            manager._apply_migration(mock_connection, 999)

        self.assertIn("알 수 없는 마이그레이션 버전", str(context.exception))


if __name__ == "__main__":
    unittest.main()
