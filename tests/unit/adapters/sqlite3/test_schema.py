"""
SchemaManager 단위 테스트.
"""

# pylint: disable=protected-access

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.sqlite3.connection import DatabaseConnection
from src.adapters.sqlite3.schema import SchemaManager


class TestSchemaManager(unittest.TestCase):
    """SchemaManager 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.schema_manager = SchemaManager(self.db_path)

    def tearDown(self):
        """테스트 정리."""
        import shutil

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
    def test_create_schema_version_table(self, mock_connection_class):
        """Given: 데이터베이스 연결이 있을 때
        When: _create_schema_version_table을 호출하면
        Then: schema_version 테이블이 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)

        # When
        manager._create_schema_version_table(mock_connection)

        # Then
        mock_connection.executescript.assert_called_once()
        call_args = mock_connection.executescript.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS schema_version", call_args)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_create_entity_tables(self, mock_connection_class):
        """Given: 데이터베이스 연결이 있을 때
        When: _create_entity_tables를 호출하면
        Then: entities 테이블과 관련 인덱스가 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        manager = SchemaManager(self.db_path)

        # When
        manager._create_entity_tables(mock_connection)

        # Then
        mock_connection.executescript.assert_called_once()
        call_args = mock_connection.executescript.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS entities", call_args)
        self.assertIn("CREATE INDEX IF NOT EXISTS idx_entities_type", call_args)
        self.assertIn("CREATE TRIGGER IF NOT EXISTS trg_entities_updated_at", call_args)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_create_edge_tables(self, mock_connection_class):
        """Given: 데이터베이스 연결이 있을 때
        When: _create_edge_tables를 호출하면
        Then: edges 테이블과 관련 인덱스가 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        manager = SchemaManager(self.db_path)

        # When
        manager._create_edge_tables(mock_connection)

        # Then
        mock_connection.executescript.assert_called_once()
        call_args = mock_connection.executescript.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS edges", call_args)
        self.assertIn("FOREIGN KEY (source_id) REFERENCES entities(id)", call_args)
        self.assertIn("CREATE INDEX IF NOT EXISTS idx_edges_source", call_args)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_create_hyperedge_tables(self, mock_connection_class):
        """Given: 데이터베이스 연결이 있을 때
        When: _create_hyperedge_tables를 호출하면
        Then: hyperedges와 hyperedge_members 테이블이 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        manager = SchemaManager(self.db_path)

        # When
        manager._create_hyperedge_tables(mock_connection)

        # Then
        mock_connection.executescript.assert_called_once()
        call_args = mock_connection.executescript.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS hyperedges", call_args)
        self.assertIn("CREATE TABLE IF NOT EXISTS hyperedge_members", call_args)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_create_document_tables(self, mock_connection_class):
        """Given: 데이터베이스 연결이 있을 때
        When: _create_document_tables를 호출하면
        Then: documents 테이블과 관련 인덱스가 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        manager = SchemaManager(self.db_path)

        # When
        manager._create_document_tables(mock_connection)

        # Then
        mock_connection.executescript.assert_called_once()
        call_args = mock_connection.executescript.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS documents", call_args)
        self.assertIn("connected_nodes JSON DEFAULT '[]'", call_args)
        self.assertIn("CREATE INDEX IF NOT EXISTS idx_documents_status", call_args)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_create_observation_tables(self, mock_connection_class):
        """Given: 데이터베이스 연결이 있을 때
        When: _create_observation_tables를 호출하면
        Then: observations 테이블이 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        manager = SchemaManager(self.db_path)

        # When
        manager._create_observation_tables(mock_connection)

        # Then
        mock_connection.executescript.assert_called_once()
        call_args = mock_connection.executescript.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS observations", call_args)
        self.assertIn("FOREIGN KEY (entity_id) REFERENCES entities(id)", call_args)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_create_embedding_tables(self, mock_connection_class):
        """Given: 데이터베이스 연결이 있을 때
        When: _create_embedding_tables를 호출하면
        Then: 임베딩 테이블들이 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        manager = SchemaManager(self.db_path)

        # When
        manager._create_embedding_tables(mock_connection)

        # Then
        mock_connection.executescript.assert_called_once()
        call_args = mock_connection.executescript.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS node_embeddings", call_args)
        self.assertIn("CREATE TABLE IF NOT EXISTS edge_embeddings", call_args)
        self.assertIn("CREATE TABLE IF NOT EXISTS hyperedge_embeddings", call_args)

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_create_sync_tables(self, mock_connection_class):
        """Given: 데이터베이스 연결이 있을 때
        When: _create_sync_tables를 호출하면
        Then: 동기화 관련 테이블들이 생성된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        manager = SchemaManager(self.db_path)

        # When
        manager._create_sync_tables(mock_connection)

        # Then
        mock_connection.executescript.assert_called_once()
        call_args = mock_connection.executescript.call_args[0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS vector_outbox", call_args)
        self.assertIn("CREATE TABLE IF NOT EXISTS sync_failures", call_args)

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

            self.assertIn("Downgrade", str(context.exception))

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

            self.assertIn("higher than latest version", str(context.exception))

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

        self.assertIn("Unknown migration version", str(context.exception))

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
            self.assertIn("Foreign key error", result["errors"][0])

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_add_json_optimization_columns_success(self, mock_connection_class):
        """Given: 데이터베이스에 JSON 최적화 컬럼이 없을 때
        When: _add_json_optimization_columns를 호출하면
        Then: JSON 최적화 컬럼이 추가된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        # entities 테이블에 json_text_content 컬럼이 없다고 가정
        mock_cursor.fetchall.side_effect = [
            [(0, "id"), (1, "name"), (2, "type")],  # entities 테이블 정보
            [(0, "id"), (1, "source_id"), (2, "target_id")],  # edges 테이블 정보
        ]
        mock_connection.cursor.return_value = mock_cursor

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)

        # When
        manager._add_json_optimization_columns(mock_connection)

        # Then
        # executescript이 호출되어 컬럼이 추가되었는지 확인
        self.assertTrue(mock_connection.executescript.called)
        call_count = mock_connection.executescript.call_count
        self.assertGreaterEqual(call_count, 2)  # entities와 edges 테이블용

    @patch("src.adapters.sqlite3.schema.DatabaseConnection")
    def test_add_json_optimization_columns_already_exists(self, mock_connection_class):
        """Given: 데이터베이스에 JSON 최적화 컬럼이 이미 있을 때
        When: _add_json_optimization_columns를 호출하면
        Then: 컬럼 추가를 건너뛴다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        # entities 테이블에 json_text_content 컬럼이 이미 있다고 가정
        mock_cursor.fetchall.side_effect = [
            [(0, "id"), (1, "name"), (2, "type"), (3, "json_text_content")],  # entities
            [(0, "id"), (1, "source_id"), (2, "target_id"), (3, "json_weight")],  # edges
        ]
        mock_connection.cursor.return_value = mock_cursor

        mock_db_conn_instance = Mock(spec=DatabaseConnection)
        mock_db_conn_instance.connect.return_value = mock_connection
        mock_db_conn_instance.__enter__ = Mock(return_value=mock_connection)
        mock_db_conn_instance.__exit__ = Mock(return_value=None)

        mock_connection_class.return_value = mock_db_conn_instance

        manager = SchemaManager(self.db_path)

        # When
        manager._add_json_optimization_columns(mock_connection)

        # Then
        # executescript이 호출되지 않았는지 확인 (컬럼이 이미 존재하므로)
        self.assertFalse(mock_connection.executescript.called)


if __name__ == "__main__":
    unittest.main()
