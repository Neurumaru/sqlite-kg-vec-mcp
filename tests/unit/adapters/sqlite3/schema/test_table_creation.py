"""
SchemaManager 개별 테이블 생성 단위 테스트.
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


class TestSchemaManagerTableCreation(unittest.TestCase):
    """SchemaManager 개별 테이블 생성 테스트."""

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
