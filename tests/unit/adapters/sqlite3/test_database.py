"""
SQLiteDatabase 어댑터 단위 테스트.
"""

# pylint: disable=protected-access

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.sqlite3.database import SQLiteDatabase
from src.common.config.database import DatabaseConfig


class TestSQLiteDatabase(unittest.IsolatedAsyncioTestCase):
    """SQLiteDatabase 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.config = DatabaseConfig(
            db_path=str(self.db_path),
            optimize=True,
            timeout=30.0,
            check_same_thread=False,
            max_connections=10,
        )

    def tearDown(self):
        """테스트 정리."""
        import shutil

        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init_with_config(self):
        """Given: DatabaseConfig가 제공될 때
        When: SQLiteDatabase를 초기화하면
        Then: 설정이 올바르게 적용된다
        """
        # Given & When
        db = SQLiteDatabase(config=self.config)

        # Then
        self.assertEqual(db.db_path, self.db_path)
        self.assertTrue(db.optimize)
        self.assertEqual(db.timeout, 30.0)
        self.assertFalse(db.check_same_thread)
        self.assertEqual(db.max_connections, 10)

    def test_init_with_legacy_params(self):
        """Given: 개별 파라미터가 제공될 때
        When: SQLiteDatabase를 초기화하면
        Then: 파라미터가 config보다 우선된다
        """
        # Given & When
        db = SQLiteDatabase(config=self.config, db_path="/different/path.db", optimize=False)

        # Then
        self.assertEqual(str(db.db_path), "/different/path.db")
        self.assertFalse(db.optimize)

    @patch("src.adapters.sqlite3.database.DatabaseConnection")
    async def test_connect_success(self, mock_connection_class):
        """Given: 정상적인 데이터베이스 연결이 가능할 때
        When: connect를 호출하면
        Then: True를 반환하고 연결이 설정된다
        """
        # Given
        mock_connection_instance = Mock()
        mock_connection_class.return_value = mock_connection_instance
        mock_connection_instance.connect.return_value = Mock(spec=sqlite3.Connection)

        db = SQLiteDatabase(config=self.config)

        # When
        result = await db.connect()

        # Then
        self.assertTrue(result)
        mock_connection_instance.connect.assert_called_once()

    @patch("src.adapters.sqlite3.database.DatabaseConnection")
    async def test_connect_failure(self, mock_connection_class):
        """Given: 데이터베이스 연결에 실패할 때
        When: connect를 호출하면
        Then: False를 반환한다
        """
        # Given
        mock_connection_instance = Mock()
        mock_connection_class.return_value = mock_connection_instance
        mock_connection_instance.connect.side_effect = Exception("Connection failed")

        db = SQLiteDatabase(config=self.config)

        # When
        result = await db.connect()

        # Then
        self.assertFalse(result)

    async def test_disconnect_success(self):
        """Given: 활성 연결이 있을 때
        When: disconnect를 호출하면
        Then: 연결이 정리되고 True를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        db._connection = mock_connection

        # 활성 트랜잭션 추가
        mock_transaction_conn = Mock(spec=sqlite3.Connection)
        db._active_transactions["tx1"] = mock_transaction_conn

        # When
        result = await db.disconnect()

        # Then
        self.assertTrue(result)
        self.assertIsNone(db._connection)
        self.assertEqual(len(db._active_transactions), 0)

    async def test_is_connected_true(self):
        """Given: 정상적인 연결이 있을 때
        When: is_connected를 호출하면
        Then: True를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_connection.execute.return_value.fetchone.return_value = (1,)
        db._connection = mock_connection

        # When
        result = await db.is_connected()

        # Then
        self.assertTrue(result)
        mock_connection.execute.assert_called_with("SELECT 1")

    async def test_is_connected_false_no_connection(self):
        """Given: 연결이 없을 때
        When: is_connected를 호출하면
        Then: False를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)

        # When
        result = await db.is_connected()

        # Then
        self.assertFalse(result)

    async def test_is_connected_false_exception(self):
        """Given: 연결이 있지만 쿼리 실행에 실패할 때
        When: is_connected를 호출하면
        Then: False를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_connection.execute.side_effect = Exception("Connection lost")
        db._connection = mock_connection

        # When
        result = await db.is_connected()

        # Then
        self.assertFalse(result)

    async def test_ping(self):
        """Given: 데이터베이스가 있을 때
        When: ping을 호출하면
        Then: is_connected와 같은 결과를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_connection.execute.return_value.fetchone.return_value = (1,)
        db._connection = mock_connection

        # When
        result = await db.ping()

        # Then
        self.assertTrue(result)

    async def test_begin_transaction(self):
        """Given: 연결이 있을 때
        When: begin_transaction을 호출하면
        Then: 트랜잭션 ID를 반환하고 트랜잭션을 시작한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        db._connection = mock_connection

        # When
        tx_id = await db.begin_transaction()

        # Then
        self.assertIsInstance(tx_id, str)
        self.assertIn(tx_id, db._active_transactions)
        mock_connection.execute.assert_called_with("BEGIN")

    async def test_begin_transaction_no_connection(self):
        """Given: 연결이 없을 때
        When: begin_transaction을 호출하면
        Then: RuntimeError가 발생한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)

        # When & Then
        with self.assertRaises(RuntimeError):
            await db.begin_transaction()

    async def test_commit_transaction_success(self):
        """Given: 활성 트랜잭션이 있을 때
        When: commit_transaction을 호출하면
        Then: 트랜잭션을 커밋하고 True를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        db._connection = mock_connection
        tx_id = "test-tx-id"
        db._active_transactions[tx_id] = mock_connection

        # When
        result = await db.commit_transaction(tx_id)

        # Then
        self.assertTrue(result)
        self.assertNotIn(tx_id, db._active_transactions)
        mock_connection.commit.assert_called_once()

    async def test_commit_transaction_not_found(self):
        """Given: 존재하지 않는 트랜잭션 ID일 때
        When: commit_transaction을 호출하면
        Then: False를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)

        # When
        result = await db.commit_transaction("nonexistent-tx")

        # Then
        self.assertFalse(result)

    async def test_rollback_transaction_success(self):
        """Given: 활성 트랜잭션이 있을 때
        When: rollback_transaction을 호출하면
        Then: 트랜잭션을 롤백하고 True를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        db._connection = mock_connection
        tx_id = "test-tx-id"
        db._active_transactions[tx_id] = mock_connection

        # When
        result = await db.rollback_transaction(tx_id)

        # Then
        self.assertTrue(result)
        self.assertNotIn(tx_id, db._active_transactions)
        mock_connection.rollback.assert_called_once()

    async def test_execute_query_success(self):
        """Given: 정상적인 연결과 쿼리가 있을 때
        When: execute_query를 호출하면
        Then: 쿼리 결과를 딕셔너리 리스트로 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.description = [("col1",), ("col2",)]
        mock_cursor.fetchall.return_value = [("val1", "val2"), ("val3", "val4")]
        db._connection = mock_connection

        # When
        result = await db.execute_query("SELECT * FROM test")

        # Then
        expected = [{"col1": "val1", "col2": "val2"}, {"col1": "val3", "col2": "val4"}]
        self.assertEqual(result, expected)
        mock_cursor.execute.assert_called_with("SELECT * FROM test")

    async def test_execute_query_with_parameters(self):
        """Given: 파라미터가 있는 쿼리일 때
        When: execute_query를 호출하면
        Then: 파라미터와 함께 쿼리를 실행한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.description = [("id",)]
        mock_cursor.fetchall.return_value = [(1,)]
        db._connection = mock_connection

        params = {"name": "test"}

        # When
        result = await db.execute_query("SELECT id FROM test WHERE name = :name", params)

        # Then
        self.assertEqual(result, [{"id": 1}])
        mock_cursor.execute.assert_called_with("SELECT id FROM test WHERE name = :name", params)

    async def test_execute_command_success(self):
        """Given: 정상적인 연결과 명령이 있을 때
        When: execute_command를 호출하면
        Then: 영향받은 행 수를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 5
        db._connection = mock_connection

        # When
        result = await db.execute_command("INSERT INTO test VALUES (1)")

        # Then
        self.assertEqual(result, 5)
        mock_cursor.execute.assert_called_with("INSERT INTO test VALUES (1)")

    async def test_execute_batch_success(self):
        """Given: 여러 명령이 있을 때
        When: execute_batch를 호출하면
        Then: 각 명령의 영향받은 행 수 리스트를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor1 = Mock()
        mock_cursor1.rowcount = 1
        mock_cursor2 = Mock()
        mock_cursor2.rowcount = 2
        mock_connection.cursor.side_effect = [mock_cursor1, mock_cursor2]
        db._connection = mock_connection

        commands = ["INSERT INTO test VALUES (1)", "UPDATE test SET value = 2"]
        params = [{"id": 1}, {"value": 2}]

        # When
        result = await db.execute_batch(commands, params)

        # Then
        self.assertEqual(result, [1, 2])
        self.assertEqual(mock_connection.cursor.call_count, 2)

    async def test_create_table_success(self):
        """Given: 테이블 스키마가 있을 때
        When: create_table을 호출하면
        Then: 테이블이 생성되고 True를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        db._connection = mock_connection

        schema = {
            "id": {"type": "INTEGER", "primary_key": True},
            "name": {"type": "TEXT", "not_null": True},
            "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        }

        # When
        result = await db.create_table("test_table", schema)

        # Then
        self.assertTrue(result)
        mock_cursor.execute.assert_called()

    async def test_table_exists_true(self):
        """Given: 테이블이 존재할 때
        When: table_exists를 호출하면
        Then: True를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        with patch.object(db, "execute_query", return_value=[{"name": "test_table"}]):
            # When
            result = await db.table_exists("test_table")

            # Then
            self.assertTrue(result)

    async def test_table_exists_false(self):
        """Given: 테이블이 존재하지 않을 때
        When: table_exists를 호출하면
        Then: False를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        with patch.object(db, "execute_query", return_value=[]):
            # When
            result = await db.table_exists("nonexistent_table")

            # Then
            self.assertFalse(result)

    async def test_vacuum_success(self):
        """Given: 정상적인 데이터베이스 연결이 있을 때
        When: vacuum을 호출하면
        Then: VACUUM 명령을 실행하고 True를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        with patch.object(db, "execute_command", return_value=0):
            # When
            result = await db.vacuum()

            # Then
            self.assertTrue(result)

    async def test_health_check_healthy(self):
        """Given: 모든 헬스체크가 통과할 때
        When: health_check를 호출하면
        Then: healthy 상태를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        db.db_path.touch()  # 파일 생성

        with (
            patch.object(db, "is_connected", return_value=True),
            patch.object(db, "execute_query", return_value=[]),
            patch.object(db, "execute_command", return_value=0),
        ):
            # When
            result = await db.health_check()

            # Then
            self.assertEqual(result["status"], "healthy")
            self.assertTrue(result["connected"])
            self.assertTrue(result["readable"])
            self.assertTrue(result["writable"])

    async def test_get_connection_info(self):
        """Given: 데이터베이스 인스턴스가 있을 때
        When: get_connection_info를 호출하면
        Then: 연결 정보를 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        db._active_transactions["tx1"] = Mock()

        with patch.object(db, "is_connected", return_value=True):
            # When
            result = await db.get_connection_info()

            # Then
            self.assertTrue(result["connected"])
            self.assertEqual(result["db_path"], str(self.db_path))
            self.assertEqual(result["active_transactions"], 1)
            self.assertEqual(result["transaction_ids"], ["tx1"])

    def test_get_connection_without_transaction(self):
        """Given: 트랜잭션 ID가 없을 때
        When: _get_connection을 호출하면
        Then: 기본 연결을 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_connection = Mock()
        db._connection = mock_connection

        # When
        result = db._get_connection()

        # Then
        self.assertEqual(result, mock_connection)

    def test_get_connection_with_transaction(self):
        """Given: 트랜잭션 ID가 있을 때
        When: _get_connection을 호출하면
        Then: 트랜잭션 연결을 반환한다
        """
        # Given
        db = SQLiteDatabase(config=self.config)
        mock_main_connection = Mock()
        mock_transaction_connection = Mock()
        db._connection = mock_main_connection
        db._active_transactions["tx1"] = mock_transaction_connection

        # When
        result = db._get_connection("tx1")

        # Then
        self.assertEqual(result, mock_transaction_connection)


if __name__ == "__main__":
    unittest.main()
