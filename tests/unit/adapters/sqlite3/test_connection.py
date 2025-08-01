"""
SQLite 데이터베이스 연결 관리자의 단위 테스트.
"""

# pylint: disable=protected-access

import datetime
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.adapters.sqlite3.connection import (
    DatabaseConnection,
    adapt_datetime,
    convert_datetime,
)
from src.adapters.sqlite3.exceptions import SQLiteConnectionException


class TestDatetimeAdapters(unittest.TestCase):
    """datetime 어댑터 함수들의 테스트."""

    def test_adapt_datetime(self):
        """Given: datetime 객체가 주어질 때
        When: adapt_datetime을 호출하면
        Then: ISO 형식 문자열이 반환된다
        """
        # Given
        dt = datetime.datetime(2023, 12, 25, 15, 30, 45)

        # When
        result = adapt_datetime(dt)

        # Then
        self.assertEqual(result, "2023-12-25T15:30:45")

    def test_convert_datetime_string(self):
        """Given: ISO 형식 문자열이 주어질 때
        When: convert_datetime을 호출하면
        Then: datetime 객체가 반환된다
        """
        # Given
        iso_string = "2023-12-25T15:30:45"

        # When
        result = convert_datetime(iso_string)

        # Then
        expected = datetime.datetime(2023, 12, 25, 15, 30, 45)
        self.assertEqual(result, expected)

    def test_convert_datetime_bytes(self):
        """Given: ISO 형식 바이트 문자열이 주어질 때
        When: convert_datetime을 호출하면
        Then: datetime 객체가 반환된다
        """
        # Given
        iso_bytes = b"2023-12-25T15:30:45"

        # When
        result = convert_datetime(iso_bytes)

        # Then
        expected = datetime.datetime(2023, 12, 25, 15, 30, 45)
        self.assertEqual(result, expected)

    def test_convert_datetime_invalid_string(self):
        """Given: 잘못된 형식의 문자열이 주어질 때
        When: convert_datetime을 호출하면
        Then: 원본 문자열이 반환되고 경고가 발생한다
        """
        # Given
        invalid_string = "invalid-datetime"

        # When
        with patch("warnings.warn") as mock_warn:
            result = convert_datetime(invalid_string)

        # Then
        self.assertEqual(result, invalid_string)
        mock_warn.assert_called_once()

    def test_convert_datetime_invalid_bytes(self):
        """Given: 잘못된 UTF-8 바이트가 주어질 때
        When: convert_datetime을 호출하면
        Then: 원본 바이트가 반환되고 경고가 발생한다
        """
        # Given
        invalid_bytes = b"\xff\xfe"

        # When
        with patch("warnings.warn") as mock_warn:
            result = convert_datetime(invalid_bytes)

        # Then
        self.assertEqual(result, invalid_bytes)
        mock_warn.assert_called_once()


class TestDatabaseConnection(unittest.TestCase):
    """DatabaseConnection 클래스의 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"

    def test_init_with_string_path(self):
        """Given: 문자열 경로가 주어질 때
        When: DatabaseConnection을 생성하면
        Then: Path 객체로 변환되어 설정된다
        """
        # Given
        db_path_str = str(self.db_path)

        # When
        conn = DatabaseConnection(db_path_str)

        # Then
        self.assertEqual(conn.db_path, Path(db_path_str))
        self.assertTrue(conn.optimize)

    def test_init_with_path_object(self):
        """Given: Path 객체가 주어질 때
        When: DatabaseConnection을 생성하면
        Then: Path 객체가 그대로 설정된다
        """
        # Given / When
        conn = DatabaseConnection(self.db_path, optimize=False)

        # Then
        self.assertEqual(conn.db_path, self.db_path)
        self.assertFalse(conn.optimize)

    @patch("src.adapters.sqlite3.connection.sqlite3.connect")
    def test_connect_success(self, mock_connect):
        """Given: 정상적인 데이터베이스 연결 상황에서
        When: connect를 호출하면
        Then: SQLite 연결이 성공한다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_connection.execute.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        conn = DatabaseConnection(self.db_path)

        # When
        result = conn.connect()

        # Then
        self.assertEqual(result, mock_connection)
        mock_connect.assert_called_once()
        self.assertEqual(mock_connection.row_factory, sqlite3.Row)
        mock_connection.execute.assert_called_with("SELECT 1")

    @patch("src.adapters.sqlite3.connection.sqlite3.connect")
    def test_connect_with_optimization(self, mock_connect):
        """Given: 최적화가 활성화된 상태에서
        When: connect를 호출하면
        Then: PRAGMA 최적화가 적용된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        conn = DatabaseConnection(self.db_path, optimize=True)

        # When
        conn.connect()

        # Then
        # PRAGMA 문들이 실행되었는지 확인
        pragma_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        expected_pragmas = [
            "PRAGMA journal_mode=WAL;",
            "PRAGMA busy_timeout=5000;",
            "PRAGMA synchronous=NORMAL;",
            "PRAGMA foreign_keys=ON;",
            "PRAGMA temp_store=MEMORY;",
            "PRAGMA cache_size=-32000;",
        ]
        for pragma in expected_pragmas:
            self.assertIn(pragma, pragma_calls)

    @patch("src.adapters.sqlite3.connection.sqlite3.connect")
    def test_connect_without_optimization(self, mock_connect):
        """Given: 최적화가 비활성화된 상태에서
        When: connect를 호출하면
        Then: PRAGMA 최적화가 적용되지 않는다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_connection.execute.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        conn = DatabaseConnection(self.db_path, optimize=False)

        # When
        conn.connect()

        # Then
        mock_connection.cursor.assert_not_called()

    @patch("src.adapters.sqlite3.connection.sqlite3.connect")
    def test_connect_operational_error(self, mock_connect):
        """Given: SQLite OperationalError가 발생할 때
        When: connect를 호출하면
        Then: SQLiteConnectionException이 발생한다
        """
        # Given
        mock_connect.side_effect = sqlite3.OperationalError("database is locked")
        conn = DatabaseConnection(self.db_path)

        # When & Then
        with self.assertRaises(SQLiteConnectionException) as context:
            conn.connect()

        self.assertIsInstance(context.exception.original_error, sqlite3.OperationalError)

    @patch("src.adapters.sqlite3.connection.sqlite3.connect")
    def test_connect_generic_sqlite_error(self, mock_connect):
        """Given: 일반적인 SQLite 에러가 발생할 때
        When: connect를 호출하면
        Then: SQLiteConnectionException이 발생한다
        """
        # Given
        mock_connect.side_effect = sqlite3.Error("generic error")
        conn = DatabaseConnection(self.db_path)

        # When & Then
        with self.assertRaises(SQLiteConnectionException) as context:
            conn.connect()

        self.assertIn("Database connection failed", context.exception.message)

    @patch("src.adapters.sqlite3.connection.sqlite3.connect")
    def test_connect_permission_error(self, mock_connect):
        """Given: 권한 에러가 발생할 때
        When: connect를 호출하면
        Then: SQLiteConnectionException이 발생한다
        """
        # Given
        mock_connect.side_effect = PermissionError("Permission denied")
        conn = DatabaseConnection(self.db_path)

        # When & Then
        with self.assertRaises(SQLiteConnectionException) as context:
            conn.connect()

        self.assertIn("Permission denied accessing database", context.exception.message)

    @patch("src.adapters.sqlite3.connection.sqlite3.connect")
    def test_connect_unexpected_error(self, mock_connect):
        """Given: 예상치 못한 에러가 발생할 때
        When: connect를 호출하면
        Then: SQLiteConnectionException이 발생한다
        """
        # Given
        mock_connect.side_effect = ValueError("Unexpected error")
        conn = DatabaseConnection(self.db_path)

        # When & Then
        with self.assertRaises(SQLiteConnectionException) as context:
            conn.connect()

        self.assertIn("An unexpected error occurred during connection", context.exception.message)

    @patch("pathlib.Path.mkdir")
    def test_connect_directory_creation_error(self, mock_mkdir):
        """Given: 디렉토리 생성에서 권한 에러가 발생할 때
        When: connect를 호출하면
        Then: PermissionError가 발생한다
        """
        # Given
        mock_mkdir.side_effect = PermissionError("Cannot create directory")
        conn = DatabaseConnection(self.db_path)

        # When & Then
        with self.assertRaises(PermissionError):
            conn.connect()

    def test_close_with_connection(self):
        """Given: 활성 연결이 있을 때
        When: close를 호출하면
        Then: 연결이 종료되고 None으로 설정된다
        """
        # Given
        conn = DatabaseConnection(self.db_path)
        mock_connection = Mock(spec=sqlite3.Connection)
        conn.connection = mock_connection

        # When
        conn.close()

        # Then
        mock_connection.close.assert_called_once()
        self.assertIsNone(conn.connection)

    def test_close_without_connection(self):
        """Given: 활성 연결이 없을 때
        When: close를 호출하면
        Then: 에러 없이 처리된다
        """
        # Given
        conn = DatabaseConnection(self.db_path)

        # When & Then (예외 발생하지 않아야 함)
        conn.close()

    @patch("src.adapters.sqlite3.connection.sqlite3.connect")
    def test_context_manager_enter_with_existing_connection(self, mock_connect):
        """Given: 이미 연결이 있는 상태에서
        When: context manager로 진입하면
        Then: 기존 연결이 반환된다
        """
        # Given
        conn = DatabaseConnection(self.db_path)
        existing_connection = Mock(spec=sqlite3.Connection)
        conn.connection = existing_connection

        # When
        with conn as result:
            # Then
            self.assertEqual(result, existing_connection)

        mock_connect.assert_not_called()

    @patch("src.adapters.sqlite3.connection.sqlite3.connect")
    def test_context_manager_enter_without_connection(self, mock_connect):
        """Given: 연결이 없는 상태에서
        When: context manager로 진입하면
        Then: 새로운 연결이 생성되어 반환된다
        """
        # Given
        mock_connection = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_connection.execute.return_value = mock_cursor
        mock_connect.return_value = mock_connection

        conn = DatabaseConnection(self.db_path)

        # When
        with conn as result:
            # Then
            self.assertEqual(result, mock_connection)

        mock_connect.assert_called_once()

    def test_context_manager_exit(self):
        """Given: context manager를 통해 연결이 생성된 상태에서
        When: context를 벗어나면
        Then: 연결이 종료된다
        """
        # Given
        conn = DatabaseConnection(self.db_path)
        mock_connection = Mock(spec=sqlite3.Connection)
        conn.connection = mock_connection

        # When
        with conn:
            pass

        # Then
        mock_connection.close.assert_called_once()
        self.assertIsNone(conn.connection)

    def test_context_manager_exit_with_exception(self):
        """Given: context manager 내에서 예외가 발생할 때
        When: context를 벗어나면
        Then: 연결이 여전히 정리된다
        """
        # Given
        conn = DatabaseConnection(self.db_path)
        mock_connection = Mock(spec=sqlite3.Connection)
        conn.connection = mock_connection

        # When
        try:
            with conn:
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Then
        mock_connection.close.assert_called_once()
        self.assertIsNone(conn.connection)

    @patch("src.adapters.sqlite3.connection.get_observable_logger")
    def test_logger_initialization(self, mock_get_logger):
        """Given: DatabaseConnection이 생성될 때
        When: 객체를 초기화하면
        Then: 로거가 올바르게 설정된다
        """
        # Given
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # When
        conn = DatabaseConnection(self.db_path)

        # Then
        mock_get_logger.assert_called_once_with("database_connection", "adapter")
        self.assertEqual(conn.logger, mock_logger)

    def test_apply_optimizations_without_connection(self):
        """Given: 연결이 없는 상태에서
        When: _apply_optimizations를 호출하면
        Then: 아무것도 실행되지 않는다
        """
        # Given
        conn = DatabaseConnection(self.db_path)

        # When & Then (예외 발생하지 않아야 함)
        conn._apply_optimizations()

    def test_connection_parameters(self):
        """Given: DatabaseConnection이 생성될 때
        When: SQLite 연결 파라미터를 확인하면
        Then: 올바른 설정값들이 적용된다
        """
        # 이 테스트는 실제 connect 메서드 호출 시 파라미터를 검증
        conn = DatabaseConnection(self.db_path)

        with patch("src.adapters.sqlite3.connection.sqlite3.connect") as mock_connect:
            mock_connection = Mock(spec=sqlite3.Connection)
            mock_cursor = Mock()
            mock_cursor.fetchone.return_value = (1,)
            mock_connection.execute.return_value = mock_cursor
            mock_connect.return_value = mock_connection

            # When
            conn.connect()

            # Then
            mock_connect.assert_called_once_with(
                str(conn.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                isolation_level=None,
                check_same_thread=False,
                timeout=30.0,
            )

    def test_connection_optimizations_applied(self):
        """Test that connection optimizations are applied."""
        # Given
        conn = DatabaseConnection(self.db_path, optimize=True)

        # When
        # Connect and apply optimizations
        sqlite_conn = conn.connect()

        # Then
        # Query current connection to verify
        cursor = sqlite_conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        result = cursor.fetchone()
        self.assertEqual(result[0].upper(), "WAL")

        # Clean up connection to ensure file is closed
        conn.close()


class TestDatabaseConnectionIntegration(unittest.TestCase):
    """DatabaseConnection의 통합 테스트 (실제 SQLite 사용)."""

    def setUp(self):
        """테스트 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "integration_test.db"

    def test_real_connection_lifecycle(self):
        """Given: 실제 임시 데이터베이스 파일을 사용할 때
        When: 연결 생성, 사용, 해제 과정을 실행하면
        Then: 정상적으로 작동한다
        """
        # Given
        conn = DatabaseConnection(self.db_path)

        # When & Then
        # 연결 생성
        sqlite_conn = conn.connect()
        self.assertIsInstance(sqlite_conn, sqlite3.Connection)

        # 간단한 작업 수행
        cursor = sqlite_conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        self.assertIsNotNone(cursor)

        # 데이터 삽입
        sqlite_conn.execute("INSERT INTO test (id) VALUES (1)")

        # 데이터 조회
        result = sqlite_conn.execute("SELECT id FROM test WHERE id = 1").fetchone()
        self.assertEqual(result["id"], 1)

        # 연결 종료
        conn.close()
        self.assertIsNone(conn.connection)

    def test_context_manager_real_usage(self):
        """Given: 실제 데이터베이스에서 context manager를 사용할 때
        When: with 문으로 작업을 수행하면
        Then: 정상적으로 작동하고 정리된다
        """
        # Given
        conn = DatabaseConnection(self.db_path)

        # When & Then
        with conn as sqlite_conn:
            # 테이블 생성
            sqlite_conn.execute("CREATE TABLE context_test (name TEXT)")

            # 데이터 삽입
            sqlite_conn.execute("INSERT INTO context_test (name) VALUES (?)", ("test",))

            # 데이터 조회
            result = sqlite_conn.execute("SELECT name FROM context_test").fetchone()
            self.assertEqual(result["name"], "test")

        # Context 종료 후 연결이 정리되었는지 확인
        self.assertIsNone(conn.connection)

    def test_datetime_conversion_real(self):
        """Given: 실제 데이터베이스에서 datetime을 사용할 때
        When: datetime 데이터를 저장하고 조회하면
        Then: 올바르게 변환된다
        """
        # Given
        conn = DatabaseConnection(self.db_path)
        test_datetime = datetime.datetime(2023, 12, 25, 15, 30, 45)

        # When & Then
        with conn as sqlite_conn:
            # 테이블 생성
            sqlite_conn.execute(
                "CREATE TABLE datetime_test (id INTEGER PRIMARY KEY, created_at TIMESTAMP)"
            )

            # datetime 데이터 삽입
            sqlite_conn.execute(
                "INSERT INTO datetime_test (created_at) VALUES (?)", (test_datetime,)
            )

            # 데이터 조회
            result = sqlite_conn.execute("SELECT created_at FROM datetime_test").fetchone()
            self.assertEqual(result["created_at"], test_datetime)

    def tearDown(self):
        """테스트 정리."""
        if self.db_path.exists():
            self.db_path.unlink()


if __name__ == "__main__":
    unittest.main()
