"""
DatabaseConnection 어댑터에 대한 단위 테스트.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.adapters.sqlite3.connection import DatabaseConnection


class TestDatabaseConnection(unittest.TestCase):
    """DatabaseConnection 어댑터에 대한 테스트 케이스."""

    def setUp(self):
        """테스트 픽스처를 설정합니다."""
        # 임시 데이터베이스 파일 생성
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"

    def tearDown(self):
        """테스트 픽스처를 정리합니다."""
        # 임시 디렉토리 및 파일 제거
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_create_database_connection(self):
        """데이터베이스 연결 생성을 테스트합니다."""
        db_conn = DatabaseConnection(self.db_path)
        self.assertEqual(db_conn.db_path, self.db_path)
        self.assertIsNone(db_conn.connection)

    def test_connect_creates_database_file(self):
        """connect()가 데이터베이스 파일을 생성하는지 테스트합니다."""
        db_conn = DatabaseConnection(self.db_path)
        connection = db_conn.connect()

        self.assertTrue(self.db_path.exists())
        self.assertIsNotNone(connection)
        self.assertIsNotNone(db_conn.connection)

        db_conn.close()

    def test_connect_creates_parent_directory(self):
        """connect()가 부모 디렉토리를 생성하는지 테스트합니다."""
        nested_path = Path(self.temp_dir) / "nested" / "test.db"
        db_conn = DatabaseConnection(nested_path)

        db_conn.connect()

        self.assertTrue(nested_path.parent.exists())
        self.assertTrue(nested_path.exists())

        db_conn.close()

    def test_context_manager(self):
        """DatabaseConnection을 컨텍스트 관리자로 사용하는 것을 테스트합니다."""
        db_conn = DatabaseConnection(self.db_path)

        with db_conn as connection:
            self.assertIsNotNone(connection)
            # 기본 SQL 작업 테스트
            result = connection.execute("SELECT 1").fetchone()
            self.assertEqual(result[0], 1)

        # 컨텍스트 종료 후 연결이 닫혀야 함
        self.assertIsNone(db_conn.connection)

    def test_connection_optimizations_applied(self):
        """연결 최적화가 적용되는지 테스트합니다."""
        db_conn = DatabaseConnection(self.db_path, optimize=True)

        with db_conn as connection:
            # WAL 모드 확인
            result = connection.execute("PRAGMA journal_mode").fetchone()
            self.assertEqual(result[0].upper(), "WAL")

            # 외래 키 활성화 확인
            result = connection.execute("PRAGMA foreign_keys").fetchone()
            self.assertEqual(result[0], 1)

    def test_no_optimizations_when_disabled(self):
        """비활성화된 경우 최적화가 적용되지 않는지 테스트합니다."""
        db_conn = DatabaseConnection(self.db_path, optimize=False)

        with db_conn as connection:
            # 여전히 연결하고 쿼리를 실행할 수 있어야 함
            result = connection.execute("SELECT 1").fetchone()
            self.assertEqual(result[0], 1)

    def test_close_connection(self):
        """명시적으로 연결을 닫는 것을 테스트합니다."""
        db_conn = DatabaseConnection(self.db_path)
        db_conn.connect()

        self.assertIsNotNone(db_conn.connection)

        db_conn.close()

        self.assertIsNone(db_conn.connection)


if __name__ == "__main__":
    unittest.main()
