"""
SQLite 예외 클래스들의 단위 테스트.
"""

import sqlite3
import unittest
from unittest.mock import Mock

from src.adapters.sqlite3.exceptions import (
    SQLiteConnectionException,
    SQLiteIntegrityException,
    SQLiteOperationalException,
    SQLiteTimeoutException,
    SQLiteTransactionException,
)


class TestSQLiteConnectionException(unittest.TestCase):
    """SQLiteConnectionException 테스트."""

    def test_init(self):
        """Given: 연결 예외 정보가 주어질 때
        When: SQLiteConnectionException을 생성하면
        Then: 예외 정보가 올바르게 설정된다
        """
        # Given
        db_path = "/test/database.db"
        message = "Connection failed"
        sqlite_error_code = "SQLITE_CANTOPEN"
        context = {"retry_count": 3}
        original_error = Exception("Original error")

        # When
        exception = SQLiteConnectionException(
            db_path=db_path,
            message=message,
            sqlite_error_code=sqlite_error_code,
            context=context,
            original_error=original_error,
        )

        # Then
        self.assertEqual(exception.db_path, db_path)
        self.assertIn(message, exception.message)  # 메시지가 포함되는지 확인
        self.assertEqual(exception.sqlite_error_code, sqlite_error_code)
        self.assertEqual(exception.error_code, "SQLITE_CONNECTION_FAILED")
        self.assertEqual(exception.context, context)
        self.assertEqual(exception.original_error, original_error)

    def test_from_sqlite_error_with_attributes(self):
        """Given: SQLite 에러 속성이 있는 예외가 있을 때
        When: from_sqlite_error를 호출하면
        Then: SQLite 에러 정보가 포함된 예외가 생성된다
        """
        # Given
        db_path = "/test/database.db"
        sqlite_error = Mock(spec=sqlite3.Error)
        sqlite_error.sqlite_errorcode = 14
        sqlite_error.sqlite_errorname = "SQLITE_CANTOPEN"
        sqlite_error.__str__ = Mock(return_value="unable to open database file")

        # When
        exception = SQLiteConnectionException.from_sqlite_error(db_path, sqlite_error)

        # Then
        self.assertEqual(exception.db_path, db_path)
        self.assertEqual(exception.sqlite_error_code, "SQLITE_CANTOPEN")
        self.assertEqual(exception.context["sqlite_error_code"], 14)
        self.assertEqual(exception.context["sqlite_error_name"], "SQLITE_CANTOPEN")
        self.assertEqual(exception.original_error, sqlite_error)

    def test_from_sqlite_error_without_attributes(self):
        """Given: SQLite 에러 속성이 없는 예외가 있을 때
        When: from_sqlite_error를 호출하면
        Then: 기본 정보만 포함된 예외가 생성된다
        """
        # Given
        db_path = "/test/database.db"
        sqlite_error = sqlite3.Error("Generic error")

        # When
        exception = SQLiteConnectionException.from_sqlite_error(db_path, sqlite_error)

        # Then
        self.assertEqual(exception.db_path, db_path)
        self.assertIsNone(exception.sqlite_error_code)
        self.assertEqual(exception.context, {})
        self.assertEqual(exception.original_error, sqlite_error)


class TestSQLiteIntegrityException(unittest.TestCase):
    """SQLiteIntegrityException 테스트."""

    def test_init_with_all_params(self):
        """Given: 모든 파라미터가 주어질 때
        When: SQLiteIntegrityException을 생성하면
        Then: 상세한 메시지가 생성된다
        """
        # Given
        constraint = "UNIQUE"
        table = "users"
        column = "email"
        value = "test@example.com"
        context = {"attempt": 2}
        original_error = sqlite3.IntegrityError("UNIQUE constraint failed")

        # When
        exception = SQLiteIntegrityException(
            constraint=constraint,
            table=table,
            column=column,
            value=value,
            context=context,
            original_error=original_error,
        )

        # Then
        self.assertEqual(exception.constraint, constraint)
        self.assertEqual(exception.table, table)
        self.assertEqual(exception.column, column)
        self.assertEqual(exception.value, value)
        self.assertIn("UNIQUE", exception.message)
        self.assertIn("users.email", exception.message)
        self.assertIn("test@example.com", exception.message)

    def test_init_table_only(self):
        """Given: 테이블만 주어질 때
        When: SQLiteIntegrityException을 생성하면
        Then: 테이블 수준 메시지가 생성된다
        """
        # Given
        constraint = "FOREIGN_KEY"
        table = "orders"

        # When
        exception = SQLiteIntegrityException(constraint=constraint, table=table)

        # Then
        self.assertIn("FOREIGN_KEY", exception.message)
        self.assertIn("orders", exception.message)
        self.assertNotIn(".", exception.message)

    def test_init_constraint_only(self):
        """Given: 제약조건만 주어질 때
        When: SQLiteIntegrityException을 생성하면
        Then: 기본 메시지가 생성된다
        """
        # Given
        constraint = "CHECK"

        # When
        exception = SQLiteIntegrityException(constraint=constraint)

        # Then
        self.assertIn("CHECK", exception.message)
        self.assertIn("constraint", exception.message)

    def test_from_sqlite_error_unique(self):
        """Given: UNIQUE 제약조건 위반 에러가 있을 때
        When: from_sqlite_error를 호출하면
        Then: UNIQUE 타입의 예외가 생성된다
        """
        # Given
        sqlite_error = sqlite3.IntegrityError("UNIQUE constraint failed: users.email")
        table = "users"

        # When
        exception = SQLiteIntegrityException.from_sqlite_error(sqlite_error, table)

        # Then
        self.assertEqual(exception.constraint, "UNIQUE")
        self.assertEqual(exception.table, table)
        self.assertEqual(exception.context["original_message"], str(sqlite_error))

    def test_from_sqlite_error_foreign_key(self):
        """Given: 외래키 제약조건 위반 에러가 있을 때
        When: from_sqlite_error를 호출하면
        Then: FOREIGN_KEY 타입의 예외가 생성된다
        """
        # Given
        sqlite_error = sqlite3.IntegrityError("FOREIGN KEY constraint failed")

        # When
        exception = SQLiteIntegrityException.from_sqlite_error(sqlite_error)

        # Then
        self.assertEqual(exception.constraint, "FOREIGN_KEY")

    def test_from_sqlite_error_check(self):
        """Given: CHECK 제약조건 위반 에러가 있을 때
        When: from_sqlite_error를 호출하면
        Then: CHECK 타입의 예외가 생성된다
        """
        # Given
        sqlite_error = sqlite3.IntegrityError("CHECK constraint failed: age >= 0")

        # When
        exception = SQLiteIntegrityException.from_sqlite_error(sqlite_error)

        # Then
        self.assertEqual(exception.constraint, "CHECK")

    def test_from_sqlite_error_not_null(self):
        """Given: NOT NULL 제약조건 위반 에러가 있을 때
        When: from_sqlite_error를 호출하면
        Then: NOT_NULL 타입의 예외가 생성된다
        """
        # Given
        sqlite_error = sqlite3.IntegrityError("NOT NULL constraint failed: users.name")

        # When
        exception = SQLiteIntegrityException.from_sqlite_error(sqlite_error)

        # Then
        self.assertEqual(exception.constraint, "NOT_NULL")

    def test_from_sqlite_error_unknown(self):
        """Given: 알 수 없는 제약조건 위반 에러가 있을 때
        When: from_sqlite_error를 호출하면
        Then: UNKNOWN 타입의 예외가 생성된다
        """
        # Given
        sqlite_error = sqlite3.IntegrityError("Some unknown constraint violation")

        # When
        exception = SQLiteIntegrityException.from_sqlite_error(sqlite_error)

        # Then
        self.assertEqual(exception.constraint, "UNKNOWN")


class TestSQLiteOperationalException(unittest.TestCase):
    """SQLiteOperationalException 테스트."""

    def test_init_with_db_path(self):
        """Given: 데이터베이스 경로가 포함된 정보가 주어질 때
        When: SQLiteOperationalException을 생성하면
        Then: 경로가 포함된 메시지가 생성된다
        """
        # Given
        operation = "SELECT"
        message = "database is locked"
        db_path = "/test/database.db"
        context = {"timeout": 30}
        original_error = sqlite3.OperationalError("database is locked")

        # When
        exception = SQLiteOperationalException(
            operation=operation,
            message=message,
            db_path=db_path,
            context=context,
            original_error=original_error,
        )

        # Then
        self.assertEqual(exception.operation, operation)
        self.assertEqual(exception.db_path, db_path)
        self.assertIn("SELECT", exception.message)
        self.assertIn("database is locked", exception.message)
        self.assertIn("/test/database.db", exception.message)
        self.assertEqual(exception.error_code, "SQLITE_OPERATIONAL_ERROR")

    def test_init_without_db_path(self):
        """Given: 데이터베이스 경로가 없을 때
        When: SQLiteOperationalException을 생성하면
        Then: 경로 없는 메시지가 생성된다
        """
        # Given
        operation = "INSERT"
        message = "disk I/O error"

        # When
        exception = SQLiteOperationalException(operation=operation, message=message)

        # Then
        self.assertIn("INSERT", exception.message)
        self.assertIn("disk I/O error", exception.message)
        self.assertNotIn("Database:", exception.message)

    def test_from_sqlite_error(self):
        """Given: SQLite OperationalError가 있을 때
        When: from_sqlite_error를 호출하면
        Then: SQLiteOperationalException이 생성된다
        """
        # Given
        operation = "CREATE TABLE"
        sqlite_error = sqlite3.OperationalError("table already exists")
        db_path = "/test/database.db"

        # When
        exception = SQLiteOperationalException.from_sqlite_error(operation, sqlite_error, db_path)

        # Then
        self.assertEqual(exception.operation, operation)
        self.assertEqual(exception.db_path, db_path)
        self.assertIn("CREATE TABLE", exception.message)
        self.assertIn("table already exists", exception.message)
        self.assertEqual(exception.original_error, sqlite_error)


class TestSQLiteTimeoutException(unittest.TestCase):
    """SQLiteTimeoutException 테스트."""

    def test_init_with_all_params(self):
        """Given: 모든 타임아웃 정보가 주어질 때
        When: SQLiteTimeoutException을 생성하면
        Then: 타임아웃 정보가 설정된다
        """
        # Given
        operation = "UPDATE"
        timeout_duration = 30.5
        db_path = "/test/database.db"
        query = "UPDATE users SET status = ? WHERE id = ?"
        context = {"retry_count": 3}
        original_error = Exception("Timeout occurred")

        # When
        exception = SQLiteTimeoutException(
            operation=operation,
            timeout_duration=timeout_duration,
            db_path=db_path,
            query=query,
            context=context,
            original_error=original_error,
        )

        # Then
        self.assertEqual(exception.db_path, db_path)
        self.assertIn("UPDATE", exception.operation)  # 'Database SQLite UPDATE' 형태로 변경됨
        self.assertEqual(exception.timeout_duration, timeout_duration)
        self.assertEqual(exception.query, query)
        self.assertEqual(exception.error_code, "SQLITE_TIMEOUT")
        self.assertEqual(exception.context, context)
        self.assertEqual(exception.original_error, original_error)

    def test_init_minimal_params(self):
        """Given: 최소 파라미터만 주어질 때
        When: SQLiteTimeoutException을 생성하면
        Then: 기본 타임아웃 예외가 생성된다
        """
        # Given
        operation = "SELECT"
        timeout_duration = 15.0

        # When
        exception = SQLiteTimeoutException(operation=operation, timeout_duration=timeout_duration)

        # Then
        self.assertIn("SELECT", exception.operation)  # 'Database SQLite SELECT' 형태로 변경됨
        self.assertEqual(exception.timeout_duration, timeout_duration)
        self.assertIsNone(exception.db_path)
        self.assertIsNone(exception.query)


class TestSQLiteTransactionException(unittest.TestCase):
    """SQLiteTransactionException 테스트."""

    def test_init(self):
        """Given: 트랜잭션 예외 정보가 주어질 때
        When: SQLiteTransactionException을 생성하면
        Then: 트랜잭션 정보가 설정된다
        """
        # Given
        transaction_id = "tx_12345"
        state = "COMMITTING"
        message = "Deadlock detected"
        context = {"retry_attempts": 2}
        original_error = sqlite3.OperationalError("database is locked")

        # When
        exception = SQLiteTransactionException(
            transaction_id=transaction_id,
            state=state,
            message=message,
            context=context,
            original_error=original_error,
        )

        # Then
        self.assertEqual(exception.transaction_id, transaction_id)
        self.assertEqual(exception.state, state)
        self.assertIn("tx_12345", exception.message)
        self.assertIn("COMMITTING", exception.message)
        self.assertIn("Deadlock detected", exception.message)
        self.assertEqual(exception.error_code, "SQLITE_TRANSACTION_FAILED")
        self.assertEqual(exception.context, context)
        self.assertEqual(exception.original_error, original_error)

    def test_init_minimal(self):
        """Given: 최소 트랜잭션 정보만 주어질 때
        When: SQLiteTransactionException을 생성하면
        Then: 기본 트랜잭션 예외가 생성된다
        """
        # Given
        transaction_id = "tx_456"
        state = "ACTIVE"
        message = "Transaction failed"

        # When
        exception = SQLiteTransactionException(
            transaction_id=transaction_id, state=state, message=message
        )

        # Then
        self.assertEqual(exception.transaction_id, transaction_id)
        self.assertEqual(exception.state, state)
        self.assertIn("tx_456", exception.message)
        self.assertIn("ACTIVE", exception.message)
        self.assertIn("Transaction failed", exception.message)


class TestExceptionHierarchy(unittest.TestCase):
    """예외 클래스 계층 구조 테스트."""

    def test_sqlite_connection_exception_inheritance(self):
        """Given: SQLiteConnectionException이 있을 때
        When: 타입을 확인하면
        Then: 올바른 상속 구조를 가진다
        """
        # Given
        exception = SQLiteConnectionException("test.db", "test message")

        # Then
        from src.adapters.exceptions.connection import DatabaseConnectionException

        self.assertIsInstance(exception, DatabaseConnectionException)

    def test_sqlite_integrity_exception_inheritance(self):
        """Given: SQLiteIntegrityException이 있을 때
        When: 타입을 확인하면
        Then: 올바른 상속 구조를 가진다
        """
        # Given
        exception = SQLiteIntegrityException("UNIQUE", "users")

        # Then
        from src.adapters.exceptions.data import DataIntegrityException

        self.assertIsInstance(exception, DataIntegrityException)

    def test_sqlite_operational_exception_inheritance(self):
        """Given: SQLiteOperationalException이 있을 때
        When: 타입을 확인하면
        Then: 올바른 상속 구조를 가진다
        """
        # Given
        exception = SQLiteOperationalException("SELECT", "test message")

        # Then
        from src.adapters.exceptions.base import InfrastructureException

        self.assertIsInstance(exception, InfrastructureException)

    def test_sqlite_timeout_exception_inheritance(self):
        """Given: SQLiteTimeoutException이 있을 때
        When: 타입을 확인하면
        Then: 올바른 상속 구조를 가진다
        """
        # Given
        exception = SQLiteTimeoutException("SELECT", 30.0)

        # Then
        from src.adapters.exceptions.timeout import DatabaseTimeoutException

        self.assertIsInstance(exception, DatabaseTimeoutException)

    def test_sqlite_transaction_exception_inheritance(self):
        """Given: SQLiteTransactionException이 있을 때
        When: 타입을 확인하면
        Then: 올바른 상속 구조를 가진다
        """
        # Given
        exception = SQLiteTransactionException("tx_1", "ACTIVE", "test message")

        # Then
        from src.adapters.exceptions.base import InfrastructureException

        self.assertIsInstance(exception, InfrastructureException)


if __name__ == "__main__":
    unittest.main()
