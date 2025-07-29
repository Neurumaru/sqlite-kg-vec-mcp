"""
SQLite-specific infrastructure exceptions.
These exceptions handle SQLite database errors and provide
meaningful abstractions for common database failure scenarios.
"""

import sqlite3
from typing import Any, Dict, Optional

from ..exceptions.base import InfrastructureException
from ..exceptions.connection import DatabaseConnectionException
from ..exceptions.data import DataIntegrityException
from ..exceptions.timeout import DatabaseTimeoutException


class SQLiteConnectionException(DatabaseConnectionException):
    """
    SQLite database connection failures.
    Handles file access issues, permissions, database locking,
    and other connection-related problems.
    """

    def __init__(
        self,
        db_path: str,
        message: str,
        sqlite_error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize SQLite connection exception.
        Args:
            db_path: Database file path
            message: Detailed error message
            sqlite_error_code: SQLite error code if available
            context: Additional context
            original_error: Original SQLite exception
        """
        self.sqlite_error_code = sqlite_error_code
        super().__init__(
            db_path=db_path,
            message=message,
            error_code="SQLITE_CONNECTION_FAILED",
            context=context,
            original_error=original_error,
        )

    @classmethod
    def from_sqlite_error(
        cls, db_path: str, sqlite_error: sqlite3.Error
    ) -> "SQLiteConnectionException":
        """
        Create exception from SQLite error.
        Args:
            db_path: Database file path
            sqlite_error: Original SQLite error
        Returns:
            SQLiteConnectionException instance
        """
        error_code = getattr(sqlite_error, "sqlite_errorcode", None)
        error_name = getattr(sqlite_error, "sqlite_errorname", None)
        context = {}
        if error_code:
            context["sqlite_error_code"] = error_code
        if error_name:
            context["sqlite_error_name"] = error_name
        return cls(
            db_path=db_path,
            message=str(sqlite_error),
            sqlite_error_code=error_name,
            context=context,
            original_error=sqlite_error,
        )


class SQLiteIntegrityException(DataIntegrityException):
    """
    SQLite integrity constraint violations.
    Handles foreign key violations, unique constraint violations,
    check constraint failures, and other integrity issues.
    """

    def __init__(
        self,
        constraint: str,
        table: Optional[str] = None,
        column: Optional[str] = None,
        value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize SQLite integrity exception.
        Args:
            constraint: Constraint type or name
            table: Table name where violation occurred
            column: Column name involved in violation
            value: Value that caused the violation
            context: Additional context
            original_error: Original SQLite exception
        """
        self.column = column
        self.value = value
        # Build detailed message
        if table and column:
            message = f"SQLite integrity constraint '{constraint}' violated on {table}.{column}"
            if value is not None:
                message += f" (value: {value})"
        elif table:
            message = f"SQLite integrity constraint '{constraint}' violated on table '{table}'"
        else:
            message = f"SQLite integrity constraint '{constraint}' violated"
        super().__init__(
            constraint=constraint,
            table=table,
            message=message,
            error_code="SQLITE_INTEGRITY_VIOLATION",
            context=context,
            original_error=original_error,
        )

    @classmethod
    def from_sqlite_error(
        cls, sqlite_error: sqlite3.IntegrityError, table: Optional[str] = None
    ) -> "SQLiteIntegrityException":
        """
        Create exception from SQLite IntegrityError.
        Args:
            sqlite_error: Original SQLite integrity error
            table: Table name if known
        Returns:
            SQLiteIntegrityException instance
        """
        error_msg = str(sqlite_error).lower()
        # Determine constraint type from error message
        if "unique" in error_msg:
            constraint = "UNIQUE"
        elif "foreign key" in error_msg:
            constraint = "FOREIGN_KEY"
        elif "check" in error_msg:
            constraint = "CHECK"
        elif "not null" in error_msg:
            constraint = "NOT_NULL"
        else:
            constraint = "UNKNOWN"
        return cls(
            constraint=constraint,
            table=table,
            context={"original_message": str(sqlite_error)},
            original_error=sqlite_error,
        )


class SQLiteOperationalException(InfrastructureException):
    """
    SQLite operational errors.
    Handles database is locked, disk I/O errors, schema changes,
    and other operational issues.
    """

    def __init__(
        self,
        operation: str,
        message: str,
        db_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize SQLite operational exception.
        Args:
            operation: Operation being performed
            message: Detailed error message
            db_path: Database file path if relevant
            context: Additional context
            original_error: Original SQLite exception
        """
        self.operation = operation
        self.db_path = db_path
        full_message = f"SQLite operational error during {operation}: {message}"
        if db_path:
            full_message += f" (Database: {db_path})"
        super().__init__(
            message=full_message,
            error_code="SQLITE_OPERATIONAL_ERROR",
            context=context,
            original_error=original_error,
        )

    @classmethod
    def from_sqlite_error(
        cls,
        operation: str,
        sqlite_error: sqlite3.OperationalError,
        db_path: Optional[str] = None,
    ) -> "SQLiteOperationalException":
        """
        Create exception from SQLite OperationalError.
        Args:
            operation: Operation being performed
            sqlite_error: Original SQLite operational error
            db_path: Database file path
        Returns:
            SQLiteOperationalException instance
        """
        return cls(
            operation=operation,
            message=str(sqlite_error),
            db_path=db_path,
            original_error=sqlite_error,
        )


class SQLiteTimeoutException(DatabaseTimeoutException):
    """
    SQLite timeout errors.
    Handles database busy/locked timeouts and operation timeouts.
    """

    def __init__(
        self,
        operation: str,
        timeout_duration: float,
        db_path: Optional[str] = None,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize SQLite timeout exception.
        Args:
            operation: Database operation that timed out
            timeout_duration: Timeout duration in seconds
            db_path: Database file path
            query: SQL query that timed out
            context: Additional context
            original_error: Original exception
        """
        self.db_path = db_path
        super().__init__(
            operation=f"SQLite {operation}",
            timeout_duration=timeout_duration,
            query=query,
            error_code="SQLITE_TIMEOUT",
            context=context,
            original_error=original_error,
        )


class SQLiteTransactionException(InfrastructureException):
    """
    SQLite transaction errors.
    Handles transaction rollback, deadlock, and state issues.
    """

    def __init__(
        self,
        transaction_id: str,
        state: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize SQLite transaction exception.
        Args:
            transaction_id: Transaction identifier
            state: Transaction state when error occurred
            message: Detailed error message
            context: Additional context
            original_error: Original exception
        """
        self.transaction_id = transaction_id
        self.state = state
        full_message = f"SQLite transaction {transaction_id} failed in state '{state}': {message}"
        super().__init__(
            message=full_message,
            error_code="SQLITE_TRANSACTION_FAILED",
            context=context,
            original_error=original_error,
        )
