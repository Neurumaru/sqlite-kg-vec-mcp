"""
SQLite 관련 인프라 예외.
이 예외들은 SQLite 데이터베이스 오류를 처리하고
일반적인 데이터베이스 실패 시나리오에 대한 의미 있는 추상화를 제공합니다.
"""

import sqlite3
from typing import Any, Optional

from ..exceptions.base import InfrastructureException
from ..exceptions.connection import DatabaseConnectionException
from ..exceptions.data import DataIntegrityException
from ..exceptions.timeout import DatabaseTimeoutException


class SQLiteConnectionException(DatabaseConnectionException):
    """
    SQLite 데이터베이스 연결 실패.
    파일 접근 문제, 권한, 데이터베이스 잠금 및
    기타 연결 관련 문제를 처리합니다.
    """

    def __init__(
        self,
        db_path: str,
        message: str,
        sqlite_error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        SQLite 연결 예외를 초기화합니다.
        Args:
            db_path: 데이터베이스 파일 경로
            message: 상세 오류 메시지
            sqlite_error_code: 사용 가능한 경우 SQLite 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 SQLite 예외
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
        SQLite 오류로부터 예외를 생성합니다.
        Args:
            db_path: 데이터베이스 파일 경로
            sqlite_error: 원래 SQLite 오류
        Returns:
            SQLiteConnectionException 인스턴스
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
    SQLite 무결성 제약 조건 위반.
    외래 키 위반, 고유 제약 조건 위반,
    검사 제약 조건 실패 및 기타 무결성 문제를 처리합니다.
    """

    def __init__(
        self,
        constraint: str,
        table: Optional[str] = None,
        column: Optional[str] = None,
        value: Optional[Any] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        SQLite 무결성 예외를 초기화합니다.
        Args:
            constraint: 제약 조건 유형 또는 이름
            table: 위반이 발생한 테이블 이름
            column: 위반에 관련된 열 이름
            value: 위반을 일으킨 값
            context: 추가 컨텍스트
            original_error: 원래 SQLite 예외
        """
        self.column = column
        self.value = value
        # 상세 메시지 빌드
        if table and column:
            message = (
                f"SQLite 무결성 제약 조건 '{constraint}'이(가) {table}.{column}에서 위반되었습니다"
            )
            if value is not None:
                message += f" (값: {value})"
        elif table:
            message = (
                f"SQLite 무결성 제약 조건 '{constraint}'이(가) 테이블 '{table}'에서 위반되었습니다"
            )
        else:
            message = f"SQLite 무결성 제약 조건 '{constraint}'이(가) 위반되었습니다"
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
        cls, sqlite_error: Optional[sqlite3.IntegrityError] = None, table: Optional[str] = None
    ) -> "SQLiteIntegrityException":
        """
        SQLite IntegrityError로부터 예외를 생성합니다.
        Args:
            sqlite_error: 원래 SQLite 무결성 오류
            table: 알려진 경우 테이블 이름
        Returns:
            SQLiteIntegrityException 인스턴스
        """
        error_msg = str(sqlite_error).lower()
        # 오류 메시지로부터 제약 조건 유형 결정
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
            table=table or "unknown",
            context={"original_message": str(sqlite_error)},
            original_error=sqlite_error,
        )


class SQLiteOperationalException(InfrastructureException):
    """
    SQLite 운영 오류.
    데이터베이스 잠금, 디스크 I/O 오류, 스키마 변경 및
    기타 운영 문제를 처리합니다.
    """

    def __init__(
        self,
        operation: str,
        message: str,
        db_path: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        SQLite 운영 예외를 초기화합니다.
        Args:
            operation: 수행 중인 작업
            message: 상세 오류 메시지
            db_path: 관련된 경우 데이터베이스 파일 경로
            context: 추가 컨텍스트
            original_error: 원래 SQLite 예외
        """
        self.operation = operation
        self.db_path = db_path
        full_message = f"{operation} 중 SQLite 운영 오류 발생: {message}"
        if db_path:
            full_message += f" (데이터베이스: {db_path})"
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
        SQLite OperationalError로부터 예외를 생성합니다.
        Args:
            operation: 수행 중인 작업
            sqlite_error: 원래 SQLite 운영 오류
            db_path: 데이터베이스 파일 경로
        Returns:
            SQLiteOperationalException 인스턴스
        """
        return cls(
            operation=operation,
            message=str(sqlite_error),
            db_path=db_path,
            original_error=sqlite_error,
        )


class SQLiteTimeoutException(DatabaseTimeoutException):
    """
    SQLite 시간 초과 오류.
    데이터베이스 busy/locked 시간 초과 및 작업 시간 초과를 처리합니다.
    """

    def __init__(
        self,
        operation: str,
        timeout_duration: float,
        db_path: Optional[str] = None,
        query: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        SQLite 시간 초과 예외를 초기화합니다.
        Args:
            operation: 시간 초과된 데이터베이스 작업
            timeout_duration: 시간 초과 기간(초)
            db_path: 데이터베이스 파일 경로
            query: 시간 초과된 SQL 쿼리
            context: 추가 컨텍스트
            original_error: 원래 예외
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
    SQLite 트랜잭션 오류.
    트랜잭션 롤백, 교착 상태 및 상태 문제를 처리합니다.
    """

    def __init__(
        self,
        transaction_id: str,
        state: str,
        message: str,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        SQLite 트랜잭션 예외를 초기화합니다.
        Args:
            transaction_id: 트랜잭션 식별자
            state: 오류 발생 시 트랜잭션 상태
            message: 상세 오류 메시지
            context: 추가 컨텍스트
            original_error: 원래 예외
        """
        self.transaction_id = transaction_id
        self.state = state
        full_message = (
            f"SQLite 트랜잭션 {transaction_id}이(가) '{state}' 상태에서 실패했습니다: {message}"
        )
        super().__init__(
            message=full_message,
            error_code="SQLITE_TRANSACTION_FAILED",
            context=context,
            original_error=original_error,
        )
