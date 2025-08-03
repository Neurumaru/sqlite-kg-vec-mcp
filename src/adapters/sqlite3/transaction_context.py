"""
표준화된 트랜잭션 컨텍스트 관리.
"""

import logging
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class IsolationLevel(Enum):
    """SQLite 트랜잭션 격리 수준."""

    DEFERRED = "DEFERRED"
    IMMEDIATE = "IMMEDIATE"
    EXCLUSIVE = "EXCLUSIVE"


class TransactionState(Enum):
    """트랜잭션 상태."""

    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class TransactionContext:
    """
    표준화된 트랜잭션 컨텍스트.
    모든 데이터베이스 작업에서 일관된 트랜잭션 관리를 제공합니다.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        isolation_level: IsolationLevel = IsolationLevel.IMMEDIATE,
        auto_commit: bool = True,
    ):
        """
        트랜잭션 컨텍스트를 초기화합니다.

        Args:
            connection: SQLite 데이터베이스 연결
            isolation_level: 트랜잭션 격리 수준
            auto_commit: 컨텍스트 종료 시 자동 커밋 여부
        """
        self.connection = connection
        self.isolation_level = isolation_level
        self.auto_commit = auto_commit
        self.transaction_id = str(uuid.uuid4())
        self.state = TransactionState.ACTIVE
        self._is_nested = False
        self._savepoint_name: Optional[str] = None

    @contextmanager
    def begin(self) -> Generator["TransactionContext", None, None]:
        """
        트랜잭션을 시작합니다.

        Yields:
            활성 트랜잭션 컨텍스트
        """
        # 이미 트랜잭션이 활성화되어 있는지 확인
        in_transaction = self.connection.in_transaction

        try:
            if in_transaction:
                # 중첩 트랜잭션의 경우 savepoint 사용
                self._is_nested = True
                self._savepoint_name = f"sp_{self.transaction_id[:8]}"
                self.connection.execute(f"SAVEPOINT {self._savepoint_name}")
                logger.debug("Savepoint 생성: %s", self._savepoint_name)
            else:
                # 새 트랜잭션 시작
                self.connection.execute(f"BEGIN {self.isolation_level.value}")
                logger.debug(
                    "트랜잭션 시작: %s (%s)", self.transaction_id, self.isolation_level.value
                )

            yield self

            # 성공적으로 완료된 경우 커밋
            if self.auto_commit and self.state == TransactionState.ACTIVE:
                self.commit()

        except Exception as e:
            # 예외 발생 시 롤백
            if self.state == TransactionState.ACTIVE:
                self.rollback()
            logger.error("트랜잭션 실패: %s, 오류: %s", self.transaction_id, e)
            raise

    def commit(self) -> bool:
        """
        트랜잭션을 커밋합니다.

        Returns:
            커밋 성공 여부
        """
        if self.state != TransactionState.ACTIVE:
            logger.warning(
                "비활성 트랜잭션 커밋 시도: %s (상태: %s)", self.transaction_id, self.state
            )
            return False

        try:
            if self._is_nested:
                # Savepoint 해제
                if self._savepoint_name:
                    self.connection.execute(f"RELEASE SAVEPOINT {self._savepoint_name}")
                    logger.debug("Savepoint 해제: %s", self._savepoint_name)
            else:
                # 트랜잭션 커밋
                self.connection.execute("COMMIT")
                logger.debug("트랜잭션 커밋: %s", self.transaction_id)

            self.state = TransactionState.COMMITTED
            return True

        except Exception as e:
            logger.error("커밋 실패: %s, 오류: %s", self.transaction_id, e)
            self.state = TransactionState.FAILED
            return False

    def rollback(self) -> bool:
        """
        트랜잭션을 롤백합니다.

        Returns:
            롤백 성공 여부
        """
        if self.state not in (TransactionState.ACTIVE, TransactionState.FAILED):
            logger.warning(
                "비활성 트랜잭션 롤백 시도: %s (상태: %s)", self.transaction_id, self.state
            )
            return False

        try:
            if self._is_nested:
                # Savepoint로 롤백
                if self._savepoint_name:
                    self.connection.execute(f"ROLLBACK TO SAVEPOINT {self._savepoint_name}")
                    logger.debug("Savepoint 롤백: %s", self._savepoint_name)
            else:
                # 트랜잭션 롤백
                self.connection.execute("ROLLBACK")
                logger.debug("트랜잭션 롤백: %s", self.transaction_id)

            self.state = TransactionState.ROLLED_BACK
            return True

        except Exception as e:
            logger.error("롤백 실패: %s, 오류: %s", self.transaction_id, e)
            self.state = TransactionState.FAILED
            return False

    def execute(self, sql: str, parameters=None) -> sqlite3.Cursor:
        """
        트랜잭션 컨텍스트에서 SQL을 실행합니다.

        Args:
            sql: 실행할 SQL 문
            parameters: SQL 매개변수

        Returns:
            SQLite 커서
        """
        if self.state != TransactionState.ACTIVE:
            raise RuntimeError(f"비활성 트랜잭션에서 SQL 실행 시도: {self.transaction_id}")

        try:
            if parameters is not None:
                return self.connection.execute(sql, parameters)
            return self.connection.execute(sql)
        except Exception as e:
            logger.error("SQL 실행 실패: %s, SQL: %s, 오류: %s", self.transaction_id, sql, e)
            raise

    def executemany(self, sql: str, parameters_list) -> sqlite3.Cursor:
        """
        트랜잭션 컨텍스트에서 여러 SQL을 실행합니다.

        Args:
            sql: 실행할 SQL 문
            parameters_list: SQL 매개변수 목록

        Returns:
            SQLite 커서
        """
        if self.state != TransactionState.ACTIVE:
            raise RuntimeError(f"비활성 트랜잭션에서 SQL 실행 시도: {self.transaction_id}")

        try:
            return self.connection.executemany(sql, parameters_list)
        except Exception as e:
            logger.error("배치 SQL 실행 실패: %s, SQL: %s, 오류: %s", self.transaction_id, sql, e)
            raise

    @property
    def is_active(self) -> bool:
        """트랜잭션이 활성 상태인지 확인합니다."""
        return self.state == TransactionState.ACTIVE

    @property
    def is_nested(self) -> bool:
        """중첩 트랜잭션인지 확인합니다."""
        return self._is_nested

    def __str__(self) -> str:
        """트랜잭션 컨텍스트의 문자열 표현."""
        return f"TransactionContext(id={self.transaction_id[:8]}, state={self.state.value}, nested={self._is_nested})"

    # Connection 인터페이스 위임 메서드들
    def cursor(self):
        """Connection의 cursor 메서드를 위임합니다."""
        return self.connection.cursor()

    def executescript(self, sql_script: str):
        """Connection의 executescript 메서드를 위임합니다."""
        return self.connection.executescript(sql_script)

    @property
    def row_factory(self):
        """Connection의 row_factory 속성을 위임합니다."""
        return self.connection.row_factory

    @row_factory.setter
    def row_factory(self, value):
        """Connection의 row_factory 속성을 위임합니다."""
        self.connection.row_factory = value

    @property
    def in_transaction(self):
        """Connection의 in_transaction 속성을 위임합니다."""
        return self.connection.in_transaction

    def close(self):
        """Connection의 close 메서드를 위임합니다."""
        return self.connection.close()

    def create_function(self, name, num_params, func):
        """Connection의 create_function 메서드를 위임합니다."""
        return self.connection.create_function(name, num_params, func)

    def create_aggregate(self, name, num_params, aggregate_class):
        """Connection의 create_aggregate 메서드를 위임합니다."""
        return self.connection.create_aggregate(name, num_params, aggregate_class)

    def isolation_level_property(self):
        """Connection의 isolation_level 속성을 위임합니다."""
        return self.connection.isolation_level

    def total_changes(self):
        """Connection의 total_changes 속성을 위임합니다."""
        return self.connection.total_changes

    def interrupt(self):
        """Connection의 interrupt 메서드를 위임합니다."""
        return self.connection.interrupt()

    def set_authorizer(self, authorizer):
        """Connection의 set_authorizer 메서드를 위임합니다."""
        return self.connection.set_authorizer(authorizer)

    def set_progress_handler(self, handler, n):
        """Connection의 set_progress_handler 메서드를 위임합니다."""
        return self.connection.set_progress_handler(handler, n)

    def set_trace_callback(self, trace_callback):
        """Connection의 set_trace_callback 메서드를 위임합니다."""
        return self.connection.set_trace_callback(trace_callback)


@contextmanager
def transaction_scope(
    connection: sqlite3.Connection,
    isolation_level: IsolationLevel = IsolationLevel.IMMEDIATE,
    auto_commit: bool = True,
) -> Generator[TransactionContext, None, None]:
    """
    트랜잭션 스코프를 위한 편의 함수.

    Args:
        connection: SQLite 데이터베이스 연결
        isolation_level: 트랜잭션 격리 수준
        auto_commit: 자동 커밋 여부

    Yields:
        트랜잭션 컨텍스트
    """
    tx_context = TransactionContext(connection, isolation_level, auto_commit)
    with tx_context.begin():
        yield tx_context
