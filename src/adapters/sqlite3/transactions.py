"""
SQLite 데이터베이스 작업에 대한 트랜잭션 관리.
"""

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from typing import Optional

from .transaction_context import TransactionContext, IsolationLevel, transaction_scope


class TransactionManager:
    """
    원자성과 일관성을 보장하기 위해 데이터베이스 트랜잭션을 관리합니다.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        트랜잭션 관리자를 초기화합니다.
        Args:
            connection: SQLite 데이터베이스 연결
        """
        self.connection = connection

    @contextmanager
    def transaction(
        self, isolation_level: IsolationLevel = IsolationLevel.IMMEDIATE
    ) -> Generator[TransactionContext, None, None]:
        """
        데이터베이스 트랜잭션을 위한 컨텍스트 관리자.

        Args:
            isolation_level: SQLite 격리 수준

        Yields:
            트랜잭션 컨텍스트

        Raises:
            트랜잭션 컨텍스트의 모든 예외
        """
        with transaction_scope(self.connection, isolation_level) as tx_context:
            yield tx_context


class UnitOfWork:
    """
    DB 변경 사항을 조정하고 추적하기 위해 작업 단위 패턴을 구현합니다.
    """

    def __init__(self, connection: sqlite3.Connection):
        """
        작업 단위를 초기화합니다.
        Args:
            connection: SQLite 데이터베이스 연결
        """
        self.connection = connection
        self.transaction_manager = TransactionManager(connection)
        self._correlation_id: Optional[str] = None

    @property
    def correlation_id(self) -> Optional[str]:
        """관련 작업을 추적하기 위한 상관 관계 ID를 가져옵니다."""
        return self._correlation_id

    @correlation_id.setter
    def correlation_id(self, value: str) -> None:
        """관련 작업을 추적하기 위한 상관 관계 ID를 설정합니다."""
        self._correlation_id = value

    @contextmanager
    def begin(
        self, isolation_level: IsolationLevel = IsolationLevel.IMMEDIATE
    ) -> Generator[TransactionContext, None, None]:
        """
        작업 단위(트랜잭션)를 시작합니다.

        Args:
            isolation_level: SQLite 격리 수준

        Yields:
            트랜잭션 컨텍스트
        """
        with self.transaction_manager.transaction(isolation_level) as tx_context:
            yield tx_context

    def register_vector_operation(
        self,
        entity_type: str,
        entity_id: int,
        operation_type: str,
        model_info: Optional[str] = None,
    ) -> int:
        """
        비동기 처리를 위해 아웃박스에 벡터 작업을 등록합니다.
        Args:
            entity_type: 엔티티 유형 ('node', 'edge', 'hyperedge')
            entity_id: 엔티티 ID
            operation_type: 작업 유형 ('insert', 'update', 'delete')
            model_info: 임베딩을 위한 선택적 모델 정보
        Returns:
            생성된 아웃박스 항목의 ID
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
        INSERT INTO vector_outbox (
            operation_type, entity_type, entity_id, model_info, correlation_id
        ) VALUES (?, ?, ?, ?, ?)
        """,
            (operation_type, entity_type, entity_id, model_info, self._correlation_id),
        )
        result = cursor.lastrowid
        if result is None:
            raise RuntimeError("vector_outbox에 삽입 실패")
        return result
