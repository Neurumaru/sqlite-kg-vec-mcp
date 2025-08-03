"""
TransactionManager 및 UnitOfWork 단위 테스트.
"""

import importlib.util
import sqlite3
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

# 직접 모듈 임포트 (전체 프로젝트 초기화 우회)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# transaction_context.py 먼저 로드 (의존성)
spec_context = importlib.util.spec_from_file_location(
    "transaction_context", project_root / "src" / "adapters" / "sqlite3" / "transaction_context.py"
)
transaction_context_module = importlib.util.module_from_spec(spec_context)
spec_context.loader.exec_module(transaction_context_module)

# transactions.py 소스 코드를 읽어서 수정된 버전으로 실행
transactions_path = project_root / "src" / "adapters" / "sqlite3" / "transactions.py"
with open(transactions_path, encoding="utf-8") as f:
    transactions_code = f.read()

# 상대 임포트를 절대 임포트로 변경
transactions_code = transactions_code.replace(
    "from .transaction_context import TransactionContext, IsolationLevel, transaction_scope",
    "# 임포트는 exec_globals에서 처리됨",
)

# 전역 네임스페이스에 클래스들 추가
exec_globals = {
    "__name__": "transactions",
    "sqlite3": sqlite3,
    "Generator": type(lambda: (yield)),
    "contextmanager": unittest.mock.Mock(),  # contextmanager를 mock으로 대체
    "Optional": type(None),  # Optional type hint
    "TransactionContext": transaction_context_module.TransactionContext,
    "IsolationLevel": transaction_context_module.IsolationLevel,
    "transaction_scope": transaction_context_module.transaction_scope,
}

# contextmanager 데코레이터를 실제로 구현

exec_globals["contextmanager"] = contextmanager

exec(transactions_code, exec_globals)  # pylint: disable=exec-used

TransactionManager = exec_globals["TransactionManager"]
UnitOfWork = exec_globals["UnitOfWork"]


class TestTransactionManager(unittest.TestCase):
    """TransactionManager 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.mock_connection.in_transaction = False  # 기본적으로 트랜잭션이 없음
        self.transaction_manager = TransactionManager(self.mock_connection)

    def test_init(self):
        """Given: SQLite 연결이 제공될 때
        When: TransactionManager를 초기화하면
        Then: 연결이 설정된다
        """
        # Given & When
        manager = TransactionManager(self.mock_connection)

        # Then
        self.assertEqual(manager.connection, self.mock_connection)

    def test_transaction_success(self):
        """Given: 정상적인 연결이 있을 때
        When: transaction 컨텍스트 매니저를 사용하면
        Then: 트랜잭션이 시작되고 커밋된다
        """
        # Given & When
        with self.transaction_manager.transaction() as tx_context:
            # Then - 트랜잭션 내부
            self.assertIsInstance(tx_context, transaction_context_module.TransactionContext)
            self.assertEqual(tx_context.connection, self.mock_connection)

        # Then - 트랜잭션 완료 후
        expected_calls = [
            unittest.mock.call("BEGIN IMMEDIATE"),
            unittest.mock.call("COMMIT"),
        ]
        self.mock_connection.execute.assert_has_calls(expected_calls)

    def test_transaction_with_custom_isolation_level(self):
        """Given: 사용자 정의 격리 수준이 제공될 때
        When: transaction을 사용하면
        Then: 지정된 격리 수준으로 트랜잭션이 시작된다
        """
        # Given & When
        with self.transaction_manager.transaction(
            isolation_level=transaction_context_module.IsolationLevel.EXCLUSIVE
        ):
            pass

        # Then
        self.mock_connection.execute.assert_any_call("BEGIN EXCLUSIVE")

    def test_transaction_with_exception(self):
        """Given: 트랜잭션 중 예외가 발생할 때
        When: transaction 컨텍스트를 사용하면
        Then: 롤백이 수행되고 예외가 재발생한다
        """
        # Given
        test_exception = Exception("Test exception")

        # When & Then
        with self.assertRaises(Exception) as context:
            with self.transaction_manager.transaction():
                raise test_exception

        self.assertEqual(context.exception, test_exception)
        expected_calls = [
            unittest.mock.call("BEGIN IMMEDIATE"),
            unittest.mock.call("ROLLBACK"),
        ]
        self.mock_connection.execute.assert_has_calls(expected_calls)

    def test_transaction_rollback_on_error(self):
        """Given: 트랜잭션 중 데이터베이스 오류가 발생할 때
        When: transaction을 사용하면
        Then: 롤백이 수행된다
        """
        # Given
        db_error = sqlite3.IntegrityError("Constraint violation")

        # When & Then
        with self.assertRaises(sqlite3.IntegrityError):
            with self.transaction_manager.transaction():
                raise db_error

        self.mock_connection.execute.assert_any_call("ROLLBACK")


class TestUnitOfWork(unittest.TestCase):
    """UnitOfWork 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.mock_connection.in_transaction = False  # 기본적으로 트랜잭션이 없음
        self.unit_of_work = UnitOfWork(self.mock_connection)

    def test_init(self):
        """Given: SQLite 연결이 제공될 때
        When: UnitOfWork를 초기화하면
        Then: 연결과 TransactionManager가 설정된다
        """
        # Given & When
        uow = UnitOfWork(self.mock_connection)

        # Then
        self.assertEqual(uow.connection, self.mock_connection)
        self.assertIsInstance(uow.transaction_manager, TransactionManager)
        self.assertIsNone(uow.correlation_id)

    def test_correlation_id_property(self):
        """Given: UnitOfWork 인스턴스가 있을 때
        When: correlation_id를 설정하고 조회하면
        Then: 올바른 값이 반환된다
        """
        # Given
        test_id = "test-correlation-id"

        # When
        self.unit_of_work.correlation_id = test_id

        # Then
        self.assertEqual(self.unit_of_work.correlation_id, test_id)

    def test_begin_success(self):
        """Given: 정상적인 연결이 있을 때
        When: begin 컨텍스트 매니저를 사용하면
        Then: 트랜잭션이 시작되고 트랜잭션 컨텍스트가 반환된다
        """
        # Given & When
        with self.unit_of_work.begin() as tx_context:
            # Then
            self.assertIsInstance(tx_context, transaction_context_module.TransactionContext)
            self.assertEqual(tx_context.connection, self.mock_connection)

        # Then
        self.mock_connection.execute.assert_any_call("BEGIN IMMEDIATE")
        self.mock_connection.execute.assert_any_call("COMMIT")

    def test_begin_with_custom_isolation_level(self):
        """Given: 사용자 정의 격리 수준이 제공될 때
        When: begin을 사용하면
        Then: 지정된 격리 수준으로 트랜잭션이 시작된다
        """
        # Given & When
        with self.unit_of_work.begin(
            isolation_level=transaction_context_module.IsolationLevel.DEFERRED
        ):
            pass

        # Then
        self.mock_connection.execute.assert_any_call("BEGIN DEFERRED")

    def test_begin_with_exception(self):
        """Given: 트랜잭션 중 예외가 발생할 때
        When: begin을 사용하면
        Then: 롤백이 수행되고 예외가 재발생한다
        """
        # Given
        test_exception = ValueError("Test error")

        # When & Then
        with self.assertRaises(ValueError):
            with self.unit_of_work.begin():
                raise test_exception

        self.mock_connection.execute.assert_any_call("ROLLBACK")

    def test_register_vector_operation_success(self):
        """Given: 벡터 연산 정보가 제공될 때
        When: register_vector_operation을 호출하면
        Then: vector_outbox에 레코드가 삽입된다
        """
        # Given
        mock_cursor = Mock()
        mock_cursor.lastrowid = 123
        self.mock_connection.cursor.return_value = mock_cursor

        entity_type = "node"
        entity_id = 456
        operation_type = "insert"
        model_info = "test-model"
        self.unit_of_work.correlation_id = "test-correlation"

        # When
        result = self.unit_of_work.register_vector_operation(
            entity_type, entity_id, operation_type, model_info
        )

        # Then
        self.assertEqual(result, 123)
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        self.assertIn("INSERT INTO vector_outbox", call_args[0][0])
        self.assertEqual(
            call_args[0][1],
            (operation_type, entity_type, entity_id, model_info, "test-correlation"),
        )

    def test_register_vector_operation_without_model_info(self):
        """Given: model_info가 제공되지 않을 때
        When: register_vector_operation을 호출하면
        Then: None으로 레코드가 삽입된다
        """
        # Given
        mock_cursor = Mock()
        mock_cursor.lastrowid = 456
        self.mock_connection.cursor.return_value = mock_cursor

        # When
        result = self.unit_of_work.register_vector_operation("edge", 789, "update")

        # Then
        self.assertEqual(result, 456)
        call_args = mock_cursor.execute.call_args[0][1]
        self.assertIsNone(call_args[3])  # model_info should be None

    def test_register_vector_operation_without_correlation_id(self):
        """Given: correlation_id가 설정되지 않았을 때
        When: register_vector_operation을 호출하면
        Then: None으로 레코드가 삽입된다
        """
        # Given
        mock_cursor = Mock()
        mock_cursor.lastrowid = 789
        self.mock_connection.cursor.return_value = mock_cursor

        # When
        result = self.unit_of_work.register_vector_operation("hyperedge", 101, "delete", "model-v1")

        # Then
        self.assertEqual(result, 789)
        call_args = mock_cursor.execute.call_args[0][1]
        self.assertIsNone(call_args[4])  # correlation_id should be None

    def test_register_vector_operation_failure(self):
        """Given: lastrowid가 None일 때
        When: register_vector_operation을 호출하면
        Then: RuntimeError가 발생한다
        """
        # Given
        mock_cursor = Mock()
        mock_cursor.lastrowid = None
        self.mock_connection.cursor.return_value = mock_cursor

        # When & Then
        with self.assertRaises(RuntimeError) as context:
            self.unit_of_work.register_vector_operation("node", 1, "insert")

        self.assertIn("vector_outbox에 삽입 실패", str(context.exception))

    def test_register_vector_operation_all_entity_types(self):
        """Given: 다양한 엔티티 타입이 있을 때
        When: register_vector_operation을 호출하면
        Then: 모든 타입이 올바르게 처리된다
        """
        # Given
        mock_cursor = Mock()
        mock_cursor.lastrowid = 1
        self.mock_connection.cursor.return_value = mock_cursor

        entity_types = ["node", "edge", "hyperedge"]
        operation_types = ["insert", "update", "delete"]

        # When & Then
        for entity_type in entity_types:
            for operation_type in operation_types:
                result = self.unit_of_work.register_vector_operation(entity_type, 1, operation_type)
                self.assertEqual(result, 1)

    def test_unit_of_work_integration(self):
        """Given: UnitOfWork와 TransactionManager의 통합이 필요할 때
        When: UnitOfWork를 사용하면
        Then: TransactionManager가 올바르게 사용된다
        """
        # Given
        uow = UnitOfWork(self.mock_connection)

        # When
        with uow.begin() as tx_context:
            # Then
            self.assertIsInstance(tx_context, transaction_context_module.TransactionContext)
            self.assertEqual(tx_context.connection, self.mock_connection)

        # Then - TransactionManager가 호출되었는지 확인
        self.assertIsInstance(uow.transaction_manager, TransactionManager)

    def test_multiple_vector_operations_in_transaction(self):
        """Given: 하나의 트랜잭션에서 여러 벡터 연산을 등록할 때
        When: register_vector_operation을 여러 번 호출하면
        Then: 모든 연산이 올바르게 등록된다
        """
        # Given
        mock_cursor = Mock()
        mock_cursor.lastrowid = 1
        self.mock_connection.cursor.return_value = mock_cursor
        self.unit_of_work.correlation_id = "batch-operation"

        # When
        with self.unit_of_work.begin():
            result1 = self.unit_of_work.register_vector_operation("node", 1, "insert")
            result2 = self.unit_of_work.register_vector_operation("edge", 2, "update")
            result3 = self.unit_of_work.register_vector_operation("node", 3, "delete")

        # Then
        self.assertEqual(result1, 1)
        self.assertEqual(result2, 1)
        self.assertEqual(result3, 1)
        self.assertEqual(mock_cursor.execute.call_count, 3)


if __name__ == "__main__":
    unittest.main()
