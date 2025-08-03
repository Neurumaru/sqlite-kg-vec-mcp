"""
TransactionContext 단위 테스트.
"""

import importlib.util
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

# 직접 모듈 임포트 (전체 프로젝트 초기화 우회)
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

spec = importlib.util.spec_from_file_location(
    "transaction_context", project_root / "src" / "adapters" / "sqlite3" / "transaction_context.py"
)
transaction_context_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transaction_context_module)

TransactionContext = transaction_context_module.TransactionContext
IsolationLevel = transaction_context_module.IsolationLevel
TransactionState = transaction_context_module.TransactionState
transaction_scope = transaction_context_module.transaction_scope


class TestIsolationLevel(unittest.TestCase):
    """IsolationLevel Enum 테스트."""

    def test_isolation_levels(self):
        """Given: IsolationLevel Enum이 있을 때
        When: 각 격리 수준을 확인하면
        Then: 올바른 SQLite 값이 반환된다
        """
        self.assertEqual(IsolationLevel.DEFERRED.value, "DEFERRED")
        self.assertEqual(IsolationLevel.IMMEDIATE.value, "IMMEDIATE")
        self.assertEqual(IsolationLevel.EXCLUSIVE.value, "EXCLUSIVE")


class TestTransactionState(unittest.TestCase):
    """TransactionState Enum 테스트."""

    def test_transaction_states(self):
        """Given: TransactionState Enum이 있을 때
        When: 각 상태를 확인하면
        Then: 올바른 문자열 값이 반환된다
        """
        self.assertEqual(TransactionState.ACTIVE.value, "active")
        self.assertEqual(TransactionState.COMMITTED.value, "committed")
        self.assertEqual(TransactionState.ROLLED_BACK.value, "rolled_back")
        self.assertEqual(TransactionState.FAILED.value, "failed")


class TestTransactionContext(unittest.TestCase):
    """TransactionContext 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.mock_connection.in_transaction = False

    def test_init_default_values(self):
        """Given: 기본 매개변수로 TransactionContext를 생성할 때
        When: 인스턴스를 초기화하면
        Then: 기본 값들이 올바르게 설정된다
        """
        # Given & When
        tx_context = TransactionContext(self.mock_connection)

        # Then
        self.assertEqual(tx_context.connection, self.mock_connection)
        self.assertEqual(tx_context.isolation_level, IsolationLevel.IMMEDIATE)
        self.assertTrue(tx_context.auto_commit)
        self.assertEqual(tx_context.state, TransactionState.ACTIVE)
        self.assertFalse(tx_context._is_nested)
        self.assertIsNone(tx_context._savepoint_name)
        self.assertIsNotNone(tx_context.transaction_id)

    def test_init_custom_values(self):
        """Given: 사용자 정의 매개변수로 TransactionContext를 생성할 때
        When: 인스턴스를 초기화하면
        Then: 사용자 정의 값들이 설정된다
        """
        # Given & When
        tx_context = TransactionContext(
            self.mock_connection,
            isolation_level=IsolationLevel.EXCLUSIVE,
            auto_commit=False,
        )

        # Then
        self.assertEqual(tx_context.isolation_level, IsolationLevel.EXCLUSIVE)
        self.assertFalse(tx_context.auto_commit)

    def test_begin_new_transaction(self):
        """Given: 활성 트랜잭션이 없을 때
        When: begin 컨텍스트 매니저를 사용하면
        Then: 새 트랜잭션이 시작되고 커밋된다
        """
        # Given
        self.mock_connection.in_transaction = False
        tx_context = TransactionContext(self.mock_connection)

        # When
        with tx_context.begin() as ctx:
            # Then - 트랜잭션 내부
            self.assertEqual(ctx, tx_context)
            self.assertEqual(ctx.state, TransactionState.ACTIVE)

        # Then - 트랜잭션 완료 후
        expected_calls = [
            unittest.mock.call("BEGIN IMMEDIATE"),
            unittest.mock.call("COMMIT"),
        ]
        self.mock_connection.execute.assert_has_calls(expected_calls)
        self.assertEqual(tx_context.state, TransactionState.COMMITTED)

    def test_begin_nested_transaction(self):
        """Given: 이미 활성 트랜잭션이 있을 때
        When: begin 컨텍스트 매니저를 사용하면
        Then: savepoint가 생성되고 해제된다
        """
        # Given
        self.mock_connection.in_transaction = True
        tx_context = TransactionContext(self.mock_connection)

        # When
        with tx_context.begin() as ctx:
            # Then - 중첩 트랜잭션 내부
            self.assertTrue(ctx._is_nested)
            self.assertIsNotNone(ctx._savepoint_name)

        # Then - 중첩 트랜잭션 완료 후
        savepoint_name = tx_context._savepoint_name
        expected_calls = [
            unittest.mock.call(f"SAVEPOINT {savepoint_name}"),
            unittest.mock.call(f"RELEASE SAVEPOINT {savepoint_name}"),
        ]
        self.mock_connection.execute.assert_has_calls(expected_calls)

    def test_begin_with_exception(self):
        """Given: 트랜잭션 중 예외가 발생할 때
        When: begin 컨텍스트를 사용하면
        Then: 롤백이 수행되고 예외가 재발생한다
        """
        # Given
        self.mock_connection.in_transaction = False
        tx_context = TransactionContext(self.mock_connection)
        test_exception = ValueError("Test exception")

        # When & Then
        with self.assertRaises(ValueError) as context:
            with tx_context.begin():
                raise test_exception

        self.assertEqual(context.exception, test_exception)
        expected_calls = [
            unittest.mock.call("BEGIN IMMEDIATE"),
            unittest.mock.call("ROLLBACK"),
        ]
        self.mock_connection.execute.assert_has_calls(expected_calls)
        self.assertEqual(tx_context.state, TransactionState.ROLLED_BACK)

    def test_begin_custom_isolation_level(self):
        """Given: 사용자 정의 격리 수준이 설정된 경우
        When: begin을 사용하면
        Then: 지정된 격리 수준으로 트랜잭션이 시작된다
        """
        # Given
        tx_context = TransactionContext(
            self.mock_connection, isolation_level=IsolationLevel.EXCLUSIVE
        )

        # When
        with tx_context.begin():
            pass

        # Then
        self.mock_connection.execute.assert_any_call("BEGIN EXCLUSIVE")

    def test_begin_auto_commit_false(self):
        """Given: auto_commit이 False로 설정된 경우
        When: begin을 사용하면
        Then: 자동 커밋이 수행되지 않는다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection, auto_commit=False)

        # When
        with tx_context.begin():
            pass

        # Then
        # 커밋이 호출되지 않았는지 확인
        commit_calls = [
            call for call in self.mock_connection.execute.call_args_list if call[0][0] == "COMMIT"
        ]
        self.assertEqual(len(commit_calls), 0)
        self.assertEqual(tx_context.state, TransactionState.ACTIVE)

    def test_commit_success(self):
        """Given: 활성 트랜잭션이 있을 때
        When: commit을 호출하면
        Then: 트랜잭션이 커밋되고 True가 반환된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE

        # When
        result = tx_context.commit()

        # Then
        self.assertTrue(result)
        self.mock_connection.execute.assert_called_with("COMMIT")
        self.assertEqual(tx_context.state, TransactionState.COMMITTED)

    def test_commit_nested_transaction(self):
        """Given: 중첩 트랜잭션이 있을 때
        When: commit을 호출하면
        Then: savepoint가 해제된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE
        tx_context._is_nested = True
        tx_context._savepoint_name = "sp_test"

        # When
        result = tx_context.commit()

        # Then
        self.assertTrue(result)
        self.mock_connection.execute.assert_called_with("RELEASE SAVEPOINT sp_test")
        self.assertEqual(tx_context.state, TransactionState.COMMITTED)

    def test_commit_inactive_transaction(self):
        """Given: 비활성 트랜잭션이 있을 때
        When: commit을 호출하면
        Then: False가 반환되고 상태가 변경되지 않는다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.COMMITTED

        # When
        result = tx_context.commit()

        # Then
        self.assertFalse(result)
        self.assertEqual(tx_context.state, TransactionState.COMMITTED)
        self.mock_connection.execute.assert_not_called()

    def test_commit_with_exception(self):
        """Given: 커밋 중 예외가 발생할 때
        When: commit을 호출하면
        Then: False가 반환되고 상태가 FAILED로 변경된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE
        self.mock_connection.execute.side_effect = sqlite3.OperationalError("Test error")

        # When
        result = tx_context.commit()

        # Then
        self.assertFalse(result)
        self.assertEqual(tx_context.state, TransactionState.FAILED)

    def test_rollback_success(self):
        """Given: 활성 트랜잭션이 있을 때
        When: rollback을 호출하면
        Then: 트랜잭션이 롤백되고 True가 반환된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE

        # When
        result = tx_context.rollback()

        # Then
        self.assertTrue(result)
        self.mock_connection.execute.assert_called_with("ROLLBACK")
        self.assertEqual(tx_context.state, TransactionState.ROLLED_BACK)

    def test_rollback_nested_transaction(self):
        """Given: 중첩 트랜잭션이 있을 때
        When: rollback을 호출하면
        Then: savepoint로 롤백된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE
        tx_context._is_nested = True
        tx_context._savepoint_name = "sp_test"

        # When
        result = tx_context.rollback()

        # Then
        self.assertTrue(result)
        self.mock_connection.execute.assert_called_with("ROLLBACK TO SAVEPOINT sp_test")
        self.assertEqual(tx_context.state, TransactionState.ROLLED_BACK)

    def test_rollback_failed_transaction(self):
        """Given: 실패한 트랜잭션이 있을 때
        When: rollback을 호출하면
        Then: 롤백이 수행되고 True가 반환된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.FAILED

        # When
        result = tx_context.rollback()

        # Then
        self.assertTrue(result)
        self.mock_connection.execute.assert_called_with("ROLLBACK")
        self.assertEqual(tx_context.state, TransactionState.ROLLED_BACK)

    def test_rollback_inactive_transaction(self):
        """Given: 비활성 트랜잭션이 있을 때
        When: rollback을 호출하면
        Then: False가 반환되고 상태가 변경되지 않는다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.COMMITTED

        # When
        result = tx_context.rollback()

        # Then
        self.assertFalse(result)
        self.assertEqual(tx_context.state, TransactionState.COMMITTED)
        self.mock_connection.execute.assert_not_called()

    def test_rollback_with_exception(self):
        """Given: 롤백 중 예외가 발생할 때
        When: rollback을 호출하면
        Then: False가 반환되고 상태가 FAILED로 변경된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE
        self.mock_connection.execute.side_effect = sqlite3.OperationalError("Test error")

        # When
        result = tx_context.rollback()

        # Then
        self.assertFalse(result)
        self.assertEqual(tx_context.state, TransactionState.FAILED)

    def test_execute_success(self):
        """Given: 활성 트랜잭션이 있을 때
        When: execute를 호출하면
        Then: SQL이 실행되고 커서가 반환된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE
        mock_cursor = Mock()
        self.mock_connection.execute.return_value = mock_cursor

        # When
        result = tx_context.execute("SELECT 1")

        # Then
        self.assertEqual(result, mock_cursor)
        self.mock_connection.execute.assert_called_with("SELECT 1")

    def test_execute_with_parameters(self):
        """Given: 활성 트랜잭션과 매개변수가 있을 때
        When: execute를 호출하면
        Then: 매개변수와 함께 SQL이 실행된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE
        mock_cursor = Mock()
        self.mock_connection.execute.return_value = mock_cursor
        parameters = {"id": 1}

        # When
        result = tx_context.execute("SELECT * FROM table WHERE id = :id", parameters)

        # Then
        self.assertEqual(result, mock_cursor)
        self.mock_connection.execute.assert_called_with(
            "SELECT * FROM table WHERE id = :id", parameters
        )

    def test_execute_inactive_transaction(self):
        """Given: 비활성 트랜잭션이 있을 때
        When: execute를 호출하면
        Then: RuntimeError가 발생한다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.COMMITTED

        # When & Then
        with self.assertRaises(RuntimeError) as context:
            tx_context.execute("SELECT 1")

        self.assertIn("비활성 트랜잭션에서 SQL 실행 시도", str(context.exception))

    def test_execute_with_exception(self):
        """Given: SQL 실행 중 예외가 발생할 때
        When: execute를 호출하면
        Then: 예외가 재발생한다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE
        self.mock_connection.execute.side_effect = sqlite3.OperationalError("Test error")

        # When & Then
        with self.assertRaises(sqlite3.OperationalError):
            tx_context.execute("SELECT 1")

    def test_executemany_success(self):
        """Given: 활성 트랜잭션이 있을 때
        When: executemany를 호출하면
        Then: 배치 SQL이 실행되고 커서가 반환된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE
        mock_cursor = Mock()
        self.mock_connection.executemany.return_value = mock_cursor
        parameters_list = [{"id": 1}, {"id": 2}]

        # When
        result = tx_context.executemany("INSERT INTO table VALUES (:id)", parameters_list)

        # Then
        self.assertEqual(result, mock_cursor)
        self.mock_connection.executemany.assert_called_with(
            "INSERT INTO table VALUES (:id)", parameters_list
        )

    def test_executemany_inactive_transaction(self):
        """Given: 비활성 트랜잭션이 있을 때
        When: executemany를 호출하면
        Then: RuntimeError가 발생한다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.COMMITTED

        # When & Then
        with self.assertRaises(RuntimeError) as context:
            tx_context.executemany("INSERT INTO table VALUES (:id)", [])

        self.assertIn("비활성 트랜잭션에서 SQL 실행 시도", str(context.exception))

    def test_executemany_with_exception(self):
        """Given: 배치 SQL 실행 중 예외가 발생할 때
        When: executemany를 호출하면
        Then: 예외가 재발생한다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)
        tx_context.state = TransactionState.ACTIVE
        self.mock_connection.executemany.side_effect = sqlite3.OperationalError("Test error")

        # When & Then
        with self.assertRaises(sqlite3.OperationalError):
            tx_context.executemany("INSERT INTO table VALUES (:id)", [])

    def test_is_active_property(self):
        """Given: TransactionContext가 있을 때
        When: is_active 속성을 확인하면
        Then: 올바른 활성 상태가 반환된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)

        # When & Then
        tx_context.state = TransactionState.ACTIVE
        self.assertTrue(tx_context.is_active)

        tx_context.state = TransactionState.COMMITTED
        self.assertFalse(tx_context.is_active)

        tx_context.state = TransactionState.ROLLED_BACK
        self.assertFalse(tx_context.is_active)

        tx_context.state = TransactionState.FAILED
        self.assertFalse(tx_context.is_active)

    def test_is_nested_property(self):
        """Given: TransactionContext가 있을 때
        When: is_nested 속성을 확인하면
        Then: 올바른 중첩 상태가 반환된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)

        # When & Then
        self.assertFalse(tx_context.is_nested)

        tx_context._is_nested = True
        self.assertTrue(tx_context.is_nested)

    def test_str_representation(self):
        """Given: TransactionContext가 있을 때
        When: 문자열 표현을 요청하면
        Then: 유용한 정보가 포함된 문자열이 반환된다
        """
        # Given
        tx_context = TransactionContext(self.mock_connection)

        # When
        str_repr = str(tx_context)

        # Then
        self.assertIn("TransactionContext", str_repr)
        self.assertIn(tx_context.transaction_id[:8], str_repr)
        self.assertIn("active", str_repr)
        self.assertIn("nested=False", str_repr)


class TestTransactionScope(unittest.TestCase):
    """transaction_scope 함수 테스트."""

    def setUp(self):
        """테스트 설정."""
        self.mock_connection = Mock(spec=sqlite3.Connection)
        self.mock_connection.in_transaction = False

    def test_transaction_scope_success(self):
        """Given: 정상적인 연결이 있을 때
        When: transaction_scope를 사용하면
        Then: 트랜잭션이 생성되고 커밋된다
        """
        # Given & When
        with transaction_scope(self.mock_connection) as tx_context:
            # Then - 트랜잭션 내부
            self.assertIsInstance(tx_context, TransactionContext)
            self.assertEqual(tx_context.connection, self.mock_connection)
            self.assertEqual(tx_context.isolation_level, IsolationLevel.IMMEDIATE)
            self.assertTrue(tx_context.auto_commit)

        # Then - 트랜잭션 완료 후
        expected_calls = [
            unittest.mock.call("BEGIN IMMEDIATE"),
            unittest.mock.call("COMMIT"),
        ]
        self.mock_connection.execute.assert_has_calls(expected_calls)

    def test_transaction_scope_custom_isolation(self):
        """Given: 사용자 정의 격리 수준이 있을 때
        When: transaction_scope를 사용하면
        Then: 지정된 격리 수준으로 트랜잭션이 시작된다
        """
        # Given & When
        with transaction_scope(
            self.mock_connection, isolation_level=IsolationLevel.EXCLUSIVE, auto_commit=False
        ) as tx_context:
            self.assertEqual(tx_context.isolation_level, IsolationLevel.EXCLUSIVE)
            self.assertFalse(tx_context.auto_commit)

        # Then
        self.mock_connection.execute.assert_called_with("BEGIN EXCLUSIVE")

    def test_transaction_scope_with_exception(self):
        """Given: 트랜잭션 중 예외가 발생할 때
        When: transaction_scope를 사용하면
        Then: 롤백이 수행되고 예외가 재발생한다
        """
        # Given
        test_exception = ValueError("Test exception")

        # When & Then
        with self.assertRaises(ValueError) as context:
            with transaction_scope(self.mock_connection):
                raise test_exception

        self.assertEqual(context.exception, test_exception)
        expected_calls = [
            unittest.mock.call("BEGIN IMMEDIATE"),
            unittest.mock.call("ROLLBACK"),
        ]
        self.mock_connection.execute.assert_has_calls(expected_calls)


if __name__ == "__main__":
    unittest.main()
