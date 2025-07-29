"""
최적화된 Database 포트 인터페이스 단위 테스트.
"""

import unittest
from abc import ABC

from src.ports.database import Database, DatabaseMaintenance


class TestDatabaseInterface(unittest.TestCase):
    """최적화된 Database 포트 인터페이스 테스트 케이스."""

    def test_database_is_abstract(self):
        """Database가 추상 클래스인지 테스트."""
        self.assertTrue(issubclass(Database, ABC))

        # 직접 인스턴스화할 수 없어야 함
        with self.assertRaises(TypeError):
            Database()

    def test_core_database_methods_exist(self):
        """핵심 Database 인터페이스의 추상 메서드들이 정의되어 있는지 테스트."""
        abstract_methods = Database.__abstractmethods__

        # 핵심 Database 인터페이스는 7개 메서드만 포함
        expected_core_methods = {
            "execute_query",
            "execute_command",
            "transaction",
            "connect",
            "is_connected",
            "table_exists",
            "get_table_schema",
        }

        # 모든 핵심 메서드가 추상 메서드로 정의되어 있는지 확인
        for method in expected_core_methods:
            self.assertIn(
                method,
                abstract_methods,
                f"'{method}' should be abstract in core Database interface",
            )

        # 핵심 인터페이스에 불필요한 메서드가 없는지 확인
        self.assertEqual(
            abstract_methods,
            expected_core_methods,
            "Database interface should only contain core methods",
        )

    def test_maintenance_interface_exists(self):
        """DatabaseMaintenance 인터페이스가 존재하는지 테스트."""
        self.assertTrue(issubclass(DatabaseMaintenance, ABC))

        maintenance_methods = DatabaseMaintenance.__abstractmethods__
        expected_maintenance_methods = {
            "vacuum",
            "analyze",
            "create_table",
            "drop_table",
            "create_index",
            "drop_index",
        }

        for method in expected_maintenance_methods:
            self.assertIn(
                method,
                maintenance_methods,
                f"'{method}' should be abstract in DatabaseMaintenance interface",
            )

    def test_interface_separation(self):
        """인터페이스 분리가 올바르게 되었는지 테스트."""
        # 각 인터페이스가 독립적이어야 함
        self.assertFalse(issubclass(Database, DatabaseMaintenance))

        # 각 인터페이스를 독립적으로 인스턴스화할 수 없어야 함
        with self.assertRaises(TypeError):
            DatabaseMaintenance()


if __name__ == "__main__":
    unittest.main()
