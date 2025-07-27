"""
Database 포트 인터페이스 단위 테스트.
"""

import unittest
from abc import ABC

from src.ports.database import Database


class TestDatabaseInterface(unittest.TestCase):
    """Database 포트 인터페이스 테스트 케이스."""

    def test_database_is_abstract(self):
        """Database가 추상 클래스인지 테스트."""
        self.assertTrue(issubclass(Database, ABC))
        
        # 직접 인스턴스화할 수 없어야 함
        with self.assertRaises(TypeError):
            Database()
            
    def test_abstract_methods_exist(self):
        """추상 메서드들이 정의되어 있는지 테스트."""
        abstract_methods = Database.__abstractmethods__
        
        expected_methods = {
            'connect', 'disconnect', 'is_connected', 'ping',
            'transaction', 'begin_transaction', 'commit_transaction', 'rollback_transaction',
            'execute_query', 'execute_command', 'execute_batch',
            'create_table', 'drop_table', 'table_exists', 'get_table_schema',
            'create_index', 'drop_index',
            'vacuum', 'analyze', 'get_database_info', 'get_table_info',
            'health_check', 'get_connection_info', 'get_performance_stats'
        }
        
        # 모든 필수 메서드가 추상 메서드로 정의되어 있는지 확인
        for method in expected_methods:
            self.assertIn(method, abstract_methods, f"'{method}' should be abstract")


if __name__ == "__main__":
    unittest.main()