"""
VectorStore 포트 인터페이스 단위 테스트.
"""

import unittest
from abc import ABC

from src.ports.vector_store import VectorStore


class TestVectorStoreInterface(unittest.TestCase):
    """VectorStore 포트 인터페이스 테스트 케이스."""

    def test_vector_store_is_abstract(self):
        """VectorStore가 추상 클래스인지 테스트."""
        self.assertTrue(issubclass(VectorStore, ABC))
        
        # 직접 인스턴스화할 수 없어야 함
        with self.assertRaises(TypeError):
            VectorStore()
            
    def test_abstract_methods_exist(self):
        """추상 메서드들이 정의되어 있는지 테스트."""
        abstract_methods = VectorStore.__abstractmethods__
        
        expected_methods = {
            'initialize_store', 'connect', 'disconnect', 'is_connected',
            'add_vector', 'add_vectors', 'get_vector', 'get_vectors',
            'update_vector', 'delete_vector', 'delete_vectors', 'vector_exists',
            'search_similar', 'search_similar_with_vectors', 'search_by_ids', 'batch_search',
            'get_metadata', 'update_metadata', 'search_by_metadata',
            'get_store_info', 'get_vector_count', 'get_dimension',
            'optimize_store', 'rebuild_index', 'clear_store',
            'create_snapshot', 'restore_snapshot',
            'health_check', 'get_performance_stats'
        }
        
        # 모든 필수 메서드가 추상 메서드로 정의되어 있는지 확인
        for method in expected_methods:
            self.assertIn(method, abstract_methods, f"'{method}' should be abstract")


if __name__ == "__main__":
    unittest.main()