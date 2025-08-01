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
            VectorStore()  # pylint: disable=abstract-class-instantiated

    def test_abstract_methods_exist(self):
        """추상 메서드들이 정의되어 있는지 테스트."""
        abstract_methods = VectorStore.__abstractmethods__

        expected_methods = {
            "add_documents",
            "similarity_search",
            "similarity_search_with_score",
            "similarity_search_by_vector",
            "delete",
            "from_documents",
            "from_texts",
            "as_retriever",
        }

        # 모든 필수 메서드가 추상 메서드로 정의되어 있는지 확인
        for method in expected_methods:
            self.assertIn(method, abstract_methods, f"'{method}' should be abstract")


if __name__ == "__main__":
    unittest.main()
