"""
VectorStore 포트 인터페이스 단위 테스트.
"""

import unittest
from abc import ABC

from src.ports.vector_reader import VectorReader
from src.ports.vector_retriever import VectorRetriever
from src.ports.vector_store import VectorStore
from src.ports.vector_writer import VectorWriter


class TestVectorStoreInterface(unittest.TestCase):
    """VectorStore 포트 인터페이스 테스트 케이스."""

    def test_vector_store_is_abstract(self):
        """VectorStore가 추상 클래스인지 테스트."""
        self.assertTrue(issubclass(VectorStore, ABC))

        # 직접 인스턴스화할 수 없어야 함
        with self.assertRaises(TypeError):
            VectorStore()  # pylint: disable=abstract-class-instantiated

    def test_vector_store_inherits_from_three_interfaces(self):
        """VectorStore가 세 개의 인터페이스를 상속하는지 테스트."""
        self.assertTrue(issubclass(VectorStore, VectorWriter))
        self.assertTrue(issubclass(VectorStore, VectorReader))
        self.assertTrue(issubclass(VectorStore, VectorRetriever))

    def test_vector_writer_abstract_methods(self):
        """VectorWriter 추상 메서드들이 정의되어 있는지 테스트."""
        abstract_methods = VectorWriter.__abstractmethods__

        expected_methods = {
            "add_documents",
            "add_vectors",
            "delete",
            "update_document",
        }

        for method in expected_methods:
            self.assertIn(
                method, abstract_methods, f"'{method}' should be abstract in VectorWriter"
            )

    def test_vector_reader_abstract_methods(self):
        """VectorReader 추상 메서드들이 정의되어 있는지 테스트."""
        abstract_methods = VectorReader.__abstractmethods__

        expected_methods = {
            "get_document",
            "get_vector",
            "list_documents",
            "count_documents",
            "similarity_search",
            "similarity_search_by_vector",
        }

        for method in expected_methods:
            self.assertIn(
                method, abstract_methods, f"'{method}' should be abstract in VectorReader"
            )

    def test_vector_retriever_abstract_methods(self):
        """VectorRetriever 추상 메서드들이 정의되어 있는지 테스트."""
        abstract_methods = VectorRetriever.__abstractmethods__

        expected_methods = {
            "retrieve",
            "retrieve_with_filter",
            "retrieve_mmr",
            "get_relevant_documents",
        }

        for method in expected_methods:
            self.assertIn(
                method, abstract_methods, f"'{method}' should be abstract in VectorRetriever"
            )

    def test_individual_interfaces_are_abstract(self):
        """개별 인터페이스들이 추상 클래스인지 테스트."""
        # VectorWriter
        self.assertTrue(issubclass(VectorWriter, ABC))
        with self.assertRaises(TypeError):
            VectorWriter()  # pylint: disable=abstract-class-instantiated

        # VectorReader
        self.assertTrue(issubclass(VectorReader, ABC))
        with self.assertRaises(TypeError):
            VectorReader()  # pylint: disable=abstract-class-instantiated

        # VectorRetriever
        self.assertTrue(issubclass(VectorRetriever, ABC))
        with self.assertRaises(TypeError):
            VectorRetriever()  # pylint: disable=abstract-class-instantiated


if __name__ == "__main__":
    unittest.main()
