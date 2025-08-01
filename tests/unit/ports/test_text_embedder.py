"""
최적화된 TextEmbedder 포트 인터페이스 단위 테스트.
"""

import unittest
from abc import ABC

from src.ports.text_embedder import TextEmbedder


class TestTextEmbedderInterface(unittest.TestCase):
    """최적화된 TextEmbedder 포트 인터페이스 테스트 케이스."""

    def test_text_embedder_is_abstract(self):
        """TextEmbedder가 추상 클래스인지 테스트."""
        self.assertTrue(issubclass(TextEmbedder, ABC))

        # 직접 인스턴스화할 수 없어야 함
        with self.assertRaises(TypeError):
            TextEmbedder()  # pylint: disable=abstract-class-instantiated

    def test_core_abstract_methods_exist(self):
        """핵심 추상 메서드들이 정의되어 있는지 테스트."""
        abstract_methods = TextEmbedder.__abstractmethods__

        # 최적화된 TextEmbedder 인터페이스는 4개 핵심 메서드만 포함
        expected_core_methods = {
            "embed_text",
            "embed_texts",
            "get_embedding_dimension",
            "is_available",
        }

        # 모든 핵심 메서드가 추상 메서드로 정의되어 있는지 확인
        for method in expected_core_methods:
            self.assertIn(
                method,
                abstract_methods,
                f"'{method}' should be abstract in core TextEmbedder interface",
            )

        # 핵심 인터페이스에 불필요한 메서드가 없는지 확인
        self.assertEqual(
            abstract_methods,
            expected_core_methods,
            "TextEmbedder interface should only contain core methods",
        )

    def test_interface_simplification(self):
        """인터페이스 단순화가 올바르게 되었는지 테스트."""
        # 제거된 메서드들이 더 이상 추상 메서드가 아닌지 확인
        abstract_methods = TextEmbedder.__abstractmethods__

        removed_methods = {
            "embed_with_metadata",
            "batch_embed_with_metadata",
            "get_model_name",
            "get_max_token_length",
            "truncate_text",
            "compute_similarity",
            "find_most_similar",
            "preprocess_text",
            "get_embedding_statistics",
            "validate_embedding",
            "warm_up",
        }

        for method in removed_methods:
            self.assertNotIn(
                method, abstract_methods, f"'{method}' should have been removed from the interface"
            )


if __name__ == "__main__":
    unittest.main()
