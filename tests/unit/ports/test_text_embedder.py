"""
TextEmbedder 포트 인터페이스 단위 테스트.
"""

import unittest
from abc import ABC

from src.ports.text_embedder import EmbeddingConfig, EmbeddingResult, TextEmbedder


class TestTextEmbedderInterface(unittest.TestCase):
    """TextEmbedder 포트 인터페이스 테스트 케이스."""

    def test_text_embedder_is_abstract(self):
        """TextEmbedder가 추상 클래스인지 테스트."""
        self.assertTrue(issubclass(TextEmbedder, ABC))

        # 직접 인스턴스화할 수 없어야 함
        with self.assertRaises(TypeError):
            TextEmbedder()

    def test_abstract_methods_exist(self):
        """추상 메서드들이 정의되어 있는지 테스트."""
        abstract_methods = TextEmbedder.__abstractmethods__

        expected_methods = {
            "embed_text",
            "embed_texts",
            "embed_with_metadata",
            "batch_embed_with_metadata",
            "get_embedding_dimension",
            "get_model_name",
            "get_max_token_length",
            "is_available",
            "truncate_text",
            "compute_similarity",
            "find_most_similar",
            "preprocess_text",
            "get_embedding_statistics",
            "validate_embedding",
            "warm_up",
        }

        # 모든 필수 메서드가 추상 메서드로 정의되어 있는지 확인
        for method in expected_methods:
            self.assertIn(method, abstract_methods, f"'{method}' should be abstract")


class TestEmbeddingConfig(unittest.TestCase):
    """EmbeddingConfig 데이터 클래스 테스트."""

    def test_embedding_config_creation(self):
        """EmbeddingConfig 생성 테스트."""
        config = EmbeddingConfig(model_name="test-model", dimension=384)

        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.dimension, 384)
        self.assertEqual(config.batch_size, 32)  # 기본값
        self.assertTrue(config.normalize)  # 기본값
        self.assertEqual(config.metadata, {})  # __post_init__에서 설정

    def test_embedding_config_with_custom_values(self):
        """사용자 정의 값으로 EmbeddingConfig 생성 테스트."""
        metadata = {"provider": "test"}
        config = EmbeddingConfig(
            model_name="custom-model",
            dimension=768,
            max_tokens=512,
            batch_size=16,
            normalize=False,
            metadata=metadata,
        )

        self.assertEqual(config.model_name, "custom-model")
        self.assertEqual(config.dimension, 768)
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.batch_size, 16)
        self.assertFalse(config.normalize)
        self.assertEqual(config.metadata, metadata)


class TestEmbeddingResult(unittest.TestCase):
    """EmbeddingResult 데이터 클래스 테스트."""

    def test_embedding_result_creation(self):
        """EmbeddingResult 생성 테스트."""
        from src.domain.value_objects.vector import Vector

        vector = Vector([0.1, 0.2, 0.3])
        result = EmbeddingResult(
            text="테스트 텍스트", vector=vector, model_name="test-model"
        )

        self.assertEqual(result.text, "테스트 텍스트")
        self.assertEqual(result.vector, vector)
        self.assertEqual(result.model_name, "test-model")
        self.assertIsNone(result.token_count)
        self.assertIsNone(result.processing_time_ms)

    def test_embedding_result_with_metadata(self):
        """메타데이터와 함께 EmbeddingResult 생성 테스트."""
        from src.domain.value_objects.vector import Vector

        vector = Vector([0.1, 0.2, 0.3])
        result = EmbeddingResult(
            text="테스트 텍스트",
            vector=vector,
            model_name="test-model",
            token_count=10,
            processing_time_ms=100.5,
        )

        self.assertEqual(result.token_count, 10)
        self.assertEqual(result.processing_time_ms, 100.5)


if __name__ == "__main__":
    unittest.main()
