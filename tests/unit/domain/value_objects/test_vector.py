"""
Vector 값 객체 단위 테스트.
"""

import unittest
import math

from src.domain.value_objects.vector import Vector


class TestVector(unittest.TestCase):
    """Vector 값 객체 테스트."""

    def test_create_vector_success(self):
        """Vector 생성 성공 테스트."""
        # When
        values = [1.0, 2.0, 3.0]
        vector = Vector(values)

        # Then
        self.assertEqual(vector.values, values)
        self.assertEqual(vector.dimension, 3)

    def test_create_vector_with_empty_list_error(self):
        """빈 리스트로 Vector 생성 시 오류 테스트."""
        # When & Then
        with self.assertRaises(ValueError) as context:
            Vector([])
        self.assertIn("Vector cannot be empty", str(context.exception))

    def test_create_vector_with_non_numeric_values_error(self):
        """숫자가 아닌 값으로 Vector 생성 시 오류 테스트."""
        # When & Then
        with self.assertRaises(ValueError) as context:
            Vector([1.0, "invalid", 3.0])
        self.assertIn("Vector values must be numeric", str(context.exception))

    def test_magnitude_calculation(self):
        """벡터 크기 계산 테스트."""
        vector = Vector([3.0, 4.0])

        # When
        magnitude = vector.magnitude()

        # Then
        self.assertEqual(magnitude, 5.0)  # sqrt(3^2 + 4^2) = 5

    def test_magnitude_zero_vector(self):
        """영벡터의 크기 계산 테스트."""
        vector = Vector([0.0, 0.0, 0.0])

        # When
        magnitude = vector.magnitude()

        # Then
        self.assertEqual(magnitude, 0.0)

    def test_cosine_similarity_success(self):
        """코사인 유사도 계산 성공 테스트."""
        vector1 = Vector([1.0, 0.0])
        vector2 = Vector([0.0, 1.0])

        # When
        similarity = vector1.cosine_similarity(vector2)

        # Then
        self.assertEqual(similarity, 0.0)  # 직교 벡터

    def test_cosine_similarity_identical_vectors(self):
        """동일한 벡터의 코사인 유사도 테스트."""
        vector1 = Vector([1.0, 2.0, 3.0])
        vector2 = Vector([1.0, 2.0, 3.0])

        # When
        similarity = vector1.cosine_similarity(vector2)

        # Then
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_cosine_similarity_different_dimensions_error(self):
        """차원이 다른 벡터의 코사인 유사도 계산 시 오류 테스트."""
        vector1 = Vector([1.0, 2.0])
        vector2 = Vector([1.0, 2.0, 3.0])

        # When & Then
        with self.assertRaises(ValueError) as context:
            vector1.cosine_similarity(vector2)
        self.assertIn("Vectors must have the same dimension", str(context.exception))

    def test_cosine_similarity_zero_magnitude_vector(self):
        """영벡터의 코사인 유사도 계산 테스트."""
        vector1 = Vector([0.0, 0.0])
        vector2 = Vector([1.0, 2.0])

        # When
        similarity = vector1.cosine_similarity(vector2)

        # Then
        self.assertEqual(similarity, 0.0)

    def test_euclidean_distance_success(self):
        """유클리드 거리 계산 성공 테스트."""
        vector1 = Vector([1.0, 1.0])
        vector2 = Vector([4.0, 5.0])

        # When
        distance = vector1.euclidean_distance(vector2)

        # Then
        expected = math.sqrt((4-1)**2 + (5-1)**2)  # sqrt(9 + 16) = 5
        self.assertEqual(distance, expected)

    def test_euclidean_distance_identical_vectors(self):
        """동일한 벡터의 유클리드 거리 테스트."""
        vector1 = Vector([1.0, 2.0, 3.0])
        vector2 = Vector([1.0, 2.0, 3.0])

        # When
        distance = vector1.euclidean_distance(vector2)

        # Then
        self.assertEqual(distance, 0.0)

    def test_euclidean_distance_different_dimensions_error(self):
        """차원이 다른 벡터의 유클리드 거리 계산 시 오류 테스트."""
        vector1 = Vector([1.0, 2.0])
        vector2 = Vector([1.0, 2.0, 3.0])

        # When & Then
        with self.assertRaises(ValueError) as context:
            vector1.euclidean_distance(vector2)
        self.assertIn("Vectors must have the same dimension", str(context.exception))

    def test_normalize_success(self):
        """벡터 정규화 성공 테스트."""
        vector = Vector([3.0, 4.0])

        # When
        normalized = vector.normalize()

        # Then
        self.assertAlmostEqual(normalized.magnitude(), 1.0, places=5)
        self.assertAlmostEqual(normalized.values[0], 0.6, places=5)  # 3/5
        self.assertAlmostEqual(normalized.values[1], 0.8, places=5)  # 4/5

    def test_normalize_zero_vector(self):
        """영벡터 정규화 테스트."""
        vector = Vector([0.0, 0.0])

        # When
        normalized = vector.normalize()

        # Then
        self.assertEqual(normalized.values, vector.values)  # 영벡터는 그대로 반환

    def test_immutability(self):
        """Vector 불변성 테스트."""
        vector = Vector([1.0, 2.0, 3.0])

        # Then
        # values 값을 수정할 수 없어야 함
        with self.assertRaises(AttributeError):
            vector.values = [4.0, 5.0, 6.0]

    def test_str_representation(self):
        """문자열 표현 테스트."""
        vector = Vector([1.0, 2.0, 3.0])

        # When
        str_repr = str(vector)

        # Then
        self.assertIn("Vector", str_repr)
        self.assertIn("3d", str_repr)

    def test_repr_representation(self):
        """repr 표현 테스트."""
        values = [1.0, 2.0, 3.0]
        vector = Vector(values)

        # When
        repr_str = repr(vector)

        # Then
        self.assertIn("Vector", repr_str)
        self.assertIn(str(values), repr_str)


if __name__ == "__main__":
    unittest.main()