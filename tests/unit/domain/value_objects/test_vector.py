"""
Vector 값 객체 단위 테스트.
"""

import math
import unittest

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
        expected = math.sqrt((4 - 1) ** 2 + (5 - 1) ** 2)  # sqrt(9 + 16) = 5
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

    # === 경계값 및 스트레스 테스트 추가 ===

    def test_vector_extreme_large_values(self):
        """극대값 벡터 처리 테스트."""
        # Given
        large_value = 1e15
        vector = Vector([large_value, large_value])

        # When & Then
        magnitude = vector.magnitude()
        self.assertIsInstance(magnitude, float)
        self.assertFalse(math.isinf(magnitude))
        self.assertFalse(math.isnan(magnitude))

        # 정규화도 안정적으로 동작해야 함
        normalized = vector.normalize()
        self.assertAlmostEqual(normalized.magnitude(), 1.0, places=5)

    def test_vector_extreme_small_values(self):
        """극소값 벡터 처리 테스트."""
        # Given
        small_value = 1e-15
        vector = Vector([small_value, small_value])

        # When & Then
        magnitude = vector.magnitude()
        self.assertGreater(magnitude, 0)
        self.assertFalse(math.isnan(magnitude))

        # 정규화 시 수치 안정성 확인
        normalized = vector.normalize()
        self.assertFalse(any(math.isnan(v) for v in normalized.values))

    def test_vector_nan_values_handling(self):
        """NaN 값 처리 테스트."""
        # When & Then - NaN 값으로 벡터 생성 시 예외 발생
        with self.assertRaises(ValueError) as context:
            Vector([1.0, float("nan"), 3.0])
        self.assertIn("Vector values must be numeric", str(context.exception))

    def test_vector_infinity_values_handling(self):
        """무한대 값 처리 테스트."""
        # When & Then - 무한대 값으로 벡터 생성 시 예외 발생
        with self.assertRaises(ValueError) as context:
            Vector([1.0, float("inf"), 3.0])
        self.assertIn("Vector values must be numeric", str(context.exception))

        with self.assertRaises(ValueError) as context:
            Vector([1.0, float("-inf"), 3.0])
        self.assertIn("Vector values must be numeric", str(context.exception))

    def test_vector_very_high_dimensions(self):
        """고차원 벡터 처리 테스트."""
        # Given - 1000차원 벡터
        high_dim = 1000
        values = [0.001] * high_dim  # 작은 값으로 설정하여 메모리 효율성 확인
        vector = Vector(values)

        # When & Then
        self.assertEqual(vector.dimension, high_dim)

        # 크기 계산 성능 확인
        import time

        start_time = time.time()
        magnitude = vector.magnitude()
        end_time = time.time()

        self.assertIsInstance(magnitude, float)
        self.assertLess(end_time - start_time, 1.0)  # 1초 이내 완료

        # 동일 차원 벡터와의 연산 확인
        vector2 = Vector([0.002] * high_dim)
        similarity = vector.cosine_similarity(vector2)
        self.assertAlmostEqual(similarity, 1.0, places=5)  # 같은 방향

    def test_vector_numerical_stability_edge_cases(self):
        """수치 안정성 경계 케이스 테스트."""
        # Given - 매우 큰 값과 매우 작은 값이 혼재
        mixed_values = [1e10, 1e-10, 1.0, -1e10, -1e-10]  # 더 현실적인 범위로 조정
        vector1 = Vector(mixed_values)
        vector2 = Vector([v * 2 for v in mixed_values])

        # When & Then - 코사인 유사도가 안정적으로 계산되어야 함
        similarity = vector1.cosine_similarity(vector2)
        self.assertAlmostEqual(similarity, 1.0, places=5)  # 같은 방향이므로 1.0

        # 유클리드 거리도 안정적이어야 함
        distance = vector1.euclidean_distance(vector2)
        self.assertIsInstance(distance, float)
        self.assertFalse(math.isnan(distance))
        self.assertFalse(math.isinf(distance))

    def test_vector_zero_magnitude_edge_cases(self):
        """영벡터 크기 관련 경계 케이스 테스트."""
        # Given - 다양한 영벡터 상황
        test_cases = [
            ([0.0, 0.0]),
            ([0.0, 0.0, 0.0, 0.0]),
            ([-0.0, 0.0]),  # 음수 영
            ([1e-300, 1e-300]),  # 사실상 영벡터
        ]

        for values in test_cases:
            with self.subTest(values=values):
                vector = Vector(values)

                # 크기는 0이거나 0에 매우 가까워야 함
                magnitude = vector.magnitude()
                self.assertLessEqual(magnitude, 1e-100)

                # 다른 벡터와의 코사인 유사도는 0이어야 함
                other_vector = Vector([1.0] * len(values))
                similarity = vector.cosine_similarity(other_vector)
                self.assertEqual(similarity, 0.0)

    def test_vector_performance_with_large_data(self):
        """대용량 데이터 성능 테스트."""
        import time

        # Given - 10,000차원 벡터 (실용적인 임베딩 크기)
        large_dim = 1000  # 테스트 환경에서는 더 작은 크기로
        values1 = [i * 0.0001 for i in range(large_dim)]
        values2 = [(i + 1) * 0.0001 for i in range(large_dim)]

        # When - 벡터 생성 성능
        start_time = time.time()
        vector1 = Vector(values1)
        vector2 = Vector(values2)
        creation_time = time.time() - start_time

        # Then - 생성 시간 검증
        self.assertLess(creation_time, 0.1)  # 100ms 이내

        # When - 연산 성능
        start_time = time.time()
        magnitude1 = vector1.magnitude()
        similarity = vector1.cosine_similarity(vector2)
        distance = vector1.euclidean_distance(vector2)
        operation_time = time.time() - start_time

        # Then - 연산 시간 검증
        self.assertLess(operation_time, 0.5)  # 500ms 이내

        # 결과 정확성 검증
        self.assertIsInstance(magnitude1, float)
        self.assertIsInstance(similarity, float)
        self.assertIsInstance(distance, float)
        self.assertFalse(any(math.isnan(v) for v in [magnitude1, similarity, distance]))

    def test_vector_concurrent_operations_safety(self):
        """동시 연산 안전성 테스트."""
        import threading
        import time

        # Given
        vector1 = Vector([1.0, 2.0, 3.0])
        vector2 = Vector([4.0, 5.0, 6.0])
        results = []
        errors = []

        def compute_similarity():
            try:
                for _ in range(10):  # 빠른 테스트를 위해 반복 횟수 감소
                    similarity = vector1.cosine_similarity(vector2)
                    results.append(similarity)
                    time.sleep(0.001)  # 작은 지연
            except Exception as e:
                errors.append(e)

        # When - 동시 연산 실행
        threads = []
        for _ in range(3):  # 스레드 수 감소
            thread = threading.Thread(target=compute_similarity)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Then - 오류 없이 일관된 결과
        self.assertEqual(len(errors), 0)  # 예외 발생 없음
        self.assertGreater(len(results), 0)  # 결과 생성됨

        # 모든 결과가 동일해야 함 (불변 객체이므로)
        expected_similarity = vector1.cosine_similarity(vector2)
        for result in results:
            self.assertAlmostEqual(result, expected_similarity, places=10)

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
