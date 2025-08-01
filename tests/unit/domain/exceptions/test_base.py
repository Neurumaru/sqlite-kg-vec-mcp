"""
도메인 예외 기본 클래스 단위 테스트.
"""

import unittest

from src.domain.exceptions.base import DomainException


class TestDomainException(unittest.TestCase):
    """DomainException 기본 클래스 테스트."""

    def test_create_domain_exception_with_message_only(self):
        """메시지만으로 도메인 예외 생성 테스트."""
        # When
        message = "테스트 도메인 예외입니다"
        exception = DomainException(message)

        # Then
        self.assertEqual(exception.message, message)
        self.assertIsNone(exception.error_code)
        self.assertEqual(str(exception), message)

    def test_create_domain_exception_with_message_and_error_code(self):
        """메시지와 에러 코드로 도메인 예외 생성 테스트."""
        # When
        message = "테스트 도메인 예외입니다"
        error_code = "TEST_ERROR"
        exception = DomainException(message, error_code)

        # Then
        self.assertEqual(exception.message, message)
        self.assertEqual(exception.error_code, error_code)
        self.assertEqual(str(exception), f"[{error_code}] {message}")

    def test_domain_exception_inherits_from_exception(self):
        """DomainException이 Exception을 상속하는지 테스트."""
        # When
        exception = DomainException("테스트 메시지")

        # Then
        self.assertIsInstance(exception, Exception)
        self.assertIsInstance(exception, DomainException)

    def test_domain_exception_str_representation_with_error_code(self):
        """에러 코드가 있는 경우 문자열 표현 테스트."""
        # When
        message = "사용자를 찾을 수 없습니다"
        error_code = "USER_NOT_FOUND"
        exception = DomainException(message, error_code)

        # Then
        expected_str = f"[{error_code}] {message}"
        self.assertEqual(str(exception), expected_str)

    def test_domain_exception_str_representation_without_error_code(self):
        """에러 코드가 없는 경우 문자열 표현 테스트."""
        # When
        message = "일반적인 도메인 오류"
        exception = DomainException(message)

        # Then
        self.assertEqual(str(exception), message)

    def test_domain_exception_can_be_raised(self):
        """DomainException이 예외로 발생될 수 있는지 테스트."""
        # When & Then
        with self.assertRaises(DomainException) as context:
            raise DomainException("테스트 예외")

        self.assertEqual(context.exception.message, "테스트 예외")

    def test_domain_exception_with_empty_message(self):
        """빈 메시지로 도메인 예외 생성 테스트."""
        # When
        message = ""
        exception = DomainException(message)

        # Then
        self.assertEqual(exception.message, "")
        self.assertEqual(str(exception), "")

    def test_domain_exception_with_empty_error_code(self):
        """빈 에러 코드로 도메인 예외 생성 테스트."""
        # When
        message = "테스트 메시지"
        error_code = ""
        exception = DomainException(message, error_code)

        # Then
        self.assertEqual(exception.error_code, "")
        # 빈 에러 코드는 조건문에서 falsy로 처리되어 에러 코드 없이 표시됨
        self.assertEqual(str(exception), message)


if __name__ == "__main__":
    unittest.main()
