"""
문서 관련 예외 단위 테스트.
"""

import unittest

from src.domain.exceptions.document_exceptions import (
    DocumentAlreadyExistsException,
    DocumentNotFoundException,
    DocumentProcessingException,
    InvalidDocumentException,
)


class TestDocumentNotFoundException(unittest.TestCase):
    """DocumentNotFoundException 테스트."""

    def test_create_document_not_found_exception(self):
        """DocumentNotFoundException 생성 테스트."""
        # When
        document_id = "doc-123"
        exception = DocumentNotFoundException(document_id)

        # Then
        self.assertEqual(exception.document_id, document_id)
        self.assertEqual(exception.error_code, "DOCUMENT_NOT_FOUND")
        expected_message = f"Document with ID '{document_id}' not found"
        self.assertEqual(exception.message, expected_message)
        self.assertEqual(str(exception), f"[DOCUMENT_NOT_FOUND] {expected_message}")

    def test_document_not_found_exception_inheritance(self):
        """DocumentNotFoundException 상속 관계 테스트."""
        # When
        exception = DocumentNotFoundException("doc-test")

        # Then
        from src.domain.exceptions.base import DomainException
        self.assertIsInstance(exception, DomainException)
        self.assertIsInstance(exception, Exception)

    def test_document_not_found_exception_can_be_raised(self):
        """DocumentNotFoundException이 예외로 발생될 수 있는지 테스트."""
        # When & Then
        document_id = "missing-doc"
        with self.assertRaises(DocumentNotFoundException) as context:
            raise DocumentNotFoundException(document_id)

        self.assertEqual(context.exception.document_id, document_id)


class TestDocumentAlreadyExistsException(unittest.TestCase):
    """DocumentAlreadyExistsException 테스트."""

    def test_create_document_already_exists_exception(self):
        """DocumentAlreadyExistsException 생성 테스트."""
        # When
        document_id = "existing-doc-456"
        exception = DocumentAlreadyExistsException(document_id)

        # Then
        self.assertEqual(exception.document_id, document_id)
        self.assertEqual(exception.error_code, "DOCUMENT_ALREADY_EXISTS")
        expected_message = f"Document with ID '{document_id}' already exists"
        self.assertEqual(exception.message, expected_message)
        self.assertEqual(str(exception), f"[DOCUMENT_ALREADY_EXISTS] {expected_message}")

    def test_document_already_exists_exception_inheritance(self):
        """DocumentAlreadyExistsException 상속 관계 테스트."""
        # When
        exception = DocumentAlreadyExistsException("doc-test")

        # Then
        from src.domain.exceptions.base import DomainException
        self.assertIsInstance(exception, DomainException)
        self.assertIsInstance(exception, Exception)

    def test_document_already_exists_exception_can_be_raised(self):
        """DocumentAlreadyExistsException이 예외로 발생될 수 있는지 테스트."""
        # When & Then
        document_id = "duplicate-doc"
        with self.assertRaises(DocumentAlreadyExistsException) as context:
            raise DocumentAlreadyExistsException(document_id)

        self.assertEqual(context.exception.document_id, document_id)


class TestInvalidDocumentException(unittest.TestCase):
    """InvalidDocumentException 테스트."""

    def test_create_invalid_document_exception(self):
        """InvalidDocumentException 생성 테스트."""
        # When
        reason = "문서 내용이 비어있습니다"
        exception = InvalidDocumentException(reason)

        # Then
        self.assertEqual(exception.reason, reason)
        self.assertEqual(exception.error_code, "INVALID_DOCUMENT")
        expected_message = f"Invalid document: {reason}"
        self.assertEqual(exception.message, expected_message)
        self.assertEqual(str(exception), f"[INVALID_DOCUMENT] {expected_message}")

    def test_invalid_document_exception_inheritance(self):
        """InvalidDocumentException 상속 관계 테스트."""
        # When
        exception = InvalidDocumentException("테스트 이유")

        # Then
        from src.domain.exceptions.base import DomainException
        self.assertIsInstance(exception, DomainException)
        self.assertIsInstance(exception, Exception)

    def test_invalid_document_exception_can_be_raised(self):
        """InvalidDocumentException이 예외로 발생될 수 있는지 테스트."""
        # When & Then
        reason = "잘못된 파일 형식"
        with self.assertRaises(InvalidDocumentException) as context:
            raise InvalidDocumentException(reason)

        self.assertEqual(context.exception.reason, reason)

    def test_invalid_document_exception_with_empty_reason(self):
        """빈 이유로 InvalidDocumentException 생성 테스트."""
        # When
        reason = ""
        exception = InvalidDocumentException(reason)

        # Then
        self.assertEqual(exception.reason, "")
        expected_message = "Invalid document: "
        self.assertEqual(exception.message, expected_message)


class TestDocumentProcessingException(unittest.TestCase):
    """DocumentProcessingException 테스트."""

    def test_create_document_processing_exception(self):
        """DocumentProcessingException 생성 테스트."""
        # When
        document_id = "doc-789"
        reason = "지식 추출 중 오류 발생"
        exception = DocumentProcessingException(document_id, reason)

        # Then
        self.assertEqual(exception.document_id, document_id)
        self.assertEqual(exception.reason, reason)
        self.assertEqual(exception.error_code, "DOCUMENT_PROCESSING_FAILED")
        expected_message = f"Failed to process document '{document_id}': {reason}"
        self.assertEqual(exception.message, expected_message)
        self.assertEqual(str(exception), f"[DOCUMENT_PROCESSING_FAILED] {expected_message}")

    def test_document_processing_exception_inheritance(self):
        """DocumentProcessingException 상속 관계 테스트."""
        # When
        exception = DocumentProcessingException("doc-test", "테스트 이유")

        # Then
        from src.domain.exceptions.base import DomainException
        self.assertIsInstance(exception, DomainException)
        self.assertIsInstance(exception, Exception)

    def test_document_processing_exception_can_be_raised(self):
        """DocumentProcessingException이 예외로 발생될 수 있는지 테스트."""
        # When & Then
        document_id = "failed-doc"
        reason = "네트워크 연결 오류"
        with self.assertRaises(DocumentProcessingException) as context:
            raise DocumentProcessingException(document_id, reason)

        self.assertEqual(context.exception.document_id, document_id)
        self.assertEqual(context.exception.reason, reason)

    def test_document_processing_exception_with_complex_reason(self):
        """복잡한 이유로 DocumentProcessingException 생성 테스트."""
        # When
        document_id = "complex-doc"
        reason = "LLM 모델 응답 파싱 실패: JSON 형식이 잘못되었습니다"
        exception = DocumentProcessingException(document_id, reason)

        # Then
        self.assertEqual(exception.reason, reason)
        expected_message = f"Failed to process document '{document_id}': {reason}"
        self.assertEqual(exception.message, expected_message)


if __name__ == "__main__":
    unittest.main()