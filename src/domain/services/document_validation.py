"""
문서 검증 도메인 서비스.
"""

from dataclasses import dataclass
from typing import Optional

from src.common.observability.logger import ObservableLogger
from src.domain.config.validation_config import ValidationConfig
from src.domain.entities.document import Document, DocumentStatus


@dataclass(frozen=True)
class DocumentValidationResult:
    """문서 검증 결과."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]

    @classmethod
    def valid(cls) -> "DocumentValidationResult":
        """유효한 결과 생성."""
        return cls(is_valid=True, errors=[], warnings=[])

    @classmethod
    def invalid(
        cls, errors: list[str], warnings: Optional[list[str]] = None
    ) -> "DocumentValidationResult":
        """무효한 결과 생성."""
        return cls(is_valid=False, errors=errors, warnings=warnings or [])

    def add_error(self, error: str) -> "DocumentValidationResult":
        """에러 추가 (새로운 인스턴스 반환)."""
        return DocumentValidationResult(
            is_valid=False, errors=self.errors + [error], warnings=self.warnings
        )

    def add_warning(self, warning: str) -> "DocumentValidationResult":
        """경고 추가 (새로운 인스턴스 반환)."""
        return DocumentValidationResult(
            is_valid=self.is_valid, errors=self.errors, warnings=self.warnings + [warning]
        )


class DocumentValidationService:
    """
    문서 검증 도메인 서비스.

    문서의 처리 가능성과 데이터 무결성을 검증합니다.
    """

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        logger: Optional[ObservableLogger] = None,
    ):
        self.config = config or ValidationConfig.from_env()
        self.config.validate()  # 설정 검증
        from src.common.observability.logger import get_logger

        self.logger = logger or get_logger("document_validation", "domain")

    def validate_for_processing(self, document: Document) -> DocumentValidationResult:
        """문서가 처리 가능한 상태인지 검증합니다."""
        result = DocumentValidationResult.valid()

        # 상태 검증
        result = self._validate_status_for_processing(document, result)

        # 컨텐츠 검증
        result = self._validate_content(document, result)

        # 제목 검증
        result = self._validate_title(document, result)

        # 메타데이터 검증
        result = self._validate_metadata(document, result)

        return result

    def validate_for_storage(self, document: Document) -> DocumentValidationResult:
        """문서가 저장 가능한 상태인지 검증합니다."""
        result = DocumentValidationResult.valid()

        # 기본 데이터 검증
        result = self._validate_basic_data(document, result)

        # 컨텐츠 검증
        result = self._validate_content(document, result)

        # 제목 검증
        result = self._validate_title(document, result)

        # 메타데이터 검증
        result = self._validate_metadata(document, result)

        return result

    def can_be_processed(self, document: Document) -> bool:
        """문서가 처리 가능한지 간단 확인."""
        return self.validate_for_processing(document).is_valid

    def can_be_stored(self, document: Document) -> bool:
        """문서가 저장 가능한지 간단 확인."""
        return self.validate_for_storage(document).is_valid

    def _validate_status_for_processing(
        self, document: Document, result: DocumentValidationResult
    ) -> DocumentValidationResult:
        """처리를 위한 상태 검증."""
        if document.status == DocumentStatus.PROCESSING:
            if not self.config.allow_processing_while_processing:
                return result.add_error("문서가 이미 처리 중입니다")
            return result.add_warning("문서가 이미 처리 중이지만 중복 처리를 허용합니다")

        if document.status == DocumentStatus.PROCESSED:
            if not self.config.allow_reprocessing:
                return result.add_error("문서가 이미 처리되었습니다")
            return result.add_warning("문서가 이미 처리되었지만 재처리를 진행합니다")

        return result

    def _validate_content(
        self, document: Document, result: DocumentValidationResult
    ) -> DocumentValidationResult:
        """컨텐츠 검증."""
        content = document.content.strip()

        if len(content) < self.config.min_content_length:
            return result.add_error(
                f"문서 내용이 너무 짧습니다 (최소 {self.config.min_content_length}자 필요)"
            )

        if len(content) > self.config.max_content_length:
            return result.add_error(
                f"문서 내용이 너무 깁니다 (최대 {self.config.max_content_length}자 허용)"
            )

        return result

    def _validate_title(
        self, document: Document, result: DocumentValidationResult
    ) -> DocumentValidationResult:
        """제목 검증."""
        if not self.config.allow_empty_title and not document.title.strip():
            return result.add_error("문서 제목이 비어있습니다")

        return result

    def _validate_metadata(
        self, document: Document, result: DocumentValidationResult
    ) -> DocumentValidationResult:
        """메타데이터 검증."""
        # 필수 메타데이터 키 확인
        if self.config.required_metadata_keys:
            for key in self.config.required_metadata_keys:
                if key not in document.metadata:
                    result = result.add_error(f"필수 메타데이터 키 '{key}'가 누락되었습니다")

        # 메타데이터 크기 확인
        metadata_str = str(document.metadata)
        if len(metadata_str) > self.config.max_metadata_size:
            result = result.add_error(
                f"메타데이터 크기가 너무 큽니다 (최대 {self.config.max_metadata_size}자 허용)"
            )

        return result

    def _validate_basic_data(
        self, document: Document, result: DocumentValidationResult
    ) -> DocumentValidationResult:
        """기본 데이터 검증."""
        if not document.id:
            return result.add_error("문서 ID가 없습니다")

        if document.version < 1:
            return result.add_error("문서 버전은 1 이상이어야 합니다")

        return result

    def log_validation_result(self, document: Document, result: DocumentValidationResult) -> None:
        """검증 결과를 로그에 기록."""
        if result.is_valid:
            self.logger.info("validation_success", document_id=str(document.id))
            if result.warnings:
                for warning in result.warnings:
                    self.logger.warning(
                        "validation_warning", document_id=str(document.id), warning=warning
                    )
        else:
            self.logger.error(
                "validation_failed", document_id=str(document.id), error_count=len(result.errors)
            )
            for error in result.errors:
                self.logger.error("validation_error", document_id=str(document.id), error=error)
            for warning in result.warnings:
                self.logger.warning(
                    "validation_warning", document_id=str(document.id), warning=warning
                )
