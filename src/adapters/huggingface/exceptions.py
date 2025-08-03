"""
HuggingFace 관련 인프라 예외.

이 예외들은 HuggingFace SentenceTransformers 서비스 오류를 처리하고
모델 로딩 실패, 임베딩 생성 오류 등에 대한 의미 있는 추상화를 제공합니다.
"""

from typing import Any, Optional

from ..exceptions.base import InfrastructureException


class HuggingFaceModelException(InfrastructureException):
    """
    HuggingFace 모델 관련 오류.

    모델을 찾을 수 없거나, 모델 로딩 실패, 모델 설정 문제를 처리합니다.
    """

    def __init__(
        self,
        model_name: str,
        operation: str,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        HuggingFace 모델 예외를 초기화합니다.

        Args:
            model_name: 모델 이름
            operation: 수행 중인 작업
            message: 상세 오류 메시지
            error_code: 선택적 오류 코드
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.model_name = model_name
        self.operation = operation

        full_message = f"HuggingFace 모델 '{model_name}' 작업({operation}) 중 오류: {message}"

        super().__init__(
            message=full_message,
            error_code=error_code or "HUGGINGFACE_MODEL_ERROR",
            context=context,
            original_error=original_error,
        )


class HuggingFaceModelLoadException(HuggingFaceModelException):
    """
    HuggingFace 모델 로딩 실패.

    SentenceTransformer 모델 로딩 중 발생하는 오류를 처리합니다.
    """

    def __init__(
        self,
        model_name: str,
        message: str,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        모델 로딩 실패 예외를 초기화합니다.

        Args:
            model_name: 로딩에 실패한 모델의 이름
            message: 상세 오류 메시지
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        super().__init__(
            model_name=model_name,
            operation="model loading",
            message=message,
            error_code="HUGGINGFACE_MODEL_LOAD_FAILED",
            context=context,
            original_error=original_error,
        )


class HuggingFaceEmbeddingException(InfrastructureException):
    """
    HuggingFace 임베딩 생성 오류.

    텍스트 임베딩 생성 중 실패를 처리합니다.
    """

    def __init__(
        self,
        model_name: str,
        text: str,
        message: str,
        context: Optional[dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        임베딩 생성 예외를 초기화합니다.

        Args:
            model_name: 임베딩에 사용된 모델
            text: 입력 텍스트 (로깅을 위해 잘림)
            message: 상세 오류 메시지
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.model_name = model_name
        self.text = text[:200] + "..." if len(text) > 200 else text

        full_message = f"HuggingFace 임베딩 생성 실패(모델: '{model_name}'): {message}"

        super().__init__(
            message=full_message,
            error_code="HUGGINGFACE_EMBEDDING_FAILED",
            context=context,
            original_error=original_error,
        )
