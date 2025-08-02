"""
Ollama 관련 인프라 예외.

이 예외들은 Ollama LLM 서비스 오류를 처리하고
일반적인 API 실패 시나리오에 대한 의미 있는 추상화를 제공합니다.
"""

from typing import Any, Optional

import requests

from ...adapters.exceptions.base import InfrastructureException
from ...adapters.exceptions.connection import HTTPConnectionException
from ...adapters.exceptions.data import DataParsingException
from ...adapters.exceptions.timeout import HTTPTimeoutException


class OllamaConnectionException(HTTPConnectionException):
    """
    Ollama 서비스 연결 실패.

    네트워크 문제, 서비스 비가용성, 연결 설정 문제를 처리합니다.
    """

    def __init__(
        self,
        base_url: str,
        message: str,
        status_code: Optional[int] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Ollama 연결 예외를 초기화합니다.

        Args:
            base_url: Ollama 서버 기본 URL
            message: 상세 오류 메시지
            status_code: 사용 가능한 경우 HTTP 상태 코드
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        super().__init__(
            url=base_url,
            message=f"Ollama 서비스: {message}",
            status_code=status_code,
            error_code="OLLAMA_CONNECTION_FAILED",
            context=context,
            original_error=original_error,
        )
        self.service = "Ollama"

    @classmethod
    def from_requests_error(
        cls, base_url: str, requests_error: requests.RequestException
    ) -> "OllamaConnectionException":
        """
        requests 오류로부터 예외를 생성합니다.

        Args:
            base_url: Ollama 서버 기본 URL
            requests_error: 원래 requests 예외

        Returns:
            OllamaConnectionException 인스턴스
        """
        if isinstance(requests_error, requests.ConnectionError):
            message = "Ollama 서버에 연결할 수 없습니다."
        elif isinstance(requests_error, requests.HTTPError):
            status_code = getattr(requests_error.response, "status_code", None)
            message = f"HTTP 오류 {status_code}"
            return cls(
                base_url=base_url,
                message=message,
                status_code=status_code,
                original_error=requests_error,
            )
        else:
            message = str(requests_error)

        return cls(base_url=base_url, message=message, original_error=requests_error)


class OllamaTimeoutException(HTTPTimeoutException):
    """
    Ollama 요청 시간 초과.

    모델 작업, 생성, API 호출 중 시간 초과를 처리합니다.
    """

    def __init__(
        self,
        base_url: str,
        operation: str,
        timeout_duration: float,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Ollama 시간 초과 예외를 초기화합니다.

        Args:
            base_url: Ollama 서버 기본 URL
            operation: 시간 초과된 작업
            timeout_duration: 초 단위 시간 초과 기간
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        super().__init__(
            url=base_url,
            method="POST",
            timeout_duration=timeout_duration,
            error_code="OLLAMA_TIMEOUT",
            context=context,
            original_error=original_error,
        )
        self.operation = operation
        self.service = "Ollama"


class OllamaModelException(InfrastructureException):
    """
    Ollama 모델 관련 오류.

    모델을 찾을 수 없거나, 모델 로딩 실패, 모델 설정 문제를 처리합니다.
    """

    def __init__(
        self,
        model_name: str,
        operation: str,
        message: str,
        error_code: Optional[str] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Ollama 모델 예외를 초기화합니다.

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

        full_message = f"Ollama 모델 '{model_name}' 작업({operation}) 중 오류: {message}"

        super().__init__(
            message=full_message,
            error_code=error_code or "OLLAMA_MODEL_ERROR",
            context=context,
            original_error=original_error,
        )


class OllamaModelNotFoundException(OllamaModelException):
    """
    Ollama 모델을 찾을 수 없음.

    요청된 모델이 Ollama 서버에서 사용 가능하지 않을 때 발생합니다.
    """

    def __init__(
        self,
        model_name: str,
        available_models: Optional[list] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        모델을 찾을 수 없음 예외를 초기화합니다.

        Args:
            model_name: 누락된 모델의 이름
            available_models: 사용 가능한 모델 목록
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.available_models = available_models or []

        message = f"모델 '{model_name}'을(를) 찾을 수 없습니다."
        if self.available_models:
            message += f". 사용 가능한 모델: {', '.join(self.available_models)}"

        super().__init__(
            model_name=model_name,
            operation="model lookup",
            message=message,
            error_code="OLLAMA_MODEL_NOT_FOUND",
            context=context,
            original_error=original_error,
        )


class OllamaGenerationException(InfrastructureException):
    """
    Ollama 텍스트 생성 오류.

    텍스트 생성, 프롬프트 처리, 응답 파싱 중 실패를 처리합니다.
    """

    def __init__(
        self,
        model_name: str,
        prompt: str,
        message: str,
        generation_params: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        생성 예외를 초기화합니다.

        Args:
            model_name: 생성에 사용된 모델
            prompt: 입력 프롬프트 (로깅을 위해 잘림)
            message: 상세 오류 메시지
            generation_params: 사용된 생성 매개변수
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.model_name = model_name
        self.prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
        self.generation_params = generation_params or {}

        full_message = f"Ollama 생성 실패(모델: '{model_name}'): {message}"

        super().__init__(
            message=full_message,
            error_code="OLLAMA_GENERATION_FAILED",
            context=context,
            original_error=original_error,
        )


class OllamaResponseException(DataParsingException):
    """
    Ollama 응답 파싱 오류.

    JSON 응답 파싱, 잘못된 형식의 응답, 예상치 못한 응답 형식 문제를 처리합니다.
    """

    def __init__(
        self,
        response_text: str,
        expected_format: str = "JSON",
        parsing_error: Optional[str] = None,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        응답 파싱 예외를 초기화합니다.

        Args:
            response_text: 원본 응답 텍스트 (잘림)
            expected_format: 예상 응답 형식
            parsing_error: 특정 파싱 오류 메시지
            context: 추가 컨텍스트
            original_error: 원래 파싱 예외
        """
        message = f"Ollama 응답을 {expected_format}(으)로 파싱하지 못했습니다."
        if parsing_error:
            message += f": {parsing_error}"

        super().__init__(
            data_format=f"Ollama {expected_format}",
            message=message,
            raw_data=response_text,
            error_code="OLLAMA_RESPONSE_PARSING_FAILED",
            context=context,
            original_error=original_error,
        )


class OllamaConfigurationException(InfrastructureException):
    """
    Ollama 설정 오류.

    잘못된 서버 URL, 누락된 설정, 서비스 설정 문제를 처리합니다.
    """

    def __init__(
        self,
        config_parameter: str,
        invalid_value: Any,
        message: str,
        context: dict[str, Any] | None = None,
        original_error: Optional[Exception] = None,
    ):
        """
        설정 예외를 초기화합니다.

        Args:
            config_parameter: 문제가 있는 설정 매개변수
            invalid_value: 잘못된 설정 값
            message: 상세 오류 메시지
            context: 추가 컨텍스트
            original_error: 원래 발생한 예외
        """
        self.config_parameter = config_parameter
        self.invalid_value = invalid_value

        full_message = (
            f"'{config_parameter}' = '{invalid_value}'에 대한 Ollama 설정 오류: {message}"
        )

        super().__init__(
            message=full_message,
            error_code="OLLAMA_CONFIGURATION_ERROR",
            context=context,
            original_error=original_error,
        )
