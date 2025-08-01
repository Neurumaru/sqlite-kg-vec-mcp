"""
Ollama-specific infrastructure exceptions.

These exceptions handle Ollama LLM service errors and provide
meaningful abstractions for common API failure scenarios.
"""

from typing import Any

import requests

from ...adapters.exceptions.base import InfrastructureException
from ...adapters.exceptions.connection import HTTPConnectionException
from ...adapters.exceptions.data import DataParsingException
from ...adapters.exceptions.timeout import HTTPTimeoutException


class OllamaConnectionException(HTTPConnectionException):
    """
    Ollama service connection failures.

    Handles network issues, service unavailability,
    and connection setup problems.
    """

    def __init__(
        self,
        base_url: str,
        message: str,
        status_code: int | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize Ollama connection exception.

        Args:
            base_url: Ollama server base URL
            message: Detailed error message
            status_code: HTTP status code if available
            context: Additional context
            original_error: Original exception
        """
        super().__init__(
            url=base_url,
            message=f"Ollama service: {message}",
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
        Create exception from requests error.

        Args:
            base_url: Ollama server base URL
            requests_error: Original requests exception

        Returns:
            OllamaConnectionException instance
        """
        if isinstance(requests_error, requests.ConnectionError):
            message = "Cannot connect to Ollama server"
        elif isinstance(requests_error, requests.HTTPError):
            status_code = getattr(requests_error.response, "status_code", None)
            message = f"HTTP error {status_code}"
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
    Ollama request timeouts.

    Handles timeouts during model operations, generation,
    and API calls.
    """

    def __init__(
        self,
        base_url: str,
        operation: str,
        timeout_duration: float,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize Ollama timeout exception.

        Args:
            base_url: Ollama server base URL
            operation: Operation that timed out
            timeout_duration: Timeout duration in seconds
            context: Additional context
            original_error: Original exception
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
    Ollama model-related errors.

    Handles model not found, model loading failures,
    and model configuration issues.
    """

    def __init__(
        self,
        model_name: str,
        operation: str,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize Ollama model exception.

        Args:
            model_name: Name of the model
            operation: Operation being performed
            message: Detailed error message
            error_code: Optional error code
            context: Additional context
            original_error: Original exception
        """
        self.model_name = model_name
        self.operation = operation

        full_message = f"Ollama model '{model_name}' error during {operation}: {message}"

        super().__init__(
            message=full_message,
            error_code=error_code or "OLLAMA_MODEL_ERROR",
            context=context,
            original_error=original_error,
        )


class OllamaModelNotFoundException(OllamaModelException):
    """
    Ollama model not found.

    Raised when a requested model is not available
    on the Ollama server.
    """

    def __init__(
        self,
        model_name: str,
        available_models: list | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize model not found exception.

        Args:
            model_name: Name of the missing model
            available_models: List of available models
            context: Additional context
            original_error: Original exception
        """
        self.available_models = available_models or []

        message = f"Model '{model_name}' not found"
        if self.available_models:
            message += f". Available models: {', '.join(self.available_models)}"

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
    Ollama text generation errors.

    Handles failures during text generation, prompt processing,
    and response parsing.
    """

    def __init__(
        self,
        model_name: str,
        prompt: str,
        message: str,
        generation_params: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize generation exception.

        Args:
            model_name: Model used for generation
            prompt: Input prompt (truncated for logging)
            message: Detailed error message
            generation_params: Generation parameters used
            context: Additional context
            original_error: Original exception
        """
        self.model_name = model_name
        self.prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
        self.generation_params = generation_params or {}

        full_message = f"Ollama generation failed with model '{model_name}': {message}"

        super().__init__(
            message=full_message,
            error_code="OLLAMA_GENERATION_FAILED",
            context=context,
            original_error=original_error,
        )


class OllamaResponseException(DataParsingException):
    """
    Ollama response parsing errors.

    Handles issues with parsing JSON responses,
    malformed responses, and unexpected response formats.
    """

    def __init__(
        self,
        response_text: str,
        expected_format: str = "JSON",
        parsing_error: str | None = None,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize response parsing exception.

        Args:
            response_text: Raw response text (truncated)
            expected_format: Expected response format
            parsing_error: Specific parsing error message
            context: Additional context
            original_error: Original parsing exception
        """
        message = f"Failed to parse Ollama response as {expected_format}"
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
    Ollama configuration errors.

    Handles invalid server URLs, missing configuration,
    and service setup issues.
    """

    def __init__(
        self,
        config_parameter: str,
        invalid_value: Any,
        message: str,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize configuration exception.

        Args:
            config_parameter: Configuration parameter with issue
            invalid_value: Invalid configuration value
            message: Detailed error message
            context: Additional context
            original_error: Original exception
        """
        self.config_parameter = config_parameter
        self.invalid_value = invalid_value

        full_message = (
            f"Ollama configuration error for '{config_parameter}' = '{invalid_value}': {message}"
        )

        super().__init__(
            message=full_message,
            error_code="OLLAMA_CONFIGURATION_ERROR",
            context=context,
            original_error=original_error,
        )
