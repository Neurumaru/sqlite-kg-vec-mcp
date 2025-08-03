"""
설정 검증자 테스트.
"""

import os
from unittest.mock import patch

import pytest
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from src.common.validation.config_validators import (
    ConfigValidationError,
    get_config_summary,
    validate_all_configs,
    validate_config_dependencies,
    validate_config_instance,
    validate_environment_variables,
)


class MockConfig(BaseSettings):
    """테스트용 모의 설정 클래스."""

    name: str = Field(default="test")
    value: int = Field(default=42)
    api_key: str = Field(default="secret_key")

    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        """값 필드 검증."""
        if v < 0:
            raise ValueError("값은 음수일 수 없습니다")
        return v


class InvalidConfig(BaseSettings):
    """검증 실패용 모의 설정 클래스."""

    required_field: str  # 필수 필드 (기본값 없음)


class TestConfigValidationError:
    """ConfigValidationError 테스트."""

    def test_error_creation(self):
        """오류 생성 테스트."""
        errors = ["error1", "error2"]
        error = ConfigValidationError("TestConfig", errors)

        assert error.config_name == "TestConfig"
        assert error.errors == errors
        assert "TestConfig 설정 검증 실패: error1, error2" in str(error)


class TestValidateConfigInstance:
    """단일 설정 인스턴스 검증 테스트."""

    def test_valid_config(self):
        """유효한 설정 테스트."""
        config = MockConfig(name="test", value=10)
        # 예외가 발생하지 않아야 함
        validate_config_instance(config, "MockConfig")

    def test_invalid_config_validation_error(self):
        """Pydantic 검증 오류 테스트."""
        with pytest.raises(ConfigValidationError) as exc_info:
            config = InvalidConfig()
            validate_config_instance(config, "InvalidConfig")

        assert "InvalidConfig 설정 검증 실패" in str(exc_info.value)

    def test_custom_validation_error(self):
        """커스텀 검증 오류 테스트."""
        with pytest.raises(ConfigValidationError) as exc_info:
            config = MockConfig(name="test", value=-1)  # 음수 값
            validate_config_instance(config, "MockConfig")

        assert "MockConfig 설정 검증 실패" in str(exc_info.value)
        assert "값은 음수일 수 없습니다" in str(exc_info.value)

    def test_unexpected_error(self):
        """예상치 못한 오류 테스트."""
        config = MockConfig()

        # 모델 검증에서 예상치 못한 오류 발생 시뮬레이션
        with patch.object(config, "model_validate", side_effect=RuntimeError("Unexpected error")):
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_config_instance(config, "MockConfig")

            assert "예상치 못한 오류: Unexpected error" in str(exc_info.value)


class TestValidateAllConfigs:
    """다중 설정 검증 테스트."""

    def test_all_valid_configs(self):
        """모든 설정이 유효한 경우 테스트."""
        config1 = MockConfig(name="config1", value=10)
        config2 = MockConfig(name="config2", value=20)

        # 예외가 발생하지 않아야 함
        validate_all_configs((config1, "Config1"), (config2, "Config2"))

    def test_some_invalid_configs(self):
        """일부 설정이 무효한 경우 테스트."""
        config1 = MockConfig(name="config1", value=10)  # 유효
        config2 = MockConfig(name="config2", value=-1)  # 무효

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_all_configs((config1, "Config1"), (config2, "Config2"))

        error_message = str(exc_info.value)
        assert "전체 설정 설정 검증 실패" in error_message
        assert "Config2: 값은 음수일 수 없습니다" in error_message

    def test_multiple_invalid_configs(self):
        """여러 설정이 무효한 경우 테스트."""
        config1 = MockConfig(name="config1", value=-1)  # 무효
        config2 = MockConfig(name="config2", value=-2)  # 무효

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_all_configs((config1, "Config1"), (config2, "Config2"))

        error_message = str(exc_info.value)
        assert "Config1: 값은 음수일 수 없습니다" in error_message
        assert "Config2: 값은 음수일 수 없습니다" in error_message


class TestValidateEnvironmentVariables:
    """환경 변수 검증 테스트."""

    def test_all_env_vars_present(self):
        """모든 환경 변수가 존재하는 경우 테스트."""
        required_vars = {"TEST_VAR1": "테스트 변수 1", "TEST_VAR2": "테스트 변수 2"}

        with patch.dict(os.environ, {"TEST_VAR1": "value1", "TEST_VAR2": "value2"}):
            # 예외가 발생하지 않아야 함
            validate_environment_variables(required_vars)

    def test_missing_env_vars(self):
        """환경 변수가 누락된 경우 테스트."""
        required_vars = {"MISSING_VAR1": "누락된 변수 1", "MISSING_VAR2": "누락된 변수 2"}

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_environment_variables(required_vars)

            error_message = str(exc_info.value)
            assert "누락된 환경 변수" in error_message
            assert "MISSING_VAR1" in error_message
            assert "MISSING_VAR2" in error_message

    def test_empty_env_vars(self):
        """빈 환경 변수 테스트."""
        required_vars = {"EMPTY_VAR": "빈 변수"}

        with patch.dict(os.environ, {"EMPTY_VAR": "   "}):
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_environment_variables(required_vars)

            error_message = str(exc_info.value)
            assert "잘못된 환경 변수" in error_message
            assert "EMPTY_VAR" in error_message

    def test_mixed_env_var_issues(self):
        """누락 및 빈 환경 변수 혼합 테스트."""
        required_vars = {
            "MISSING_VAR": "누락된 변수",
            "EMPTY_VAR": "빈 변수",
            "VALID_VAR": "유효한 변수",
        }

        with patch.dict(os.environ, {"EMPTY_VAR": "", "VALID_VAR": "value"}, clear=True):
            with pytest.raises(ConfigValidationError) as exc_info:
                validate_environment_variables(required_vars)

            error_message = str(exc_info.value)
            assert "누락된 환경 변수" in error_message
            assert "잘못된 환경 변수" in error_message


class TestValidateConfigDependencies:
    """설정 의존성 검증 테스트."""

    def test_openai_provider_with_api_key(self):
        """OpenAI 제공업체와 API 키가 있는 경우 테스트."""

        class MockLLMConfig:
            """OpenAI 제공업체 테스트용 모의 LLM 설정."""
            default_provider = "openai"
            openai = type("", (), {"api_key": "sk-test123"})()

        config_map = {"llm": MockLLMConfig()}

        # 예외가 발생하지 않아야 함
        validate_config_dependencies(config_map)

    def test_openai_provider_without_api_key(self):
        """OpenAI 제공업체이지만 API 키가 없는 경우 테스트."""

        class MockLLMConfig:
            """API 키 없는 OpenAI 제공업체 테스트용 모의 LLM 설정."""
            default_provider = "openai"
            openai = type("", (), {"api_key": None})()

        config_map = {"llm": MockLLMConfig()}

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dependencies(config_map)

        assert "OpenAI 제공업체 사용 시 API 키가 필요합니다" in str(exc_info.value)

    def test_anthropic_provider_without_api_key(self):
        """Anthropic 제공업체이지만 API 키가 없는 경우 테스트."""

        class MockLLMConfig:
            """API 키 없는 Anthropic 제공업체 테스트용 모의 LLM 설정."""
            default_provider = "anthropic"
            anthropic = type("", (), {"api_key": ""})()

        config_map = {"llm": MockLLMConfig()}

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dependencies(config_map)

        assert "Anthropic 제공업체 사용 시 API 키가 필요합니다" in str(exc_info.value)

    def test_backup_enabled_without_path(self):
        """백업이 활성화되었지만 경로가 없는 경우 테스트."""

        class MockDBConfig:
            """백업 경로 없는 테스트용 모의 데이터베이스 설정."""
            backup_enabled = True
            backup_path = None

        config_map = {"database": MockDBConfig()}

        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config_dependencies(config_map)

        assert "백업이 활성화된 경우 백업 경로가 필요합니다" in str(exc_info.value)

    def test_backup_enabled_with_path(self):
        """백업이 활성화되고 경로가 있는 경우 테스트."""

        class MockDBConfig:
            """백업 경로 포함 테스트용 모의 데이터베이스 설정."""
            backup_enabled = True
            backup_path = "/backup/path"

        config_map = {"database": MockDBConfig()}

        # 예외가 발생하지 않아야 함
        validate_config_dependencies(config_map)

    def test_backup_disabled(self):
        """백업이 비활성화된 경우 테스트."""

        class MockDBConfig:
            """백업 비활성화 테스트용 모의 데이터베이스 설정."""
            backup_enabled = False
            backup_path = None

        config_map = {"database": MockDBConfig()}

        # 예외가 발생하지 않아야 함
        validate_config_dependencies(config_map)


class TestGetConfigSummary:
    """설정 요약 생성 테스트."""

    def test_config_summary_with_sensitive_data(self):
        """민감한 데이터가 있는 설정 요약 테스트."""
        config = MockConfig(name="test", api_key="secret_api_key_12345", value=42)

        summary = get_config_summary(config)

        assert summary["name"] == "test"
        assert summary["value"] == 42
        assert summary["api_key"] == "secr****"  # 마스킹됨

    def test_config_summary_with_empty_sensitive_data(self):
        """빈 민감한 데이터가 있는 설정 요약 테스트."""
        config = MockConfig(name="test", api_key="", value=42)

        summary = get_config_summary(config)

        assert summary["name"] == "test"
        assert summary["value"] == 42
        assert summary["api_key"] is None  # 빈 값은 None으로

    def test_config_summary_without_sensitive_data(self):
        """민감한 데이터가 없는 설정 요약 테스트."""

        class SimpleConfig(BaseSettings):
            """민감한 데이터가 없는 단순 설정 클래스."""
            name: str = "test"
            count: int = 10

        config = SimpleConfig()
        summary = get_config_summary(config)

        assert summary["name"] == "test"
        assert summary["count"] == 10
        # 민감한 필드가 없으므로 마스킹되지 않음
