"""
ValidationManager 테스트.
"""

import logging
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseSettings, Field

from src.common.config.validation_manager import ConfigValidationManager


class MockValidConfig(BaseSettings):
    """테스트용 유효한 설정 클래스."""

    name: str = Field(default="test")
    value: int = Field(default=42)

    def validate(self):
        """커스텀 검증 메서드."""
        if self.value < 0:
            raise ValueError("값은 0 이상이어야 합니다")


class MockInvalidConfig(BaseSettings):
    """테스트용 무효한 설정 클래스."""

    name: str = Field(default="test")
    value: int = Field(default=-1)  # 무효한 기본값

    def validate(self):
        """커스텀 검증 메서드."""
        if self.value < 0:
            raise ValueError("값은 0 이상이어야 합니다")


class MockConfigWithoutValidation(BaseSettings):
    """검증 메서드가 없는 테스트용 설정 클래스."""

    name: str = Field(default="test")
    value: int = Field(default=10)


class MockLLMConfig:
    """LLM 설정 모의 클래스."""

    def __init__(self, embedding_dimension=None):
        self.embedding_dimension = embedding_dimension

    def get_active_provider_config(self):
        return self

    def model_validate(self, data):
        pass

    def model_dump(self):
        return {"embedding_dimension": self.embedding_dimension}


class MockDBConfig:
    """데이터베이스 설정 모의 클래스."""

    def __init__(self, vector_dimension=384, backup_enabled=False, backup_path=None):
        self.vector_dimension = vector_dimension
        self.backup_enabled = backup_enabled
        self.backup_path = backup_path

    def model_validate(self, data):
        pass

    def model_dump(self):
        return {
            "vector_dimension": self.vector_dimension,
            "backup_enabled": self.backup_enabled,
            "backup_path": self.backup_path,
        }


class TestConfigValidationManager:
    """ConfigValidationManager 테스트."""

    def test_initialization(self):
        """초기화 테스트."""
        manager = ConfigValidationManager()

        assert manager._configs == {}
        assert manager._validation_errors == []
        assert isinstance(manager.logger, logging.Logger)

    def test_initialization_with_custom_logger(self):
        """커스텀 로거로 초기화 테스트."""
        custom_logger = Mock()
        manager = ConfigValidationManager(logger=custom_logger)

        assert manager.logger == custom_logger

    def test_register_config(self):
        """설정 등록 테스트."""
        manager = ConfigValidationManager()
        config = MockValidConfig()

        manager.register_config("test_config", config)

        assert "test_config" in manager._configs
        assert manager._configs["test_config"] == config

    def test_validate_all_success(self):
        """모든 설정 검증 성공 테스트."""
        manager = ConfigValidationManager()
        config1 = MockValidConfig(name="config1", value=10)
        config2 = MockConfigWithoutValidation(name="config2", value=20)

        manager.register_config("config1", config1)
        manager.register_config("config2", config2)

        success, errors = manager.validate_all()

        assert success is True
        assert errors == []

    def test_validate_all_with_validation_errors(self):
        """검증 오류가 있는 설정 테스트."""
        manager = ConfigValidationManager()
        valid_config = MockValidConfig(name="valid", value=10)
        invalid_config = MockInvalidConfig(name="invalid", value=-1)

        manager.register_config("valid_config", valid_config)
        manager.register_config("invalid_config", invalid_config)

        success, errors = manager.validate_all()

        assert success is False
        assert len(errors) == 1
        assert "invalid_config: 값은 0 이상이어야 합니다" in errors[0]

    def test_validate_all_with_pydantic_validation_error(self):
        """Pydantic 검증 오류 테스트."""
        manager = ConfigValidationManager()

        # 모의 설정에서 model_validate 오류 발생 시뮬레이션
        config = Mock()
        config.model_validate.side_effect = Exception("Pydantic validation error")
        config.model_dump.return_value = {}

        manager.register_config("error_config", config)

        success, errors = manager.validate_all()

        assert success is False
        assert len(errors) == 1
        assert "error_config: Pydantic validation error" in errors[0]

    def test_validate_all_with_unexpected_error(self):
        """예상치 못한 오류 테스트."""
        manager = ConfigValidationManager()

        # _validate_config_dependencies에서 오류 발생 시뮬레이션
        with patch.object(
            manager, "_validate_config_dependencies", side_effect=RuntimeError("Unexpected error")
        ):
            success, errors = manager.validate_all()

            assert success is False
            assert len(errors) == 1
            assert "예상치 못한 오류: Unexpected error" in errors[0]

    def test_validate_config_dependencies_llm_db_dimension_mismatch(self):
        """LLM과 DB 차원 불일치 테스트."""
        manager = ConfigValidationManager()
        llm_config = MockLLMConfig(embedding_dimension=512)
        db_config = MockDBConfig(vector_dimension=384)

        manager.register_config("llm", llm_config)
        manager.register_config("database", db_config)

        success, errors = manager.validate_all()

        assert success is False
        assert len(errors) == 1
        assert "LLM 임베딩 차원(512)과 데이터베이스 벡터 차원(384)이 일치하지 않습니다" in errors[0]

    def test_validate_config_dependencies_llm_db_dimension_match(self):
        """LLM과 DB 차원 일치 테스트."""
        manager = ConfigValidationManager()
        llm_config = MockLLMConfig(embedding_dimension=384)
        db_config = MockDBConfig(vector_dimension=384)

        manager.register_config("llm", llm_config)
        manager.register_config("database", db_config)

        success, errors = manager.validate_all()

        assert success is True
        assert errors == []

    def test_validate_config_dependencies_backup_enabled_without_path(self):
        """백업 활성화되었지만 경로가 없는 경우 테스트."""
        manager = ConfigValidationManager()
        db_config = MockDBConfig(backup_enabled=True, backup_path=None)

        manager.register_config("database", db_config)

        success, errors = manager.validate_all()

        assert success is False
        assert len(errors) == 1
        assert "백업이 활성화된 경우 백업 경로가 필요합니다" in errors[0]

    def test_validate_config_dependencies_backup_enabled_with_path(self):
        """백업 활성화되고 경로가 있는 경우 테스트."""
        manager = ConfigValidationManager()
        db_config = MockDBConfig(backup_enabled=True, backup_path="/backup/path")

        manager.register_config("database", db_config)

        success, errors = manager.validate_all()

        assert success is True
        assert errors == []

    def test_get_config_summary(self):
        """설정 요약 테스트."""
        manager = ConfigValidationManager()
        config = MockValidConfig(name="test_config", value=42)

        manager.register_config("test_config", config)

        summary = manager.get_config_summary()

        assert "test_config" in summary
        assert summary["test_config"]["name"] == "test_config"
        assert summary["test_config"]["value"] == 42

    def test_get_config_summary_with_sensitive_data(self):
        """민감한 데이터 마스킹 테스트."""
        manager = ConfigValidationManager()

        class ConfigWithSecret(BaseSettings):
            name: str = "test"
            api_key: str = "secret_key_12345"
            password: str = "secret_password"
            regular_field: str = "regular_value"

        config = ConfigWithSecret()
        manager.register_config("secret_config", config)

        summary = manager.get_config_summary()

        assert summary["secret_config"]["name"] == "test"
        assert summary["secret_config"]["api_key"] == "secr****"
        assert summary["secret_config"]["password"] == "secr****"
        assert summary["secret_config"]["regular_field"] == "regular_value"

    def test_get_config_summary_with_non_pydantic_config(self):
        """Pydantic이 아닌 설정 요약 테스트."""
        manager = ConfigValidationManager()

        class NonPydanticConfig:
            def __init__(self):
                self.name = "non_pydantic"

        config = NonPydanticConfig()
        manager.register_config("non_pydantic", config)

        summary = manager.get_config_summary()

        assert "non_pydantic" in summary
        assert summary["non_pydantic"]["type"] == "NonPydanticConfig"

    def test_get_config_summary_with_error(self):
        """요약 생성 오류 테스트."""
        manager = ConfigValidationManager()

        # model_dump에서 오류 발생하는 모의 설정
        config = Mock()
        config.model_dump.side_effect = Exception("Summary error")

        manager.register_config("error_config", config)

        summary = manager.get_config_summary()

        assert "error_config" in summary
        assert "error" in summary["error_config"]
        assert "요약 생성 실패: Summary error" in summary["error_config"]["error"]

    def test_clear_configs(self):
        """설정 초기화 테스트."""
        manager = ConfigValidationManager()
        config = MockValidConfig()

        manager.register_config("test_config", config)
        manager._validation_errors.append("test_error")

        assert len(manager._configs) == 1
        assert len(manager._validation_errors) == 1

        manager.clear_configs()

        assert len(manager._configs) == 0
        assert len(manager._validation_errors) == 0

    def test_mcp_observability_log_level_warning(self):
        """MCP와 관찰가능성 로그 레벨 불일치 경고 테스트."""
        manager = ConfigValidationManager()

        class MockMCPConfig:
            log_level = "DEBUG"

            def model_validate(self, data):
                pass

            def model_dump(self):
                return {"log_level": self.log_level}

        class MockObsConfig:
            log_level = "INFO"

            def model_validate(self, data):
                pass

            def model_dump(self):
                return {"log_level": self.log_level}

        mcp_config = MockMCPConfig()
        obs_config = MockObsConfig()

        manager.register_config("mcp", mcp_config)
        manager.register_config("observability", obs_config)

        with patch.object(manager.logger, "warning") as mock_warning:
            success, errors = manager.validate_all()

            # 경고는 발생하지만 검증은 성공해야 함
            assert success is True
            assert errors == []
            mock_warning.assert_called_once()

            warning_call = mock_warning.call_args[0][0]
            assert "MCP 로그 레벨(DEBUG)과 관찰가능성 로그 레벨(INFO)이 다릅니다" in warning_call


class TestCreateDefaultValidationManager:
    """create_default_validation_manager 함수 테스트."""

    @patch("src.common.config.validation_manager.AppConfig")
    @patch("src.common.config.validation_manager.DatabaseConfig")
    @patch("src.common.config.validation_manager.LLMConfig")
    @patch("src.common.config.validation_manager.MCPConfig")
    @patch("src.common.config.validation_manager.ObservabilityConfig")
    def test_create_default_validation_manager_success(
        self, mock_obs, mock_mcp, mock_llm, mock_db, mock_app
    ):
        """기본 검증 관리자 생성 성공 테스트."""
        from src.common.config.validation_manager import create_default_validation_manager

        # Mock 설정 인스턴스들
        mock_app.return_value = Mock()
        mock_db.return_value = Mock()
        mock_llm.return_value = Mock()
        mock_mcp.return_value = Mock()
        mock_obs.return_value = Mock()

        manager = create_default_validation_manager()

        assert isinstance(manager, ConfigValidationManager)
        assert len(manager._configs) == 5
        assert "app" in manager._configs
        assert "database" in manager._configs
        assert "llm" in manager._configs
        assert "mcp" in manager._configs
        assert "observability" in manager._configs

    @patch("src.common.config.validation_manager.AppConfig")
    @patch("logging.getLogger")
    def test_create_default_validation_manager_with_error(self, mock_logger, mock_app_config):
        """기본 검증 관리자 생성 오류 테스트."""
        from src.common.config.validation_manager import create_default_validation_manager

        # AppConfig 생성 시 오류 발생 시뮬레이션
        mock_app_config.side_effect = Exception("Config creation error")
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance

        manager = create_default_validation_manager()

        # 오류가 발생해도 manager는 생성되어야 함
        assert isinstance(manager, ConfigValidationManager)
        # 오류 로그가 기록되어야 함
        mock_logger_instance.error.assert_called_once()
        error_call = mock_logger_instance.error.call_args[0][0]
        assert "기본 설정 로드 실패" in error_call
