"""
통합 설정 검증 관리자.
"""

from typing import Any, Optional

from src.common.observability.logger import ObservableLogger

from .app import AppConfig
from .database import DatabaseConfig
from .llm import LLMConfig
from .mcp import MCPConfig
from .observability import ObservabilityConfig


class ConfigValidationManager:
    """모든 설정의 검증을 관리하는 클래스."""

    def __init__(self, logger: Optional[ObservableLogger] = None):
        from src.common.observability.logger import get_logger

        self.logger = logger or get_logger("config_validation_manager", "common")
        self._configs: dict[str, Any] = {}
        self._validation_errors: list[str] = []

    def register_config(self, name: str, config: Any) -> None:
        """설정을 등록합니다."""
        self._configs[name] = config
        self.logger.debug("config_registered", config_name=name)

    def validate_all(self) -> tuple[bool, list[str]]:
        """등록된 모든 설정을 검증합니다."""
        self._validation_errors.clear()

        try:
            # 각 설정을 개별적으로 검증
            for name, config in self._configs.items():
                try:
                    if hasattr(config, "model_validate"):
                        # Pydantic 설정
                        config.__class__.model_validate(config.model_dump())

                    # 커스텀 검증 메서드 실행
                    if hasattr(config, "validate_all"):
                        config.validate_all()
                    elif hasattr(config, "validate_provider_config"):
                        config.validate_provider_config()
                    # ValidationConfig의 validate 메서드는 value 매개변수가 필요하므로 제외

                    self.logger.info("config_validation_success", config_name=name)

                except Exception as e:
                    error_msg = f"{name}: {str(e)}"
                    self._validation_errors.append(error_msg)
                    self.logger.error("config_validation_failed", config_name=name, error=str(e))

            # 설정 간 의존성 검증
            self._validate_config_dependencies()

            success = len(self._validation_errors) == 0
            if success:
                self.logger.info("all_config_validation_completed")
            else:
                self.logger.error(
                    "config_validation_failed", error_count=len(self._validation_errors)
                )

            return success, self._validation_errors.copy()

        except Exception as e:
            self.logger.error("config_validation_unexpected_error", error=str(e))
            return False, [f"예상치 못한 오류: {str(e)}"]

    def _validate_config_dependencies(self) -> None:
        """설정 간 의존성을 검증합니다."""
        try:
            self._validate_llm_database_compatibility()
            self._validate_backup_settings()
            self._validate_mcp_observability_compatibility()
        except Exception as e:
            self._validation_errors.append(f"의존성 검증 중 오류: {str(e)}")

    def _validate_llm_database_compatibility(self) -> None:
        """LLM과 데이터베이스 호환성 검증."""
        if "llm" not in self._configs or "database" not in self._configs:
            return

        llm_config = self._configs["llm"]
        db_config = self._configs["database"]

        if not hasattr(llm_config, "get_active_provider_config"):
            return

        active_provider = llm_config.get_active_provider_config()
        if not (
            hasattr(active_provider, "embedding_dimension") and active_provider.embedding_dimension
        ):
            return

        if not hasattr(db_config, "vector_dimension"):
            return

        if active_provider.embedding_dimension != db_config.vector_dimension:
            self._validation_errors.append(
                f"LLM 임베딩 차원({active_provider.embedding_dimension})과 "
                f"데이터베이스 벡터 차원({db_config.vector_dimension})이 일치하지 않습니다"
            )

    def _validate_backup_settings(self) -> None:
        """백업 설정 의존성 검증."""
        if "database" not in self._configs:
            return

        db_config = self._configs["database"]
        has_backup_enabled = hasattr(db_config, "backup_enabled") and db_config.backup_enabled
        has_backup_path = hasattr(db_config, "backup_path") and db_config.backup_path

        if has_backup_enabled and not has_backup_path:
            self._validation_errors.append("백업이 활성화된 경우 백업 경로가 필요합니다")

    def _validate_mcp_observability_compatibility(self) -> None:
        """MCP와 관찰가능성 설정 호환성 검증."""
        if "mcp" not in self._configs or "observability" not in self._configs:
            return

        mcp_config = self._configs["mcp"]
        obs_config = self._configs["observability"]

        if not (hasattr(mcp_config, "log_level") and hasattr(obs_config, "log_level")):
            return

        if mcp_config.log_level != obs_config.log_level:
            self.logger.warning(
                "log_level_mismatch",
                mcp_log_level=mcp_config.log_level,
                observability_log_level=obs_config.log_level,
            )

    def get_config_summary(self) -> dict[str, dict]:
        """모든 설정의 요약 정보를 반환합니다."""
        summary = {}

        for name, config in self._configs.items():
            try:
                summary[name] = self._create_config_summary(config)
            except Exception as e:
                summary[name] = {"error": f"요약 생성 실패: {str(e)}"}

        return summary

    def _create_config_summary(self, config) -> dict:
        """개별 설정의 요약을 생성합니다."""
        if not hasattr(config, "model_dump"):
            return {"type": type(config).__name__}

        config_dict = config.model_dump()
        return self._mask_sensitive_data(config_dict)

    def _mask_sensitive_data(self, config_dict: dict) -> dict:
        """민감한 정보를 마스킹합니다."""
        masked_dict = {}
        sensitive_keys = ["key", "password", "secret", "token"]

        for key, value in config_dict.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                masked_dict[key] = f"{str(value)[:4]}****" if value else "****"
            else:
                masked_dict[key] = value

        return masked_dict

    def clear_configs(self) -> None:
        """등록된 모든 설정을 지웁니다."""
        self._configs.clear()
        self._validation_errors.clear()
        self.logger.debug("all_configs_cleared")


def create_default_validation_manager() -> ConfigValidationManager:
    """기본 설정 검증 관리자를 생성합니다."""
    manager = ConfigValidationManager()

    try:
        # 기본 설정들을 로드하고 등록
        manager.register_config("app", AppConfig())
        manager.register_config("database", DatabaseConfig())
        manager.register_config("llm", LLMConfig())
        manager.register_config("mcp", MCPConfig())
        manager.register_config("observability", ObservabilityConfig())

    except Exception as e:
        from src.common.observability.logger import get_logger

        logger = get_logger("config_validation_manager", "common")
        logger.error("default_config_load_failed", error=str(e))

    return manager
