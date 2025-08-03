"""
통합 설정 검증 관리자.
"""

import logging
from typing import Dict, List, Optional, Tuple

from ..validation.config_validators import ConfigValidationError, validate_all_configs
from .app import AppConfig
from .database import DatabaseConfig
from .llm import LLMConfig
from .mcp import MCPConfig
from .observability import ObservabilityConfig


class ConfigValidationManager:
    """모든 설정의 검증을 관리하는 클래스."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._configs: Dict[str, object] = {}
        self._validation_errors: List[str] = []

    def register_config(self, name: str, config: object) -> None:
        """설정을 등록합니다."""
        self._configs[name] = config
        self.logger.debug(f"설정 '{name}' 등록됨")

    def validate_all(self) -> Tuple[bool, List[str]]:
        """등록된 모든 설정을 검증합니다."""
        self._validation_errors.clear()

        try:
            # 각 설정을 개별적으로 검증
            for name, config in self._configs.items():
                try:
                    if hasattr(config, "model_validate"):
                        # Pydantic 설정
                        config.model_validate(config.model_dump())

                    # 커스텀 검증 메서드 실행
                    if hasattr(config, "validate_all"):
                        config.validate_all()
                    elif hasattr(config, "validate"):
                        config.validate()
                    elif hasattr(config, "validate_provider_config"):
                        config.validate_provider_config()

                    self.logger.info(f"설정 '{name}' 검증 성공")

                except Exception as e:
                    error_msg = f"{name}: {str(e)}"
                    self._validation_errors.append(error_msg)
                    self.logger.error(f"설정 '{name}' 검증 실패: {e}")

            # 설정 간 의존성 검증
            self._validate_config_dependencies()

            success = len(self._validation_errors) == 0
            if success:
                self.logger.info("모든 설정 검증 완료")
            else:
                self.logger.error(f"설정 검증 실패: {len(self._validation_errors)}개 오류")

            return success, self._validation_errors.copy()

        except Exception as e:
            self.logger.error(f"설정 검증 중 예상치 못한 오류: {e}")
            return False, [f"예상치 못한 오류: {str(e)}"]

    def _validate_config_dependencies(self) -> None:
        """설정 간 의존성을 검증합니다."""
        try:
            # LLM과 데이터베이스 호환성 검증
            if "llm" in self._configs and "database" in self._configs:
                llm_config = self._configs["llm"]
                db_config = self._configs["database"]

                # 임베딩 차원 일치성 검증
                if hasattr(llm_config, "get_active_provider_config"):
                    active_provider = llm_config.get_active_provider_config()
                    if (
                        hasattr(active_provider, "embedding_dimension")
                        and active_provider.embedding_dimension
                    ):
                        if hasattr(db_config, "vector_dimension"):
                            if active_provider.embedding_dimension != db_config.vector_dimension:
                                self._validation_errors.append(
                                    f"LLM 임베딩 차원({active_provider.embedding_dimension})과 "
                                    f"데이터베이스 벡터 차원({db_config.vector_dimension})이 일치하지 않습니다"
                                )

            # 백업 설정 의존성 검증
            if "database" in self._configs:
                db_config = self._configs["database"]
                if hasattr(db_config, "backup_enabled") and db_config.backup_enabled:
                    if not hasattr(db_config, "backup_path") or not db_config.backup_path:
                        self._validation_errors.append(
                            "백업이 활성화된 경우 백업 경로가 필요합니다"
                        )

            # MCP와 관찰가능성 설정 호환성
            if "mcp" in self._configs and "observability" in self._configs:
                mcp_config = self._configs["mcp"]
                obs_config = self._configs["observability"]

                # 로깅 레벨 호환성 검증
                if hasattr(mcp_config, "log_level") and hasattr(obs_config, "log_level"):
                    if mcp_config.log_level != obs_config.log_level:
                        self.logger.warning(
                            f"MCP 로그 레벨({mcp_config.log_level})과 "
                            f"관찰가능성 로그 레벨({obs_config.log_level})이 다릅니다"
                        )

        except Exception as e:
            self._validation_errors.append(f"의존성 검증 중 오류: {str(e)}")

    def get_config_summary(self) -> Dict[str, Dict]:
        """모든 설정의 요약 정보를 반환합니다."""
        summary = {}

        for name, config in self._configs.items():
            try:
                if hasattr(config, "model_dump"):
                    # Pydantic 설정
                    config_dict = config.model_dump()
                    # 민감한 정보 마스킹
                    masked_dict = {}
                    for key, value in config_dict.items():
                        if any(
                            sensitive in key.lower()
                            for sensitive in ["key", "password", "secret", "token"]
                        ):
                            if value:
                                masked_dict[key] = f"{str(value)[:4]}****"
                            else:
                                masked_dict[key] = None
                        else:
                            masked_dict[key] = value
                    summary[name] = masked_dict
                else:
                    # 일반 설정 객체
                    summary[name] = {"type": type(config).__name__}

            except Exception as e:
                summary[name] = {"error": f"요약 생성 실패: {str(e)}"}

        return summary

    def clear_configs(self) -> None:
        """등록된 모든 설정을 지웁니다."""
        self._configs.clear()
        self._validation_errors.clear()
        self.logger.debug("모든 설정이 지워졌습니다")


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
        logging.getLogger(__name__).error(f"기본 설정 로드 실패: {e}")

    return manager
