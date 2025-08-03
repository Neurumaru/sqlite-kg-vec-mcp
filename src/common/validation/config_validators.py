"""
설정 검증 관련 유틸리티.
"""

import os
from typing import Any

from pydantic import ValidationError
from pydantic_settings import BaseSettings


class ConfigValidationError(Exception):
    """설정 검증 오류."""

    def __init__(self, config_name: str, errors: list[str]):
        self.config_name = config_name
        self.errors = errors
        super().__init__(f"{config_name} 설정 검증 실패: {', '.join(errors)}")


def validate_config_instance(config: BaseSettings, config_name: str) -> None:
    """단일 설정 인스턴스를 검증합니다."""
    errors = []

    try:
        # Pydantic 검증 실행
        config.__class__.model_validate(config.model_dump())

        # 추가 커스텀 검증 메서드가 있다면 실행
        if hasattr(config, "validate_all"):
            config.validate_all()
        elif hasattr(config, "validate_provider_config"):
            config.validate_provider_config()
        # ValidationConfig의 validate 메서드는 value 매개변수가 필요하므로 제외

    except ValidationError as e:
        for error in e.errors():
            field = ".".join(str(x) for x in error["loc"])
            message = error["msg"]
            errors.append(f"{field}: {message}")
    except ValueError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"예상치 못한 오류: {str(e)}")

    if errors:
        raise ConfigValidationError(config_name, errors)


def validate_all_configs(*configs: tuple[BaseSettings, str]) -> None:
    """여러 설정을 한 번에 검증합니다."""
    all_errors = {}

    for config, config_name in configs:
        try:
            validate_config_instance(config, config_name)
        except ConfigValidationError as e:
            all_errors[config_name] = e.errors

    if all_errors:
        error_messages = []
        for config_name, errors in all_errors.items():
            error_messages.extend([f"{config_name}: {error}" for error in errors])
        raise ConfigValidationError("전체 설정", error_messages)


def validate_environment_variables(required_vars: dict[str, str]) -> None:
    """필수 환경 변수들을 검증합니다."""
    missing_vars = []
    invalid_vars = []

    for var_name, description in required_vars.items():
        value = os.getenv(var_name)
        if value is None:
            missing_vars.append(f"{var_name} ({description})")
        elif not value.strip():
            invalid_vars.append(f"{var_name} ({description}) - 빈 값")

    errors = []
    if missing_vars:
        errors.append(f"누락된 환경 변수: {', '.join(missing_vars)}")
    if invalid_vars:
        errors.append(f"잘못된 환경 변수: {', '.join(invalid_vars)}")

    if errors:
        raise ConfigValidationError("환경 변수", errors)


def validate_config_dependencies(config_map: dict[str, Any]) -> None:
    """설정 간 의존성을 검증합니다."""
    errors = []

    # 예: OpenAI 설정이 활성화되어 있으면 API 키가 필요
    if "llm" in config_map:
        llm_config = config_map["llm"]
        if hasattr(llm_config, "default_provider"):
            if llm_config.default_provider == "openai":
                if not hasattr(llm_config, "openai") or not llm_config.openai.api_key:
                    errors.append("OpenAI 제공업체 사용 시 API 키가 필요합니다")
            elif llm_config.default_provider == "anthropic":
                if not hasattr(llm_config, "anthropic") or not llm_config.anthropic.api_key:
                    errors.append("Anthropic 제공업체 사용 시 API 키가 필요합니다")

    # 예: 백업이 활성화되어 있으면 백업 경로가 필요
    if "database" in config_map:
        db_config = config_map["database"]
        if hasattr(db_config, "backup_enabled") and db_config.backup_enabled:
            if not hasattr(db_config, "backup_path") or not db_config.backup_path:
                errors.append("백업이 활성화된 경우 백업 경로가 필요합니다")

    if errors:
        raise ConfigValidationError("설정 의존성", errors)


def get_config_summary(config: BaseSettings) -> dict[str, Any]:
    """설정의 요약 정보를 반환합니다 (민감한 정보 마스킹)."""
    summary = {}

    for field_name, field_value in config.model_dump().items():
        # API 키나 비밀번호 같은 민감한 정보는 마스킹
        if any(
            sensitive in field_name.lower() for sensitive in ["key", "password", "secret", "token"]
        ):
            if field_value:
                summary[field_name] = f"{str(field_value)[:4]}****"
            else:
                summary[field_name] = "****"
        else:
            summary[field_name] = field_value

    return summary
