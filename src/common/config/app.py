"""
모든 컴포넌트 구성을 결합하는 메인 애플리케이션 구성.
"""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings

from .database import DatabaseConfig
from .llm import LLMConfig
from .mcp import MCPConfig
from .observability import ObservabilityConfig


class AppConfig(BaseSettings):
    """
    메인 애플리케이션 구성.

    벡터 검색 기능을 갖춘 SQLite 지식 그래프를 위한 모든 컴포넌트 구성을
    단일 중앙 집중식 구성 클래스로 결합합니다.
    """

    # 애플리케이션 메타데이터
    app_name: str = Field(default="sqlite-kg-vec-mcp", description="애플리케이션 이름")

    app_version: str = Field(default="0.2.0", description="애플리케이션 버전")

    environment: str = Field(
        default="development", description="환경 (development, staging, production)"
    )

    debug: bool = Field(default=False, description="디버그 모드 활성화")

    # 데이터 디렉토리
    data_dir: str = Field(default="data", description="데이터 파일을 위한 기본 디렉토리")

    @property
    def data_directory(self) -> Path:
        """필요한 경우 데이터 디렉토리 경로를 가져오고 생성합니다."""
        path = Path(self.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # 컴포넌트 구성
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig, description="데이터베이스 구성"
    )

    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM 구성")

    mcp: MCPConfig = Field(default_factory=MCPConfig, description="MCP 서버 구성")

    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig, description="관찰 가능성 구성"
    )

    def model_post_init(self, __context: Any) -> None:  # pylint: disable=arguments-differ
        """초기화 후 유효성 검사 및 설정."""
        # 환경별 설정으로 중첩 구성 업데이트
        self._update_nested_configs()

    def _update_nested_configs(self):
        """앱 수준 설정으로 중첩 구성을 업데이트합니다."""
        # 앱 메타데이터로 관찰 가능성 구성 업데이트
        self.observability.service_name = self.app_name
        self.observability.service_version = self.app_version
        self.observability.environment = self.environment

        # 데이터베이스 경로가 절대 경로가 아닌 경우 data_dir에 상대적으로 업데이트
        if not Path(self.database.db_path).is_absolute():
            self.database.db_path = str(self.data_directory / self.database.db_path)

        # MCP 벡터 인덱스 디렉토리가 설정되지 않은 경우 업데이트
        if self.mcp.vector_index_dir is None:
            self.mcp.vector_index_dir = str(self.data_directory / "vector_index")

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "AppConfig":
        """
        환경 변수 및 .env 파일로부터 구성을 생성합니다.

        인자:
            env_file: .env 파일 경로 (기본값은 현재 디렉토리의 .env)

        반환:
            AppConfig 인스턴스
        """
        env_file = env_file or ".env"

        # 환경에서 각 컴포넌트 구성 로드
        database_config = DatabaseConfig()
        llm_config = LLMConfig()
        mcp_config = MCPConfig()
        observability_config = ObservabilityConfig()

        # 메인 구성 생성
        return cls(
            database=database_config,
            llm=llm_config,
            mcp=mcp_config,
            observability=observability_config,
        )

    def to_dict(self) -> dict:
        """구성을 딕셔너리로 변환합니다."""
        return self.model_dump()

    def get_database_url(self) -> str:
        """SQLite를 위한 데이터베이스 URL을 가져옵니다."""
        return f"sqlite:///{self.database.db_path}"

    def is_production(self) -> bool:
        """프로덕션 환경에서 실행 중인지 확인합니다."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """개발 환경에서 실행 중인지 확인합니다."""
        return self.environment == "development"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "validate_assignment": True,
        "extra": "ignore",
    }
