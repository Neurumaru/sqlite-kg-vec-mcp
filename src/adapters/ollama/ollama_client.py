"""
SQLite KG Vec MCP와 LLM 통합을 위한 Ollama 클라이언트.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Optional, cast

import requests

from src.common.config.llm import OllamaConfig
from src.common.observability import get_observable_logger, with_observability
from src.domain.config.timeout_config import TimeoutConfig

from .exceptions import (
    OllamaConnectionException,
    OllamaGenerationException,
    OllamaResponseException,
    OllamaTimeoutException,
)

# Langfuse 통합 제거됨 - 대체 프롬프트 사용


@dataclass
class LLMResponse:
    """LLM으로부터의 응답."""

    text: str
    model: str
    tokens_used: int
    response_time: float
    metadata: Optional[dict[str, Any]] = None


class OllamaClient:
    """Ollama LLM 모델과 상호 작용하기 위한 클라이언트."""

    def __init__(
        self,
        config: Optional[OllamaConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        """
        Ollama 클라이언트를 초기화합니다.

        Args:
            config: Ollama 설정 객체
            timeout_config: 타임아웃 설정 객체
            base_url: Ollama 서버 URL (더 이상 사용되지 않음, 대신 config 사용)
            model: 사용할 모델 이름 (더 이상 사용되지 않음, 대신 config 사용)
            timeout: 초 단위 요청 시간 초과 (더 이상 사용되지 않음, 대신 timeout_config 사용)
        """
        if config is None:
            config = OllamaConfig()

        if timeout_config is None:
            timeout_config = TimeoutConfig.from_env()

        # 이전 버전과의 호환성을 위해 제공된 경우 개별 매개변수로 설정 재정의
        self.base_url = (base_url or f"http://{config.host}:{config.port}").rstrip("/")
        self.model = model or config.model
        self.timeout = timeout or int(timeout_config.ollama_standard_timeout)
        self.timeout_config = timeout_config
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.session = requests.Session()
        self.logger = get_observable_logger("ollama_client", "adapter")

        # 연결 테스트
        self._test_connection()

    def _test_connection(self) -> bool:
        """Ollama 서버에 대한 연결을 테스트합니다."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags", timeout=self.timeout_config.ollama_quick_timeout
            )
            response.raise_for_status()
            return True
        except requests.ConnectionError as exception:
            self.logger.warning(
                "ollama_connection_failed",
                base_url=self.base_url,
                error_type="connection_error",
                error_message=str(exception),
            )
            return False
        except requests.Timeout as exception:
            self.logger.warning(
                "ollama_connection_timeout",
                base_url=self.base_url,
                timeout_duration=self.timeout_config.ollama_quick_timeout,
                error_message=str(exception),
            )
            return False
        except requests.HTTPError as exception:
            status_code = getattr(exception.response, "status_code", None)
            self.logger.warning(
                "ollama_http_error",
                base_url=self.base_url,
                status_code=status_code,
                error_message=str(exception),
            )
            return False
        except (requests.RequestException, ValueError, KeyError) as exception:
            self.logger.warning(
                "ollama_unexpected_error",
                base_url=self.base_url,
                error_type=type(exception).__name__,
                error_message=str(exception),
            )
            return False

    @with_observability(operation="llm_generate", include_args=True, include_result=True)
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Ollama 모델을 사용하여 텍스트를 생성합니다.

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 선택적 시스템 프롬프트
            temperature: 샘플링 온도 (0.0-1.0)
            max_tokens: 생성할 최대 토큰 수
            stream: 응답을 스트리밍할지 여부

        Returns:
            생성된 텍스트와 메타데이터가 포함된 LLMResponse
        """
        start_time = time.time()

        # 요청 데이터 준비
        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        data: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": options,
        }

        if system_prompt:
            data["system"] = system_prompt

        # 잠재적 예외를 처리하기 위해 response_text 초기화
        response_text = ""

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate", json=data, timeout=self.timeout
            )
            response.raise_for_status()
            response_text = response.text

            # 응답 파싱
            generated_text = ""
            total_tokens = 0

            if stream:
                # 스트리밍 응답 처리
                text_chunks = []
                for line in response.iter_lines():
                    if line:
                        chunk_data = json.loads(line.decode("utf-8"))
                        if "response" in chunk_data:
                            text_chunks.append(chunk_data["response"])

                        if chunk_data.get("done", False):
                            total_tokens = chunk_data.get("eval_count", 0)
                            break

                generated_text = "".join(text_chunks)
            else:
                # 비스트리밍 응답 처리
                result = response.json()
                generated_text = result.get("response", "")
                total_tokens = result.get("eval_count", 0)

            response_time = time.time() - start_time

            return LLMResponse(
                text=generated_text,
                model=self.model,
                tokens_used=total_tokens,
                response_time=response_time,
                metadata={"temperature": temperature},
            )

        except requests.ConnectionError as exception:
            raise OllamaConnectionException.from_requests_error(
                self.base_url, exception
            ) from exception
        except requests.Timeout as exception:
            raise OllamaTimeoutException(
                base_url=self.base_url,
                operation="text generation",
                timeout_duration=self.timeout,
                original_error=exception,
            ) from exception
        except requests.HTTPError as exception:
            status_code = getattr(exception.response, "status_code", None)
            raise OllamaConnectionException(
                base_url=self.base_url,
                message=f"생성 중 HTTP {status_code} 오류",
                status_code=status_code,
                original_error=exception,
            ) from exception
        except json.JSONDecodeError as exception:
            raise OllamaResponseException(
                response_text=response_text, parsing_error=str(exception), original_error=exception
            ) from exception
        except (ValueError, KeyError, TypeError) as exception:
            raise OllamaGenerationException(
                model_name=self.model,
                prompt=prompt,
                message=f"생성 중 데이터 처리 오류: {exception}",
                generation_params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                original_error=exception,
            ) from exception

    def extract_entities_and_relationships(self, text: str) -> dict[str, Any]:
        """
        LLM을 사용하여 텍스트에서 엔티티와 관계를 추출합니다.

        Args:
            text: 분석할 입력 텍스트

        Returns:
            추출된 엔티티와 관계가 포함된 사전
        """
        # 기본 프롬프트 사용 (Langfuse 제거됨)
        system_prompt = """당신은 전문 지식 그래프 추출 시스템입니다.
            주어진 텍스트를 분석하고 JSON 형식으로 엔티티와 관계를 추출하십시오.
            다음 구조의 JSON 객체를 반환하십시오:
            {
                "entities": [
                    {
                        "id": "unique_id",
                        "name": "entity_name",
                        "type": "entity_type",
                        "properties": {"key": "value"}
                    }
                ],
                "relationships": [
                    {
                        "source": "source_entity_id",
                        "target": "target_entity_id",
                        "type": "relationship_type",
                        "properties": {"key": "value"}
                    }
                ]
            }
            추출에 집중할 대상:
            - 사람, 조직, 장소, 개념, 이벤트
            - 엔티티 간의 명확한 관계
            - 중요한 속성 및 특성
            정확하게 텍스트에 명시적으로 언급된 정보만 추출하십시오."""

        prompt = f"""이 텍스트에서 엔티티와 관계를 추출하십시오:

{text}

유효한 JSON만 반환하고 추가 텍스트나 설명은 제외하십시오."""

        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # 보다 일관된 추출을 위해 낮은 온도
            max_tokens=2000,
        )

        try:
            # 응답 텍스트 정리 (마크다운 코드 블록이 있는 경우 제거)
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # ```json 제거
            if response_text.startswith("```"):
                response_text = response_text[3:]  # ``` 제거
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # 닫는 ``` 제거
            response_text = response_text.strip()

            # JSON 응답 파싱
            result = cast(dict[str, Any], json.loads(response_text))

            # 구조 유효성 검사
            if "entities" not in result:
                result["entities"] = []
            if "relationships" not in result:
                result["relationships"] = []

            # 프롬프트 사용 로깅 제거됨 (Langfuse 통합 제거됨)

            return result

        except json.JSONDecodeError as exception:
            self.logger.error(
                "llm_response_parse_failed",
                error_message=str(exception),
                response_text=response.text,
            )

            # 파싱 실패 시 빈 구조 반환
            return {"entities": [], "relationships": []}

    def generate_embeddings_description(self, entity: dict[str, Any]) -> str:
        """
        임베딩 개선을 위해 엔티티에 대한 풍부한 설명을 생성합니다.

        Args:
            entity: 이름, 유형, 속성을 가진 엔티티 사전

        Returns:
            향상된 설명 문자열
        """
        system_prompt = """당신은 지식 그래프 엔티티에 대한 풍부하고 유익한 설명을 만드는 전문가입니다.
        이름, 유형, 속성이 주어진 엔티티에 대해 벡터 임베딩 및 의미 검색에 이상적인 포괄적이면서도 간결한 설명을 만드십시오.
        설명은 다음을 포함해야 합니다:
        - 엔티티의 주요 특성
        - 주요 관계 및 컨텍스트 언급
        - 1-3 문장 길이
        - 검색 및 매칭에 유익함
        설명만 반환하고 추가 텍스트는 제외하십시오."""

        entity_info = f"""엔티티 이름: {entity.get('name', '알 수 없음')}
엔티티 유형: {entity.get('type', '알 수 없음')}
속성: {json.dumps(entity.get('properties', {}), indent=2)}"""

        prompt = f"이 엔티티에 대한 풍부한 설명을 만드십시오:\n\n{entity_info}"

        response = self.generate(
            prompt=prompt, system_prompt=system_prompt, temperature=0.3, max_tokens=150
        )

        return str(response.text).strip()

    def list_available_models(self) -> list[str]:
        """Ollama에서 사용 가능한 모델 목록을 가져옵니다."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags", timeout=self.timeout_config.ollama_quick_timeout
            )
            response.raise_for_status()

            data = response.json()
            return [model["name"] for model in data.get("models", [])]

        except (requests.RequestException, ValueError, KeyError) as exception:
            self.logger.error("model_list_fetch_failed", error_message=str(exception))
            return []

    def pull_model(self, model_name: str) -> bool:
        """
        Ollama에서 모델을 가져오거나 다운로드합니다.

        Args:
            model_name: 가져올 모델의 이름

        Returns:
            성공하면 True, 그렇지 않으면 False
        """
        try:
            data = {"name": model_name}
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=data,
                timeout=self.timeout_config.ollama_download_timeout,  # 모델 다운로드
            )
            response.raise_for_status()

            # 가져오기가 성공했는지 확인
            for line in response.iter_lines():
                if line:
                    chunk_data = json.loads(line.decode("utf-8"))
                    if chunk_data.get("status") == "success":
                        return True

            return False

        except (requests.RequestException, ValueError, json.JSONDecodeError) as exception:
            self.logger.error(
                "model_pull_failed", model_name=model_name, error_message=str(exception)
            )
            return False
