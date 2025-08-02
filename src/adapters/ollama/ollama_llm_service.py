"""
LLMService 인터페이스의 Ollama 기반 구현.
"""

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from typing import Any, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage

from src.common.config.llm import OllamaConfig
from src.ports.llm import LLM

from .ollama_client import OllamaClient

# 순환 종속성을 피하기 위한 임시 유형 별칭
SearchResult = Any


class OllamaLLMService(LLM):
    """
    LLM 포트의 Ollama 기반 구현.

    이 어댑터는 검색 안내, 지식 추출 및 분석을 위해 Ollama 모델을 사용하여
    LLM 기능을 제공합니다.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        config: Optional[OllamaConfig] = None,
        default_temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Ollama LLM 서비스를 초기화합니다.

        Args:
            ollama_client: 설정된 Ollama 클라이언트 (사용 중단됨, 대신 config를 사용하세요)
            config: Ollama 설정 객체
            default_temperature: 기본 샘플링 온도 (사용 중단됨, 대신 config를 사용하세요)
            max_tokens: 응답의 기본 최대 토큰 수 (사용 중단됨, 대신 config를 사용하세요)
        """
        if config is None:
            config = OllamaConfig()

        if ollama_client is None:
            ollama_client = OllamaClient(config=config)

        self.ollama_client = ollama_client
        self.default_temperature = default_temperature or config.temperature
        self.max_tokens = max_tokens or config.max_tokens

    # LangChain 호환 메서드

    async def invoke(
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> BaseMessage:
        """
        메시지 기반으로 응답을 생성합니다 (LangChain invoke 스타일).

        Args:
            input: LangChain BaseMessage 리스트
            **kwargs: 모델 매개변수 (temperature, max_tokens, stop 등)

        Returns:
            AIMessage 응답
        """
        # 메시지를 텍스트로 변환
        prompt = self._messages_to_text(messages)

        # 매개변수 추출
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Ollama 클라이언트를 사용하여 생성
        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return AIMessage(content=response.text)

    async def stream(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 모드로 응답을 생성합니다 (LangChain stream 스타일).

        Args:
            input: LangChain BaseMessage 리스트
            **kwargs: 모델 매개변수

        Yields:
            응답 텍스트 청크
        """
        # 메시지를 텍스트로 변환
        prompt = self._messages_to_text(messages)

        # 매개변수 추출
        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Ollama 클라이언트를 사용하여 생성 (현재 스트리밍 시뮬레이션)
        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 스트리밍 시뮬레이션을 위해 응답을 청크로 분할
        text = response.text
        chunk_size = 50
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            yield chunk
            await asyncio.sleep(0.05)  # 스트리밍 지연 시뮬레이션

    async def batch(
        self,
        inputs: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> list[BaseMessage]:
        """
        여러 메시지 시퀀스를 일괄 처리합니다 (LangChain batch 스타일).

        Args:
            inputs: 메시지 시퀀스 리스트
            **kwargs: 모델 매개변수

        Returns:
            AIMessage 응답 리스트
        """
        # 동시 처리를 위한 태스크 생성
        tasks = []
        for message_list in inputs:
            task = asyncio.create_task(self.invoke(message_list, **kwargs))
            tasks.append(task)

        # 모든 태스크가 완료될 때까지 대기
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 처리 및 예외 핸들링
        processed_results: list[BaseMessage] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # 실패한 요청에 대한 오류 메시지 생성
                error_msg = f"일괄 처리 항목 {i} 실패: {str(result)}"
                processed_results.append(AIMessage(content=error_msg))
            elif isinstance(result, BaseMessage):
                processed_results.append(result)
            else:
                # 예기치 않은 결과 유형 처리
                processed_results.append(
                    AIMessage(content=f"예기치 않은 결과 유형: {type(result)}")
                )

        return processed_results

    def _messages_to_text(self, messages: list[BaseMessage]) -> str:
        """BaseMessage 리스트를 텍스트로 변환합니다."""
        text_parts = []
        for message in messages:
            role = message.__class__.__name__.replace("Message", "").lower()
            if role == "ai":
                role = "assistant"
            elif role == "human":
                role = "user"

            text_parts.append(f"{role}: {message.content}")

        return "\n".join(text_parts)

    # 대화형 검색 안내

    async def analyze_query(
        self, query: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """검색 쿼리를 분석하여 최적의 검색 전략을 결정합니다."""
        system_prompt = """당신은 전문 검색 분석가입니다. 주어진 쿼리를 분석하고 최상의 검색 전략을 추천하세요.

        쿼리 의도를 분류하고 다음 전략 중 하나를 추천하세요:
        - SEMANTIC: 개념적, 의미 기반 검색용
        - STRUCTURAL: 특정 엔티티/관계 쿼리용
        - HYBRID: 두 가지 접근 방식이 모두 필요한 복잡한 쿼리용
        - STOP: 쿼리가 불분명하거나 유효하지 않은 경우

        다음 구조의 JSON을 반환하세요:
        {
            "strategy": "SEMANTIC|STRUCTURAL|HYBRID|STOP",
            "confidence": 0.0-1.0,
            "reasoning": "설명",
            "suggested_filters": ["필터1", "필터2"],
            "query_type": "사실적|탐색적|탐색적|트랜잭션"
        }"""

        context_str = f"컨텍스트: {json.dumps(context)}\n" if context else ""
        prompt = f"{context_str}분석할 쿼리: {query}"

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=500,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, dict):
                return result
            raise ValueError("dict 응답이 예상되었습니다") from None
        except Exception as exception:
            logging.warning("쿼리 분석 응답 파싱 실패: %s", exception)
            return {
                "strategy": "SEMANTIC",
                "confidence": 0.5,
                "reasoning": "파싱 오류로 인해 시맨틱 검색으로 대체",
                "suggested_filters": [],
                "query_type": "exploratory",
            }

    async def guide_search_navigation(
        self,
        current_results: list[SearchResult],
        original_query: str,
        search_history: list[dict[str, Any]],
        step_number: int,
    ) -> dict[str, Any]:
        """대화형 검색 탐색의 다음 단계를 안내합니다."""
        system_prompt = """당신은 검색 탐색 가이드입니다. 현재 결과와 검색 기록을 바탕으로 다음 검색 작업을 추천하세요.

        다음 구조의 JSON을 반환하세요:
        {
            "next_action": "refine|expand|pivot|stop",
            "strategy": "SEMANTIC|STRUCTURAL|HYBRID|STOP",
            "suggested_query": "새 쿼리 또는 수정",
            "reasoning": "추천에 대한 설명",
            "focus_areas": ["영역1", "영역2"],
            "confidence": 0.0-1.0
        }"""

        # 현재 결과 요약
        results_summary = []
        for i, result in enumerate(current_results[:5]):  # 상위 5개로 제한
            results_summary.append(
                {
                    "rank": i + 1,
                    "score": result.score,
                    "type": (result.entity_type if hasattr(result, "entity_type") else "unknown"),
                }
            )

        prompt = f"""원본 쿼리: {original_query}
        단계 번호: {step_number}
        현재 결과 요약: {json.dumps(results_summary)}
        검색 기록: {json.dumps(search_history[-3:])}  # 마지막 3단계

        다음 검색 작업은 무엇이어야 합니까?"""

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=400,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, dict):
                return result
            raise ValueError("dict 응답이 예상되었습니다") from None
        except Exception as exception:
            logging.warning("탐색 안내 파싱 실패: %s", exception)
            return {
                "next_action": "stop",
                "strategy": "STOP",
                "suggested_query": original_query,
                "reasoning": "파싱 오류로 인해 안내를 생성할 수 없습니다",
                "focus_areas": [],
                "confidence": 0.0,
            }

    async def evaluate_search_results(
        self, results: list[SearchResult], query: str, search_context: dict[str, Any]
    ) -> dict[str, Any]:
        """검색 결과의 품질과 관련성을 평가합니다."""
        system_prompt = """당신은 검색 품질 평가자입니다. 검색 결과의 관련성과 품질을 평가하세요.

        다음 구조의 JSON을 반환하세요:
        {
            "overall_quality": 0.0-1.0,
            "relevance_score": 0.0-1.0,
            "coverage_score": 0.0-1.0,
            "diversity_score": 0.0-1.0,
            "recommendations": ["제안1", "제안2"],
            "best_result_index": 0,
            "quality_issues": ["문제1", "문제2"]
        }"""

        # 평가를 위한 결과 요약
        results_data = []
        for i, result in enumerate(results[:10]):  # 상위 10개로 제한
            results_data.append(
                {
                    "index": i,
                    "score": result.score,
                    "snippet": (
                        getattr(result, "snippet", "")[:100] + "..."
                        if hasattr(result, "snippet")
                        else ""
                    ),
                }
            )

        prompt = f"""쿼리: {query}
        컨텍스트: {json.dumps(search_context)}
        평가할 결과: {json.dumps(results_data)}

        이 검색 결과의 품질과 관련성을 평가하세요."""

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=600,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, dict):
                return result
            raise ValueError("dict 응답이 예상되었습니다") from None
        except Exception as exception:
            logging.warning("결과 평가 파싱 실패: %s", exception)
            return {
                "overall_quality": 0.5,
                "relevance_score": 0.5,
                "coverage_score": 0.5,
                "diversity_score": 0.5,
                "recommendations": ["검색 쿼리를 수정해 보세요"],
                "best_result_index": 0,
                "quality_issues": ["파싱 오류로 인해 평가할 수 없습니다"],
            }

    # 지식 추출

    async def extract_knowledge_from_text(
        self,
        text: str,
        extraction_schema: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """비정형 텍스트에서 구조화된 지식을 추출합니다."""
        # 기존 Ollama 클라이언트 메서드 사용
        result = await asyncio.to_thread(
            self.ollama_client.extract_entities_and_relationships, text
        )

        result["extraction_metadata"] = {
            "text_length": len(text),
            "schema_used": extraction_schema is not None,
            "context_provided": context is not None,
            "model": self.ollama_client.model,
        }

        return result

    async def generate_entity_summary(
        self,
        entity_data: dict[str, Any],
        related_entities: list[dict[str, Any]] | None = None,
    ) -> str:
        """데이터와 관계를 기반으로 엔티티에 대한 요약을 생성합니다."""
        # 기존 Ollama 클라이언트 메서드 사용
        summary = await asyncio.to_thread(
            self.ollama_client.generate_embeddings_description, entity_data
        )

        return summary

    async def suggest_relationships(
        self,
        source_entity: dict[str, Any],
        target_entities: list[dict[str, Any]],
        context: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """엔티티 간의 잠재적인 관계를 제안합니다."""
        system_prompt = """당신은 지식 그래프 관계 전문가입니다. 엔티티 간의 잠재적인 관계를 제안하세요.

        각 관계에 대해 다음 구조의 JSON 배열을 반환하세요:
        {
            "target_entity": "대상_엔티티_이름",
            "relationship_type": "관계_유형",
            "confidence": 0.0-1.0,
            "reasoning": "설명",
            "properties": {"키": "값"}
        }"""

        target_names = [entity.get("name", "Unknown") for entity in target_entities[:10]]
        context_str = f"컨텍스트: {context}\n" if context else ""

        prompt = f"""{context_str}소스 엔티티: {json.dumps(source_entity)}
        대상 엔티티: {json.dumps(target_names)}

        소스 엔티티와 대상 엔티티 간의 의미 있는 관계를 제안하세요."""

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,
            max_tokens=800,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, list):
                return result
            return []
        except Exception as exception:
            logging.warning("관계 제안 파싱 실패: %s", exception)
            return []

    # 쿼리 향상

    async def expand_query(
        self, original_query: str, search_context: dict[str, Any] | None = None
    ) -> list[str]:
        """관련 용어 및 개념으로 쿼리를 확장합니다."""
        system_prompt = """당신은 쿼리 확장 전문가입니다. 주어진 쿼리에 대한 관련 용어 및 개념을 생성하세요.

        확장된 쿼리 용어의 JSON 배열을 반환하세요:
        ["용어1", "용어2", "용어3", ...]

        동의어, 관련 개념 및 문맥적으로 관련된 용어에 중점을 둡니다."""

        context_str = f"컨텍스트: {json.dumps(search_context)}\n" if search_context else ""
        prompt = f"{context_str}원본 쿼리: {original_query}\n\n5-10개의 관련 용어를 생성하세요."

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.6,
            max_tokens=300,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, list):
                return result
            return [original_query]
        except Exception as exception:
            logging.warning("쿼리 확장 파싱 실패: %s", exception)
            return [original_query]

    async def generate_search_suggestions(
        self, partial_query: str, search_history: list[str] | None = None
    ) -> list[str]:
        """부분 쿼리에 대한 검색 제안을 생성합니다."""
        system_prompt = """당신은 검색 제안 생성기입니다. 부분 쿼리를 완성하고 변형을 제안하세요.

        검색 제안의 JSON 배열을 반환하세요:
        ["제안1", "제안2", "제안3", ...]

        일반적인 검색 패턴과 사용자 의도를 고려하세요."""

        history_str = f"최근 검색어: {json.dumps(search_history[-5:])}\n" if search_history else ""
        prompt = f"{history_str}부분 쿼리: {partial_query}\n\n3-7개의 검색 제안을 생성하세요."

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=200,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, list):
                return result
            return [partial_query]
        except Exception as exception:
            logging.warning("검색 제안 파싱 실패: %s", exception)
            return [partial_query]

    # 콘텐츠 분석

    async def classify_content(
        self, content: str, classification_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """주어진 스키마에 따라 콘텐츠를 분류합니다."""
        system_prompt = f"""당신은 콘텐츠 분류기입니다. 제공된 스키마에 따라 주어진 콘텐츠를 분류하세요.

        분류 스키마: {json.dumps(classification_schema)}

        분류 결과와 신뢰도 점수를 포함한 JSON을 반환하세요."""

        prompt = f"분류할 콘텐츠: {content[:1000]}..."  # 콘텐츠 길이 제한

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=400,
        )

        try:
            result = self._parse_json_response(response.text)
            if isinstance(result, dict):
                return result
            raise ValueError("dict 응답이 예상되었습니다") from None
        except Exception as exception:
            logging.warning("콘텐츠 분류 파싱 실패: %s", exception)
            return {"error": "분류 실패", "confidence": 0.0}

    async def detect_language(self, text: str) -> str:
        """주어진 텍스트의 언어를 감지합니다."""
        system_prompt = """주어진 텍스트의 언어를 감지합니다. ISO 639-1 언어 코드만 반환하세요 (예: 'en', 'es', 'fr', 'de', 'ko' 등)."""

        prompt = f"텍스트: {text[:500]}..."  # 텍스트 길이 제한

        response = await asyncio.to_thread(
            self.ollama_client.generate,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=10,
        )

        # 응답에서 언어 코드 추출
        lang_code = str(response.text).strip().lower()
        # 합리적인 언어 코드인지 확인
        if len(lang_code) == 2 and lang_code.isalpha():
            return lang_code
        return "en"  # 기본값으로 영어 반환

    # 스트리밍 응답

    async def stream_analysis(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> AsyncGenerator[str, None]:
        """실시간 처리를 위해 분석 결과를 스트리밍합니다."""
        context_str = f"컨텍스트: {json.dumps(context)}\n" if context else ""
        full_prompt = f"{context_str}{prompt}"

        # 지원되는 경우 스트리밍을 시도하고, 그렇지 않으면 청크 시뮬레이션으로 대체
        try:
            # 스트리밍 생성 시도 (향후 Ollama 클라이언트가 지원하는 경우)
            response = await asyncio.to_thread(
                self.ollama_client.generate,
                prompt=full_prompt,
                temperature=self.default_temperature,
                max_tokens=self.max_tokens,
                stream=True,  # 먼저 스트리밍 시도
            )

            # 스트리밍이 지원된다면 실제 청크를 생성할 것입니다
            # 지금은 더 나은 청킹 로직으로 시뮬레이션합니다
            text = response.text

            # 콘텐츠 기반 동적 청크 크기 조정
            words = text.split()
            current_chunk = ""
            word_count = 0

            for word in words:
                current_chunk += word + " "
                word_count += 1

                # 최적 크기에 도달하거나 자연스러운 중단점에서 청크 생성
                if word_count >= 5 and (
                    word.endswith(".")
                    or word.endswith("!")
                    or word.endswith("?")
                    or len(current_chunk) > 100
                ):
                    yield current_chunk.strip()
                    current_chunk = ""
                    word_count = 0
                    await asyncio.sleep(0.05)  # 자연스러운 타이핑 지연

            # 남은 콘텐츠 생성
            if current_chunk.strip():
                yield current_chunk.strip()

        except Exception:
            # 모든 오류에 대해 간단한 청킹으로 대체
            response = await asyncio.to_thread(
                self.ollama_client.generate,
                prompt=full_prompt,
                temperature=self.default_temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )

            text = response.text
            chunk_size = 50
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                yield chunk
                await asyncio.sleep(0.1)

    # 설정 및 상태

    async def get_model_info(self) -> dict[str, Any]:
        """현재 LLM 모델에 대한 정보를 가져옵니다."""
        available_models = await asyncio.to_thread(self.ollama_client.list_available_models)

        return {
            "current_model": self.ollama_client.model,
            "base_url": self.ollama_client.base_url,
            "available_models": available_models,
            "default_temperature": self.default_temperature,
            "max_tokens": self.max_tokens,
            "provider": "ollama",
        }

    async def health_check(self) -> dict[str, Any]:
        """LLM 서비스의 상태를 확인합니다."""
        try:
            # 간단한 요청으로 연결 테스트
            test_response = await asyncio.to_thread(
                self.ollama_client.generate, prompt="Hello", max_tokens=10
            )

            return {
                "status": "healthy",
                "model": self.ollama_client.model,
                "base_url": self.ollama_client.base_url,
                "response_time": test_response.response_time,
                "last_check": "now",
            }
        except Exception as exception:
            return {
                "status": "unhealthy",
                "error": str(exception),
                "model": self.ollama_client.model,
                "base_url": self.ollama_client.base_url,
                "last_check": "now",
            }

    async def get_usage_stats(self) -> dict[str, Any]:
        """LLM 서비스의 사용 통계를 가져옵니다."""
        # 참고: 실제 구현에서는 사용량 추적이 필요합니다
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "model": self.ollama_client.model,
            "note": "사용량 추적은 아직 구현되지 않았습니다",
        }

    # 헬퍼 메서드

    def _parse_json_response(self, response_text: str) -> dict[str, Any] | list[Any]:
        """LLM의 JSON 응답을 파싱하고 일반적인 형식 문제를 처리합니다."""
        response_text = response_text.strip()

        # 마크다운 코드 블록이 있는 경우 제거
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]

        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        try:
            parsed = json.loads(response_text)
            return cast(dict[str, Any] | list[Any], parsed)
        except json.JSONDecodeError as exception:
            # 응답에서 JSON 추출 시도
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    return cast(dict[str, Any] | list[Any], parsed)
                except json.JSONDecodeError as inner_exception:
                    raise ValueError(
                        f"응답에서 추출한 JSON을 파싱할 수 없습니다: {response_text}"
                    ) from inner_exception
            raise ValueError(
                f"응답에서 유효한 JSON을 찾을 수 없습니다: {response_text}"
            ) from exception
