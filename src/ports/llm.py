"""
LLM 서비스 포트.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, List

from langchain_core.messages import BaseMessage



class LLM(ABC):
    """
    LLM 서비스 포트.

    범용적인 대화형 언어 모델 기능을 제공합니다 (LangChain 호환).
    """

    @abstractmethod
    async def invoke(
        self,
        input: List[BaseMessage],
        **kwargs: Any,
    ) -> BaseMessage:
        """
        메시지들을 기반으로 응답을 생성합니다 (LangChain invoke 스타일).

        Args:
            input: LangChain BaseMessage 리스트
            **kwargs: 모델 파라미터 (temperature, max_tokens, stop 등)

        Returns:
            AIMessage 응답
        """
        pass

    @abstractmethod
    async def stream(
        self, 
        input: List[BaseMessage],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        스트리밍 방식으로 응답을 생성합니다 (LangChain stream 스타일).

        Args:
            input: LangChain BaseMessage 리스트
            **kwargs: 모델 파라미터

        Yields:
            응답 텍스트 청크들
        """
        pass

    @abstractmethod
    async def batch(
        self,
        inputs: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> List[BaseMessage]:
        """
        여러 메시지 시퀀스를 배치로 처리합니다 (LangChain batch 스타일).

        Args:
            inputs: 메시지 시퀀스들의 리스트
            **kwargs: 모델 파라미터

        Returns:
            AIMessage 응답들의 리스트
        """
        pass

