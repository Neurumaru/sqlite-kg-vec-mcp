"""
LLM 서비스 포트.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    """메시지 역할."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """LLM 메시지."""
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """LLM 응답."""
    content: str
    model_name: str
    token_usage: Optional[Dict[str, int]] = None
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeExtractionPrompt:
    """지식 추출 프롬프트."""
    system_prompt: str
    user_prompt_template: str
    expected_format: str
    examples: Optional[List[Dict[str, str]]] = None


class LLMService(ABC):
    """
    LLM 서비스 포트.
    
    문서에서 지식을 추출하기 위한 LLM 기능을 제공합니다.
    """
    
    @abstractmethod
    async def generate_response(self, messages: List[LLMMessage], 
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None) -> LLMResponse:
        """
        메시지들을 기반으로 응답을 생성합니다.
        
        Args:
            messages: 대화 메시지들
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
            
        Returns:
            LLM 응답
        """
        pass
    
    @abstractmethod
    async def extract_entities(self, text: str, 
                             entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        텍스트에서 엔티티를 추출합니다.
        
        Args:
            text: 추출할 텍스트
            entity_types: 추출할 엔티티 타입들
            
        Returns:
            추출된 엔티티들
        """
        pass
    
    @abstractmethod
    async def extract_relationships(self, text: str, 
                                  entities: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        텍스트에서 관계를 추출합니다.
        
        Args:
            text: 추출할 텍스트
            entities: 이미 추출된 엔티티들
            
        Returns:
            추출된 관계들
        """
        pass
    
    @abstractmethod
    async def extract_knowledge(self, text: str) -> Dict[str, Any]:
        """
        텍스트에서 지식(엔티티 + 관계)을 통합 추출합니다.
        
        Args:
            text: 추출할 텍스트
            
        Returns:
            추출된 지식 {"entities": [...], "relationships": [...]}
        """
        pass
    
    @abstractmethod
    async def summarize_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        텍스트를 요약합니다.
        
        Args:
            text: 요약할 텍스트
            max_length: 최대 요약 길이
            
        Returns:
            요약된 텍스트
        """
        pass
    
    @abstractmethod
    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """
        텍스트를 분류합니다.
        
        Args:
            text: 분류할 텍스트
            categories: 분류 카테고리들
            
        Returns:
            카테고리별 점수
        """
        pass
    
    @abstractmethod
    async def generate_questions(self, text: str, num_questions: int = 5) -> List[str]:
        """
        텍스트를 기반으로 질문을 생성합니다.
        
        Args:
            text: 기반 텍스트
            num_questions: 생성할 질문 수
            
        Returns:
            생성된 질문들
        """
        pass
    
    @abstractmethod
    async def answer_question(self, question: str, context: str) -> str:
        """
        컨텍스트를 기반으로 질문에 답변합니다.
        
        Args:
            question: 질문
            context: 컨텍스트
            
        Returns:
            답변
        """
        pass
    
    @abstractmethod
    async def stream_response(self, messages: List[LLMMessage],
                            temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """
        스트리밍 방식으로 응답을 생성합니다.
        
        Args:
            messages: 대화 메시지들
            temperature: 생성 온도
            
        Yields:
            응답 청크들
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        사용 중인 모델명을 반환합니다.
        
        Returns:
            모델명
        """
        pass
    
    @abstractmethod
    def get_max_context_length(self) -> int:
        """
        모델의 최대 컨텍스트 길이를 반환합니다.
        
        Returns:
            최대 컨텍스트 길이
        """
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """
        LLM 서비스가 사용 가능한지 확인합니다.
        
        Returns:
            사용 가능 여부
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        텍스트의 토큰 수를 추정합니다.
        
        Args:
            text: 추정할 텍스트
            
        Returns:
            추정 토큰 수
        """
        pass
    
    @abstractmethod
    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        토큰 제한에 맞게 텍스트를 자릅니다.
        
        Args:
            text: 원본 텍스트
            max_tokens: 최대 토큰 수
            
        Returns:
            잘린 텍스트
        """
        pass
    
    @abstractmethod
    async def get_usage_statistics(self) -> Dict[str, Any]:
        """
        사용량 통계를 반환합니다.
        
        Returns:
            사용량 통계
        """
        pass
    
    @abstractmethod
    async def validate_response_format(self, response: str, 
                                     expected_format: str) -> bool:
        """
        응답 형식이 예상과 일치하는지 검증합니다.
        
        Args:
            response: 검증할 응답
            expected_format: 예상 형식
            
        Returns:
            형식 일치 여부
        """
        pass