"""
Langfuse 기반 프롬프트 관리 모듈.
"""

import os
import logging
from typing import Dict, Any, Optional
from langfuse import Langfuse


class LangfusePromptManager:
    """Langfuse를 사용한 프롬프트 관리자."""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        public_key: Optional[str] = None,
        host: Optional[str] = None
    ):
        """
        Langfuse 프롬프트 관리자 초기화.
        
        Args:
            secret_key: Langfuse secret key (환경변수 LANGFUSE_SECRET_KEY에서도 읽음)
            public_key: Langfuse public key (환경변수 LANGFUSE_PUBLIC_KEY에서도 읽음)
            host: Langfuse host URL (환경변수 LANGFUSE_HOST에서도 읽음)
        """
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        # Langfuse 클라이언트 초기화
        if self.secret_key and self.public_key:
            self.client = Langfuse(
                secret_key=self.secret_key,
                public_key=self.public_key,
                host=self.host
            )
            self.enabled = True
            logging.info("Langfuse prompt manager initialized successfully")
        else:
            self.client = None
            self.enabled = False
            logging.warning(
                "Langfuse credentials not found. "
                "Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY environment variables "
                "to enable Langfuse prompt management."
            )
        
        # 기본 프롬프트 템플릿 (Langfuse가 비활성화된 경우 사용)
        self.fallback_prompts = {
            "knowledge_extraction": {
                "system": """당신은 텍스트에서 지식을 추출하는 전문 AI입니다. 주어진 텍스트를 분석하여 개체(entities)와 관계(relationships)를 JSON 형태로 추출해주세요.

개체는 다음과 같은 유형이 될 수 있습니다:
- Person: 사람
- Organization: 조직, 기관, 회사
- Location: 장소, 지역
- Event: 사건, 행사
- Concept: 개념, 이론
- Product: 제품, 서비스
- Award: 상, 상금
- Date: 날짜, 시간

관계는 개체들 간의 연결을 나타냅니다. 예: "worked_at", "founded", "located_in", "awarded", "developed" 등

응답은 반드시 다음 JSON 형식을 따라주세요:
```json
{
  "entities": [
    {
      "id": "고유식별자",
      "name": "개체명",
      "type": "개체유형",
      "properties": {"추가속성": "값"}
    }
  ],
  "relationships": [
    {
      "source": "출발개체ID",
      "target": "도착개체ID", 
      "type": "관계유형",
      "properties": {"추가속성": "값"}
    }
  ]
}
```""",
                "user": "다음 텍스트에서 개체와 관계를 추출해주세요:\n\n{text}"
            }
        }
    
    def get_prompt(
        self,
        prompt_name: str,
        version: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Langfuse에서 프롬프트를 가져옵니다.
        
        Args:
            prompt_name: 프롬프트 이름
            version: 프롬프트 버전 (지정하지 않으면 최신 버전)
            variables: 프롬프트 변수들
            
        Returns:
            시스템 프롬프트와 사용자 프롬프트를 포함한 딕셔너리
        """
        if not self.enabled:
            logging.warning("Langfuse not enabled, using fallback prompt")
            return self._get_fallback_prompt(prompt_name, variables)
        
        try:
            # Langfuse에서 프롬프트 가져오기
            if version:
                prompt = self.client.get_prompt(name=prompt_name, version=version)
            else:
                prompt = self.client.get_prompt(name=prompt_name)
            
            # 프롬프트 컴파일 (변수 치환)
            if variables:
                compiled_prompt = prompt.compile(**variables)
            else:
                compiled_prompt = prompt.compile()
            
            # 시스템 프롬프트와 사용자 프롬프트 분리
            if hasattr(compiled_prompt, 'messages'):
                # 멀티턴 대화 형식
                system_prompt = ""
                user_prompt = ""
                
                for message in compiled_prompt.messages:
                    if message.get('role') == 'system':
                        system_prompt = message.get('content', '')
                    elif message.get('role') == 'user':
                        user_prompt = message.get('content', '')
                
                return {
                    "system": system_prompt,
                    "user": user_prompt
                }
            else:
                # 단일 프롬프트 형식
                content = compiled_prompt.prompt if hasattr(compiled_prompt, 'prompt') else str(compiled_prompt)
                return {
                    "system": "",
                    "user": content
                }
                
        except Exception as e:
            logging.error(f"Failed to get prompt from Langfuse: {e}")
            logging.warning("Falling back to default prompt")
            return self._get_fallback_prompt(prompt_name, variables)
    
    def _get_fallback_prompt(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        기본 프롬프트를 반환합니다.
        
        Args:
            prompt_name: 프롬프트 이름
            variables: 프롬프트 변수들
            
        Returns:
            시스템 프롬프트와 사용자 프롬프트를 포함한 딕셔너리
        """
        if prompt_name not in self.fallback_prompts:
            raise ValueError(f"Unknown prompt name: {prompt_name}")
        
        prompt_template = self.fallback_prompts[prompt_name]
        
        # 변수 치환
        if variables:
            system_prompt = prompt_template["system"]
            user_prompt = prompt_template["user"].format(**variables)
        else:
            system_prompt = prompt_template["system"]
            user_prompt = prompt_template["user"]
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def create_or_update_prompt(
        self,
        name: str,
        prompt_text: str,
        config: Optional[Dict[str, Any]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Langfuse에 프롬프트를 생성하거나 업데이트합니다.
        
        Args:
            name: 프롬프트 이름
            prompt_text: 프롬프트 텍스트
            config: 프롬프트 설정
            labels: 프롬프트 라벨
            
        Returns:
            성공 여부
        """
        if not self.enabled:
            logging.warning("Langfuse not enabled, cannot create/update prompt")
            return False
        
        try:
            self.client.create_prompt(
                name=name,
                prompt=prompt_text,
                config=config or {},
                labels=labels or {}
            )
            logging.info(f"Successfully created/updated prompt: {name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create/update prompt {name}: {e}")
            return False
    
    def log_prompt_usage(
        self,
        prompt_name: str,
        version: Optional[str],
        input_variables: Dict[str, Any],
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        프롬프트 사용을 로깅합니다.
        
        Args:
            prompt_name: 사용된 프롬프트 이름
            version: 사용된 프롬프트 버전
            input_variables: 입력 변수들
            response: LLM 응답
            metadata: 추가 메타데이터
        """
        if not self.enabled:
            return
        
        try:
            # 로깅 (향후 개선 예정)
            logging.info(f"Prompt usage logged: {prompt_name}, entities: {metadata.get('entities_count', 0)}, relationships: {metadata.get('relationships_count', 0)}")
        except Exception as e:
            logging.error(f"Failed to log prompt usage: {e}")


# 전역 프롬프트 관리자 인스턴스
_prompt_manager = None


def get_prompt_manager() -> LangfusePromptManager:
    """전역 프롬프트 관리자 인스턴스를 반환합니다."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = LangfusePromptManager()
    return _prompt_manager


def get_knowledge_extraction_prompt(text: str) -> Dict[str, str]:
    """
    지식 추출용 프롬프트를 가져옵니다.
    
    Args:
        text: 분석할 텍스트
        
    Returns:
        시스템 프롬프트와 사용자 프롬프트
    """
    manager = get_prompt_manager()
    return manager.get_prompt(
        prompt_name="knowledge_extraction",
        variables={"text": text}
    )