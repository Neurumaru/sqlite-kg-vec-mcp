"""
Langfuse 기반 프롬프트 관리 모듈.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
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
        
        # prompts 폴더 경로
        self.prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        
        # 지원하는 프롬프트 이름들
        self.supported_prompts = [
            "query_analysis",
            "search_navigation", 
            "search_evaluation",
            "knowledge_extraction"
        ]
    
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
            # Langfuse에서 프롬프트 가져오기 시도
            if version:
                prompt = self.client.get_prompt(name=prompt_name, version=version)
            else:
                prompt = self.client.get_prompt(name=prompt_name)
            
            if variables:
                compiled_prompt = prompt.compile(**variables)
            else:
                compiled_prompt = prompt.compile()
            
            if hasattr(compiled_prompt, 'messages'):
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
                content = compiled_prompt.prompt if hasattr(compiled_prompt, 'prompt') else str(compiled_prompt)
                return {
                    "system": "",
                    "user": content
                }
                
        except Exception as e:
            logging.warning(f"Failed to get prompt from Langfuse: {e}")
            logging.info("Trying to create prompt from default if it doesn't exist")
            
            # Langfuse에 프롬프트가 없다면 default에서 생성 시도 (한 번만)
            if not hasattr(self, '_creation_attempted'):
                self._creation_attempted = set()
            
            if prompt_name not in self._creation_attempted:
                self._creation_attempted.add(prompt_name)
                try:
                    default_prompt = self._load_default_prompt(prompt_name)
                    if default_prompt and self._create_prompt_in_langfuse(prompt_name, default_prompt):
                        logging.info(f"Created prompt '{prompt_name}' in Langfuse from default")
                        # 다시 시도
                        prompt = self.client.get_prompt(name=prompt_name, version=version)
                        if variables:
                            compiled_prompt = prompt.compile(**variables)
                        else:
                            compiled_prompt = prompt.compile()
                        
                        if hasattr(compiled_prompt, 'messages'):
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
                            content = compiled_prompt.prompt if hasattr(compiled_prompt, 'prompt') else str(compiled_prompt)
                            return {
                                "system": "",
                                "user": content
                            }
                except Exception as create_error:
                    logging.error(f"Failed to create prompt in Langfuse: {create_error}")
            
            logging.warning("Using fallback prompt from file system")
            return self._get_fallback_prompt(prompt_name, variables)
    
    def _load_default_prompt(self, prompt_name: str) -> Optional[str]:
        """
        prompts 폴더에서 기본 프롬프트를 로드합니다.
        
        Args:
            prompt_name: 프롬프트 이름
            
        Returns:
            프롬프트 내용 또는 None
        """
        if prompt_name not in self.supported_prompts:
            logging.error(f"Unsupported prompt name: {prompt_name}")
            return None
        
        prompt_file = self.prompts_dir / f"{prompt_name}.md"
        
        try:
            if prompt_file.exists():
                return prompt_file.read_text(encoding='utf-8')
            else:
                logging.error(f"Prompt file not found: {prompt_file}")
                return None
        except Exception as e:
            logging.error(f"Failed to read prompt file {prompt_file}: {e}")
            return None
    
    def _create_prompt_in_langfuse(self, prompt_name: str, prompt_content: str) -> bool:
        """
        Langfuse에 프롬프트를 생성합니다.
        
        Args:
            prompt_name: 프롬프트 이름
            prompt_content: 프롬프트 내용
            
        Returns:
            성공 여부
        """
        if not self.enabled:
            return False
            
        try:
            self.client.create_prompt(
                name=prompt_name,
                prompt=prompt_content,
                labels=["interactive_search", "v1.0", "auto_created"]
            )
            logging.info(f"Successfully created prompt '{prompt_name}' in Langfuse")
            return True
        except Exception as e:
            logging.error(f"Failed to create prompt '{prompt_name}' in Langfuse: {e}")
            return False
    
    def _get_fallback_prompt(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        파일 시스템에서 기본 프롬프트를 반환합니다.
        
        Args:
            prompt_name: 프롬프트 이름
            variables: 프롬프트 변수들
            
        Returns:
            시스템 프롬프트와 사용자 프롬프트를 포함한 딕셔너리
        """
        prompt_content = self._load_default_prompt(prompt_name)
        
        if not prompt_content:
            raise ValueError(f"Cannot load default prompt for: {prompt_name}")
        
        # 간단한 변수 치환 (필요시 더 정교한 템플릿 엔진 사용 가능)
        if variables:
            try:
                formatted_content = prompt_content.format(**variables)
            except KeyError as e:
                logging.warning(f"Variable {e} not found in variables, using original content")
                formatted_content = prompt_content
        else:
            formatted_content = prompt_content
        
        # markdown에서 시스템/사용자 프롬프트 분리 (간단한 구현)
        # 더 정교한 파싱이 필요하면 markdown 파서 사용 가능
        return {
            "system": formatted_content,
            "user": "Process the above instructions with the given input."
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
    
    def create_search_trace(
        self,
        session_id: str,
        query: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        검색 세션을 위한 Langfuse Trace를 생성합니다.
        
        Args:
            session_id: 검색 세션 ID
            query: 원본 쿼리
            user_id: 사용자 ID
            metadata: 추가 메타데이터
            
        Returns:
            Langfuse Trace 객체
        """
        if not self.enabled:
            return None
        
        try:
            from datetime import datetime
            
            # Langfuse 3.x API 사용
            trace = self.client.trace(
                name="interactive_knowledge_graph_search",
                input={"original_query": query, "user_id": user_id},
                metadata={
                    "search_type": "interactive_navigation",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    **(metadata or {})
                }
            )
            return trace
        except Exception as e:
            logging.error(f"Failed to create search trace: {e}")
            return None
    
    def log_navigation_step(
        self,
        trace,
        step_number: int,
        navigation_decision: Dict[str, Any],
        execution_result: Dict[str, Any],
        evaluation: Dict[str, Any]
    ):
        """
        네비게이션 단계를 로깅합니다.
        
        Args:
            trace: Langfuse Trace 객체
            step_number: 단계 번호
            navigation_decision: LLM의 네비게이션 결정
            execution_result: 실행 결과
            evaluation: 결과 평가
        """
        if not self.enabled or not trace:
            return
        
        try:
            span = trace.span(
                name=f"navigation_step_{step_number}",
                input={
                    "decision": navigation_decision,
                    "step_number": step_number
                },
                output={
                    "execution_result": execution_result,
                    "evaluation": evaluation
                },
                metadata={
                    "step_type": navigation_decision.get("action", "unknown"),
                    "confidence": navigation_decision.get("confidence", 0.0),
                    "results_count": len(execution_result.get("findings", [])),
                    "should_continue": not evaluation.get("should_stop", False)
                }
            )
            
            # 각 단계의 품질 점수 기록
            if "relevance_score" in evaluation:
                span.score(
                    name="step_relevance",
                    value=evaluation["relevance_score"],
                    comment="이 단계에서 발견한 결과의 관련성"
                )
            
            return span
        except Exception as e:
            logging.error(f"Failed to log navigation step: {e}")
            return None
    
    def log_llm_generation(
        self,
        span,
        generation_type: str,
        prompt: Dict[str, str],
        response: str,
        model: str,
        usage: Optional[Dict[str, int]] = None
    ):
        """
        LLM 생성을 로깅합니다.
        
        Args:
            span: 상위 Span 객체
            generation_type: 생성 타입 (navigation, evaluation, analysis 등)
            prompt: 사용된 프롬프트
            response: LLM 응답
            model: 사용된 모델
            usage: 토큰 사용량
        """
        if not self.enabled or not span:
            return
        
        try:
            generation = span.generation(
                name=generation_type,
                model=model,
                input=[
                    {"role": "system", "content": prompt.get("system", "")},
                    {"role": "user", "content": prompt.get("user", "")}
                ],
                output=response,
                usage=usage or {}
            )
            return generation
        except Exception as e:
            logging.error(f"Failed to log LLM generation: {e}")
            return None
    
    def finalize_search_trace(
        self,
        trace,
        final_results: list,
        total_steps: int,
        success: bool,
        user_feedback: Optional[Dict[str, Any]] = None
    ):
        """
        검색 Trace를 완료합니다.
        
        Args:
            trace: Langfuse Trace 객체
            final_results: 최종 검색 결과
            total_steps: 총 단계 수
            success: 성공 여부
            user_feedback: 사용자 피드백
        """
        if not self.enabled or not trace:
            return
        
        try:
            trace.update(
                output={
                    "final_results": final_results,
                    "total_steps": total_steps,
                    "success": success,
                    "results_count": len(final_results)
                }
            )
            
            # 전체 검색 품질 점수
            if user_feedback:
                if "overall_satisfaction" in user_feedback:
                    trace.score(
                        name="overall_satisfaction",
                        value=user_feedback["overall_satisfaction"],
                        comment=user_feedback.get("feedback_comment", "")
                    )
                
                if "result_relevance" in user_feedback:
                    trace.score(
                        name="result_relevance",
                        value=user_feedback["result_relevance"],
                        comment="최종 결과의 관련성"
                    )
                    
                if "search_efficiency" in user_feedback:
                    trace.score(
                        name="search_efficiency", 
                        value=user_feedback["search_efficiency"],
                        comment="검색 효율성 (단계 수 대비 품질)"
                    )
        except Exception as e:
            logging.error(f"Failed to finalize search trace: {e}")


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


def get_search_navigation_prompt(
    query: str,
    current_entities: list,
    exploration_history: str,
    step_number: int
) -> Dict[str, str]:
    """
    검색 네비게이션용 프롬프트를 가져옵니다.
    
    Args:
        query: 원본 쿼리
        current_entities: 현재 발견한 엔티티들
        exploration_history: 탐색 히스토리
        step_number: 현재 단계 번호
        
    Returns:
        시스템 프롬프트와 사용자 프롬프트
    """
    manager = get_prompt_manager()
    return manager.get_prompt(
        prompt_name="search_navigation",
        variables={
            "query": query,
            "current_entities": current_entities,
            "exploration_history": exploration_history,
            "step_number": step_number
        }
    )


def get_search_evaluation_prompt(
    query: str,
    findings: list,
    context: dict,
    step_number: int
) -> Dict[str, str]:
    """
    검색 결과 평가용 프롬프트를 가져옵니다.
    
    Args:
        query: 원본 쿼리
        findings: 현재까지 발견한 결과들
        context: 탐색 컨텍스트
        step_number: 현재 단계 번호
        
    Returns:
        시스템 프롬프트와 사용자 프롬프트
    """
    manager = get_prompt_manager()
    return manager.get_prompt(
        prompt_name="search_evaluation",
        variables={
            "query": query,
            "findings": findings,
            "context": context,
            "step_number": step_number
        }
    )


def get_query_analysis_prompt(query: str) -> Dict[str, str]:
    """
    쿼리 분석용 프롬프트를 가져옵니다.
    
    Args:
        query: 분석할 쿼리
        
    Returns:
        시스템 프롬프트와 사용자 프롬프트
    """
    manager = get_prompt_manager()
    return manager.get_prompt(
        prompt_name="query_analysis",
        variables={"query": query}
    )