"""
LLM 기반 Interactive Knowledge Graph Search 모듈.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

from src.adapters.llm.langfuse_prompts import (
    LangfusePromptManager,
    get_search_navigation_prompt,
    get_search_evaluation_prompt,
    get_query_analysis_prompt
)


class SearchContext:
    """검색 컨텍스트를 관리하는 클래스."""
    
    def __init__(self, original_query: str):
        self.original_query = original_query
        self.entities = []
        self.relationships = []
        self.history = []
        self.current_step = 0
        self.metadata = {}
    
    def add_findings(self, entities: List[Dict], relationships: List[Dict]):
        """새로운 발견 사항을 추가합니다."""
        self.entities.extend(entities)
        self.relationships.extend(relationships)
    
    def add_history_step(self, step_info: Dict):
        """탐색 히스토리에 단계를 추가합니다."""
        self.history.append(step_info)
        self.current_step += 1
    
    def get_history_summary(self) -> str:
        """탐색 히스토리 요약을 반환합니다."""
        if not self.history:
            return "탐색 시작"
        
        summary_parts = []
        for i, step in enumerate(self.history[-3:]):  # 최근 3단계만
            action = step.get("action", "unknown")
            result_count = step.get("result_count", 0)
            summary_parts.append(f"Step {i+1}: {action} -> {result_count}개 결과")
        
        return " | ".join(summary_parts)
    
    def get_entity_names(self) -> List[str]:
        """현재 발견한 엔티티 이름 목록을 반환합니다."""
        return [entity.get("name", entity.get("id", "unknown")) for entity in self.entities]


class InteractiveSearchEngine:
    """LLM 기반 Interactive 검색 엔진."""
    
    def __init__(
        self,
        knowledge_graph,
        llm_client,
        max_steps: int = 10,
        enable_langfuse: bool = True
    ):
        """
        Interactive 검색 엔진 초기화.
        
        Args:
            knowledge_graph: 지식 그래프 인스턴스
            llm_client: LLM 클라이언트
            max_steps: 최대 탐색 단계 수
            enable_langfuse: Langfuse 추적 활성화 여부
        """
        self.kg = knowledge_graph
        self.llm = llm_client
        self.max_steps = max_steps
        self.enable_langfuse = enable_langfuse
        
        # Langfuse 프롬프트 매니저
        self.prompt_manager = LangfusePromptManager() if enable_langfuse else None
        
        self.logger = logging.getLogger(__name__)
    
    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Interactive 검색을 수행합니다.
        
        Args:
            query: 검색 쿼리
            user_id: 사용자 ID
            session_metadata: 세션 메타데이터
            
        Returns:
            검색 결과 및 메타데이터
        """
        session_id = str(uuid.uuid4())
        context = SearchContext(query)
        
        # Langfuse 추적 시작
        trace = None
        if self.prompt_manager:
            trace = self.prompt_manager.create_search_trace(
                session_id=session_id,
                query=query,
                user_id=user_id,
                metadata=session_metadata
            )
        
        try:
            # 1. 쿼리 분석
            query_analysis = await self._analyze_query(query, trace)
            context.metadata["query_analysis"] = query_analysis
            
            # 2. Interactive 탐색 수행
            for step in range(self.max_steps):
                self.logger.info(f"Interactive search step {step + 1}/{self.max_steps}")
                
                # 네비게이션 결정
                navigation_decision = await self._get_navigation_decision(
                    context, step, trace
                )
                
                if navigation_decision.get("action") == "CONCLUDE":
                    self.logger.info("LLM decided to conclude search")
                    break
                
                # 탐색 실행
                execution_result = await self._execute_navigation(
                    navigation_decision, context, trace
                )
                
                # 결과 평가
                evaluation = await self._evaluate_results(
                    context, execution_result, step, trace
                )
                
                # 컨텍스트 업데이트
                context.add_findings(
                    execution_result.get("entities", []),
                    execution_result.get("relationships", [])
                )
                
                context.add_history_step({
                    "step": step,
                    "action": navigation_decision.get("action"),
                    "result_count": len(execution_result.get("entities", [])),
                    "evaluation": evaluation
                })
                
                # Langfuse 로깅
                if self.prompt_manager and trace:
                    self.prompt_manager.log_navigation_step(
                        trace=trace,
                        step_number=step,
                        navigation_decision=navigation_decision,
                        execution_result=execution_result,
                        evaluation=evaluation
                    )
                
                # 종료 조건 체크
                if evaluation.get("should_stop", False):
                    self.logger.info("Evaluation suggests stopping search")
                    break
            
            # 3. 최종 결과 정리
            final_results = self._prepare_final_results(context)
            
            # Langfuse 추적 완료
            if self.prompt_manager and trace:
                self.prompt_manager.finalize_search_trace(
                    trace=trace,
                    final_results=final_results["entities"],
                    total_steps=context.current_step,
                    success=True
                )
            
            return {
                "session_id": session_id,
                "original_query": query,
                "final_results": final_results,
                "total_steps": context.current_step,
                "search_metadata": context.metadata,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Interactive search failed: {e}")
            
            # 실패 시 Langfuse 추적 완료
            if self.prompt_manager and trace:
                self.prompt_manager.finalize_search_trace(
                    trace=trace,
                    final_results=[],
                    total_steps=context.current_step,
                    success=False
                )
            
            return {
                "session_id": session_id,
                "original_query": query,
                "final_results": {"entities": [], "relationships": []},
                "total_steps": context.current_step,
                "error": str(e),
                "success": False
            }
    
    async def _analyze_query(
        self,
        query: str,
        trace = None
    ) -> Dict[str, Any]:
        """쿼리를 분석하여 검색 전략을 수립합니다."""
        try:
            prompt = get_query_analysis_prompt(query)
            
            # LLM 호출
            response = await self.llm.generate(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"]
            )
            
            # Langfuse 로깅
            if self.prompt_manager and trace:
                span = trace.span(
                    name="query_analysis",
                    input={"query": query},
                    output={"analysis": response}
                )
                
                self.prompt_manager.log_llm_generation(
                    span=span,
                    generation_type="query_analysis",
                    prompt=prompt,
                    response=response,
                    model=self.llm.model_name
                )
            
            return json.loads(response)
            
        except Exception as e:
            self.logger.error(f"Query analysis failed: {e}")
            return {
                "query_type": "CONCEPT_SEMANTIC",
                "search_strategy": "SEMANTIC_FIRST",
                "reasoning": f"Fallback due to analysis error: {e}"
            }
    
    async def _get_navigation_decision(
        self,
        context: SearchContext,
        step: int,
        trace = None
    ) -> Dict[str, Any]:
        """다음 네비게이션 결정을 가져옵니다."""
        try:
            prompt = get_search_navigation_prompt(
                query=context.original_query,
                current_entities=context.get_entity_names(),
                exploration_history=context.get_history_summary(),
                step_number=step
            )
            
            # LLM 호출
            response = await self.llm.generate(
                system_prompt=prompt["system"],  
                user_prompt=prompt["user"]
            )
            
            navigation_decision = json.loads(response)
            
            # Langfuse 로깅
            if self.prompt_manager and trace:
                generation = self.prompt_manager.log_llm_generation(
                    span=trace,
                    generation_type="navigation_decision",
                    prompt=prompt,
                    response=response,
                    model=self.llm.model_name
                )
            
            return navigation_decision
            
        except Exception as e:
            self.logger.error(f"Navigation decision failed: {e}")
            # 기본 행동 반환
            return {
                "action": "SEMANTIC_EXPAND",
                "reasoning": f"Fallback due to decision error: {e}",
                "parameters": {"max_results": 10},
                "confidence": 0.5
            }
    
    async def _execute_navigation(
        self,
        navigation: Dict[str, Any],
        context: SearchContext,
        trace = None
    ) -> Dict[str, Any]:
        """네비게이션 결정을 실행합니다."""
        action = navigation.get("action", "SEMANTIC_EXPAND")
        parameters = navigation.get("parameters", {})
        
        try:
            if action == "SEMANTIC_EXPAND":
                return await self._semantic_expand(context, parameters)
            elif action == "STRUCTURAL_TRAVERSE":
                return await self._structural_traverse(context, parameters)
            elif action == "FILTER_REFINE":
                return await self._filter_refine(context, parameters)
            elif action == "PIVOT_STRATEGY":
                return await self._pivot_strategy(context, parameters)
            else:
                return {"entities": [], "relationships": []}
                
        except Exception as e:
            self.logger.error(f"Navigation execution failed: {e}")
            return {"entities": [], "relationships": [], "error": str(e)}
    
    async def _semantic_expand(
        self,
        context: SearchContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """의미적 확장 검색을 수행합니다."""
        # 벡터 유사도 검색 수행
        max_results = parameters.get("max_results", 10)
        threshold = parameters.get("similarity_threshold", 0.7)
        
        # 쿼리 텍스트 또는 기존 엔티티들을 기반으로 검색
        if context.entities:
            # 기존 엔티티들의 평균 벡터로 검색
            results = await self.kg.search_similar_nodes_by_entities(
                entity_ids=[e.get("id") for e in context.entities[-3:]],  # 최근 3개
                limit=max_results,
                threshold=threshold
            )
        else:
            # 원본 쿼리로 검색
            results = await self.kg.search_by_text(
                query=context.original_query,
                limit=max_results
            )
        
        return {
            "entities": [r.entity.to_dict() for r in results],
            "relationships": [],
            "method": "semantic_expand"
        }
    
    async def _structural_traverse(
        self,
        context: SearchContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """구조적 순회를 수행합니다."""
        if not context.entities:
            return {"entities": [], "relationships": []}
        
        direction = parameters.get("direction", "outgoing")
        relation_types = parameters.get("relation_types", [])
        max_results = parameters.get("max_results", 20)
        
        all_entities = []
        all_relationships = []
        
        # 최근 발견한 엔티티들에서 탐색
        for entity in context.entities[-5:]:  # 최근 5개 엔티티
            entity_id = entity.get("id")
            if not entity_id:
                continue
            
            neighbors = await self.kg.get_neighbors(
                node_id=entity_id,
                relation_types=relation_types if relation_types else None,
                direction=direction,
                limit=max_results // len(context.entities[-5:])
            )
            
            all_entities.extend([n.to_dict() for n in neighbors])
            # 관계 정보도 수집 (실제 구현 필요)
        
        return {
            "entities": all_entities[:max_results],
            "relationships": all_relationships,
            "method": "structural_traverse"
        }
    
    async def _filter_refine(
        self,
        context: SearchContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """결과를 필터링하고 정제합니다."""
        # 현재 결과에서 조건에 맞는 것만 필터링
        entity_types = parameters.get("entity_types", [])
        
        filtered_entities = []
        if entity_types:
            filtered_entities = [
                e for e in context.entities
                if e.get("type") in entity_types
            ]
        else:
            filtered_entities = context.entities
        
        return {
            "entities": filtered_entities,
            "relationships": context.relationships,
            "method": "filter_refine"
        }
    
    async def _pivot_strategy(
        self,
        context: SearchContext,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """전략을 전환합니다."""
        # 원본 쿼리에 대해 다른 방식으로 검색
        return await self._semantic_expand(context, parameters)
    
    async def _evaluate_results(
        self,
        context: SearchContext,
        execution_result: Dict[str, Any],
        step: int,
        trace = None
    ) -> Dict[str, Any]:
        """결과를 평가합니다."""
        try:
            prompt = get_search_evaluation_prompt(
                query=context.original_query,
                findings=execution_result.get("entities", []),
                context={
                    "total_entities": len(context.entities),
                    "total_relationships": len(context.relationships),
                    "history": context.get_history_summary()
                },
                step_number=step
            )
            
            # LLM 호출
            response = await self.llm.generate(
                system_prompt=prompt["system"],
                user_prompt=prompt["user"]
            )
            
            evaluation = json.loads(response)
            
            # Langfuse 로깅
            if self.prompt_manager and trace:
                self.prompt_manager.log_llm_generation(
                    span=trace,
                    generation_type="result_evaluation",
                    prompt=prompt,
                    response=response,
                    model=self.llm.model_name
                )
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Result evaluation failed: {e}")
            # 기본 평가 반환
            return {
                "relevance_score": 0.5,
                "should_continue": step < self.max_steps - 1,
                "should_stop": False,
                "feedback": f"Evaluation error: {e}"
            }
    
    def _prepare_final_results(self, context: SearchContext) -> Dict[str, Any]:
        """최종 결과를 정리합니다."""
        # 중복 제거 및 정렬
        unique_entities = {}
        for entity in context.entities:
            entity_id = entity.get("id")
            if entity_id and entity_id not in unique_entities:
                unique_entities[entity_id] = entity
        
        unique_relationships = {}
        for rel in context.relationships:
            rel_key = f"{rel.get('source')}_{rel.get('target')}_{rel.get('type')}"
            if rel_key not in unique_relationships:
                unique_relationships[rel_key] = rel
        
        return {
            "entities": list(unique_entities.values()),
            "relationships": list(unique_relationships.values()),
            "metadata": {
                "total_entities": len(unique_entities),
                "total_relationships": len(unique_relationships),
                "search_steps": context.current_step
            }
        }