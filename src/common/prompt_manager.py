"""
프롬프트 관리 유틸리티.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..adapters.llm.langfuse_prompts import get_prompt_manager


def ensure_prompts_in_langfuse():
    """
    모든 default 프롬프트가 Langfuse에 있는지 확인하고, 없으면 생성합니다.
    """
    manager = get_prompt_manager()
    
    if not manager.enabled:
        logging.info("Langfuse not enabled, skipping prompt creation")
        return
    
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    
    for prompt_name in manager.supported_prompts:
        try:
            # 먼저 Langfuse에서 프롬프트 가져오기 시도
            manager.get_prompt(prompt_name)
            logging.info(f"Prompt '{prompt_name}' already exists in Langfuse")
        except Exception:
            # 프롬프트가 없으면 default에서 생성
            logging.info(f"Creating prompt '{prompt_name}' in Langfuse from default")
            prompt_file = prompts_dir / f"{prompt_name}.md"
            
            if prompt_file.exists():
                try:
                    content = prompt_file.read_text(encoding='utf-8')
                    manager._create_prompt_in_langfuse(prompt_name, content)
                    logging.info(f"Successfully created prompt '{prompt_name}'")
                except Exception as e:
                    logging.error(f"Failed to create prompt '{prompt_name}': {e}")
            else:
                logging.error(f"Default prompt file not found: {prompt_file}")


def get_interactive_search_prompt(
    prompt_type: str,
    **variables
) -> Dict[str, str]:
    """
    대화형 검색을 위한 프롬프트를 가져옵니다.
    
    Args:
        prompt_type: 프롬프트 타입 (query_analysis, search_navigation, search_evaluation, knowledge_extraction)
        **variables: 프롬프트 변수들
        
    Returns:
        시스템 프롬프트와 사용자 프롬프트를 포함한 딕셔너리
    """
    manager = get_prompt_manager()
    return manager.get_prompt(prompt_type, variables=variables)


def test_prompt_system():
    """
    프롬프트 시스템을 테스트합니다.
    """
    logging.info("Testing prompt system...")
    
    manager = get_prompt_manager()
    
    # 각 프롬프트 타입 테스트
    test_cases = [
        {
            "name": "query_analysis",
            "variables": {"query": "What are the connections between AI and healthcare?"}
        },
        {
            "name": "search_navigation", 
            "variables": {
                "query": "AI healthcare connections",
                "current_entities": ["AI", "Healthcare"],
                "exploration_history": "Started with semantic search",
                "step_number": 2
            }
        },
        {
            "name": "search_evaluation",
            "variables": {
                "query": "AI applications",
                "findings": ["Machine Learning", "Neural Networks"],
                "context": {"domain": "technology"},
                "step_number": 3
            }
        },
        {
            "name": "knowledge_extraction",
            "variables": {"text": "Dr. John Smith works at Stanford University researching artificial intelligence."}
        }
    ]
    
    for test_case in test_cases:
        try:
            logging.info(f"Testing prompt: {test_case['name']}")
            prompt = manager.get_prompt(test_case["name"], variables=test_case["variables"])
            
            if prompt and "system" in prompt:
                logging.info(f"✅ {test_case['name']}: Success (system: {len(prompt['system'])} chars)")
            else:
                logging.error(f"❌ {test_case['name']}: Failed - invalid response")
                
        except Exception as e:
            logging.error(f"❌ {test_case['name']}: Failed - {e}")
    
    logging.info("Prompt system test completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 프롬프트 시스템 테스트
    test_prompt_system()
    
    # Langfuse에 프롬프트 생성 (필요시)
    ensure_prompts_in_langfuse()