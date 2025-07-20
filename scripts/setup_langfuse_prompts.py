#!/usr/bin/env python3
"""
Langfuse에 지식그래프 추출용 프롬프트를 생성하는 스크립트.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlite_kg_vec_mcp.llm.langfuse_prompts import LangfusePromptManager


def setup_knowledge_extraction_prompt():
    """지식 추출용 프롬프트를 Langfuse에 생성합니다."""
    
    # 환경변수 설정
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-a4337e80-2ca0-4f79-9443-91e3730c1be5"
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-f4e7f521-9f22-41a5-9859-75b9904b8ece"
    os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"
    
    # 프롬프트 매니저 초기화
    manager = LangfusePromptManager()
    
    if not manager.enabled:
        print("❌ Langfuse 설정에 실패했습니다.")
        return False
    
    # 지식 추출 프롬프트 생성
    prompt_config = {
        "model": "gemma3n",
        "temperature": 0.1,
        "max_tokens": 2000
    }
    
    # 단일 프롬프트 문자열
    knowledge_extraction_prompt = """당신은 텍스트에서 지식을 추출하는 전문 AI입니다. 주어진 텍스트를 분석하여 개체(entities)와 관계(relationships)를 JSON 형태로 추출해주세요.

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
```

정확하고 명시적으로 언급된 정보만 추출하며, JSON 외의 추가 텍스트나 설명은 포함하지 마세요.

다음 텍스트에서 개체와 관계를 추출해주세요:

{{text}}"""
    
    try:
        # 프롬프트 생성
        success = manager.client.create_prompt(
            name="knowledge_extraction",
            prompt=knowledge_extraction_prompt,
            config=prompt_config,
            labels=["production"]
        )
        
        print("✅ 지식 추출 프롬프트가 Langfuse에 성공적으로 생성되었습니다!")
        print(f"   프롬프트 이름: knowledge_extraction")
        print(f"   설정: {prompt_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ 프롬프트 생성 실패: {e}")
        return False


def main():
    """메인 실행 함수"""
    print("🚀 Langfuse 프롬프트 설정 시작...")
    
    success = setup_knowledge_extraction_prompt()
    
    if success:
        print("\n✅ 모든 프롬프트가 성공적으로 설정되었습니다!")
        print("\n이제 지식그래프 추출 시 Langfuse 프롬프트가 사용됩니다.")
        print("Langfuse 대시보드에서 프롬프트 사용 현황을 확인할 수 있습니다.")
    else:
        print("\n❌ 프롬프트 설정에 실패했습니다.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())