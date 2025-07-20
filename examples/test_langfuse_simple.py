#!/usr/bin/env python3
"""
Langfuse 통합 테스트를 위한 간단한 예제.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlite_kg_vec_mcp.db.connection import DatabaseConnection
from sqlite_kg_vec_mcp.db.schema import SchemaManager
from sqlite_kg_vec_mcp.llm.ollama_client import OllamaClient
from sqlite_kg_vec_mcp.llm.langfuse_prompts import get_prompt_manager


def main():
    """간단한 Langfuse 테스트"""
    
    # Langfuse 환경변수 설정
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-a4337e80-2ca0-4f79-9443-91e3730c1be5"
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-f4e7f521-9f22-41a5-9859-75b9904b8ece"
    os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"
    
    print("🔧 Langfuse 테스트 시작...")
    
    # 프롬프트 매니저 테스트
    prompt_manager = get_prompt_manager()
    print(f"Langfuse 활성화: {prompt_manager.enabled}")
    
    if not prompt_manager.enabled:
        print("❌ Langfuse가 비활성화되어 있습니다.")
        return
    
    # Ollama 클라이언트 초기화
    print("🤖 Ollama 클라이언트 초기화...")
    ollama_client = OllamaClient(model="gemma3n")
    
    # 간단한 텍스트로 테스트
    test_text = "알버트 아인슈타인은 상대성 이론을 개발했습니다."
    
    print(f"📝 테스트 텍스트: {test_text}")
    print("🔍 지식 추출 시작...")
    
    try:
        # 지식 추출 실행
        result = ollama_client.extract_entities_and_relationships(test_text)
        
        print("✅ 지식 추출 완료!")
        print(f"개체 수: {len(result.get('entities', []))}")
        print(f"관계 수: {len(result.get('relationships', []))}")
        
        # 결과 출력
        if result.get('entities'):
            print("\n📊 추출된 개체들:")
            for entity in result['entities']:
                print(f"  - {entity.get('name')} ({entity.get('type')})")
        
        if result.get('relationships'):
            print("\n🔗 추출된 관계들:")
            for rel in result['relationships']:
                print(f"  - {rel.get('source')} → {rel.get('type')} → {rel.get('target')}")
        
        print("\n✅ Langfuse 통합 테스트 성공!")
        print("Langfuse 대시보드에서 로깅 데이터를 확인하세요.")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())