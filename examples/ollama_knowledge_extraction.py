#!/usr/bin/env python3
"""
Ollama Gemma3와 Nomic Embed Text를 사용한 지식그래프 구축 예제.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlite_kg_vec_mcp.db.connection import DatabaseConnection
from sqlite_kg_vec_mcp.db.schema import SchemaManager
from sqlite_kg_vec_mcp.llm.ollama_client import OllamaClient
from sqlite_kg_vec_mcp.llm.knowledge_extractor import KnowledgeExtractor
from sqlite_kg_vec_mcp.vector.text_embedder import create_embedder
from sqlite_kg_vec_mcp.vector.embeddings import EmbeddingManager
from sqlite_kg_vec_mcp.vector.search import VectorSearch


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def setup_langfuse():
    """Langfuse 환경변수 설정"""
    # Langfuse API 키 설정 (예제용)
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-a4337e80-2ca0-4f79-9443-91e3730c1be5"
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-f4e7f521-9f22-41a5-9859-75b9904b8ece"
    if not os.getenv("LANGFUSE_HOST"):
        os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"
    
    print("🔧 Langfuse 설정 완료")
    print(f"   Host: {os.getenv('LANGFUSE_HOST')}")
    print(f"   Public Key: {os.getenv('LANGFUSE_PUBLIC_KEY')[:20]}...")
    print(f"   Secret Key: {'*' * 20}...")


def check_ollama_models():
    """Ollama 모델 확인 및 설치"""
    try:
        ollama_client = OllamaClient()
        available_models = ollama_client.list_available_models()
        
        print("사용 가능한 Ollama 모델:")
        for model in available_models:
            print(f"  - {model}")
        
        # Gemma3n 모델 확인
        gemma3n_available = any("gemma3n" in model for model in available_models)
        
        if not gemma3n_available:
            print("\n❌ Gemma3n 모델이 없습니다.")
            print("다음 명령어로 설치하세요:")
            print("  ollama pull gemma3n")
            return False
        
        # Nomic 모델은 선택사항으로 처리
        nomic_available = any("nomic-embed-text" in model for model in available_models)
        if not nomic_available:
            print("\n⚠️ nomic-embed-text 모델이 없습니다. sentence-transformers를 사용합니다.")
            print("Nomic 모델을 사용하려면: ollama pull nomic-embed-text")
        
        print("✅ 모든 필요한 모델이 설치되어 있습니다.")
        return True
        
    except Exception as e:
        print(f"❌ Ollama 연결 실패: {e}")
        print("Ollama 서버가 실행 중인지 확인하세요: ollama serve")
        return False


def main():
    """메인 실행 함수"""
    setup_logging()
    setup_langfuse()
    
    # Ollama 모델 확인
    if not check_ollama_models():
        return
    
    # 데이터베이스 설정
    db_path = "ollama_knowledge_example.db"
    
    # 기존 데이터베이스 삭제 (테스트용)
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print(f"\n📊 데이터베이스 초기화: {db_path}")
    
    # 데이터베이스 연결 및 스키마 초기화
    db_connection = DatabaseConnection(db_path)
    connection = db_connection.connect()
    
    schema_manager = SchemaManager(db_path)
    schema_manager.initialize_schema()
    
    # Ollama 클라이언트 초기화
    print("🤖 Ollama 클라이언트 초기화...")
    ollama_client = OllamaClient(model="gemma3n")
    
    # Nomic 임베더 초기화
    print("🧮 Nomic 임베더 초기화...")
    try:
        nomic_embedder = create_embedder("nomic")
        print(f"임베딩 차원: {nomic_embedder.dimension}")
    except Exception as e:
        print(f"Nomic 임베더 초기화 실패: {e}")
        print("Sentence Transformers로 대체합니다...")
        nomic_embedder = create_embedder("sentence-transformers")
    
    # 임베딩 매니저 초기화
    embedding_manager = EmbeddingManager(connection)
    embedding_manager.text_embedder = nomic_embedder
    
    # 벡터 검색 초기화
    print(f"벡터 검색 초기화 (차원: {nomic_embedder.dimension})...")
    vector_search = VectorSearch(
        connection=connection,
        embedding_dim=nomic_embedder.dimension,
        space="cosine",
        index_dir="ollama_example_index",  # 새로운 인덱스 디렉토리 사용
        text_embedder=nomic_embedder  # 임베더 직접 전달
    )
    
    # 지식 추출기 초기화
    knowledge_extractor = KnowledgeExtractor(
        connection=connection,
        ollama_client=ollama_client,
        auto_embed=True
    )
    knowledge_extractor.embedding_manager = embedding_manager
    
    # 예제 텍스트들
    sample_texts = [
        """
        알버트 아인슈타인은 1879년 독일에서 태어난 이론물리학자입니다. 
        그는 상대성 이론으로 유명하며, 1921년 노벨 물리학상을 수상했습니다. 
        아인슈타인은 프린스턴 대학교에서 연구했으며, 현대 물리학의 아버지로 불립니다.
        """,
        """
        마리 퀴리는 1867년 폴란드에서 태어난 물리학자이자 화학자입니다.
        그녀는 방사능 연구의 선구자였으며, 노벨상을 두 번 수상한 최초의 여성입니다.
        1903년 물리학상과 1911년 화학상을 받았으며, 라듐과 폴로늄을 발견했습니다.
        """,
        """
        스탠포드 대학교는 1885년 캘리포니아에 설립된 명문 사립대학입니다.
        실리콘밸리의 중심에 위치하여 기술 혁신의 허브 역할을 하고 있습니다.
        구글, 야후, 넷플릭스 등 많은 기술 기업의 창업자들이 이 대학 출신입니다.
        """,
        """
        인공지능은 인간의 지능을 모방하는 컴퓨터 시스템입니다.
        머신러닝과 딥러닝 기술의 발전으로 많은 분야에서 활용되고 있습니다.
        자연어 처리, 컴퓨터 비전, 로보틱스 등 다양한 응용 분야가 있습니다.
        """
    ]
    
    # 지식 추출 실행
    print("\n🔍 지식 추출 시작...")
    all_results = []
    
    for i, text in enumerate(sample_texts):
        print(f"\n--- 텍스트 {i+1} 처리 중 ---")
        print(f"내용: {text.strip()[:100]}...")
        
        result = knowledge_extractor.extract_from_text(
            text, 
            source_id=f"doc_{i+1}",
            enhance_descriptions=True
        )
        
        all_results.append(result)
        
        print(f"✅ 생성된 개체: {result.entities_created}")
        print(f"✅ 생성된 관계: {result.relationships_created}")
        print(f"⏱️ 처리 시간: {result.processing_time:.2f}초")
        
        if result.errors:
            print(f"⚠️ 오류: {len(result.errors)}개")
            for error in result.errors[:3]:  # 처음 3개만 표시
                print(f"   - {error}")
    
    # 통계 출력
    print("\n📈 전체 통계:")
    stats = knowledge_extractor.get_extraction_statistics()
    
    print(f"총 개체 수: {stats['entities']['total']}")
    print("개체 유형별:")
    for entity_type, count in stats['entities']['by_type'].items():
        print(f"  - {entity_type}: {count}")
    
    print(f"\n총 관계 수: {stats['relationships']['total']}")
    print("관계 유형별:")
    for rel_type, count in stats['relationships']['by_type'].items():
        print(f"  - {rel_type}: {count}")
    
    if 'total_embeddings' in stats['embeddings']:
        print(f"\n총 임베딩 수: {stats['embeddings']['total_embeddings']}")
    
    print(f"\n사용된 모델: {stats['model']}")
    
    # 벡터 검색 테스트
    print("\n🔍 벡터 검색 테스트:")
    
    # 벡터 인덱스 업데이트
    vector_search.update_index()
    
    # 검색 쿼리들
    test_queries = [
        "노벨상을 받은 과학자",
        "대학교와 교육기관", 
        "물리학과 화학 연구",
        "인공지능과 기술"
    ]
    
    for query in test_queries:
        print(f"\n쿼리: '{query}'")
        
        try:
            results = vector_search.search_by_text(
                query_text=query,
                k=3,
                include_entities=True
            )
            
            if results:
                print("검색 결과:")
                for i, result in enumerate(results[:3]):
                    entity = result.entity
                    print(f"  {i+1}. {entity.name} ({entity.type}) - 유사도: {result.similarity:.3f}")
                    if hasattr(entity, 'properties') and entity.properties.get('llm_description'):
                        desc = entity.properties['llm_description'][:100]
                        print(f"     설명: {desc}...")
            else:
                print("  검색 결과 없음")
                
        except Exception as e:
            print(f"  검색 오류: {e}")
    
    # 리소스 정리
    print("\n🧹 리소스 정리...")
    db_connection.close()
    
    print("✅ 예제 완료!")
    print(f"\n데이터베이스 파일: {db_path}")
    print("생성된 지식그래프를 확인해보세요.")


if __name__ == "__main__":
    main()