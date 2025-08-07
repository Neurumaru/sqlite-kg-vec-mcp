# 남은 테스트 실패 항목들

## 우선순위 High - 즉시 수정 필요

### 1. Ollama Adapter 테스트 실패
**파일**: `tests/unit/adapters/ollama/test_nomic_embedder.py`

#### 1.1 test_embed_texts_partial_failure
- **에러**: `Exception: API Error` (일반 Exception이 발생하지만 `OllamaModelException` 예상)
- **원인**: Mock `side_effect`가 부분 실패 시나리오에 맞게 구성되지 않음
- **해결방법**: `side_effect`를 여러 텍스트 중 일부만 실패하도록 설정

#### 1.2 test_is_available_failure  
- **에러**: Exception 타입 불일치
- **원인**: Mock에서 발생하는 예외가 실제 구현에서 처리하는 예외 타입과 다름
- **해결방법**: 올바른 예외 타입(`ConnectionError`, `TimeoutError` 등) 사용

### 2. Graph Traversal StopIteration 에러
**파일**: `src/adapters/sqlite3/graph/traversal.py`
- **에러**: `neighbor_query_failed_bfs component=graph_traversal entity_id=2 error_type=StopIteration`
- **원인**: BFS 알고리즘에서 빈 결과 처리 시 StopIteration 발생
- **해결방법**: 빈 결과에 대한 적절한 예외 처리 추가

## 우선순위 Medium

### 3. Query Processing JSON 파싱 실패
**관련 파일**: `src/domain/services/`
- **에러**: 
  - `쿼리 분석 응답 파싱 실패: 응답에서 유효한 JSON을 찾을 수 없습니다: Invalid JSON response`
  - `쿼리 확장 파싱 실패: 응답에서 유효한 JSON을 찾을 수 없습니다: Invalid JSON`
- **원인**: LLM 응답이 예상된 JSON 형식이 아님
- **해결방법**: JSON 파싱 실패에 대한 fallback 처리 추가

### 4. Document Processor 경고/에러
**관련 파일**: `src/domain/services/document_processor/`
- **에러**: 
  - `문서 doc1에 0개의 오류가 있습니다`
  - `Failed to process document: 추출 실패`
- **원인**: 문서 처리 과정에서 예외 상황 처리 부족
- **해결방법**: 문서 처리 실패 시나리오에 대한 로깅 및 에러 처리 개선

### 5. Transaction Context 에러
**파일**: `src/adapters/sqlite3/transaction_context.py`
- **에러**: 
  - `트랜잭션 실패: 엣지 삽입 실패`
  - `비활성 트랜잭션 커밋/롤백 시도`
- **원인**: 트랜잭션 상태 관리 문제
- **해결방법**: 트랜잭션 상태 검증 로직 강화

## 우선순위 Low

### 6. Integration Warning
- **경고**: `Langfuse integration has been removed`
- **해결방법**: 관련 경고 메시지 제거 또는 설정으로 비활성화

## 수정 전략

1. **단위 테스트부터 시작**: Mock 설정 문제들을 우선 해결
2. **점진적 수정**: 각 실패 항목을 개별적으로 수정하고 테스트
3. **통합 테스트**: 개별 수정 후 전체 테스트 스위트 실행
4. **커밋 분리**: 각 수정사항을 논리적 단위로 커밋

## 예상 작업 시간
- High 우선순위: 2-3 시간
- Medium 우선순위: 3-4 시간  
- Low 우선순위: 1 시간

**총 예상 시간**: 6-8 시간