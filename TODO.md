# TODO List

이 문서는 sqlite-kg-vec-mcp 프로젝트의 개선 작업 목록입니다.

## 🚨 Critical Issues (즉시 수정 필요)

### ✅ VectorStore LangChain 의존성 제거
- **위치**: `src/ports/vector_store.py:8`
- **문제**: 포트 인터페이스가 외부 라이브러리에 의존하여 헥사고날 아키텍처 위반
- **해결**: 순수한 추상 인터페이스로 재정의
- **상태**: ✅ 완료

### ✅ VectorStore Fat Interface 분리
- **위치**: `src/ports/vector_store.py` (156라인, 10개 메서드)
- **문제**: 인터페이스 분리 원칙(ISP) 위반
- **해결**: VectorWriter, VectorReader, VectorRetriever로 분리
- **상태**: ✅ 완료

### ✅ HuggingFace Adapter 오류
- **파일**: `tests/unit/adapters/huggingface/test_text_embedder.py`
- **문제**: `test_initialization_model_load_error` - SentenceTransformer 모델 로딩 실패 시 예외 처리 미흡
- **원인**: Mock 설정 문제로 Exception 대신 적절한 예외 타입이 발생하지 않음
- **해결**: HuggingFaceModelLoadException, HuggingFaceEmbeddingException 예외 클래스 생성 및 적용
- **상태**: ✅ 완료

### ✅ Ollama Client 연결 오류
- **파일**: `tests/unit/adapters/ollama/client/test_connection.py`
- **문제**: 다양한 연결 실패 시나리오에서 예외 처리 미흡
- **세부사항**: HTTP 연결 실패, 응답 파싱 오류, 타임아웃 처리 문제
- **해결**: `test_connection()` 공개 메서드 추가, 적절한 예외 타입별 처리 구현
- **상태**: ✅ 완료

### 📋 Document Processing 실패
- **파일**: 여러 도메인 서비스 테스트
- **문제**: 문서 처리 중 노드 유효성 검증 실패: "nodes[0]는 Node 인스턴스여야 합니다"
- **원인**: 문서 중복 처리 시 검증 로직 문제
- **상태**: 📋 대기 중

### ✅ Tests 폴더 Pylint 오류 수정
- **파일**: `tests/unit/` 전반
- **문제**: 
  - BaseSettings import 오류 (pydantic → pydantic_settings)
  - 테스트 클래스/메서드 docstring 누락 (C0115, C0116)
  - exec 사용으로 인한 보안 경고 (W0122)
  - implicit boolean 비교 (== [], == {} → not list, not dict)
  - import outside toplevel (C0415)
- **해결**: 모든 Pylint 오류 수정 및 Pydantic v2 호환성 개선
- **최종 점수**: 10.0/10 달성 (src/ 및 tests/ 모두 완벽)
- **상태**: ✅ 완료

## 🔥 High Priority (우선 수정)

### ✅ 타입 힌트 및 None 기본값 개선
- **위치**: 전 프로젝트 (400+ 건 변환)
- **문제**: `T | None` → `Optional[T]` 형태로 일관성 개선 필요
- **해결**: 전체 코드베이스에서 Optional 스타일로 통일, ruff UP045 규칙 비활성화
- **상태**: ✅ 완료

### ✅ 검색 가중치 하드코딩 제거
- **위치**: `src/domain/services/knowledge_search.py`
- **문제**: 0.7, 0.3, 0.6 등 가중치 하드코딩
- **해결**: SearchConfig 클래스 생성
- **상태**: ✅ 완료

### 📋 SQLite Transaction 오류
- **파일**: `tests/unit/adapters/sqlite3/` 관련 테스트들
- **문제**: 
  - 트랜잭션 실패: 엣지 삽입 실패
  - 커밋/롤백 상태 관리 문제
  - 비활성 트랜잭션 처리 경고
- **상태**: 📋 대기 중

### 📋 FastMCP Server CRUD 오류
- **파일**: 
  - `tests/unit/adapters/fastmcp/server/test_edge_crud.py`
  - `tests/unit/adapters/fastmcp/server/test_node_crud.py`
- **문제**: JSON-RPC 프로토콜 처리 및 CRUD 작업 실패
- **상태**: 📋 대기 중

### 📋 Vector Store/Search 오류
- **파일**: 벡터 검색 관련 테스트들
- **문제**: 
  - 유사도 검색 실패
  - 벡터 임베딩 처리 오류
  - HNSW 인덱스 관련 문제
- **상태**: 📋 대기 중

## ⚡ Medium Priority (기능 강화)

### ✅ 예외 처리 표준화
- **위치**: 전 프로젝트 (35건의 광범위한 except Exception)
- **문제**: `src/adapters/sqlite3/vector_store.py` 등에서 구체적 예외 타입 누락
- **해결**: 구체적 예외 타입별 처리, 적절한 에러 전파
- **상태**: ✅ 완료

### ✅ 검증 규칙 하드코딩 제거
- **위치**: `src/domain/services/document_validation.py:17-27`
- **문제**: 문서 길이, 메타데이터 크기 등 하드코딩
- **해결**: 환경 변수나 설정 파일로 외부화
- **상태**: ✅ 완료

### ✅ 타임아웃 값 표준화
- **위치**: 여러 파일
- **문제**: 60초, 300초, 10초 등 다양한 타임아웃 값 산재
- **해결**: 통일된 타임아웃 설정 관리
- **상태**: ✅ 완료

### ✅ 배치 크기 설정화
- **위치**: `src/adapters/ollama/nomic_embedder.py:209`
- **문제**: batch_size=32 하드코딩
- **해결**: 설정으로 외부화
- **상태**: ✅ 완료

### ✅ 설정 검증 강화
- **위치**: `src/domain/services/document_validation.py`
- **문제**: 일부 설정값만 검증
- **해결**: 모든 설정값에 대한 포괄적 검증
- **상태**: ✅ 완료

### ✅ 트랜잭션 경계 명확화
- **위치**: `src/adapters/sqlite3/database.py`
- **문제**: 트랜잭션 관리가 일관되지 않음
- **해결**: 명확한 트랜잭션 경계 정의
- **상태**: ✅ 완료

### 📋 로깅 패턴 일관성
- **위치**: 전 프로젝트
- **문제**: logging.getLogger(__name__) vs 파라미터 주입 혼재
- **해결**: 의존성 주입 패턴으로 통일
- **상태**: 📋 대기 중

### 📋 메트릭 수집 구현
- **위치**: `src/common/observability/integration.py`
- **문제**: 메트릭 수집 로직이 부분적으로만 구현됨
- **해결**: 완전한 메트릭 수집 시스템 구현 (`src/common/observability/metrics.py`)
- **상태**: 📋 대기 중

### 📋 Query Processing 오류
- **문제**: 
  - 쿼리 분석 응답 파싱 실패: "응답에서 유효한 JSON을 찾을 수 없습니다"
  - 쿼리 확장 파싱 실패
  - LLM 응답 처리 로직 개선 필요
- **상태**: 📋 대기 중

### 📋 Knowledge Graph 오류
- **문제**: 
  - 그래프 순회 및 관계 처리 실패
  - 엔티티 간 관계 생성/수정 오류
- **상태**: 📋 대기 중

### 📋 Configuration 및 Integration 오류
- **문제**: 
  - Langfuse 통합 제거 관련 경고 처리
  - 설정 관리 및 환경 변수 처리 개선
- **상태**: 📋 대기 중

## 🔧 Low Priority (최적화, 선택적)

### 📋 중복 검증 로직 통합
- **위치**: 여러 DTO 파일
- **문제**: 유사한 검증 로직이 중복됨
- **해결**: 공통 검증 유틸리티 클래스 생성
- **상태**: 📋 대기 중

### 📋 테스트에서 time.sleep 제거
- **위치**: 여러 테스트 파일
- **문제**: 테스트에서 실제 시간 지연 사용
- **해결**: Mock 시간이나 더 나은 동기화 메커니즘 사용
- **상태**: 📋 대기 중

### 📋 네이밍 규칙 통일
- **위치**: 전 프로젝트
- **문제**: Service 접미사 일관성 부족, import 순서 불일치
- **해결**: 네이밍 컨벤션 표준화 및 isort 설정 강화
- **상태**: 📋 대기 중

### 📋 문서화 개선
- **위치**: 여러 파일
- **문제**: 일부 메서드의 독스트링 부족
- **해결**: 포괄적 독스트링 추가
- **상태**: 📋 대기 중

### 📋 Repository 인터페이스 통합
- **문제**: NodeRepository와 RelationshipRepository가 거의 동일한 CRUD 패턴
- **해결**: 공통 베이스 인터페이스 도입
- **상태**: 📋 대기 중

## 📝 참고사항
- 모든 수정 후 `make check` (lint + test) 실행 필요
- 헥사고날 아키텍처 원칙 준수: 도메인 로직과 어댑터 분리 유지
- 의존성 주입 패턴 일관성 확보
- 에러 로깅 및 모니터링 개선

## 범례
- ✅ 완료
- 🔄 진행 중  
- 📋 대기 중
- 🚨 긴급
- 🔥 높은 우선순위
- ⚡ 중간 우선순위
- 🔧 낮은 우선순위