# TODO List

이 문서는 sqlite-kg-vec-mcp 프로젝트의 개선 작업 목록입니다.

## 🚨 Critical Issues (즉시 수정 필요)

### ✅ 1. VectorStore LangChain 의존성 제거
- **위치**: `src/ports/vector_store.py:8`
- **문제**: 포트 인터페이스가 외부 라이브러리에 의존하여 헥사고날 아키텍처 위반
- **해결**: 순수한 추상 인터페이스로 재정의
- **상태**: ✅ 완료
- **예상 시간**: 45분

### ✅ 2. VectorStore Fat Interface 분리
- **위치**: `src/ports/vector_store.py` (156라인, 10개 메서드)
- **문제**: 인터페이스 분리 원칙(ISP) 위반
- **해결**: VectorWriter, VectorReader, VectorRetriever로 분리
- **상태**: ✅ 완료
- **예상 시간**: 60분

## 🔥 High Priority (개선 필요)

### 🔄 3. 타입 힌트 및 None 기본값 개선
- **위치**: 전 프로젝트 (32건 발견)
- **문제**: `Optional[list[T]]` → `list[T] | None` 형태로 변경 필요
- **예시**: 
  - `src/domain/services/knowledge_search.py:53-54`
  - `src/domain/services/document_processor.py:209-212`
  - `src/domain/services/document_validation.py:26,59`
- **상태**: 🔄 진행 중
- **예상 시간**: 60분

### ✅ 4. 검색 가중치 하드코딩 제거
- **위치**: `src/domain/services/knowledge_search.py`
- **문제**: 0.7, 0.3, 0.6 등 가중치 하드코딩
- **해결**: SearchConfig 클래스 생성
- **상태**: ✅ 완료
- **예상 시간**: 30분

## ⚡ Medium Priority (기능 강화)

### 📋 5. 예외 처리 표준화
- **위치**: 전 프로젝트 (35건의 광범위한 except Exception)
- **문제**: `src/adapters/sqlite3/vector_store.py` 등에서 구체적 예외 타입 누락
- **해결**: 구체적 예외 타입별 처리, 적절한 에러 전파
- **상태**: 📋 대기 중
- **예상 시간**: 120분

### ✅ 6. 검증 규칙 하드코딩 제거
- **위치**: `src/domain/services/document_validation.py:17-27`
- **문제**: 문서 길이, 메타데이터 크기 등 하드코딩
- **해결**: 환경 변수나 설정 파일로 외부화
- **상태**: ✅ 완료
- **예상 시간**: 25분

### ✅ 7. 타임아웃 값 표준화
- **위치**: 여러 파일
- **문제**: 60초, 300초, 10초 등 다양한 타임아웃 값 산재
- **해결**: 통일된 타임아웃 설정 관리
- **상태**: ✅ 완료
- **예상 시간**: 30분

### 🔄 8. 배치 크기 설정화
- **위치**: `src/adapters/ollama/nomic_embedder.py:209`
- **문제**: batch_size=32 하드코딩
- **해결**: 설정으로 외부화
- **상태**: 🔄 진행 중
- **예상 시간**: 15분

### 📋 9. 로깅 패턴 일관성
- **위치**: 전 프로젝트
- **문제**: logging.getLogger(__name__) vs 파라미터 주입 혼재
- **해결**: 의존성 주입 패턴으로 통일
- **상태**: 📋 대기 중
- **예상 시간**: 45분

### 📋 10. 설정 검증 강화
- **위치**: `src/domain/services/document_validation.py`
- **문제**: 일부 설정값만 검증
- **해결**: 모든 설정값에 대한 포괄적 검증
- **상태**: 📋 대기 중
- **예상 시간**: 30분

### 📋 11. 메트릭 수집 구현
- **위치**: `src/common/observability/integration.py`
- **문제**: 메트릭 수집 로직이 부분적으로만 구현됨
- **해결**: 완전한 메트릭 수집 시스템 구현 (`src/common/observability/metrics.py`)
- **상태**: 📋 대기 중
- **예상 시간**: 60분

### 📋 12. 트랜잭션 경계 명확화
- **위치**: `src/adapters/sqlite3/database.py`
- **문제**: 트랜잭션 관리가 일관되지 않음
- **해결**: 명확한 트랜잭션 경계 정의
- **상태**: 📋 대기 중
- **예상 시간**: 40분

## 🔧 Low Priority (최적화, 선택적)

### 📋 13. 중복 검증 로직 통합
- **위치**: 여러 DTO 파일
- **문제**: 유사한 검증 로직이 중복됨
- **해결**: 공통 검증 유틸리티 클래스 생성
- **상태**: 📋 대기 중
- **예상 시간**: 50분

### 📋 14. 테스트에서 time.sleep 제거
- **위치**: 여러 테스트 파일
- **문제**: 테스트에서 실제 시간 지연 사용
- **해결**: Mock 시간이나 더 나은 동기화 메커니즘 사용
- **상태**: 📋 대기 중
- **예상 시간**: 30분

### 📋 15. 네이밍 규칙 통일
- **위치**: 전 프로젝트
- **문제**: Service 접미사 일관성 부족, import 순서 불일치
- **해결**: 네이밍 컨벤션 표준화 및 isort 설정 강화
- **상태**: 📋 대기 중
- **예상 시간**: 40분

### 📋 16. 문서화 개선
- **위치**: 여러 파일
- **문제**: 일부 메서드의 독스트링 부족
- **해결**: 포괄적 독스트링 추가
- **상태**: 📋 대기 중
- **예상 시간**: 60분

### 📋 17. Repository 인터페이스 통합
- **문제**: NodeRepository와 RelationshipRepository가 거의 동일한 CRUD 패턴
- **해결**: 공통 베이스 인터페이스 도입
- **상태**: 📋 대기 중
- **예상 시간**: 60분

## 📊 예상 총 작업 시간

- **Critical Issues**: ✅ 완료 (2.5시간)
- **High Priority**: 1시간  
- **Medium Priority**: 7시간
- **Low Priority**: 4.5시간
- **총합**: 약 15시간 (완료된 작업 제외)

## 📋 권장 작업 순서

### **Phase 1 (Critical Issues - 완료) ✅**
1. ✅ VectorStore LangChain 의존성 제거 (45분)
2. ✅ VectorStore Fat Interface 분리 (60분)

### **Phase 2 (High Priority - 1시간)**
3. 🔄 타입 힌트 및 None 기본값 개선 (60분)
4. ✅ 검색 가중치 하드코딩 제거 (30분)

### **Phase 3 (Medium Priority - 7시간)**
5. 📋 예외 처리 표준화 (120분)
6. 📋 검증 규칙 및 타임아웃 표준화 (55분)
7. 📋 로깅 패턴 일관성 및 설정 검증 강화 (75분)
8. 📋 메트릭 수집 구현 (60분)
9. 📋 트랜잭션 경계 명확화 (40분)

### **Phase 4 (Low Priority - 4.5시간)**
10. 📋 중복 검증 로직 통합 및 기타 최적화

## 범례

- ✅ 완료
- 🔄 진행 중  
- 📋 대기 중
- 🚨 긴급
- 🔥 높은 우선순위
- ⚡ 중간 우선순위
- 🔧 낮은 우선순위