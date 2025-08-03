# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개발 명령어

### 의존성 설치
```bash
# 개발 의존성 설치
uv sync --group dev

# 프로덕션 의존성만 설치
uv sync --no-dev
```

### 코드 품질 검사
```bash
# 전체 린트 실행 (black, isort, flake8, ruff, pylint)
make lint

# 개별 도구 실행
make format      # black + isort 포매팅
make type-check  # mypy 타입 검사
make flake8      # flake8 린팅
make ruff        # ruff 린팅
```

### 테스트
```bash
# 기본 테스트 실행
make test
# 또는
uv run python -m unittest discover -s tests -p "test_*.py"

# 커버리지 포함 테스트
make test-cov

# 단일 테스트 파일 실행
uv run python -m unittest tests.unit.domain.entities.test_document
```

### 빌드 및 정리
```bash
make build    # 패키지 빌드
make clean    # 빌드 아티팩트 정리
make check    # lint + test 전체 실행
```

## 코드 아키텍처

이 프로젝트는 **헥사고날 아키텍처 (Ports and Adapters)**를 사용하며, 다음과 같은 구조를 따릅니다:

### 핵심 구조
- **`src/domain/`**: 비즈니스 로직의 핵심. 외부 의존성 없는 순수한 도메인 코드
  - `entities/`: Document, Node, Relationship 등 핵심 비즈니스 엔티티
  - `value_objects/`: DocumentId, NodeId, Vector 등 불변 값 객체
  - `services/`: DocumentProcessor, KnowledgeSearch 등 도메인 서비스
  - `events/`: 도메인 이벤트 (DocumentProcessed, NodeCreated 등)
  - `exceptions/`: 도메인별 예외

- **`src/ports/`**: 추상 인터페이스 (계약)
  - `repositories/`: 데이터 영속성 추상화
  - 다양한 포트 인터페이스들

- **`src/adapters/`**: 구체적인 기술 구현체
  - `sqlite3/`: SQLite 데이터베이스 어댑터
  - `openai/`, `ollama/`, `huggingface/`: 각종 AI 서비스 어댑터
  - `hnsw/`: 벡터 검색 어댑터
  - `fastmcp/`: MCP 서버 어댑터
  - `testing/`: 테스트용 어댑터

### 의존성 주입 패턴
- 도메인 서비스는 포트 인터페이스에만 의존
- 실제 어댑터는 런타임에 주입
- 예: `DocumentProcessor(knowledge_extractor: LLM)`

### 데이터베이스 스키마
- **Property Graph 모델** 기반
- 엔티티: `entities` 테이블 (id, uuid, name, type, properties JSON)
- 관계: `edges` 테이블 (이진 관계) + `hyperedges`/`hyperedge_members` (N진 관계)
- 벡터: 별도 테이블 (`node_embeddings`, `relationship_embeddings`)
- SQLite + HNSW/Faiss 하이브리드 벡터 검색

### MCP 서버 통합
- **JSON-RPC 2.0** 프로토콜
- WebSocket 전송
- 엔티티/관계 CRUD, 유사도 검색, 그래프 순회 API 제공

## 개발 가이드라인

### 새 어댑터 추가시
1. `src/ports/`에 해당 추상 인터페이스 정의
2. `src/adapters/`에 구체 구현체 작성
3. 도메인 서비스에서는 포트만 사용
4. 테스트용 mock 어댑터도 `src/adapters/testing/`에 제공

### 테스트 작성
- 유닛 테스트: `tests/unit/`
- 통합 테스트: `tests/integration/`
- 도메인 로직은 의존성 주입으로 순수하게 테스트
- 어댑터는 실제 외부 서비스와 통합 테스트

### 코드 스타일
- Black (line-length=100) + isort로 포매팅
- flake8 + ruff + pylint로 린팅
- mypy로 타입 검사 (strict=false 설정)
- 모든 변경사항은 `make check` 통과 필요

## 현재 개선 작업 (TODO)

다음은 남은 개선 작업 목록입니다. (업데이트: 2025-08-02)

### 🚨 Critical Issues (즉시 수정 필요)

#### 1. VectorStore LangChain 의존성 제거
- **위치**: `src/ports/vector_store.py:8`
- **문제**: 포트 인터페이스가 외부 라이브러리에 의존하여 헥사고날 아키텍처 위반
- **해결**: 순수한 추상 인터페이스로 재정의
- **예상 시간**: 45분

#### 2. VectorStore Fat Interface 분리
- **위치**: `src/ports/vector_store.py` (156라인, 10개 메서드)
- **문제**: 인터페이스 분리 원칙(ISP) 위반
- **해결**: VectorWriter, VectorReader, VectorRetriever로 분리
- **예상 시간**: 60분

### 🔥 High Priority (개선 필요)

#### 3. 타입 힌트 및 None 기본값 개선 (부분 완료)
- **위치**: 전 프로젝트 (32건 발견)
- **문제**: `List[T] = None` → `Optional[List[T]] = None` 형태로 변경 필요
- **예시**: `src/domain/services/document_validation.py:26`, `src/domain/services/document_processor.py:205-208`
- **예상 시간**: 60분

#### 4. 검색 가중치 하드코딩 제거
- **위치**: `src/domain/services/knowledge_search.py`
- **문제**: 0.7, 0.3, 0.6 등 가중치 하드코딩
- **해결**: SearchConfig 클래스 생성
- **예상 시간**: 30분

### ⚡ Medium Priority (기능 강화)

#### 5. 예외 처리 표준화
- **위치**: 전 프로젝트 (35건의 광범위한 except Exception)
- **문제**: `src/adapters/sqlite3/vector_store.py` 등에서 구체적 예외 타입 누락
- **해결**: 구체적 예외 타입별 처리, 적절한 에러 전파
- **예상 시간**: 120분

#### 6. 검증 규칙 하드코딩 제거
- **위치**: `src/domain/services/document_validation.py:17-27`
- **문제**: 문서 길이, 메타데이터 크기 등 하드코딩
- **해결**: 환경 변수나 설정 파일로 외부화
- **예상 시간**: 25분

#### 7. 타임아웃 값 표준화
- **위치**: 여러 파일
- **문제**: 60초, 300초, 10초 등 다양한 타임아웃 값 산재
- **해결**: 통일된 타임아웃 설정 관리
- **예상 시간**: 30분

#### 8. 배치 크기 설정화
- **위치**: `src/adapters/ollama/nomic_embedder.py:209`
- **문제**: batch_size=32 하드코딩
- **해결**: 설정으로 외부화
- **예상 시간**: 15분

#### 9. 로깅 패턴 일관성
- **위치**: 전 프로젝트
- **문제**: logging.getLogger(__name__) vs 파라미터 주입 혼재
- **해결**: 의존성 주입 패턴으로 통일
- **예상 시간**: 45분

#### 10. 설정 검증 강화
- **위치**: `src/domain/services/document_validation.py`
- **문제**: 일부 설정값만 검증
- **해결**: 모든 설정값에 대한 포괄적 검증
- **예상 시간**: 30분

#### 11. 메트릭 수집 구현
- **위치**: `src/common/observability/integration.py`
- **문제**: 메트릭 수집 로직이 부분적으로만 구현됨
- **해결**: 완전한 메트릭 수집 시스템 구현 (`src/common/observability/metrics.py`)
- **예상 시간**: 60분

#### 12. 트랜잭션 경계 명확화
- **위치**: `src/adapters/sqlite3/database.py`
- **문제**: 트랜잭션 관리가 일관되지 않음
- **해결**: 명확한 트랜잭션 경계 정의
- **예상 시간**: 40분

#### 13. 중복 검증 로직 통합
- **위치**: 여러 DTO 파일
- **문제**: 유사한 검증 로직이 중복됨
- **해결**: 공통 검증 유틸리티 클래스 생성
- **예상 시간**: 50분

### 🔧 Low Priority (최적화, 선택적)

#### 14. 테스트에서 time.sleep 제거
- **위치**: 여러 테스트 파일
- **문제**: 테스트에서 실제 시간 지연 사용
- **해결**: Mock 시간이나 더 나은 동기화 메커니즘 사용
- **예상 시간**: 30분

#### 15. 네이밍 규칙 통일
- **위치**: 전 프로젝트
- **문제**: Service 접미사 일관성 부족, import 순서 불일치
- **해결**: 네이밍 컨벤션 표준화 및 isort 설정 강화
- **예상 시간**: 40분

#### 16. 문서화 개선
- **위치**: 여러 파일
- **문제**: 일부 메서드의 독스트링 부족
- **해결**: 포괄적 독스트링 추가
- **예상 시간**: 60분

#### 17. Repository 인터페이스 통합
- **문제**: NodeRepository와 RelationshipRepository가 거의 동일한 CRUD 패턴
- **해결**: 공통 베이스 인터페이스 도입
- **예상 시간**: 60분

#### 18. 워크플로우 캡슐화 (대규모 리팩토링)
- **목표**: 문서 처리 단계를 체계적으로 관리
- **새 파일들**:
  - `src/domain/value_objects/processing_step.py` (처리 단계 enum)
  - `src/domain/services/document_processing_context.py` (상태 관리)
  - `src/domain/services/document_processing_workflow.py` (워크플로우 매니저)
- **기능**: 단계별 처리, 진행 상황 추적, 재시도 로직, 에러 처리
- **예상 시간**: 3.5시간

#### 19. 배치 처리 기능 추가
- **목표**: 다중 문서 처리 성능 개선
- **기능**: 병렬 처리, 진행률 리포팅, 부분 실패 처리
- **예상 시간**: 120분

#### 20. 에러 처리 및 재시도 로직 강화
- **목표**: 시스템 안정성 향상
- **기능**: 커스텀 예외, 재시도 메커니즘, Circuit breaker 패턴
- **예상 시간**: 90분

#### 21. 리소스 누수 방지
- **문제**: DB connection, HTTP session 등의 명시적 정리 부족
- **해결**: Context manager 활용
- **예상 시간**: 60분

### 📋 권장 작업 순서

#### **Phase 1 (Critical Issues - 2.5시간)**
1. VectorStore LangChain 의존성 제거 (45분)
2. VectorStore Fat Interface 분리 (60분)
- **목표**: 아키텍처 무결성 확보, ISP 원칙 준수

#### **Phase 2 (High Priority - 1.5시간)**
3. 타입 힌트 및 None 기본값 개선 완료 (60분)
4. 검색 가중치 하드코딩 제거 (30분)
- **목표**: 코드 품질 향상, 설정 외부화

#### **Phase 3 (Medium Priority - 7시간)**
5. 예외 처리 표준화 (120분)
6. 검증 규칙 및 타임아웃 표준화 (55분)
7. 로깅 패턴 일관성 및 설정 검증 강화 (75분)
8. 메트릭 수집 구현 (60분)
9. 트랜잭션 경계 명확화 (40분)
10. 중복 검증 로직 통합 (50분)
- **목표**: 시스템 안정성 및 관찰가능성 향상

#### **Phase 4 (Low Priority - 별도 계획)**
11. 테스트 최적화 및 네이밍 표준화 (70분)
12. 문서화 개선 (60분)
13. Repository 통합 및 워크플로우 캡슐화 (5-6시간)
14. 배치 처리 및 재시도 로직 (3.5시간)
- **목표**: 성능 최적화 및 장기적 유지보수성

### 📊 예상 총 작업 시간
- **Critical Issues**: 2.5시간
- **High Priority**: 1.5시간  
- **Medium Priority**: 7시간
- **Low Priority**: 9-10시간
- **총합**: 약 20-21시간

### 💡 개발 시 주의사항
- **아키텍처 원칙 준수**: 모든 변경사항은 헥사고날 아키텍처 원칙 준수
- **점진적 개선**: 각 단계마다 테스트 코드 작성 및 검증
- **품질 검사**: `make check` 통과 확인 후 커밋
- **단계별 검증**: Critical Issues부터 순차적으로 해결하여 안정성 확보