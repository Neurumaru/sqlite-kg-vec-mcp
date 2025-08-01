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

다음은 전체 codebase 안티패턴 분석 결과를 바탕으로 수립한 체계적인 개선 작업 목록입니다. 
(분석 일자: 2025-07-31, 총 134개 안티패턴 발견 - Critical 3개, High 32개, Medium 67개, Low 32개)

### 🚨 Critical Issues (즉시 수정 필요)

#### 1. DTO 불변성 위반 수정
- **위치**: `src/dto/event.py:102-103`
- **문제**: `correlation_id` 필드를 초기화 후 수정하여 불변성 위반
- **해결**: `__post_init__`에서 필드 수정 대신 factory 메서드 사용
- **영향도**: 데이터 무결성 위험, 동시성 문제 가능성
- **예상 시간**: 30분

#### 2. 도메인-DTO 의존성 역전 원칙 위반 수정
- **위치**: `src/domain/services/document_processor.py:15-23`
- **문제**: 도메인 서비스가 DTO에 직접 의존 (헥사고날 아키텍처 위반)
- **해결**: 매퍼 패턴 도입으로 의존성 분리
- **영향도**: 아키텍처 원칙 위반, 테스트 어려움
- **예상 시간**: 60분

#### 3. 전역 상태 및 싱글톤 패턴 제거
- **위치**: `src/common/observability/integration.py:363, 366-381`
- **문제**: 전역 변수와 싱글톤 패턴으로 테스트 격리 불가
- **해결**: 의존성 주입 패턴으로 변경
- **영향도**: 테스트 가능성 저해, 동시성 문제
- **예상 시간**: 45분

### 🔥 High Priority (즉시 개선 필요)

#### 4. KnowledgeExtractionResult 값 객체 분리
- **현재 위치**: `src/domain/services/document_processor.py` 내부 클래스
- **이동 위치**: `src/domain/value_objects/knowledge_extraction_result.py`
- **이유**: 도메인 서비스에서 값 객체를 분리하여 헥사고날 아키텍처 원칙 준수
- **예상 시간**: 30분

#### 5. 타입 힌트 및 None 기본값 개선
- **범위**: 전 프로젝트 (32건 발견)
- **변경**: `List[NodeId] = None` → `Optional[List[NodeId]] = None`
- **적용**: `Union[T, None]` → `Optional[T]` 형태로 통일
- **예상 시간**: 90분

#### 6. VectorStore Fat Interface 분리
- **위치**: `src/ports/vector_store.py` (156라인, 10개 메서드)
- **문제**: 인터페이스 분리 원칙(ISP) 위반
- **해결**: VectorWriter, VectorReader, VectorFactory로 분리
- **예상 시간**: 120분

#### 7. DocumentProcessor 단일 책임 원칙 위반 해결
- **위치**: `src/domain/services/document_processor.py:49-394`
- **문제**: 문서 처리 + 영속성 + DTO 변환 + 통계 등 다중 책임
- **해결**: 책임별 클래스 분리
- **예상 시간**: 90분

#### 8. 문서 검증 로직 분리
- **새 파일**: `src/domain/services/document_validation.py`
- **내용**: `DocumentValidationRules` 클래스 생성
- **변경**: 하드코딩된 검증 로직을 설정 가능한 상수로 변경
- **예상 시간**: 45분

### ⚡ Medium Priority (기능 강화)

#### 9. 하드코딩된 값 설정화 (47건)
- **대상**: 검색 가중치, 타임아웃, 차원, 임계값 등
- **예시**: `src/domain/services/knowledge_search.py` 가중치, `src/adapters/ollama/nomic_embedder.py` 차원
- **해결**: 설정 클래스나 환경 변수로 외부화
- **예상 시간**: 150분

#### 10. 예외 처리 표준화 (28건)
- **문제**: 광범위한 `except Exception`, 구체적 예외 타입 누락
- **해결**: 구체적 예외 타입별 처리, 적절한 에러 전파
- **예상 시간**: 120분

#### 11. 트랜잭션 경계 명확화
- **목표**: 데이터 일관성 보장을 위한 트랜잭션 관리 개선
- **작업**: Repository 인터페이스에 트랜잭션 메서드 추가
- **적용**: `_process_document_with_persistence`에 트랜잭션 컨텍스트 적용
- **예상 시간**: 60분

#### 12. 중복 검증 로직 통합
- **대상**: DTO 파일들, 값 객체들의 유사한 검증 패턴
- **해결**: 공통 검증 유틸리티 클래스 생성
- **예상 시간**: 75분

#### 13. 모니터링 메트릭 구현
- **목표**: TODO 주석으로 남겨진 성능 모니터링 기능 구현
- **새 파일**: `src/common/observability/metrics.py`
- **측정 항목**: 처리 시간, 메모리 사용량, 성공/실패율
- **예상 시간**: 75분

#### 14. LangChain 의존성 제거
- **위치**: `src/ports/vector_store.py:8`, `src/ports/llm.py:9`
- **문제**: 포트 인터페이스가 특정 라이브러리에 의존
- **해결**: 추상화 순수성 확보를 위한 자체 인터페이스 정의
- **예상 시간**: 90분

### 🔧 Low Priority (최적화, 선택적)

#### 15. Repository 인터페이스 통합
- **문제**: NodeRepository와 RelationshipRepository가 거의 동일한 CRUD 패턴
- **해결**: 공통 베이스 인터페이스 도입
- **예상 시간**: 60분

#### 16. 워크플로우 캡슐화 (대규모 리팩토링)
- **목표**: 문서 처리 단계를 체계적으로 관리
- **새 파일들**:
  - `src/domain/value_objects/processing_step.py` (처리 단계 enum)
  - `src/domain/services/document_processing_context.py` (상태 관리)
  - `src/domain/services/document_processing_workflow.py` (워크플로우 매니저)
- **기능**: 단계별 처리, 진행 상황 추적, 재시도 로직, 에러 처리
- **예상 시간**: 3.5시간

#### 17. 배치 처리 기능 추가
- **목표**: 다중 문서 처리 성능 개선
- **기능**: 병렬 처리, 진행률 리포팅, 부분 실패 처리
- **예상 시간**: 120분

#### 18. 에러 처리 및 재시도 로직 강화
- **목표**: 시스템 안정성 향상
- **기능**: 커스텀 예외, 재시도 메커니즘, Circuit breaker 패턴
- **예상 시간**: 90분

#### 19. 네이밍 규칙 통일
- **문제**: Service 접미사 일관성 부족, import 순서 불일치
- **해결**: 코딩 컨벤션 표준화
- **예상 시간**: 45분

#### 20. 리소스 누수 방지
- **문제**: DB connection, HTTP session 등의 명시적 정리 부족
- **해결**: Context manager 활용
- **예상 시간**: 60분

### 📋 권장 작업 순서

#### **Phase 1 (Critical - 2-3시간)**
1. DTO 불변성 위반 수정
2. 도메인-DTO 의존성 분리  
3. 전역 상태 제거
- **목표**: 아키텍처 무결성 확보

#### **Phase 2 (High Priority - 6-8시간)**
4. KnowledgeExtractionResult 분리
5. 타입 힌트 표준화
6. VectorStore 인터페이스 분리
7. DocumentProcessor 책임 분리
8. 문서 검증 로직 분리
- **목표**: 코드 구조 개선, SOLID 원칙 준수

#### **Phase 3 (Medium Priority - 8-10시간)**
9. 하드코딩된 값 설정화
10. 예외 처리 표준화
11. 트랜잭션 경계 명확화
12. 중복 로직 통합
13. 모니터링 메트릭 구현
14. LangChain 의존성 제거
- **목표**: 시스템 안정성 및 관찰가능성 향상

#### **Phase 4 (Low Priority - 별도 계획)**
15-20. 최적화 및 대규모 리팩토링
- **목표**: 성능 최적화 및 장기적 유지보수성

### 💡 개발 시 주의사항
- **아키텍처 원칙 준수**: 모든 변경사항은 헥사고날 아키텍처 원칙 준수
- **점진적 개선**: 각 단계마다 테스트 코드 작성 및 검증
- **품질 검사**: `make check` 통과 확인 후 커밋
- **단계별 검증**: Critical Issues부터 순차적으로 해결하여 안정성 확보