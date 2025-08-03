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