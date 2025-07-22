# SQLite KG Vec MCP

MCP 서버 인터페이스를 통한 통합 지식 그래프 및 벡터 데이터베이스.

이 프로젝트는 SQLite 기반 지식 그래프와 벡터 저장소(선택적으로 HNSW 인덱스 사용)를 결합하여 
MCP(Model Context Protocol) 서버를 통한 인터페이스를 제공합니다.

*Read this in other languages: [English](README.md), [한국어](README-KR.md)*

## 주요 기능

- 노드와 엣지를 사용한 지식 그래프 저장
- 지식 그래프의 노드와 엣지에 대한 벡터 임베딩을 통한 의미적 유사도 검색
- 단일 SQLite 파일에 모든 데이터 저장
- 간단한 MCP(Model Context Protocol)을 통한 그래프 조작

## 설치

```bash
# 저장소 복제
git clone https://github.com/Neurumaru/sqlite-kg-vec-mcp.git
cd sqlite-kg-vec-mcp

# pip로 설치 (개발 의존성 포함)
pip install -e ".[dev]"

# 또는 더 빠른 의존성 해결을 위해 uv 사용
uv pip install -e ".[dev]"
```

## 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하여 필요한 환경 변수를 설정하세요:

```bash
# .env.example 파일을 복사하여 시작
cp .env.example .env
```

`.env` 파일에 다음 설정을 포함해야 합니다:

```env
# Langfuse 설정 (프롬프트 관리용)
LANGFUSE_SECRET_KEY=your-secret-key-here
LANGFUSE_PUBLIC_KEY=your-public-key-here
LANGFUSE_HOST=https://us.cloud.langfuse.com

# Ollama 설정
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3n

# 데이터베이스 설정
DATABASE_PATH=./knowledge_graph.db

# 임베딩 설정
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

## 빠른 시작

```python
from src import KnowledgeGraphServer

# SQLite 데이터베이스 파일로 서버 초기화
server = KnowledgeGraphServer(db_path="knowledge_graph.db")

# MCP 서버 시작
server.start(host="127.0.0.1", port=8080)
```

## 프로젝트 구조

```
sqlite-kg-vec-mcp/
├── src/                    # 소스 코드 디렉토리
├── tests/                  # 테스트 스위트
├── examples/               # 사용 예제
├── pyproject.toml          # 프로젝트 메타데이터 및 의존성
├── LICENSE                 # MIT 라이센스
└── README.md               # 영문 설명서
```

### 주요 구성 요소

- **db/**: SQLite 연결 관리, 스키마 정의, 트랜잭션 처리, 데이터베이스 마이그레이션 등 데이터베이스 관련 코드

- **graph/**: 엔티티/노드 작업, 관계/엣지 관리, 하이퍼엣지 처리, 그래프 순회 알고리즘 등 지식 그래프 구현

- **vector/**: 벡터 임베딩 관리, HNSW/Faiss 인덱스 작업, 유사도 검색 알고리즘, 벡터-데이터베이스 동기화 등 벡터 저장 및 검색 기능

- **server/**: API 엔드포인트, 요청 핸들러, WebSocket 통신 등 MCP 서버 구현

## 개발

### 개발 환경 설정

```bash
# 개발 의존성 설치
pip install -e ".[dev]"

# 테스트 실행
pytest

# 코드 포맷팅
black .
isort .

# 타입 체크
mypy src
```

### 테스트 실행

```bash
# 모든 테스트 실행
pytest

# 특정 테스트 파일 실행
pytest tests/test_graph.py

# 커버리지와 함께 실행
pytest --cov=sqlite_kg_vec_mcp
```

## 사용 예제

### 노드 및 관계 생성

```python
from src import KnowledgeGraph

# 지식 그래프 열기 또는 생성
kg = KnowledgeGraph("example.db")

# 노드 생성
person_node = kg.create_node(
    name="홍길동",
    type="Person",
    properties={"age": 30, "occupation": "엔지니어"}
)

company_node = kg.create_node(
    name="테크코프",
    type="Company",
    properties={"founded": 2010, "industry": "기술"}
)

# 관계 생성
kg.create_edge(
    source_id=person_node.id,
    target_id=company_node.id,
    relation_type="WORKS_FOR",
    properties={"since": 2020, "position": "선임 엔지니어"}
)

# 벡터 검색
similar_engineers = kg.search_similar_nodes(
    query_vector=person_node.embedding,
    limit=5,
    node_types=["Person"]
)
```

### MCP 서버 API 사용

Model Context Protocol 클라이언트를 사용하여 MCP 서버에 연결:

```python
from fastmcp import MCPClient

# MCP 서버에 연결
client = MCPClient("ws://localhost:8080")

# 노드 생성
response = await client.request("create_node", {
    "name": "제품 X",
    "type": "Product",
    "properties": {"price": 99.99, "category": "전자제품"}
})

# 그래프 쿼리
neighbors = await client.request("get_neighbors", {
    "node_id": response["node_id"],
    "relation_types": ["MANUFACTURES", "SELLS"],
    "direction": "incoming"
})
```

## 설계 고려 사항

다음은 시스템의 주요 구성 요소에 대한 핵심 설계 결정 및 고려 사항입니다.

### 1. SQLite 스키마 및 지식 그래프 모델링

- **기본 모델:** **속성 그래프(Property Graph)** 모델을 기반으로 합니다.
- **엔티티 테이블:**
    - `entities` (또는 `nodes`): `id` (INTEGER PRIMARY KEY - 성능 최적화), `uuid` (TEXT UNIQUE NOT NULL - 외부 연동용 안정적 식별자), `name` (TEXT, 선택 사항), `type` (TEXT NOT NULL), `properties` (JSON - 유연한 속성 저장), `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `updated_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP).
    - **권장:** 자주 쿼리되는 '핫(hot)' 속성은 성능을 위해 **별도 컬럼**으로 분리하고, JSON 내 특정 경로 쿼리 시 **생성된 컬럼(Generated Columns)**과 그에 대한 인덱스를 활용합니다. `type` 필드는 별도 참조 테이블로 분리하여 일관성 및 계층 구조 지원을 고려할 수 있습니다.
- **기본 관계 (Binary):**
    - `edges`: `id` (INTEGER PRIMARY KEY), `source_id` (INTEGER NOT NULL REFERENCES entities(id)), `target_id` (INTEGER NOT NULL REFERENCES entities(id)), `relation_type` (TEXT NOT NULL), `properties` (JSON), `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `updated_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP).
    - **권장:** `properties` JSON은 보조 정보 저장에 사용하고, 핵심 필터링/조인 조건은 네이티브 컬럼에 둡니다.
- **N-ary 관계 모델링: 하이퍼엣지(Hyperedge) 채택**
    - 복잡한 관계(참여자 셋 이상)는 일관성을 위해 **하이퍼엣지 모델**을 사용합니다.
    - `hyperedges`: `id` (INTEGER PRIMARY KEY), `hyperedge_type` (TEXT NOT NULL), `properties` (JSON - 메타데이터 및 속성 통합 고려), `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `updated_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP).
    - `hyperedge_members`: `hyperedge_id` (FK), `entity_id` (FK), `role` (TEXT NOT NULL).
    - **단순화 방안:** 기존 `relationship_metadata` 및 `relationship_properties` 테이블 대신, 관련 정보를 `hyperedges.properties` JSON 컬럼에 통합하는 것을 고려합니다. 이는 스키마를 단순화하지만, 쿼리 복잡성 및 성능 영향을 평가해야 합니다. 필요한 경우 생성된 컬럼을 활용합니다.
- **관찰(Observations) 처리 (하이브리드 방식):**
    - 자주 접근하거나 최신 정보("핫")는 `entities.properties` JSON 필드 내 (또는 별도 컬럼)에 저장합니다.
    - 오래되거나 전체 기록("콜드")은 별도의 `observations` 테이블(`id` PK, `entity_id` FK, `content` TEXT, `created_at` DATETIME)에 저장합니다.
    - **동기화:** **애플리케이션 레벨 로직** 또는 **주기적인 배치(Batch) 처리**를 기본으로 구현합니다. DB 트리거는 성능 영향을 고려하여 필요시에만 제한적으로 사용합니다. 데이터 이동 기준(Staleness criteria) 정의가 필요합니다.
- **인덱싱:**
    - **필수:** `INTEGER PK`, `uuid`, 외래 키 (`source_id`, `target_id`, `hyperedge_id`, `entity_id`)에는 **명시적으로 인덱스**를 생성합니다.
    - **권장:** 엔티티/관계 타입(`type`, `relation_type`, `hyperedge_type`), 역할(`hyperedge_members.role`) 등 자주 필터링되는 컬럼에 인덱스를 생성합니다.
    - **성능 향상:** 그래프 순회를 위해 `(source_id, relation_type)`, `(target_id, relation_type)` 등 **복합 인덱스(Composite Index)**를 활용합니다.
    - **JSON 최적화:** `properties` JSON 내 자주 쿼리되는 경로는 **생성된 컬럼**을 정의하고 해당 컬럼에 인덱스를 생성합니다.
    - **크기 최적화:** 특정 타입의 데이터만 인덱싱하는 **부분 인덱스(Partial Index)** 사용을 고려합니다.
- **데이터베이스 설정 및 관리:**
    - **기본:** `PRAGMA journal_mode=WAL;`, `PRAGMA busy_timeout=5000;`, `PRAGMA synchronous=NORMAL;` 사용을 권장합니다.
    - **추가 권장:** `PRAGMA foreign_keys=ON;` (참조 무결성), `PRAGMA temp_store=MEMORY;` 설정을 고려합니다. `PRAGMA mmap_size`는 시스템 메모리 및 DB 크기에 맞게 조정합니다.
    - **스키마 관리:** 스키마 변경 이력 관리를 위해 `schema_version` 테이블 도입을 고려합니다. 제약 조건, 타임스탬프를 적극 활용합니다.

### 2. 벡터 저장 및 검색

- **저장 방식 (분리):** 벡터 임베딩은 핵심 그래프 데이터와 분리하여 별도의 테이블에 저장합니다.
    - `node_embeddings`: `node_id` (FK to entities.id), `embedding` (BLOB), `model_info` (TEXT 또는 JSON) 등.
    - `relationship_embeddings`: `embedding_id` (PK), `embedding` (BLOB), `model_info` (TEXT 또는 JSON) 등. 이 테이블의 ID는 `relationship_metadata.embedding_id`를 통해 하이퍼엣지와 연결됩니다.
- **검색 방식 (하이브리드):**
    - 벡터 데이터는 SQLite의 임베딩 테이블에 영구 저장합니다.
    - 빠른 유사도 검색을 위해, 벡터 데이터를 메모리로 로드하여 **Faiss** 또는 **HNSW** 라이브러리 기반의 외부 인덱스를 구축하고 사용합니다.
    - SQLite의 ID와 외부 벡터 인덱스를 매핑하여 관리합니다.
- **임베딩 생성 및 업데이트:** 일반적으로 **사전 계산**합니다. 관계 임베딩은 참여 엔티티와 역할을 조합하여 생성할 수 있습니다. 엔티티/관계 변경 시 임베딩 업데이트 전략이 필요합니다.

### 3. 그래프 순회

- **기본 방식:** SQLite의 **재귀적 CTE (`WITH RECURSIVE`)** 를 사용하여 이진 관계(`edges`) 및 하이퍼엣지(`hyperedge_members` 조인) 기반의 순회를 구현합니다.
- **최적화:** 성능 향상을 위해 순환 감지, 깊이 제한, 적절한 인덱스 활용이 중요합니다. 필요시 경로 미리 계산, 인접 리스트, 뷰 정의 등을 고려합니다.

### 4. MCP 서버 인터페이스

- **프로토콜:** **JSON-RPC 2.0** 표준 메시지 형식을 사용합니다.
- **전송 방식:** **WebSocket**을 주 전송 방식으로 사용합니다.
- **주요 API 엔드포인트:**
    - 엔티티 CRUD
    - 관계 생성/조회/수정/삭제 (API 레벨에서는 하이퍼엣지 모델의 복잡성을 추상화하여 제공)
    - 관찰 추가/삭제
    - 유사도 기반 검색 (`search_similar_nodes`, `search_similar_relationships`)
    - 그래프 순회 (`get_neighbors`, `find_paths` 등)
    - 속성 기반 쿼리
- **핵심 역할:** MCP 서버는 하이퍼엣지 관련 테이블(members, metadata, properties) 조인 및 데이터 조작의 복잡성을 **추상화**하여 클라이언트에게는 일관되고 사용하기 쉬운 그래프 인터페이스를 제공해야 합니다.

### 5. 성능 튜닝 및 확장성 전략

- **SQLite 최적화:**
    - **기본:** `PRAGMA journal_mode=WAL;`, `PRAGMA busy_timeout=5000;` (적절한 값 설정), `PRAGMA synchronous=NORMAL;` (상황에 따라 고려) 등을 기본으로 사용합니다.
    - **고급 인덱싱:** 부분 인덱스, 커버링 인덱스, 표현식/생성된 컬럼 기반 인덱스를 활용하여 쿼리 성능을 극대화합니다.
    - **쿼리 플랜 분석:** `EXPLAIN QUERY PLAN`, `ANALYZE`를 주기적으로 사용하여 쿼리 성능 병목을 식별하고 인덱스 설계를 개선합니다.
    - **메모리/저장소:** `PRAGMA cache_size`, `PRAGMA temp_store=MEMORY` 등을 시스템 환경에 맞게 조정합니다. `mmap_size`는 읽기 성능 향상에 도움이 될 수 있으나, 충돌 시 데이터 손상 위험을 인지하고 사용해야 합니다.
    - **KG 특화 최적화:** 재귀 CTE 사용 시 관련 컬럼 인덱싱 및 깊이 제한 적용, 양방향 순회를 위한 인덱스 생성(`(source_id)`, `(target_id)` 모두), 문 캐시(Statement Cache) 및 애플리케이션 레벨 캐시(LRU) 활용을 고려합니다.
- **대규모 데이터 처리:**
    - **현실적 방안:** 그래프 데이터의 연결성을 고려하여 임의 샤딩은 피합니다. 대신, **노드 중심 파티셔닝**(노드 ID/도메인 기준 분리) 또는 **전체 읽기 복제본**(`Litestream`, `rqlite` 등 활용)을 통한 읽기 확장성을 고려합니다.
    - **한계 인지:** SQLite 단일 파일 크기 및 쓰기 성능 한계를 인지하고, 필요시 외부 분산 시스템(분산 SQL, 그래프 DB) 연동 또는 하이브리드 아키텍처를 고려합니다.
- **외부 벡터 인덱스 성능 관리:**
    - **튜닝:** Faiss/HNSW 인덱스 파라미터(`M`, `efSearch`, `efConstruction`)를 조정하고, 메모리 효율적인 인덱스 타입(예: PQ) 사용을 고려합니다.
    - **영속성/백업:** 인덱스 파일 직렬화 및 버전 관리, SQLite DB와의 **통합 스냅샷**(쓰기 중지 후 동시 스냅샷), WAL 체크포인트 활용, 클라우드 스토리지 백업 등을 통해 일관성을 유지합니다.
    - **복구:** 임베딩 버전 관리(`embedding_version` 컬럼 활용)를 통해 롤백 시 오래된 임베딩을 감지하고 재처리할 수 있도록 합니다. 복구 시나리오에 대한 테스트가 중요합니다.
- **쓰기 동시성:**
    - **기본 전략:** 튜닝된 WAL 모드와 짧은 트랜잭션(관련 쓰기를 묶어 배치 처리)을 기본으로 합니다.
    - **추가 방안:** 매우 빈번한 업데이트 경로에 대한 가벼운 비정규화(예: 노드 `properties`에 카운트 저장), 쓰기 경합 시 연쇄 실패 방지를 위한 서킷 브레이커 패턴 적용을 고려합니다. (앱 레벨 큐/샤딩은 복잡성 증가로 인해 신중히 접근합니다.)

### 6. 데이터 일관성 및 트랜잭션 관리

- **SQLite-벡터 인덱스 동기화:**
    - **기본 패턴:** **비동기 + Transactional Outbox** 패턴을 기본으로 채택합니다. DB 트랜잭션은 관계형 데이터에 집중하고, 벡터 연산 의도는 동일 트랜잭션 내 Outbox 테이블에 기록 후 별도 프로세스에서 비동기 처리합니다. 이를 통해 쓰기 성능과 인덱싱 작업을 분리합니다.
    - **벡터 연산:** 벡터 인덱스 추가/수정/삭제 작업은 **멱등성(Idempotent)** 있게 설계하여 안전한 재시도를 보장합니다.
- **복합 트랜잭션 원자성:**
    - 애플리케이션 레벨에서 **Unit of Work** 패턴을 사용하여 관련된 여러 테이블의 DB 변경 작업을 단일 SQLite 트랜잭션(`BEGIN IMMEDIATE ... COMMIT/ROLLBACK`)으로 묶어 처리합니다.
- **복구 및 일관성 유지:**
    - **실패 처리:** 동기화 실패 시, 상세 정보(상관관계 ID, 재시도 횟수, 오류 내용 등)를 포함하여 **`sync_failures` 로그 테이블**에 기록합니다. 별도의 복구 서비스가 이를 처리하며, 재시도 횟수 초과 시 **데드-레터 큐(Dead-Letter Queue)**로 보내 수동 개입을 유도합니다.
    - **주기적 조정(Reconciliation):** 정기적으로 DB와 벡터 인덱스 간의 불일치를 검사하고(예: 임베딩 버전 비교, 해시 비교) 자동으로 수정하는 프로세스를 실행하여 최종 일관성을 보장합니다.
    - **롤백:** Command-Log 방식 대신, 멱등성 연산과 조정 프로세스에 의존하여 복잡성을 줄입니다. DB 롤백 시 관련 Outbox 항목도 제거하거나 처리되지 않도록 관리합니다.
- **선택적 동기 경로:** 즉각적인 일관성이 매우 중요한 소수의 **핵심 작업**(예: 초기 데이터 로딩)에 한해 **제한적으로 동기식 처리 경로**를 허용할 수 있습니다. 단, 이 경로는 **엄격한 타임아웃**과 **서킷 브레이커** 패턴을 적용하여 시스템 전체 영향 최소화해야 합니다.
- **모니터링:** 트랜잭션 시간, 잠금 대기, Outbox 큐 길이, `sync_failures` 발생 빈도 등을 모니터링하여 병목 및 이상 징후를 조기에 감지합니다.

이러한 설계 고려 사항은 SQLite의 관계형 강점과 성능을 활용하면서도, 유연한 지식 그래프 표현과 벡터 검색 기능을 통합하고, MCP를 통해 효과적인 인터페이스를 제공하는 것을 목표로 합니다.

## 의존성

- `numpy`: 벡터 임베딩의 효율적인 배열 연산을 위함
- `fastmcp`: 서버 및 클라이언트 통신을 위한 Model Context Protocol의 Python 구현
- SQLite: 내장 데이터베이스 저장소 (추가 설치 필요 없음)

선택적 의존성:
- `hnswlib`: 계층적 탐색 가능한 작은 세계(HNSW) 그래프를 사용한 효율적인 근사 최근접 이웃 검색을 위함

## 기여하기

기여는 언제나 환영합니다! 다음 단계를 따라주세요:

1. 저장소 포크하기
2. 기능 브랜치 생성하기 (`git checkout -b feature/amazing-feature`)
3. 변경사항 작업하기
4. 테스트 실행하기 (`pytest`)
5. 변경사항 커밋하기 (`git commit -m 'Add amazing feature'`)
6. 브랜치에 푸시하기 (`git push origin feature/amazing-feature`)
7. Pull Request 열기

코드 형식화를 위해 Black과 isort를 사용하고, mypy 타입 검사를 통과하는 등 코딩 표준을 따라주세요.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 참고 자료

- [MCP 메모리 서버 구현 예시 (modelcontextprotocol/servers memory)](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)