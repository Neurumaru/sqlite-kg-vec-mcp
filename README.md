# sqlite-kg-vec-mcp
MCP Server that integrates a knowledge graph and a vector database based on SQLite

## 프로젝트 개요
SQLite를 기반으로 지식 그래프와 벡터 데이터베이스를 통합하는 MCP 서버입니다. 이 서버는 지식을 효율적으로 저장하고 검색할 수 있는 기능을 제공합니다.

## 주요 기능
- 지식 그래프 관리
  - 엔티티(Entities) 생성, 조회, 수정, 삭제
  - 관계(Relations) 관리
  - 관찰(Observations) 저장 및 조회
  - 설명(Description) 수정
    - 엔티티에 대한 설명, 관계, 관찰 등을 요약해 적어둔 문서 (벡터 데이터베이스 검색에 사용)
- 벡터 데이터베이스
  - 벡터 데이터 저장
  - 유사도 기반 검색
  - 효율적인 벡터 인덱싱
- MCP (Model Context Protocol) 서버로 제공

## 기술 스택
- 데이터베이스: SQLite
- 서버: Python
- API 프레임워크: FastMCP
- 벡터 처리: vectorlite

## 구현 계획

### 1단계: 데이터베이스 스키마 설계
- 지식 그래프 스키마
  - Entities 테이블: 엔티티 정보 저장
  - Relations 테이블: 엔티티 간 관계 저장
  - Observations 테이블: 엔티티에 대한 관찰 저장
- 벡터 데이터베이스 스키마
  - Description 테이블: 엔티티에 대한 설명 저장
  - Vectors 테이블: 벡터 데이터 저장
  - 효율적인 검색을 위한 인덱스 구조

### 2단계: 코어 기능 구현
- SQLite 데이터베이스 관리자
  - 데이터베이스 연결 및 초기화
  - 트랜잭션 관리
- 지식 그래프 관리자
  - CRUD 작업 구현
  - 관계 탐색 기능
- 벡터 데이터베이스 관리자
  - 벡터 저장 및 검색
  - 유사도 계산 최적화

### 3단계: MCP 서버 구현
- MCP 서버 구현
  - 지식 그래프 관련 엔드포인트
  - 벡터 데이터베이스 관련 엔드포인트

## 프로젝트 구조
```
sqlite-kg-vec-mcp/
├── src/
│   ├── core/
│   │   ├── db/
│   │   ├── kg/
│   │   └── vec/
│   └── utils/
└── tests/
```
## 라이선스
MIT License
