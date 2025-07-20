# HNSW 성능 벤치마크

이 문서는 SQLite KG Vec MCP 프로젝트의 HNSW 인덱스 성능을 측정하고 분석하는 방법을 설명합니다.

## 빠른 시작

### 1. 의존성 설치

```bash
# 벤치마크 의존성 설치
uv pip install psutil matplotlib

# 또는 개발 환경 전체 설치
uv pip install -e ".[dev,benchmark]"
```

### 2. 빠른 성능 확인

```bash
# 빠른 성능 테스트 (약 10초)
python -m pytest tests/test_hnsw_benchmark.py::HNSWPerformanceTest::test_quick_performance_check -v

# 또는 인터랙티브 실행
python scripts/run_hnsw_benchmark.py
```

### 3. 상세 벤치마크 실행

```bash
# 대화형 벤치마크 실행기
python scripts/run_hnsw_benchmark.py

# 직접 벤치마크 클래스 실행
python tests/test_hnsw_benchmark.py
```

## 벤치마크 종류

### 1. 빠른 성능 확인 (권장)
- **실행 시간**: ~10초
- **데이터셋**: 1,000개 벡터, 128차원
- **목적**: CI/CD 및 기본 성능 확인

**성능 기준**:
- 인덱스 구축 시간: < 10초
- 검색 속도: > 100 QPS
- 재현율: > 80%

### 2. 차원 스케일링 테스트
- **실행 시간**: ~5분
- **테스트 차원**: 128, 384, 768, 1536
- **목적**: 벡터 차원 증가에 따른 성능 변화 분석

### 3. 데이터셋 크기 테스트
- **실행 시간**: ~10분
- **데이터셋 크기**: 1K, 5K, 10K, 50K
- **목적**: 데이터 규모 증가에 따른 성능 변화 분석

### 4. 파라미터 튜닝 테스트
- **실행 시간**: ~15분
- **테스트 설정**: 
  - 작은 설정: M=16, ef_construction=200
  - 중간 설정: M=32, ef_construction=400
  - 큰 설정: M=64, ef_construction=800
- **목적**: 최적 HNSW 파라미터 찾기

### 5. 종합 벤치마크
- **실행 시간**: ~30분
- **데이터셋**: 50,000개 벡터, 768차원
- **목적**: 실제 사용 시나리오 성능 측정

## 성능 메트릭

### 구축 성능
- **인덱스 구축 시간**: 전체 벡터를 인덱스에 추가하는 시간
- **메모리 사용량**: 인덱스 구축 후 추가 메모리 사용량
- **초당 벡터 추가**: vectors/second

### 검색 성능
- **QPS (Queries Per Second)**: 초당 검색 쿼리 수
- **평균 검색 시간**: 단일 검색의 평균 시간 (ms)
- **재현율 (Recall)**: 정확한 nearest neighbors 찾기 비율

### 정확도 메트릭
- **평균 재현율**: 모든 쿼리의 평균 재현율
- **최소/최대 재현율**: 재현율의 범위
- **재현율 표준편차**: 재현율의 일관성

## 벤치마크 결과 해석

### 일반적인 성능 지표

| 데이터셋 크기 | 구축 시간 | 검색 속도 | 재현율 | 메모리 사용 |
|------------|----------|----------|--------|------------|
| 1K vectors | < 1초 | > 5000 QPS | > 95% | < 10MB |
| 10K vectors | < 10초 | > 1000 QPS | > 90% | < 100MB |
| 50K vectors | < 60초 | > 500 QPS | > 85% | < 500MB |

### 파라미터 설정 가이드

#### 소규모 데이터셋 (< 10K)
```python
config = {
    "M": 16,
    "ef_construction": 200,
    "ef_search": 50
}
```
- **특징**: 빠른 구축, 적은 메모리 사용
- **적합한 경우**: 프로토타입, 소규모 서비스

#### 중간 규모 데이터셋 (10K-50K)
```python
config = {
    "M": 32,
    "ef_construction": 400,
    "ef_search": 100
}
```
- **특징**: 균형잡힌 성능과 정확도
- **적합한 경우**: 일반적인 프로덕션 환경

#### 대규모 데이터셋 (> 50K)
```python
config = {
    "M": 64,
    "ef_construction": 800,
    "ef_search": 200
}
```
- **특징**: 높은 정확도, 많은 메모리 사용
- **적합한 경우**: 고정확도가 필요한 시스템

## 성능 최적화 팁

### 1. 메모리 최적화
- 벡터 차원을 가능한 낮게 유지
- 불필요한 메타데이터 제거
- 배치 단위로 벡터 추가

### 2. 검색 속도 최적화
- `ef_search` 값 조정 (정확도 vs 속도 트레이드오프)
- 적절한 `M` 값 선택
- 쿼리 배치 처리

### 3. 정확도 최적화
- `ef_construction` 값 증가
- `M` 값 증가
- 벡터 정규화 확인

### 4. 하드웨어 고려사항
- **CPU**: 멀티코어 활용 (hnswlib은 OpenMP 지원)
- **메모리**: 인덱스 크기의 1.5-2배 여유 메모리 권장
- **저장소**: SSD 사용 시 인덱스 로딩 속도 향상

## 문제 해결

### 일반적인 문제들

#### 1. 재현율이 낮음 (< 80%)
```python
# 해결방법: ef_construction 증가
config["ef_construction"] = 800
config["ef_search"] = 200
```

#### 2. 검색 속도가 느림 (< 100 QPS)
```python
# 해결방법: ef_search 감소
config["ef_search"] = 50
# 또는 M 값 감소
config["M"] = 16
```

#### 3. 메모리 사용량이 많음
```python
# 해결방법: M 값 감소
config["M"] = 16
# 또는 벡터 차원 축소 고려
```

#### 4. 인덱스 구축이 느림
```python
# 해결방법: ef_construction 감소
config["ef_construction"] = 200
# 배치 단위로 벡터 추가
```

## 커스텀 벤치마크

### 자체 데이터셋으로 벤치마크

```python
from tests.test_hnsw_benchmark import HNSWBenchmarkTest
import numpy as np

# 커스텀 벤치마크 클래스
class CustomBenchmark(HNSWBenchmarkTest):
    def test_my_dataset(self):
        # 자체 데이터 로드
        vectors = np.load("my_vectors.npy")
        queries = np.load("my_queries.npy")
        
        # 벤치마크 실행
        config = {"M": 32, "ef_construction": 400, "ef_search": 100}
        construction_result, index = self.benchmark_index_construction(vectors, config)
        search_results = self.benchmark_search_performance(
            index, queries, vectors, [10]
        )
        
        # 결과 출력
        print(f"QPS: {search_results['k10']['queries_per_second']:.1f}")
        print(f"Recall: {search_results['k10']['average_recall']:.3f}")

# 실행
benchmark = CustomBenchmark()
benchmark.setUp()
benchmark.test_my_dataset()
benchmark.tearDown()
```

## 연속 통합 (CI)

### GitHub Actions 예제

```yaml
name: HNSW Performance Test

on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv pip install -e ".[benchmark]"
    
    - name: Run quick performance test
      run: |
        python -m pytest tests/test_hnsw_benchmark.py::HNSWPerformanceTest::test_quick_performance_check -v
```

## 추가 리소스

- [hnswlib 공식 문서](https://github.com/nmslib/hnswlib)
- [HNSW 알고리즘 논문](https://arxiv.org/abs/1603.09320)
- [벡터 검색 최적화 가이드](https://www.pinecone.io/learn/hnsw/)

---

벤치마크 관련 질문이나 문제가 있으면 GitHub Issues에 문의해주세요.