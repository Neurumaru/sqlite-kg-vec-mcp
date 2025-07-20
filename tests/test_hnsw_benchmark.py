"""
HNSW 인덱스 성능 벤치마크 테스트.
"""

import time
import os
import tempfile
import unittest
import numpy as np
import psutil
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlite_kg_vec_mcp.vector.hnsw import HNSWIndex


class PerformanceMetrics:
    """성능 측정 메트릭 클래스."""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """타이머 시작."""
        self.metrics[f"{name}_start"] = time.time()
    
    def end_timer(self, name: str):
        """타이머 종료 및 소요 시간 기록."""
        if f"{name}_start" in self.metrics:
            duration = time.time() - self.metrics[f"{name}_start"]
            self.metrics[f"{name}_duration"] = duration
            return duration
        return None
    
    def record_memory(self, name: str):
        """현재 메모리 사용량 기록."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics[f"{name}_memory_mb"] = memory_mb
        return memory_mb
    
    def record_value(self, name: str, value: Any):
        """임의의 값 기록."""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """모든 메트릭 반환."""
        return self.metrics.copy()


class HNSWBenchmarkTest(unittest.TestCase):
    """HNSW 인덱스 벤치마크 테스트."""
    
    def setUp(self):
        """테스트 설정."""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics = PerformanceMetrics()
        self.results = []
        
        # 테스트 설정
        self.dimensions = [128, 384, 768, 1536]  # 다양한 벡터 차원
        self.dataset_sizes = [1000, 5000, 10000, 50000]  # 다양한 데이터셋 크기
        self.search_k_values = [1, 5, 10, 50]  # 다양한 k 값
        
        # HNSW 파라미터 설정
        self.hnsw_configs = [
            {"M": 16, "ef_construction": 200, "ef_search": 50},
            {"M": 32, "ef_construction": 400, "ef_search": 100},
            {"M": 64, "ef_construction": 800, "ef_search": 200},
        ]
    
    def tearDown(self):
        """테스트 정리."""
        # 임시 디렉토리 정리
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def generate_random_vectors(self, n: int, dim: int) -> np.ndarray:
        """랜덤 벡터 생성."""
        vectors = np.random.randn(n, dim).astype(np.float32)
        # L2 정규화
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-8)
        return vectors
    
    def generate_clustered_vectors(self, n: int, dim: int, n_clusters: int = 10) -> np.ndarray:
        """클러스터된 벡터 생성 (더 현실적인 데이터)."""
        vectors = []
        vectors_per_cluster = n // n_clusters
        
        for i in range(n_clusters):
            # 클러스터 중심점
            center = np.random.randn(dim).astype(np.float32)
            center = center / np.linalg.norm(center)
            
            # 클러스터 주변의 벡터들
            cluster_vectors = np.random.randn(vectors_per_cluster, dim).astype(np.float32) * 0.3
            cluster_vectors += center
            
            # 정규화
            norms = np.linalg.norm(cluster_vectors, axis=1, keepdims=True)
            cluster_vectors = cluster_vectors / (norms + 1e-8)
            
            vectors.append(cluster_vectors)
        
        # 남은 벡터들
        remaining = n - len(vectors) * vectors_per_cluster
        if remaining > 0:
            extra_vectors = self.generate_random_vectors(remaining, dim)
            vectors.append(extra_vectors)
        
        return np.vstack(vectors)
    
    def calculate_recall(self, true_neighbors: List[int], found_neighbors: List[int]) -> float:
        """재현율(Recall) 계산."""
        if not true_neighbors or not found_neighbors:
            return 0.0
        
        true_set = set(true_neighbors)
        found_set = set(found_neighbors)
        intersection = true_set.intersection(found_set)
        
        return len(intersection) / len(true_set)
    
    def brute_force_search(self, query: np.ndarray, vectors: np.ndarray, k: int) -> List[int]:
        """브루트 포스 검색 (정답 기준)."""
        distances = np.linalg.norm(vectors - query, axis=1)
        indices = np.argsort(distances)[:k]
        return indices.tolist()
    
    def benchmark_index_construction(self, vectors: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """인덱스 구축 성능 벤치마크."""
        metrics = PerformanceMetrics()
        
        metrics.record_memory("before_construction")
        metrics.start_timer("construction")
        
        # HNSW 인덱스 생성
        index = HNSWIndex(
            space="cosine",
            dim=vectors.shape[1],
            ef_construction=config["ef_construction"],
            M=config["M"],
            index_dir=self.temp_dir
        )
        
        index.init_index(max_elements=len(vectors))
        
        # 벡터 추가
        for i, vector in enumerate(vectors):
            index.add_item(
                entity_type="test",
                entity_id=i,
                vector=vector
            )
        
        construction_time = metrics.end_timer("construction")
        metrics.record_memory("after_construction")
        
        # ef_search 설정
        if hasattr(index.index, 'set_ef'):
            index.index.set_ef(config.get("ef_search", 50))
        
        result = {
            "construction_time": construction_time,
            "memory_before_mb": metrics.metrics["before_construction_memory_mb"],
            "memory_after_mb": metrics.metrics["after_construction_memory_mb"],
            "memory_used_mb": metrics.metrics["after_construction_memory_mb"] - metrics.metrics["before_construction_memory_mb"],
            "vectors_count": len(vectors),
            "dimension": vectors.shape[1],
            "config": config
        }
        
        return result, index
    
    def benchmark_search_performance(
        self, 
        index: HNSWIndex, 
        query_vectors: np.ndarray, 
        all_vectors: np.ndarray,
        k_values: List[int]
    ) -> Dict[str, Any]:
        """검색 성능 벤치마크."""
        results = {}
        
        for k in k_values:
            metrics = PerformanceMetrics()
            recalls = []
            
            metrics.start_timer(f"search_k{k}")
            
            for query in query_vectors:
                # HNSW 검색
                hnsw_results = index.search(query, k)
                hnsw_indices = [result[1] for result in hnsw_results]
                
                # 브루트 포스 검색 (정답)
                true_indices = self.brute_force_search(query, all_vectors, k)
                
                # 재현율 계산
                recall = self.calculate_recall(true_indices, hnsw_indices)
                recalls.append(recall)
            
            search_time = metrics.end_timer(f"search_k{k}")
            
            results[f"k{k}"] = {
                "total_search_time": search_time,
                "avg_search_time_ms": (search_time / len(query_vectors)) * 1000,
                "queries_per_second": len(query_vectors) / search_time,
                "average_recall": np.mean(recalls),
                "min_recall": np.min(recalls),
                "max_recall": np.max(recalls),
                "recall_std": np.std(recalls)
            }
        
        return results
    
    def test_dimension_scaling_benchmark(self):
        """차원 증가에 따른 성능 변화 벤치마크."""
        print("\n🔬 차원 스케일링 벤치마크 시작...")
        
        dataset_size = 10000
        query_size = 100
        config = self.hnsw_configs[1]  # 중간 설정 사용
        
        for dim in self.dimensions:
            print(f"\n📊 차원: {dim}")
            
            # 데이터 생성
            vectors = self.generate_clustered_vectors(dataset_size, dim)
            query_vectors = self.generate_random_vectors(query_size, dim)
            
            # 인덱스 구축 벤치마크
            construction_result, index = self.benchmark_index_construction(vectors, config)
            
            # 검색 성능 벤치마크
            search_results = self.benchmark_search_performance(
                index, query_vectors, vectors, [10]
            )
            
            result = {
                "dimension": dim,
                "dataset_size": dataset_size,
                "construction": construction_result,
                "search": search_results["k10"]
            }
            
            self.results.append(result)
            
            print(f"   구축 시간: {construction_result['construction_time']:.2f}초")
            print(f"   메모리 사용: {construction_result['memory_used_mb']:.1f}MB")
            print(f"   검색 속도: {search_results['k10']['queries_per_second']:.1f} QPS")
            print(f"   평균 재현율: {search_results['k10']['average_recall']:.3f}")
    
    def test_dataset_size_scaling_benchmark(self):
        """데이터셋 크기 증가에 따른 성능 변화 벤치마크."""
        print("\n🔬 데이터셋 크기 스케일링 벤치마크 시작...")
        
        dim = 384  # 고정 차원
        query_size = 100
        config = self.hnsw_configs[1]  # 중간 설정 사용
        
        for size in self.dataset_sizes:
            print(f"\n📊 데이터셋 크기: {size:,}")
            
            # 데이터 생성
            vectors = self.generate_clustered_vectors(size, dim)
            query_vectors = self.generate_random_vectors(query_size, dim)
            
            # 인덱스 구축 벤치마크
            construction_result, index = self.benchmark_index_construction(vectors, config)
            
            # 검색 성능 벤치마크
            search_results = self.benchmark_search_performance(
                index, query_vectors, vectors, [10]
            )
            
            result = {
                "dataset_size": size,
                "dimension": dim,
                "construction": construction_result,
                "search": search_results["k10"]
            }
            
            self.results.append(result)
            
            print(f"   구축 시간: {construction_result['construction_time']:.2f}초")
            print(f"   메모리 사용: {construction_result['memory_used_mb']:.1f}MB")
            print(f"   검색 속도: {search_results['k10']['queries_per_second']:.1f} QPS")
            print(f"   평균 재현율: {search_results['k10']['average_recall']:.3f}")
    
    def test_parameter_tuning_benchmark(self):
        """HNSW 파라미터 튜닝 벤치마크."""
        print("\n🔬 파라미터 튜닝 벤치마크 시작...")
        
        dataset_size = 20000
        dim = 384
        query_size = 100
        
        # 데이터 생성
        vectors = self.generate_clustered_vectors(dataset_size, dim)
        query_vectors = self.generate_random_vectors(query_size, dim)
        
        for i, config in enumerate(self.hnsw_configs):
            print(f"\n📊 설정 {i+1}: M={config['M']}, ef_construction={config['ef_construction']}, ef_search={config['ef_search']}")
            
            # 인덱스 구축 벤치마크
            construction_result, index = self.benchmark_index_construction(vectors, config)
            
            # 검색 성능 벤치마크
            search_results = self.benchmark_search_performance(
                index, query_vectors, vectors, [1, 10, 50]
            )
            
            result = {
                "config_index": i,
                "config": config,
                "dataset_size": dataset_size,
                "dimension": dim,
                "construction": construction_result,
                "search": search_results
            }
            
            self.results.append(result)
            
            print(f"   구축 시간: {construction_result['construction_time']:.2f}초")
            print(f"   메모리 사용: {construction_result['memory_used_mb']:.1f}MB")
            
            for k in [1, 10, 50]:
                search_result = search_results[f"k{k}"]
                print(f"   k={k}: {search_result['queries_per_second']:.1f} QPS, 재현율 {search_result['average_recall']:.3f}")
    
    def test_comprehensive_benchmark(self):
        """종합 성능 벤치마크."""
        print("\n🔬 종합 성능 벤치마크 시작...")
        
        # 실제 사용 시나리오와 유사한 설정
        config = {"M": 32, "ef_construction": 400, "ef_search": 100}
        dataset_size = 50000
        dim = 768  # OpenAI embedding 차원과 유사
        query_size = 1000
        
        print(f"📊 설정: {dataset_size:,}개 벡터, {dim}차원")
        
        # 클러스터된 데이터 생성 (더 현실적)
        vectors = self.generate_clustered_vectors(dataset_size, dim, n_clusters=20)
        query_vectors = self.generate_random_vectors(query_size, dim)
        
        # 인덱스 구축
        print("🔨 인덱스 구축 중...")
        construction_result, index = self.benchmark_index_construction(vectors, config)
        
        # 다양한 k 값으로 검색 성능 테스트
        print("🔍 검색 성능 테스트 중...")
        search_results = self.benchmark_search_performance(
            index, query_vectors, vectors, self.search_k_values
        )
        
        # 결과 저장
        comprehensive_result = {
            "test_type": "comprehensive",
            "config": config,
            "dataset_size": dataset_size,
            "dimension": dim,
            "query_size": query_size,
            "construction": construction_result,
            "search": search_results
        }
        
        self.results.append(comprehensive_result)
        
        # 결과 출력
        print(f"\n✅ 종합 벤치마크 결과:")
        print(f"   인덱스 구축 시간: {construction_result['construction_time']:.2f}초")
        print(f"   메모리 사용량: {construction_result['memory_used_mb']:.1f}MB")
        print(f"   초당 벡터 추가: {dataset_size / construction_result['construction_time']:.0f} vectors/sec")
        
        print(f"\n   검색 성능:")
        for k in self.search_k_values:
            result = search_results[f"k{k}"]
            print(f"   k={k:2d}: {result['queries_per_second']:6.1f} QPS, 재현율 {result['average_recall']:.3f}")
    
    def save_benchmark_results(self):
        """벤치마크 결과를 파일로 저장."""
        if not self.results:
            return
        
        # JSON 파일로 저장
        results_file = os.path.join(self.temp_dir, "hnsw_benchmark_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 벤치마크 결과가 저장되었습니다: {results_file}")
        return results_file
    
    def generate_performance_report(self):
        """성능 보고서 생성."""
        if not self.results:
            return
        
        print("\n📊 HNSW 성능 벤치마크 보고서")
        print("=" * 50)
        
        # 최고 성능 찾기
        best_qps = 0
        best_recall = 0
        best_construction = float('inf')
        
        for result in self.results:
            if 'search' in result:
                if 'k10' in result['search']:
                    qps = result['search']['k10']['queries_per_second']
                    recall = result['search']['k10']['average_recall']
                    construction_time = result['construction']['construction_time']
                    
                    if qps > best_qps:
                        best_qps = qps
                    if recall > best_recall:
                        best_recall = recall
                    if construction_time < best_construction:
                        best_construction = construction_time
        
        print(f"🏆 최고 검색 속도: {best_qps:.1f} QPS")
        print(f"🎯 최고 재현율: {best_recall:.3f}")
        print(f"⚡ 최단 구축 시간: {best_construction:.2f}초")
        
        # 권장 설정
        print(f"\n💡 권장 설정:")
        print(f"   - 소규모 데이터셋 (< 10K): M=16, ef_construction=200")
        print(f"   - 중간 데이터셋 (10K-50K): M=32, ef_construction=400")
        print(f"   - 대규모 데이터셋 (> 50K): M=64, ef_construction=800")
        print(f"   - 정확도 우선시: ef_search를 높게 설정 (100-200)")
        print(f"   - 속도 우선시: ef_search를 낮게 설정 (50-100)")
    
    def run_all_benchmarks(self):
        """모든 벤치마크 실행."""
        print("🚀 HNSW 성능 벤치마크 시작")
        print("=" * 50)
        
        # 개별 벤치마크 실행
        self.test_dimension_scaling_benchmark()
        self.test_dataset_size_scaling_benchmark()
        self.test_parameter_tuning_benchmark()
        self.test_comprehensive_benchmark()
        
        # 결과 저장 및 보고서 생성
        self.save_benchmark_results()
        self.generate_performance_report()
        
        print(f"\n✅ 모든 벤치마크가 완료되었습니다!")
        print(f"총 {len(self.results)}개의 테스트 결과가 수집되었습니다.")


class HNSWPerformanceTest(HNSWBenchmarkTest):
    """단일 성능 테스트 (CI/CD에서 빠른 실행용)."""
    
    def test_quick_performance_check(self):
        """빠른 성능 확인 테스트."""
        print("\n⚡ 빠른 HNSW 성능 확인...")
        
        # 작은 데이터셋으로 빠른 테스트
        vectors = self.generate_random_vectors(1000, 128)
        query_vectors = self.generate_random_vectors(10, 128)
        config = {"M": 16, "ef_construction": 200, "ef_search": 50}
        
        # 인덱스 구축
        construction_result, index = self.benchmark_index_construction(vectors, config)
        
        # 검색 성능
        search_results = self.benchmark_search_performance(
            index, query_vectors, vectors, [10]
        )
        
        # 성능 기준 검증
        self.assertLess(construction_result['construction_time'], 10.0, "인덱스 구축 시간이 너무 길음")
        self.assertGreater(search_results['k10']['queries_per_second'], 100, "검색 속도가 너무 느림")
        self.assertGreater(search_results['k10']['average_recall'], 0.8, "재현율이 너무 낮음")
        
        print(f"✅ 성능 확인 완료:")
        print(f"   구축 시간: {construction_result['construction_time']:.2f}초")
        print(f"   검색 속도: {search_results['k10']['queries_per_second']:.1f} QPS")
        print(f"   재현율: {search_results['k10']['average_recall']:.3f}")


if __name__ == "__main__":
    # 벤치마크 실행
    benchmark = HNSWBenchmarkTest()
    benchmark.setUp()
    
    try:
        benchmark.run_all_benchmarks()
    finally:
        benchmark.tearDown()