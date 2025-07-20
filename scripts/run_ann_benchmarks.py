#!/usr/bin/env python3
"""
ann-benchmarks 통합 실행 스크립트.
표준 ANN 벤치마크 데이터셋으로 sqlite-kg-vec-mcp HNSW 성능 측정.
"""

import sys
import os
import argparse
import tempfile
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import h5py

# Add paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external" / "ann-benchmarks"))

# Import our adapter
from ann_benchmarks_adapter.sqlite_kg_hnsw import SqliteKgHNSW

# Import ann-benchmarks utilities if available
try:
    from ann_benchmarks.datasets import get_dataset
    from ann_benchmarks.distance import metrics as ann_metrics
    from ann_benchmarks.results import store_results
    ANN_BENCHMARKS_AVAILABLE = True
except ImportError:
    ANN_BENCHMARKS_AVAILABLE = False
    print("Warning: ann-benchmarks not fully available. Using fallback implementation.")


class SimpleDatasetDownloader:
    """간단한 데이터셋 다운로더 (ann-benchmarks가 없는 경우)."""
    
    @staticmethod
    def create_random_dataset(n_samples: int, n_features: int, name: str = "random") -> Tuple[np.ndarray, np.ndarray]:
        """랜덤 데이터셋 생성."""
        np.random.seed(42)  # 재현 가능한 결과
        
        # 클러스터된 데이터 생성
        n_clusters = min(10, n_samples // 100)
        if n_clusters < 1:
            n_clusters = 1
        
        data = []
        samples_per_cluster = n_samples // n_clusters
        
        for i in range(n_clusters):
            # 클러스터 중심
            center = np.random.randn(n_features)
            
            # 클러스터 주변 데이터
            cluster_data = np.random.randn(samples_per_cluster, n_features) * 0.3 + center
            data.append(cluster_data)
        
        # 남은 데이터
        remaining = n_samples - len(data) * samples_per_cluster
        if remaining > 0:
            extra_data = np.random.randn(remaining, n_features)
            data.append(extra_data)
        
        X = np.vstack(data).astype(np.float32)
        
        # L2 정규화 (cosine distance를 위해)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (norms + 1e-8)
        
        # 쿼리 셋 (데이터의 일부)
        n_queries = min(1000, n_samples // 10)
        query_indices = np.random.choice(n_samples, n_queries, replace=False)
        queries = X[query_indices].copy()
        
        return X, queries
    
    @staticmethod
    def get_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """데이터셋 가져오기."""
        datasets = {
            "random-xs": (1000, 50),
            "random-s": (10000, 100),
            "random-m": (50000, 200),
            "random-l": (100000, 500),
        }
        
        if name not in datasets:
            name = "random-s"  # 기본값
        
        n_samples, n_features = datasets[name]
        X, queries = SimpleDatasetDownloader.create_random_dataset(n_samples, n_features, name)
        
        return X, queries, "cosine"


class AnnBenchmarkRunner:
    """ann-benchmarks 실행기."""
    
    def __init__(self, output_dir: str = None):
        """
        초기화.
        
        Args:
            output_dir: 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir) if output_dir else Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 사용 가능한 데이터셋
        if ANN_BENCHMARKS_AVAILABLE:
            self.available_datasets = [
                "fashion-mnist-784-euclidean",
                "gist-960-euclidean", 
                "glove-25-angular",
                "glove-100-angular",
                "mnist-784-euclidean",
                "sift-128-euclidean"
            ]
        else:
            self.available_datasets = ["random-xs", "random-s", "random-m"]
    
    def download_dataset(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        데이터셋 다운로드.
        
        Args:
            dataset_name: 데이터셋 이름
            
        Returns:
            (train_data, test_queries, metric)
        """
        print(f"📥 데이터셋 다운로드: {dataset_name}")
        
        if ANN_BENCHMARKS_AVAILABLE:
            try:
                dataset = get_dataset(dataset_name)
                return dataset['train'], dataset['test'], dataset['distance']
            except Exception as e:
                print(f"⚠️ ann-benchmarks 데이터셋 로드 실패: {e}")
                print("대체 데이터셋을 사용합니다.")
        
        # Fallback to simple dataset
        return SimpleDatasetDownloader.get_dataset(dataset_name)
    
    def run_benchmark(
        self, 
        dataset_name: str, 
        algorithm_configs: List[Dict[str, Any]] = None,
        ef_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        벤치마크 실행.
        
        Args:
            dataset_name: 데이터셋 이름
            algorithm_configs: 알고리즘 설정 리스트
            ef_values: ef 파라미터 값들
            
        Returns:
            벤치마크 결과
        """
        # 기본 설정
        if algorithm_configs is None:
            algorithm_configs = [
                {"M": 16, "efConstruction": 200},
                {"M": 32, "efConstruction": 400},
                {"M": 64, "efConstruction": 800},
            ]
        
        if ef_values is None:
            ef_values = [100, 200, 400, 800]
        
        # 데이터셋 로드
        train_data, test_queries, metric = self.download_dataset(dataset_name)
        
        print(f"📊 데이터셋 정보:")
        print(f"   훈련 데이터: {train_data.shape}")
        print(f"   쿼리 데이터: {test_queries.shape}")
        print(f"   거리 메트릭: {metric}")
        
        results = {
            "dataset": dataset_name,
            "metric": metric,
            "train_size": len(train_data),
            "test_size": len(test_queries),
            "dimension": train_data.shape[1],
            "timestamp": time.time(),
            "configurations": []
        }
        
        # 각 설정에 대해 벤치마크 실행
        for config_idx, config in enumerate(algorithm_configs):
            print(f"\n🔧 설정 {config_idx + 1}/{len(algorithm_configs)}: {config}")
            
            # 인덱스 구축
            build_start = time.time()
            
            algo = SqliteKgHNSW(metric, config)
            
            try:
                algo.fit(train_data)
                build_time = time.time() - build_start
                
                print(f"   인덱스 구축 시간: {build_time:.2f}초")
                
                config_results = {
                    "config": config,
                    "build_time": build_time,
                    "memory_usage_kb": algo.get_memory_usage(),
                    "additional_info": algo.get_additional(),
                    "ef_results": []
                }
                
                # 각 ef 값에 대해 쿼리 성능 측정
                for ef in ef_values:
                    print(f"   🔍 ef={ef} 테스트 중...")
                    
                    algo.set_query_arguments(ef)
                    
                    # 쿼리 성능 측정
                    query_start = time.time()
                    
                    # 배치 쿼리 실행
                    algo.batch_query(test_queries, 10)
                    all_results = algo.get_batch_results()
                    
                    query_time = time.time() - query_start
                    
                    # 재현율 계산 (브루트 포스와 비교)
                    recall = self.calculate_recall(train_data, test_queries, all_results, metric)
                    
                    qps = len(test_queries) / query_time
                    avg_query_time = (query_time / len(test_queries)) * 1000
                    
                    ef_result = {
                        "ef": ef,
                        "total_query_time": query_time,
                        "queries_per_second": qps,
                        "avg_query_time_ms": avg_query_time,
                        "recall_at_10": recall,
                        "memory_usage_kb": algo.get_memory_usage()
                    }
                    
                    config_results["ef_results"].append(ef_result)
                    
                    memory_kb = ef_result['memory_usage_kb'] or 0
                    print(f"      QPS: {qps:.1f}, 재현율: {recall:.3f}, 메모리: {memory_kb:.0f}KB")
                
                results["configurations"].append(config_results)
                
            except Exception as e:
                print(f"   ❌ 설정 실행 실패: {e}")
                continue
            
            finally:
                algo.done()
        
        return results
    
    def calculate_recall(
        self, 
        train_data: np.ndarray, 
        test_queries: np.ndarray, 
        ann_results: List[List[int]],
        metric: str,
        k: int = 10
    ) -> float:
        """
        재현율 계산 (브루트 포스와 비교).
        
        Args:
            train_data: 훈련 데이터
            test_queries: 쿼리 데이터
            ann_results: ANN 검색 결과 (리스트의 리스트)
            metric: 거리 메트릭
            k: 상위 k개 결과
            
        Returns:
            평균 재현율
        """
        recalls = []
        
        # 모든 쿼리에 대해 계산 (작은 데이터셋이므로)
        max_queries = min(len(test_queries), len(ann_results))
        
        for i in range(max_queries):
            query = test_queries[i]
            ann_neighbors = ann_results[i][:k] if len(ann_results[i]) >= k else ann_results[i]
            
            # 브루트 포스 검색
            if metric in ["cosine", "angular"]:
                # 정규화된 벡터에 대한 코사인 유사도
                # train_data와 query 모두 이미 정규화되어 있음
                similarities = np.dot(train_data, query)
                true_neighbors = np.argsort(-similarities)[:k]
            else:
                # 유클리드 거리
                distances = np.linalg.norm(train_data - query, axis=1)
                true_neighbors = np.argsort(distances)[:k]
            
            # 재현율 계산
            intersection = set(ann_neighbors) & set(true_neighbors)
            recall = len(intersection) / k if k > 0 else 0.0
            recalls.append(recall)
            
            # 디버깅 정보 (첫 번째 쿼리에 대해서만)
            if i == 0:
                print(f"   디버그: ANN={ann_neighbors[:5]}, 실제={true_neighbors[:5]}, 교집합={len(intersection)}")
        
        return np.mean(recalls)
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        결과 저장.
        
        Args:
            results: 벤치마크 결과
            filename: 파일명 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"benchmark_{results['dataset']}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장: {filepath}")
        return str(filepath)
    
    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력."""
        print(f"\n📈 벤치마크 요약 - {results['dataset']}")
        print("=" * 60)
        
        for config_result in results["configurations"]:
            config = config_result["config"]
            print(f"\n🔧 M={config['M']}, efConstruction={config['efConstruction']}")
            print(f"   구축 시간: {config_result['build_time']:.2f}초")
            memory_kb = config_result['memory_usage_kb'] or 0
            print(f"   메모리 사용: {memory_kb:.0f}KB")
            
            best_qps = 0
            best_recall = 0
            
            for ef_result in config_result["ef_results"]:
                qps = ef_result["queries_per_second"]
                recall = ef_result["recall_at_10"]
                
                if qps > best_qps:
                    best_qps = qps
                if recall > best_recall:
                    best_recall = recall
            
            print(f"   최고 QPS: {best_qps:.1f}")
            print(f"   최고 재현율: {best_recall:.3f}")


def main():
    """메인 실행 함수."""
    parser = argparse.ArgumentParser(description="ann-benchmarks를 사용한 HNSW 벤치마크")
    parser.add_argument("--dataset", type=str, default="random-s", 
                       help="벤치마크 데이터셋")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="결과 저장 디렉토리")
    parser.add_argument("--quick", action="store_true",
                       help="빠른 테스트 (작은 설정만)")
    
    args = parser.parse_args()
    
    print("🚀 ann-benchmarks 통합 벤치마크 시작")
    print("=" * 60)
    
    runner = AnnBenchmarkRunner(args.output_dir)
    
    # 설정 선택
    if args.quick:
        configs = [{"M": 16, "efConstruction": 200}]
        ef_values = [100, 200]
    else:
        configs = [
            {"M": 16, "efConstruction": 200},
            {"M": 32, "efConstruction": 400},
            {"M": 64, "efConstruction": 800},
        ]
        ef_values = [50, 100, 200, 400]
    
    try:
        # 벤치마크 실행
        results = runner.run_benchmark(
            dataset_name=args.dataset,
            algorithm_configs=configs,
            ef_values=ef_values
        )
        
        # 결과 저장 및 요약
        runner.save_results(results)
        runner.print_summary(results)
        
        print(f"\n✅ 벤치마크 완료!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n❌ 벤치마크 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())