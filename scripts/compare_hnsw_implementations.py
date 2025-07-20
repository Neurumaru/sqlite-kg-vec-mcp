#!/usr/bin/env python3
"""
다양한 HNSW 구현체들의 성능 비교 스크립트.
sqlite-kg-vec-mcp vs hnswlib vs FAISS 성능 비교.
"""

import sys
import time
import numpy as np
import psutil
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Add project path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import libraries
import hnswlib
import faiss
from ann_benchmarks_adapter.sqlite_kg_hnsw import SqliteKgHNSW

# 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class HNSWComparator:
    """HNSW 구현체 성능 비교기."""
    
    def __init__(self, output_dir: str = "comparison_results"):
        """
        초기화.
        
        Args:
            output_dir: 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.implementations = {
            "sqlite-kg-vec-mcp-hnswlib": lambda metric, config: SQLiteKgHNSWWrapper(metric, config, "hnswlib"),
            "sqlite-kg-vec-mcp-faiss": lambda metric, config: SQLiteKgHNSWWrapper(metric, config, "faiss"),
            "hnswlib": HnswlibWrapper,
            "faiss": FAISSWrapper
        }
    
    def generate_dataset(self, n_samples: int, n_features: int, metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
        """
        테스트 데이터셋 생성.
        
        Args:
            n_samples: 샘플 수
            n_features: 특성 수
            metric: 거리 메트릭
            
        Returns:
            (train_data, test_queries)
        """
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
        if metric in ["cosine", "angular"]:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / (norms + 1e-8)
        
        # 쿼리 셋 (데이터의 일부 + 새로운 쿼리)
        n_queries = min(100, n_samples // 10)
        query_indices = np.random.choice(n_samples, n_queries // 2, replace=False)
        queries_from_data = X[query_indices].copy()
        
        # 새로운 쿼리 생성
        new_queries = np.random.randn(n_queries // 2, n_features).astype(np.float32)
        if metric in ["cosine", "angular"]:
            norms = np.linalg.norm(new_queries, axis=1, keepdims=True)
            new_queries = new_queries / (norms + 1e-8)
        
        queries = np.vstack([queries_from_data, new_queries])
        
        return X, queries
    
    def run_comparison(
        self, 
        dataset_size: int = 10000,
        dimension: int = 128,
        metric: str = "cosine",
        configs: List[Dict[str, Any]] = None,
        ef_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        비교 벤치마크 실행.
        
        Args:
            dataset_size: 데이터셋 크기
            dimension: 벡터 차원
            metric: 거리 메트릭
            configs: HNSW 설정들
            ef_values: ef 파라미터 값들
            
        Returns:
            비교 결과
        """
        if configs is None:
            configs = [
                {"M": 16, "efConstruction": 200},
                {"M": 32, "efConstruction": 400}
            ]
        
        if ef_values is None:
            ef_values = [50, 100, 200]
        
        print(f"📊 HNSW 구현체 성능 비교")
        print(f"   데이터셋: {dataset_size} 벡터, {dimension} 차원")
        print(f"   메트릭: {metric}")
        print("=" * 60)
        
        # 데이터셋 생성
        train_data, test_queries = self.generate_dataset(dataset_size, dimension, metric)
        
        results = {
            "dataset_info": {
                "size": dataset_size,
                "dimension": dimension,
                "metric": metric,
                "train_size": len(train_data),
                "test_size": len(test_queries)
            },
            "timestamp": time.time(),
            "implementations": {}
        }
        
        # 각 구현체별로 벤치마크 실행
        for impl_name, impl_class in self.implementations.items():
            print(f"\\n🔧 {impl_name} 테스트 중...")
            
            try:
                impl_results = self.benchmark_implementation(
                    impl_class, impl_name, train_data, test_queries, 
                    configs, ef_values, metric
                )
                results["implementations"][impl_name] = impl_results
                
            except Exception as e:
                print(f"   ❌ {impl_name} 테스트 실패: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def benchmark_implementation(
        self,
        impl_class,
        impl_name: str,
        train_data: np.ndarray,
        test_queries: np.ndarray,
        configs: List[Dict[str, Any]],
        ef_values: List[int],
        metric: str
    ) -> Dict[str, Any]:
        """
        특정 구현체 벤치마크.
        
        Args:
            impl_class: 구현체 클래스
            impl_name: 구현체 이름
            train_data: 훈련 데이터
            test_queries: 쿼리 데이터
            configs: 설정들
            ef_values: ef 값들
            metric: 거리 메트릭
            
        Returns:
            벤치마크 결과
        """
        impl_results = {"configurations": []}
        
        for config in configs:
            print(f"   설정: {config}")
            
            # 인덱스 구축
            build_start = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                impl = impl_class(metric, config)
                impl.fit(train_data)
                
                build_time = time.time() - build_start
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
                
                print(f"   구축 시간: {build_time:.3f}초, 메모리: {memory_used:.1f}MB")
                
                config_result = {
                    "config": config,
                    "build_time": build_time,
                    "memory_usage_mb": memory_used,
                    "ef_results": []
                }
                
                # ef 값별 성능 측정
                for ef in ef_values:
                    print(f"     ef={ef} 측정 중...")
                    
                    impl.set_query_arguments(ef)
                    
                    # 쿼리 성능 측정
                    query_start = time.time()
                    
                    # 배치 쿼리 실행
                    impl.batch_query(test_queries, 10)
                    all_results = impl.get_batch_results()
                    
                    query_time = time.time() - query_start
                    
                    # 재현율 계산
                    recall = self.calculate_recall(train_data, test_queries, all_results, metric)
                    
                    qps = len(test_queries) / query_time
                    avg_query_time = (query_time / len(test_queries)) * 1000
                    
                    ef_result = {
                        "ef": ef,
                        "queries_per_second": qps,
                        "avg_query_time_ms": avg_query_time,
                        "recall_at_10": recall,
                        "total_query_time": query_time
                    }
                    
                    config_result["ef_results"].append(ef_result)
                    print(f"       QPS: {qps:.1f}, 재현율: {recall:.3f}")
                
                impl_results["configurations"].append(config_result)
                impl.done()
                
            except Exception as e:
                print(f"   ❌ 설정 실행 실패: {e}")
                continue
        
        return impl_results
    
    def calculate_recall(
        self, 
        train_data: np.ndarray, 
        test_queries: np.ndarray, 
        ann_results: List[List[int]],
        metric: str,
        k: int = 10
    ) -> float:
        """재현율 계산."""
        recalls = []
        max_queries = min(len(test_queries), len(ann_results), 50)  # 샘플링
        
        for i in range(max_queries):
            query = test_queries[i]
            ann_neighbors = ann_results[i][:k] if len(ann_results[i]) >= k else ann_results[i]
            
            # 브루트 포스 검색
            if metric in ["cosine", "angular"]:
                similarities = np.dot(train_data, query)
                true_neighbors = np.argsort(-similarities)[:k]
            else:
                distances = np.linalg.norm(train_data - query, axis=1)
                true_neighbors = np.argsort(distances)[:k]
            
            # 재현율 계산
            intersection = set(ann_neighbors) & set(true_neighbors)
            recall = len(intersection) / k if k > 0 else 0.0
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """결과 저장."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"hnsw_comparison_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 결과 저장: {filepath}")
        return str(filepath)
    
    def create_comparison_plots(self, results: Dict[str, Any]):
        """비교 플롯 생성."""
        implementations = list(results["implementations"].keys())
        
        if len(implementations) < 2:
            print("⚠️ 비교할 구현체가 부족합니다.")
            return
        
        # QPS vs Recall 비교
        plt.figure(figsize=(15, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(implementations)))
        
        for i, (impl_name, impl_data) in enumerate(results["implementations"].items()):
            if not impl_data.get("configurations"):
                continue
                
            for config_idx, config_result in enumerate(impl_data["configurations"]):
                config = config_result["config"]
                
                qps_values = []
                recall_values = []
                ef_values = []
                
                for ef_result in config_result["ef_results"]:
                    qps_values.append(ef_result["queries_per_second"])
                    recall_values.append(ef_result["recall_at_10"])
                    ef_values.append(ef_result["ef"])
                
                label = f"{impl_name} (M={config['M']})"
                
                plt.plot(recall_values, qps_values, 'o-', 
                        color=colors[i], label=label, linewidth=2, markersize=8)
                
                # ef 값 표시
                for recall, qps, ef in zip(recall_values, qps_values, ef_values):
                    plt.annotate(f'ef={ef}', (recall, qps), 
                               textcoords="offset points", xytext=(5,5), ha='left',
                               fontsize=8, alpha=0.7)
        
        plt.xlabel('Recall@10', fontsize=14)
        plt.ylabel('Queries Per Second (QPS)', fontsize=14)
        plt.title('HNSW 구현체 성능 비교: QPS vs Recall', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        comparison_plot = self.output_dir / "hnsw_comparison_qps_recall.png"
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"📊 비교 플롯 저장: {comparison_plot}")
    
    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력."""
        print(f"\\n📋 HNSW 구현체 성능 비교 요약")
        print("=" * 80)
        
        dataset_info = results["dataset_info"]
        print(f"데이터셋: {dataset_info['size']} 벡터, {dataset_info['dimension']} 차원")
        print(f"메트릭: {dataset_info['metric']}")
        print()
        
        # 헤더
        header = f"{'구현체':<20} {'설정':<15} {'구축시간(s)':<12} {'메모리(MB)':<12} {'최고QPS':<12} {'최고재현율':<12}"
        print(header)
        print("-" * len(header))
        
        for impl_name, impl_data in results["implementations"].items():
            if not impl_data.get("configurations"):
                print(f"{impl_name:<20} {'실패':<15} {'-':<12} {'-':<12} {'-':<12} {'-':<12}")
                continue
                
            for config_result in impl_data["configurations"]:
                config = config_result["config"]
                config_str = f"M={config['M']}"
                
                build_time = config_result["build_time"]
                memory_mb = config_result["memory_usage_mb"]
                
                if config_result["ef_results"]:
                    best_qps = max(ef_result["queries_per_second"] for ef_result in config_result["ef_results"])
                    best_recall = max(ef_result["recall_at_10"] for ef_result in config_result["ef_results"])
                else:
                    best_qps = 0
                    best_recall = 0
                
                row = f"{impl_name:<20} {config_str:<15} {build_time:<12.3f} {memory_mb:<12.1f} {best_qps:<12.1f} {best_recall:<12.3f}"
                print(row)


class SQLiteKgHNSWWrapper:
    """sqlite-kg-vec-mcp HNSW 래퍼."""
    
    def __init__(self, metric: str, config: Dict[str, Any], backend: str = "faiss"):
        self.metric = metric
        self.config = config
        self.backend = backend
        self.impl = SqliteKgHNSW(metric, config, backend=backend)
        
    def fit(self, X: np.ndarray):
        self.impl.fit(X)
        
    def set_query_arguments(self, ef: int):
        self.impl.set_query_arguments(ef)
        
    def batch_query(self, queries: np.ndarray, k: int):
        self.impl.batch_query(queries, k)
        
    def get_batch_results(self):
        return self.impl.get_batch_results()
        
    def done(self):
        self.impl.done()


class HnswlibWrapper:
    """hnswlib 래퍼."""
    
    def __init__(self, metric: str, config: Dict[str, Any]):
        self.metric = "cosine" if metric in ["cosine", "angular"] else "l2"
        self.config = config
        self.impl = None
        self.batch_results = []
        
    def fit(self, X: np.ndarray):
        self.impl = hnswlib.Index(space=self.metric, dim=X.shape[1])
        self.impl.init_index(
            max_elements=len(X),
            M=self.config["M"],
            ef_construction=self.config["efConstruction"]
        )
        
        # 라벨 생성 (0부터 시작)
        labels = np.arange(len(X))
        self.impl.add_items(X, labels)
        
    def set_query_arguments(self, ef: int):
        if self.impl:
            self.impl.set_ef(ef)
        
    def batch_query(self, queries: np.ndarray, k: int):
        self.batch_results = []
        for query in queries:
            labels, distances = self.impl.knn_query(query, k=k)
            # labels가 1D 배열이므로 tolist()로 변환
            if isinstance(labels, np.ndarray):
                self.batch_results.append(labels.flatten().tolist())
            else:
                self.batch_results.append(labels)
        
    def get_batch_results(self):
        return self.batch_results
        
    def done(self):
        if self.impl:
            del self.impl


class FAISSWrapper:
    """FAISS HNSW 래퍼."""
    
    def __init__(self, metric: str, config: Dict[str, Any]):
        self.metric = metric
        self.config = config
        self.impl = None
        self.batch_results = []
        
    def fit(self, X: np.ndarray):
        dimension = X.shape[1]
        M = self.config["M"]
        
        if self.metric in ["cosine", "angular"]:
            # Inner product for normalized vectors
            self.impl = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
        else:
            # L2 distance
            self.impl = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2)
        
        self.impl.hnsw.efConstruction = self.config["efConstruction"]
        self.impl.add(X)
        
    def set_query_arguments(self, ef: int):
        if self.impl:
            self.impl.hnsw.efSearch = ef
        
    def batch_query(self, queries: np.ndarray, k: int):
        distances, indices = self.impl.search(queries, k)
        self.batch_results = indices.tolist()
        
    def get_batch_results(self):
        return self.batch_results
        
    def done(self):
        if self.impl:
            del self.impl


def main():
    """메인 실행 함수."""
    import argparse
    
    parser = argparse.ArgumentParser(description="HNSW 구현체 성능 비교")
    parser.add_argument("--size", type=int, default=10000, help="데이터셋 크기")
    parser.add_argument("--dimension", type=int, default=128, help="벡터 차원")
    parser.add_argument("--metric", type=str, default="cosine", help="거리 메트릭")
    parser.add_argument("--quick", action="store_true", help="빠른 테스트")
    
    args = parser.parse_args()
    
    comparator = HNSWComparator()
    
    # 설정
    if args.quick:
        configs = [{"M": 16, "efConstruction": 200}]
        ef_values = [50, 100]
    else:
        configs = [
            {"M": 16, "efConstruction": 200},
            {"M": 32, "efConstruction": 400}
        ]
        ef_values = [50, 100, 200]
    
    try:
        # 비교 실행
        results = comparator.run_comparison(
            dataset_size=args.size,
            dimension=args.dimension,
            metric=args.metric,
            configs=configs,
            ef_values=ef_values
        )
        
        # 결과 저장 및 출력
        comparator.save_results(results)
        comparator.print_summary(results)
        comparator.create_comparison_plots(results)
        
        print(f"\\n✅ HNSW 구현체 비교 완료!")
        
    except KeyboardInterrupt:
        print(f"\\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\\n❌ 비교 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())