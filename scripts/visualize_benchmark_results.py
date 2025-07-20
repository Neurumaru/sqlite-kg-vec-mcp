#!/usr/bin/env python3
"""
ann-benchmarks 결과 시각화 스크립트.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns

# 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BenchmarkVisualizer:
    """벤치마크 결과 시각화기."""
    
    def __init__(self, output_dir: str = "plots"):
        """
        초기화.
        
        Args:
            output_dir: 플롯 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_results(self, result_file: str) -> Dict[str, Any]:
        """
        벤치마크 결과 로드.
        
        Args:
            result_file: 결과 파일 경로
            
        Returns:
            벤치마크 결과 딕셔너리
        """
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def plot_qps_vs_recall(self, results: Dict[str, Any], save_path: str = None):
        """
        QPS vs Recall 플롯 생성.
        
        Args:
            results: 벤치마크 결과
            save_path: 저장 경로
        """
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(results['configurations'])))
        
        for i, config_result in enumerate(results['configurations']):
            config = config_result['config']
            
            qps_values = []
            recall_values = []
            ef_values = []
            
            for ef_result in config_result['ef_results']:
                qps_values.append(ef_result['queries_per_second'])
                recall_values.append(ef_result['recall_at_10'])
                ef_values.append(ef_result['ef'])
            
            label = f"M={config['M']}, efConstruction={config['efConstruction']}"
            
            # QPS vs Recall 곡선
            plt.plot(recall_values, qps_values, 'o-', color=colors[i], 
                    label=label, linewidth=2, markersize=8)
            
            # ef 값 표시
            for j, (recall, qps, ef) in enumerate(zip(recall_values, qps_values, ef_values)):
                plt.annotate(f'ef={ef}', (recall, qps), 
                           textcoords="offset points", xytext=(5,5), ha='left',
                           fontsize=8, alpha=0.7)
        
        plt.xlabel('Recall@10', fontsize=14)
        plt.ylabel('Queries Per Second (QPS)', fontsize=14)
        plt.title(f'QPS vs Recall - {results["dataset"]}\n'
                 f'Dataset: {results["train_size"]} vectors, {results["dimension"]} dimensions', 
                 fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"qps_vs_recall_{results['dataset']}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 QPS vs Recall 플롯 저장: {save_path}")
    
    def plot_build_time_comparison(self, results: Dict[str, Any], save_path: str = None):
        """
        인덱스 구축 시간 비교 플롯.
        
        Args:
            results: 벤치마크 결과
            save_path: 저장 경로
        """
        plt.figure(figsize=(10, 6))
        
        configs = []
        build_times = []
        memory_usage = []
        
        for config_result in results['configurations']:
            config = config_result['config']
            config_name = f"M={config['M']}\nefC={config['efConstruction']}"
            configs.append(config_name)
            build_times.append(config_result['build_time'])
            memory_usage.append(config_result['memory_usage_kb'] / 1024)  # MB로 변환
        
        x = np.arange(len(configs))
        width = 0.35
        
        # 구축 시간
        plt.subplot(1, 2, 1)
        bars1 = plt.bar(x, build_times, width, label='Build Time', color='skyblue', alpha=0.8)
        plt.xlabel('Configuration')
        plt.ylabel('Build Time (seconds)')
        plt.title('Index Build Time')
        plt.xticks(x, configs)
        plt.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, time in zip(bars1, build_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time:.3f}s', ha='center', va='bottom', fontsize=10)
        
        # 메모리 사용량
        plt.subplot(1, 2, 2)
        bars2 = plt.bar(x, memory_usage, width, label='Memory Usage', color='lightcoral', alpha=0.8)
        plt.xlabel('Configuration')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage')
        plt.xticks(x, configs)
        plt.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, mem in zip(bars2, memory_usage):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{mem:.1f}MB', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"build_comparison_{results['dataset']}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 구축 시간 비교 플롯 저장: {save_path}")
    
    def plot_ef_parameter_analysis(self, results: Dict[str, Any], save_path: str = None):
        """
        ef 파라미터 분석 플롯.
        
        Args:
            results: 벤치마크 결과
            save_path: 저장 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(results['configurations'])))
        
        for i, config_result in enumerate(results['configurations']):
            config = config_result['config']
            label = f"M={config['M']}, efConstruction={config['efConstruction']}"
            
            ef_values = []
            qps_values = []
            recall_values = []
            memory_values = []
            latency_values = []
            
            for ef_result in config_result['ef_results']:
                ef_values.append(ef_result['ef'])
                qps_values.append(ef_result['queries_per_second'])
                recall_values.append(ef_result['recall_at_10'])
                memory_values.append(ef_result['memory_usage_kb'] / 1024)
                latency_values.append(ef_result['avg_query_time_ms'])
            
            # ef vs QPS
            axes[0, 0].plot(ef_values, qps_values, 'o-', color=colors[i], 
                           label=label, linewidth=2, markersize=6)
            axes[0, 0].set_xlabel('ef Parameter')
            axes[0, 0].set_ylabel('Queries Per Second')
            axes[0, 0].set_title('ef vs QPS')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # ef vs Recall
            axes[0, 1].plot(ef_values, recall_values, 'o-', color=colors[i], 
                           label=label, linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('ef Parameter')
            axes[0, 1].set_ylabel('Recall@10')
            axes[0, 1].set_title('ef vs Recall')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # ef vs Memory
            axes[1, 0].plot(ef_values, memory_values, 'o-', color=colors[i], 
                           label=label, linewidth=2, markersize=6)
            axes[1, 0].set_xlabel('ef Parameter')
            axes[1, 0].set_ylabel('Memory Usage (MB)')
            axes[1, 0].set_title('ef vs Memory Usage')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # ef vs Latency
            axes[1, 1].plot(ef_values, latency_values, 'o-', color=colors[i], 
                           label=label, linewidth=2, markersize=6)
            axes[1, 1].set_xlabel('ef Parameter')
            axes[1, 1].set_ylabel('Average Query Time (ms)')
            axes[1, 1].set_title('ef vs Query Latency')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.suptitle(f'ef Parameter Analysis - {results["dataset"]}', fontsize=16)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"ef_analysis_{results['dataset']}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 ef 파라미터 분석 플롯 저장: {save_path}")
    
    def create_summary_table(self, results: Dict[str, Any]):
        """
        요약 테이블 생성 및 출력.
        
        Args:
            results: 벤치마크 결과
        """
        print(f"\n📋 벤치마크 요약 테이블 - {results['dataset']}")
        print("=" * 80)
        
        header = f"{'Configuration':<20} {'Build Time':<12} {'Memory (MB)':<12} {'Best QPS':<12} {'Best Recall':<12}"
        print(header)
        print("-" * len(header))
        
        for config_result in results['configurations']:
            config = config_result['config']
            config_name = f"M={config['M']}, efC={config['efConstruction']}"
            
            build_time = config_result['build_time']
            memory_mb = config_result['memory_usage_kb'] / 1024
            
            # 최고 성능 찾기
            best_qps = max(ef_result['queries_per_second'] for ef_result in config_result['ef_results'])
            best_recall = max(ef_result['recall_at_10'] for ef_result in config_result['ef_results'])
            
            row = f"{config_name:<20} {build_time:<12.3f} {memory_mb:<12.1f} {best_qps:<12.1f} {best_recall:<12.3f}"
            print(row)
        
        print()
    
    def generate_all_plots(self, result_file: str):
        """
        모든 플롯 생성.
        
        Args:
            result_file: 결과 파일 경로
        """
        print(f"📊 벤치마크 결과 시각화: {result_file}")
        
        results = self.load_results(result_file)
        
        # 요약 테이블
        self.create_summary_table(results)
        
        # 모든 플롯 생성
        self.plot_qps_vs_recall(results)
        self.plot_build_time_comparison(results)
        self.plot_ef_parameter_analysis(results)
        
        print(f"\n✅ 모든 플롯이 {self.output_dir}에 저장되었습니다!")


def main():
    """메인 실행 함수."""
    parser = argparse.ArgumentParser(description="벤치마크 결과 시각화")
    parser.add_argument("result_file", type=str, help="벤치마크 결과 JSON 파일")
    parser.add_argument("--output-dir", type=str, default="plots", 
                       help="플롯 저장 디렉토리")
    
    args = parser.parse_args()
    
    if not Path(args.result_file).exists():
        print(f"❌ 결과 파일을 찾을 수 없습니다: {args.result_file}")
        return 1
    
    try:
        visualizer = BenchmarkVisualizer(args.output_dir)
        visualizer.generate_all_plots(args.result_file)
        
    except Exception as e:
        print(f"❌ 시각화 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())