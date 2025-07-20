#!/usr/bin/env python3
"""
HNSW 벤치마크 실행 스크립트.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_hnsw_benchmark import HNSWBenchmarkTest, HNSWPerformanceTest


def main():
    """메인 실행 함수"""
    print("🚀 HNSW 성능 벤치마크 시작")
    print("=" * 60)
    
    # 벤치마크 인스턴스 생성
    benchmark = HNSWBenchmarkTest()
    benchmark.setUp()
    
    try:
        # 선택적 벤치마크 실행
        choice = input("\n실행할 벤치마크를 선택하세요:\n"
                      "1. 빠른 성능 확인 (권장)\n"
                      "2. 차원 스케일링 테스트\n"
                      "3. 데이터셋 크기 테스트\n"
                      "4. 파라미터 튜닝 테스트\n"
                      "5. 종합 벤치마크\n"
                      "6. 모든 벤치마크\n"
                      "선택 (1-6): ").strip()
        
        if choice == "1":
            print("\n⚡ 빠른 성능 확인 실행...")
            quick_test = HNSWPerformanceTest()
            quick_test.setUp()
            try:
                quick_test.test_quick_performance_check()
            finally:
                quick_test.tearDown()
                
        elif choice == "2":
            benchmark.test_dimension_scaling_benchmark()
            
        elif choice == "3":
            benchmark.test_dataset_size_scaling_benchmark()
            
        elif choice == "4":
            benchmark.test_parameter_tuning_benchmark()
            
        elif choice == "5":
            benchmark.test_comprehensive_benchmark()
            
        elif choice == "6":
            benchmark.run_all_benchmarks()
            
        else:
            print("❌ 잘못된 선택입니다.")
            return 1
        
        # 결과 저장
        if benchmark.results:
            benchmark.save_benchmark_results()
            benchmark.generate_performance_report()
        
        print(f"\n✅ 벤치마크 완료!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
        
    except Exception as e:
        print(f"\n❌ 벤치마크 실행 중 오류 발생: {e}")
        return 1
        
    finally:
        benchmark.tearDown()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())