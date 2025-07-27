"""
Tests package for sqlite-kg-vec-mcp.

unittest 기반 테스트 패키지입니다.
"""

import os
import sys
import unittest

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))


def load_tests(loader, tests, pattern):
    """unittest 테스트 디스커버리 설정"""
    suite = unittest.TestSuite()

    # 단위 테스트
    unit_tests = loader.discover("tests/unit", pattern="test_*.py")
    suite.addTests(unit_tests)

    # 통합 테스트
    integration_tests = loader.discover("tests/integration", pattern="test_*.py")
    suite.addTests(integration_tests)

    # E2E 테스트
    e2e_tests = loader.discover("tests/e2e", pattern="test_*.py")
    suite.addTests(e2e_tests)

    return suite
