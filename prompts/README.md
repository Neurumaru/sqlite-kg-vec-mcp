# Default Prompts

이 폴더에는 SQLite KG Vec MCP 시스템에서 사용하는 기본 프롬프트들이 저장되어 있습니다.

## 프롬프트 파일들

### 1. `query_analysis.md`
사용자 쿼리를 분석하여 최적의 검색 전략을 결정하는 프롬프트입니다.

**용도**: 
- 쿼리 타입 분석 (factual, exploratory, relational, analytical)
- 검색 전략 추천 (semantic, structural, hybrid)
- 대상 엔티티 타입 식별

### 2. `search_navigation.md`
대화형 검색에서 다음 탐색 단계를 결정하는 프롬프트입니다.

**용도**:
- 현재 결과 기반 다음 행동 결정
- 탐색 방향 설정 (expand, drill_down, pivot, etc.)
- 검색 매개변수 조정

### 3. `search_evaluation.md`
검색 결과의 품질과 완성도를 평가하는 프롬프트입니다.

**용도**:
- 결과 관련성, 완성도, 품질 평가
- 부족한 측면 식별
- 검색 계속 여부 결정

### 4. `knowledge_extraction.md`
텍스트에서 구조화된 지식(엔티티, 관계)을 추출하는 프롬프트입니다.

**용도**:
- 텍스트에서 엔티티 추출
- 엔티티 간 관계 식별
- 지식 그래프 구축을 위한 구조화

## 사용 방법

### 1. Langfuse와 함께 사용
```python
from src.llm.langfuse_prompts import get_prompt_manager

manager = get_prompt_manager()
prompt = manager.get_prompt("query_analysis", variables={"query": "사용자 쿼리"})
```

### 2. 직접 파일 사용
```python
from pathlib import Path

prompts_dir = Path("prompts")
content = (prompts_dir / "query_analysis.md").read_text()
```

## 작동 방식

1. **Langfuse 우선**: 먼저 Langfuse에서 프롬프트를 가져오려고 시도
2. **자동 생성**: Langfuse에 프롬프트가 없으면 이 폴더의 파일에서 자동 생성
3. **Fallback**: Langfuse가 비활성화되었거나 실패하면 로컬 파일 직접 사용

## 프롬프트 수정

1. **로컬 수정**: 이 폴더의 `.md` 파일을 수정
2. **Langfuse 동기화**: 
   ```python
   from src.utils.prompt_manager import ensure_prompts_in_langfuse
   ensure_prompts_in_langfuse()
   ```
3. **테스트**: 
   ```python
   from src.utils.prompt_manager import test_prompt_system
   test_prompt_system()
   ```

## 변수 치환

프롬프트에서 `{variable_name}` 형식으로 변수를 사용할 수 있습니다.

예시:
```markdown
사용자 쿼리: "{query}"
현재 단계: {step_number}
```

## 환경 설정

Langfuse 사용을 위해 다음 환경변수를 설정하세요:
```bash
export LANGFUSE_SECRET_KEY="your-secret-key"
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"  # 선택적
```