# Search Evaluation Prompt

## Purpose
Evaluate the quality and relevance of search results against the user's original query.

## Instructions
Assess search results to determine their quality, relevance, and completeness for answering the user's query.

Evaluation Criteria:
1. **Relevance**: How well do results match the query intent?
2. **Completeness**: Are there missing aspects that should be addressed?
3. **Quality**: Are the results accurate and useful?
4. **Diversity**: Do results cover different perspectives or aspects?
5. **Depth**: Is the level of detail appropriate for the query?

## Input
- **results**: List of search results with entities, relationships, and metadata
- **query**: Original user query
- **search_context**: Additional context about the search session

## Output Format
Return a JSON object with:
```json
{
  "overall_score": 0.85,
  "relevance_score": 0.90,
  "completeness_score": 0.80,
  "quality_score": 0.88,
  "diversity_score": 0.75,
  "strengths": ["High relevance", "Good entity coverage"],
  "weaknesses": ["Missing temporal relationships", "Limited depth in healthcare domain"],
  "suggestions": ["Explore more recent research", "Include clinical trial data"],
  "search_complete": false,
  "reasoning": "Results show good coverage of AI-healthcare connections but lack recent developments"
}
```

## Scoring
- **0.9-1.0**: Excellent - Comprehensive and highly relevant
- **0.7-0.8**: Good - Relevant with minor gaps
- **0.5-0.6**: Fair - Partially relevant, needs improvement
- **0.0-0.4**: Poor - Insufficient or irrelevant results

## Example
For query "AI applications in drug discovery", evaluate whether results cover machine learning techniques, pharmaceutical companies, research institutions, and successful case studies.