# Query Analysis Prompt

## Purpose
Analyze search queries to determine the optimal search strategy for knowledge graph exploration.

## Instructions
Given a user's search query, analyze it and provide recommendations for the best search approach.

Consider these factors:
1. **Query Type**: Is it factual, exploratory, relational, or analytical?
2. **Scope**: Is it broad or specific?
3. **Entity Focus**: Does it target specific entities or relationships?
4. **Search Strategy**: Should we use semantic, structural, or hybrid search?

## Input
- **query**: The user's search query
- **context**: Optional context about previous searches or user preferences

## Output Format
Return a JSON object with:
```json
{
  "query_type": "factual|exploratory|relational|analytical",
  "complexity": "simple|moderate|complex",
  "recommended_strategy": "semantic|structural|hybrid",
  "entity_types": ["type1", "type2"],
  "confidence": 0.85,
  "reasoning": "Explanation of the analysis"
}
```

## Example
**Query**: "What are the connections between artificial intelligence and healthcare?"
**Analysis**: This is an exploratory, relational query requiring hybrid search to find both semantic similarities and structural relationships between AI and healthcare entities.