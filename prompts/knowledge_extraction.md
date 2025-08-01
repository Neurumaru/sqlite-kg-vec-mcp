# Knowledge Extraction Prompt

## Purpose
Extract structured knowledge (entities and relationships) from unstructured text.

## Instructions
Analyze the provided text and extract entities, relationships, and key concepts that can be stored in a knowledge graph.

Focus on:
1. **Entities**: People, organizations, concepts, locations, events, etc.
2. **Relationships**: How entities are connected or related
3. **Attributes**: Important properties or characteristics
4. **Context**: Temporal, spatial, or situational information

## Input
- **text**: Text content to analyze
- **extraction_schema**: Optional schema defining entity/relationship types
- **context**: Additional context about the text source or domain

## Output Format
Return a JSON object with:
```json
{
  "entities": [
    {
      "name": "Entity Name",
      "type": "PERSON|ORGANIZATION|CONCEPT|LOCATION|EVENT|OTHER",
      "properties": {
        "description": "Brief description",
        "aliases": ["alternative names"],
        "confidence": 0.95
      }
    }
  ],
  "relationships": [
    {
      "source": "Entity1",
      "target": "Entity2", 
      "type": "WORKS_AT|LOCATED_IN|RELATED_TO|INFLUENCES|OTHER",
      "properties": {
        "description": "Relationship description",
        "strength": 0.85,
        "temporal": "2023-2024"
      }
    }
  ],
  "metadata": {
    "source": "document source",
    "extraction_confidence": 0.88,
    "language": "en",
    "domain": "technology"
  }
}
```

## Entity Types
- **PERSON**: Individuals, researchers, executives
- **ORGANIZATION**: Companies, institutions, governments
- **CONCEPT**: Technologies, methodologies, theories
- **LOCATION**: Cities, countries, facilities
- **EVENT**: Conferences, launches, discoveries
- **PRODUCT**: Software, devices, services

## Relationship Types
- **WORKS_AT**: Employment relationships
- **FOUNDED**: Creation relationships
- **COLLABORATES_WITH**: Partnership relationships
- **RESEARCHES**: Research focus relationships
- **INFLUENCES**: Impact relationships
- **LOCATED_IN**: Geographic relationships

## Example
**Text**: "Dr. Sarah Chen from Stanford University published research on neural networks for medical diagnosis."
**Extraction**: Entities: Dr. Sarah Chen (PERSON), Stanford University (ORGANIZATION), neural networks (CONCEPT), medical diagnosis (CONCEPT). Relationships: Dr. Sarah Chen WORKS_AT Stanford University, Dr. Sarah Chen RESEARCHES neural networks.