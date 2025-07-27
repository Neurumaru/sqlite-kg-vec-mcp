"""
Ollama client for LLM integration with SQLite KG Vec MCP.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
import requests
from dataclasses import dataclass

from src.common.observability import get_observable_logger, with_observability
# Langfuse integration removed - using fallback prompts


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    model: str
    tokens_used: int
    response_time: float
    metadata: Optional[Dict[str, Any]] = None


class OllamaClient:
    """Client for interacting with Ollama LLM models."""
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "gemma2",
        timeout: int = 60
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            model: Model name to use (default: gemma2)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.logger = get_observable_logger("ollama_client", "adapter")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return True
        except requests.ConnectionError as e:
            from .exceptions import OllamaConnectionException
            self.logger.warning("ollama_connection_failed",
                              base_url=self.base_url,
                              error_type="connection_error",
                              error_message=str(e))
            return False
        except requests.Timeout as e:
            from .exceptions import OllamaTimeoutException
            self.logger.warning("ollama_connection_timeout",
                              base_url=self.base_url,
                              timeout_duration=5,
                              error_message=str(e))
            return False
        except requests.HTTPError as e:
            from .exceptions import OllamaConnectionException
            status_code = getattr(e.response, 'status_code', None)
            self.logger.warning("ollama_http_error",
                              base_url=self.base_url,
                              status_code=status_code,
                              error_message=str(e))
            return False
        except Exception as e:
            self.logger.warning("ollama_unexpected_error",
                              base_url=self.base_url,
                              error_type=type(e).__name__,
                              error_message=str(e))
            return False
    
    @with_observability(operation="llm_generate", include_args=True, include_result=True)
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Generate text using Ollama model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            LLMResponse with generated text and metadata
        """
        start_time = time.time()
        
        # Prepare request data
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        if max_tokens:
            data["options"]["num_predict"] = max_tokens
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            if stream:
                # Handle streaming response
                text_chunks = []
                total_tokens = 0
                
                for line in response.iter_lines():
                    if line:
                        chunk_data = json.loads(line.decode('utf-8'))
                        if 'response' in chunk_data:
                            text_chunks.append(chunk_data['response'])
                        
                        if chunk_data.get('done', False):
                            total_tokens = chunk_data.get('eval_count', 0)
                            break
                
                generated_text = ''.join(text_chunks)
            else:
                # Handle non-streaming response
                result = response.json()
                generated_text = result.get('response', '')
                total_tokens = result.get('eval_count', 0)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                text=generated_text,
                model=self.model,
                tokens_used=total_tokens,
                response_time=response_time,
                metadata={"temperature": temperature}
            )
            
        except requests.ConnectionError as e:
            from .exceptions import OllamaConnectionException
            raise OllamaConnectionException.from_requests_error(self.base_url, e)
        except requests.Timeout as e:
            from .exceptions import OllamaTimeoutException
            raise OllamaTimeoutException(
                base_url=self.base_url,
                operation="text generation",
                timeout_duration=self.timeout,
                original_error=e
            )
        except requests.HTTPError as e:
            from .exceptions import OllamaConnectionException
            status_code = getattr(e.response, 'status_code', None)
            raise OllamaConnectionException(
                base_url=self.base_url,
                message=f"HTTP {status_code} error during generation",
                status_code=status_code,
                original_error=e
            )
        except json.JSONDecodeError as e:
            from .exceptions import OllamaResponseException
            response_text = getattr(e, 'doc', 'Unknown response')
            raise OllamaResponseException(
                response_text=response_text,
                parsing_error=str(e),
                original_error=e
            )
        except Exception as e:
            from .exceptions import OllamaGenerationException
            raise OllamaGenerationException(
                model_name=self.model,
                prompt=prompt,
                message=f"Unexpected error during generation: {e}",
                generation_params={"temperature": temperature, "max_tokens": max_tokens},
                original_error=e
            )
    
    def extract_entities_and_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from text using LLM.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing extracted entities and relationships
        """
        # 기본 프롬프트 사용 (Langfuse 제거됨)
        system_prompt = """You are an expert knowledge graph extraction system. 
            Analyze the given text and extract entities and relationships in JSON format.
            
            Return a JSON object with this structure:
            {
                "entities": [
                    {
                        "id": "unique_id",
                        "name": "entity_name", 
                        "type": "entity_type",
                        "properties": {"key": "value"}
                    }
                ],
                "relationships": [
                    {
                        "source": "source_entity_id",
                        "target": "target_entity_id", 
                        "type": "relationship_type",
                        "properties": {"key": "value"}
                    }
                ]
            }
            
            Focus on extracting:
            - People, organizations, places, concepts, events
            - Clear relationships between entities
            - Important properties and attributes
            
            Be precise and only extract information explicitly mentioned in the text."""
            
            prompt = f"""Extract entities and relationships from this text:

            {text}

            Return only valid JSON, no additional text or explanation."""
        
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Lower temperature for more consistent extraction
            max_tokens=2000
        )
        
        try:
            # Clean the response text (remove markdown code blocks if present)
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]   # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove closing ```
            response_text = response_text.strip()
            
            # Parse the JSON response
            result = json.loads(response_text)
            
            # Validate structure
            if "entities" not in result:
                result["entities"] = []
            if "relationships" not in result:
                result["relationships"] = []
            
            # Prompt usage logging removed (Langfuse integration removed)
                
            return result
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse LLM response as JSON: {e}")
            logging.error(f"Response text: {response.text}")
            
            # Return empty structure on parse failure
            return {"entities": [], "relationships": []}
    
    def generate_embeddings_description(self, entity: Dict[str, Any]) -> str:
        """
        Generate a rich description for an entity to improve embeddings.
        
        Args:
            entity: Entity dictionary with name, type, and properties
            
        Returns:
            Enhanced description string
        """
        system_prompt = """You are an expert at creating rich, informative descriptions for knowledge graph entities.
        Given an entity with its name, type, and properties, create a comprehensive but concise description
        that would be ideal for vector embeddings and semantic search.
        
        The description should:
        - Include the entity's main characteristics
        - Mention key relationships and context
        - Be 1-3 sentences long
        - Be informative for search and matching
        
        Return only the description, no additional text."""
        
        entity_info = f"""Entity Name: {entity.get('name', 'Unknown')}
Entity Type: {entity.get('type', 'Unknown')}
Properties: {json.dumps(entity.get('properties', {}), indent=2)}"""
        
        prompt = f"Create a rich description for this entity:\n\n{entity_info}"
        
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=150
        )
        
        return response.text.strip()
    
    def list_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
            
        except Exception as e:
            logging.error(f"Error fetching available models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {"name": model_name}
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=data,
                timeout=300  # 5 minutes for model download
            )
            response.raise_for_status()
            
            # Check if pull was successful
            for line in response.iter_lines():
                if line:
                    chunk_data = json.loads(line.decode('utf-8'))
                    if chunk_data.get('status') == 'success':
                        return True
                        
            return False
            
        except Exception as e:
            logging.error(f"Error pulling model {model_name}: {e}")
            return False