"""
Prompt management port interface.

This port defines how the domain interacts with prompt management systems
for template storage, retrieval, and variable substitution.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class PromptManager(ABC):
    """
    Port interface for prompt management operations.
    
    This interface abstracts prompt template management, allowing the domain
    to work with different prompt management systems (Langfuse, local files, etc.)
    """

    @abstractmethod
    def get_prompt(
        self,
        prompt_name: str,
        version: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Retrieve a prompt template and compile it with variables.
        
        Args:
            prompt_name: Name/identifier of the prompt
            version: Specific version to retrieve (None for latest)
            variables: Variables to substitute in the template
            
        Returns:
            Dictionary containing 'system' and 'user' prompt content
            
        Raises:
            PromptNotFoundError: If prompt doesn't exist
            PromptCompilationError: If variable substitution fails
        """
        pass

    @abstractmethod
    def create_or_update_prompt(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new prompt template or update an existing one.
        
        Args:
            name: Prompt identifier
            template: Template content with variable placeholders
            description: Human-readable description
            labels: Metadata labels for categorization
            config: Additional configuration options
            
        Returns:
            Version identifier of the created/updated prompt
        """
        pass

    @abstractmethod
    def list_prompts(
        self,
        labels: Optional[Dict[str, str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List available prompt templates.
        
        Args:
            labels: Filter by labels
            limit: Maximum number of results
            
        Returns:
            List of prompt metadata dictionaries
        """
        pass

    @abstractmethod
    def delete_prompt(self, name: str, version: Optional[str] = None) -> bool:
        """
        Delete a prompt template.
        
        Args:
            name: Prompt identifier
            version: Specific version to delete (None for all versions)
            
        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    def get_prompt_versions(self, name: str) -> List[str]:
        """
        Get all versions of a prompt template.
        
        Args:
            name: Prompt identifier
            
        Returns:
            List of version identifiers, ordered by creation time
        """
        pass

    @abstractmethod
    def validate_template(self, template: str, variables: Dict[str, Any]) -> bool:
        """
        Validate that a template can be compiled with given variables.
        
        Args:
            template: Template content
            variables: Variables for substitution
            
        Returns:
            True if template is valid and can be compiled
        """
        pass


class PromptNotFoundError(Exception):
    """Raised when a requested prompt template is not found."""
    pass


class PromptCompilationError(Exception):
    """Raised when prompt template compilation fails."""
    pass


class PromptValidationError(Exception):
    """Raised when prompt template validation fails."""
    pass