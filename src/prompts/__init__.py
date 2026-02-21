"""
Prompt Registry for versioned prompt management.

Usage:
    from src.prompts import PromptRegistry
    
    registry = PromptRegistry()
    prompt = registry.get_prompt("mcq", version="v1")
    prompt = registry.get_prompt("mcq", version="latest")
"""

from pathlib import Path
from typing import Optional
import re

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class PromptRegistry:
    """
    Load and manage versioned prompts from files.
    
    Prompts are stored in:
        src/prompts/{agent_name}/v{N}.txt
    
    Example:
        src/prompts/mcq/v1.txt
        src/prompts/art/v1.txt
        src/prompts/art/v2.txt
    """
    
    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            # Default to src/prompts directory
            self.base_path = Path(__file__).parent
        else:
            self.base_path = Path(base_path)
    
    def get_prompt(
        self,
        agent: str,
        version: str = "latest",
        **format_kwargs
    ) -> str:
        """
        Get a prompt by agent name and version.
        
        Args:
            agent: Agent name (e.g., "mcq", "art", "moral", "science")
            version: Version string ("v1", "v2") or "latest"
            **format_kwargs: Variables to interpolate into the prompt template
        
        Returns:
            The prompt string with variables interpolated
        
        Example:
            prompt = registry.get_prompt(
                "mcq",
                version="v1",
                age=8,
                summary="Once upon a time...",
                language="English"
            )
        """
        if version == "latest":
            version = self._get_latest_version(agent)
        
        prompt_path = self.base_path / agent / f"{version}.txt"
        
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt not found: {prompt_path}. "
                f"Available versions: {self.list_versions(agent)}"
            )
        
        prompt_template = prompt_path.read_text(encoding="utf-8")
        
        # Interpolate variables if provided
        if format_kwargs:
            prompt = prompt_template.format(**format_kwargs)
        else:
            prompt = prompt_template
        
        logger.debug(f"Loaded prompt: {agent}/{version}")
        return prompt
    
    def _get_latest_version(self, agent: str) -> str:
        """Get the latest version number for an agent."""
        versions = self.list_versions(agent)
        if not versions:
            raise FileNotFoundError(f"No prompts found for agent: {agent}")
        return versions[-1]
    
    def list_versions(self, agent: str) -> list[str]:
        """List all available versions for an agent, sorted."""
        agent_path = self.base_path / agent
        
        if not agent_path.exists():
            return []
        
        # Find all v*.txt files
        version_files = list(agent_path.glob("v*.txt"))
        
        # Extract version numbers and sort
        def version_key(path: Path) -> int:
            match = re.search(r"v(\d+)", path.stem)
            return int(match.group(1)) if match else 0
        
        versions = sorted(version_files, key=version_key)
        return [v.stem for v in versions]
    
    def list_agents(self) -> list[str]:
        """List all agents with prompts."""
        return [
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]


# Singleton instance for convenience
_registry: Optional[PromptRegistry] = None


def get_registry() -> PromptRegistry:
    """Get the singleton PromptRegistry instance."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry()
    return _registry
