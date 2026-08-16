"""
skills/base.py

The generic Skill / SkillRegistry abstraction. This rarely needs to
change — new capabilities go in their own module and call
`registry.register(...)`, not in here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Skill:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., str]  # takes **kwargs, returns a string result

    def to_tool_param(self) -> dict[str, Any]:
        """Convert to the format the Anthropic API expects for a tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(self, **kwargs) -> str:
        try:
            return self.handler(**kwargs)
        except Exception as e:
            # Tool errors should be reported back to the model as text,
            # not raised — that lets Claude see the failure and decide
            # whether to retry, try a different approach, or ask the user.
            return f"Error running skill '{self.name}': {e}"


class SkillRegistry:
    """Holds all skills the agent knows about."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> Callable:
        """Decorator form: @registry.register(...) def my_skill(...): ..."""

        def decorator(fn: Callable[..., str]) -> Callable[..., str]:
            self._skills[name] = Skill(name, description, input_schema, fn)
            return fn

        return decorator

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def as_tool_params(self) -> list[dict[str, Any]]:
        return [s.to_tool_param() for s in self._skills.values()]


def build_default_registry(embedder: Any = None) -> SkillRegistry:

    from . import fetch_url, knowledge_base, web_search
    from ..embedder import Embedder

    if embedder is None:
        embedder = Embedder()

    registry = SkillRegistry()

    knowledge_base.register(registry, embedder=embedder)
    web_search.register(registry)
    fetch_url.register(registry)

    return registry
