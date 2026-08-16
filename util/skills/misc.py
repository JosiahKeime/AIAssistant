"""
skills/misc.py

Small standalone skills that don't warrant their own module yet.
Split one out into its own file once it grows a real dependency.
"""

from __future__ import annotations

from .base import SkillRegistry


def register(registry: SkillRegistry) -> None:
    @registry.register(
        name="get_word_count",
        description="Count words in a piece of text.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )
    def get_word_count(text: str) -> str:
        return str(len(text.split()))
