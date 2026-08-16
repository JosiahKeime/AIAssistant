"""
skills/__init__.py
 
Assembles the default SkillRegistry from every skill module in this
package. Each module exposes `register(registry) -> None` and owns its
own name, description, schema, and handler.
 
To add a new skill:
    1. Create skills/your_thing.py with a `register(registry)` function
       (copy the shape of web_search.py or misc.py).
    2. Import it below and add one line to build_default_registry().
 
Nothing in agent.py, or any other skill module, needs to change.
"""
 
from __future__ import annotations
 
from .base import Skill, SkillRegistry, build_default_registry
 
__all__ = ["Skill", "SkillRegistry", "build_default_registry"]
