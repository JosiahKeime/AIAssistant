"""
skills/web_search.py

Search the web via Tavily. Requires TAVILY_API_KEY in the environment.
pip install tavily-python
"""

from __future__ import annotations

import os

from .base import SkillRegistry

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

_client = None  # lazy singleton, shared with fetch_url.py's own singleton


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable is not set")
        from tavily import TavilyClient
        _client = TavilyClient(api_key=api_key)
    return _client


def _format_results(results: list[dict], answer: str | None) -> str:
    if not results:
        return "No results found."

    blocks = []
    if answer:
        blocks.append(f"Quick answer: {answer}")

    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("content", "")
        snippet = snippet[:400] + ("..." if len(snippet) > 400 else "")
        blocks.append(f"[{title}]({url})\n{snippet}")

    return "\n\n".join(blocks)


def register(registry: SkillRegistry) -> None:
    @registry.register(
        name="web_search",
        description=(
            "Search the web for current information. Returns a ranked list "
            "of results, each with a title, URL, and short snippet. Use "
            "this for anything time-sensitive or outside your training data."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "default": 5,
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": (
                        "'basic' is fast and cheap. 'advanced' does deeper "
                        "content analysis and costs more credits."
                    ),
                    "default": "basic",
                },
            },
            "required": ["query"],
        },
    )
    def web_search(query: str, num_results: int = 5, search_depth: str = "basic") -> str:
        num_results = max(1, min(num_results, 10))
        response = _get_client().search(
            query=query,
            search_depth=search_depth,
            max_results=num_results,
            include_answer=True,
        )
        return _format_results(response.get("results", []), response.get("answer"))
