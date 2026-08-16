"""
skills/fetch_url.py

Fetch and extract full page content for a specific URL, via Tavily's
extract endpoint (cleaned text, boilerplate stripped — no separate
scraping/parsing layer needed).

Companion to web_search.py: the model typically calls web_search first,
then fetch_url on a specific result when the snippet isn't enough.
"""

from __future__ import annotations

import os

from .base import SkillRegistry

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable is not set")
        from tavily import TavilyClient

        _client = TavilyClient(api_key=api_key)
    return _client


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "... [truncated]"


def register(registry: SkillRegistry) -> None:
    @registry.register(
        name="fetch_url",
        description=(
            "Fetch and extract the full text content of a specific web "
            "page. Use this after web_search when a snippet isn't enough "
            "detail — pass the exact URL from a search result."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The exact URL to fetch, including scheme (https://...).",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Truncate extracted content to this many characters (default 6000).",
                    "default": 6000,
                },
            },
            "required": ["url"],
        },
    )
    def fetch_url(url: str, max_chars: int = 6000) -> str:
        response = _get_client().extract(urls=[url])
        results = response.get("results", [])
        failed = response.get("failed_results", [])

        if not results:
            reason = failed[0].get("error") if failed else "unknown error"
            return f"Could not extract content from {url}: {reason}"

        content = results[0].get("raw_content", "")
        if not content:
            return f"No extractable content found at {url}."

        return f"Content from {url}:\n\n{_truncate(content, max_chars)}"
