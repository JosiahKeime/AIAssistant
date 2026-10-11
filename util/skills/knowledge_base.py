"""Flat search tools for the local ChromaDB knowledge base."""

from __future__ import annotations

from .base import SkillRegistry


def register(registry: SkillRegistry, vector_store) -> None:
    @registry.register(
        name="list_knowledge_base_collections",
        description="List the available document collections in the knowledge base.",
        input_schema={"type": "object", "properties": {}},
    )
    def list_knowledge_base_collections() -> str:
        collections = vector_store.list_collections()
        if not collections:
            return "No collections found."
        return "Available collections:\n" + "\n".join(
            f"- {name}" for name in collections
        )

    @registry.register(
        name="search_knowledge_base",
        description=(
            "Search document chunks directly for passages relevant to a query. "
            "Optionally limit the search to named collections or filter by "
            'metadata, such as {"source": "guide.txt"}.'
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "n_results": {
                    "type": "integer",
                    "description": "How many chunks to return",
                    "default": 5,
                },
                "collection_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Collections to search; omit to search all.",
                },
                "metadata_filter": {
                    "type": "object",
                    "description": "Optional Chroma metadata filter.",
                    "additionalProperties": True,
                },
            },
            "required": ["query"],
        },
    )
    def search_knowledge_base(
        query: str,
        n_results: int = 5,
        collection_names: list[str] | None = None,
        metadata_filter: dict | None = None,
    ) -> str:
        available = vector_store.list_collections()
        if not available:
            return "No embedded collections found."

        if collection_names:
            unknown = set(collection_names) - set(available)
            if unknown:
                return (
                    f"Unknown collection(s): {', '.join(sorted(unknown))}. "
                    f"Available: {', '.join(available)}"
                )

        try:
            results = vector_store.search_chunks(
                query,
                k=n_results,
                collection_names=collection_names,
                metadata_filter=metadata_filter,
            )
        except Exception as exc:
            return f"Error searching knowledge base: {exc}"

        passages = []
        for result in results:
            text = getattr(result, "page_content", "")
            if text:
                source = getattr(result, "metadata", {}).get(
                    "source", "unknown source"
                )
                passages.append(f"[{source}] {text}")

        return "\n\n---\n\n".join(passages) if passages else "No results found."