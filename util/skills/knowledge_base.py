"""
skills/knowledge_base.py

Three-tier search over the local ChromaDB vector database:
    1. Collection summaries — identify relevant collections
    2. Document summaries — identify relevant documents within collections
    3. Chunk search — retrieve specific passages
"""

from __future__ import annotations

from .base import SkillRegistry


def register(registry: SkillRegistry, vector_store) -> None:

    @registry.register(
        name="list_knowledge_base_collections",
        description=(
            "List all user collections in the knowledge base, each with a "
            "short summary of what it contains. Use this to get an overview "
            "of available collections and their contents."
        ),
        input_schema={"type": "object", "properties": {}},
    )
    def list_knowledge_base_collections() -> str:
        collections = vector_store.list_collections()
        if not collections:
            return "No collections found."

        lines = []
        for name in collections:
            summary = vector_store.get_collection_summary(name)
            lines.append(f"- {name}: {summary or '(no description yet)'}")
        return "\n".join(lines)

    @registry.register(
        name="search_collection_summaries",
        description=(
            "Search collection-level summaries to identify which collections "
            "are most relevant to a query. Returns collection names with their "
            "summary descriptions. Use this as the FIRST step when answering a "
            "question about the user's documents — it narrows down which "
            "collections to search before drilling into document summaries "
            "and chunks."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "n_results": {
                    "type": "integer",
                    "description": "How many collection summaries to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    )
    def search_collection_summaries(query: str, n_results: int = 5) -> str:
        try:
            results = vector_store.search_collection_summaries(
                query, k=n_results
            )
        except Exception as exc:
            return f"Error searching collection summaries: {exc}"

        if not results:
            return "No collection summaries found."

        lines = []
        for result in results:
            summary = getattr(result, "page_content", "")
            collection = getattr(result, "metadata", {}).get(
                "collection", "unknown"
            )
            if summary:
                lines.append(f"[{collection}] {summary}")

        if not lines:
            return "No collection summaries found."
        return "\n\n---\n\n".join(lines)

    @registry.register(
        name="search_document_summaries",
        description=(
            "Search one-sentence summaries of the user's embedded documents. "
            "Use this AFTER search_collection_summaries to identify which "
            "specific documents within a collection are relevant. Pass "
            "collection_name to scope results to a specific collection "
            "identified in the previous step."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "n_results": {
                    "type": "integer",
                    "description": "How many document summaries to return",
                    "default": 5,
                },
                "collection_name": {
                    "type": "string",
                    "description": (
                        "Scope results to this collection. Omit to search "
                        "across all collections."
                    ),
                },
            },
            "required": ["query"],
        },
    )
    def search_document_summaries(
        query: str,
        n_results: int = 5,
        collection_name: str | None = None,
    ) -> str:
        try:
            results = vector_store.search_document_summaries(
                query, k=n_results, collection_name=collection_name
            )
        except Exception as exc:
            return f"Error searching document summaries: {exc}"

        if not results:
            return "No document summaries found."

        summaries = []
        for result in results:
            summary = getattr(result, "page_content", "")
            metadata = getattr(result, "metadata", {})
            source = metadata.get("source", "unknown source")
            collection = metadata.get("collection", "unknown collection")
            if summary:
                summaries.append(f"[{collection}/{source}] {summary}")

        if not summaries:
            return "No document summaries found."
        return "\n\n---\n\n".join(summaries)

    @registry.register(
        name="search_knowledge_base",
        description=(
            "Search the local ChromaDB vector database for passages "
            "relevant to a query. Use this AFTER identifying relevant "
            "collections (via search_collection_summaries) and documents "
            "(via search_document_summaries). Pass collection_names to "
            "scope the search, and metadata_filter to target a specific "
            'document (e.g. {"source": "guide.txt"}).'
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
                    "description": (
                        "Which collections to search, by name. Omit to "
                        "search all collections."
                    ),
                },
                "metadata_filter": {
                    "type": "object",
                    "description": (
                        "Optional Chroma metadata filter. Use this to limit "
                        'results to a specific document, for example '
                        '{"source": "guide.txt"}.'
                    ),
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
        print(
            f"--- Searching knowledge base for query: {query} "
            f"(n_results={n_results})"
        )

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

        if not results:
            return "No results found."

        collected = []
        for result in results:
            chunk_text = getattr(result, "page_content", "")
            if chunk_text:
                source = getattr(result, "metadata", {}).get(
                    "source", "unknown source"
                )
                collected.append(f"[{source}] {chunk_text}")

        if not collected:
            return "No results found."
        return "\n\n---\n\n".join(collected[: n_results * 2])