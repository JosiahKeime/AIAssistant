"""
skills/knowledge_base.py

Search the local ChromaDB vector database (via your Embedder wrapper)
for passages relevant to a query.
"""

from __future__ import annotations

from .base import SkillRegistry


def register(registry: SkillRegistry, embedder) -> None:
    @registry.register(
        name="list_knowledge_base_collections",
        description=(
            "List the ChromaDB collections available to search, with a "
            "short description of what each one contains. Call this "
            "before search_knowledge_base when you're not sure which "
            "collection(s) are relevant, or when there are many "
            "collections and searching all of them would be wasteful."
        ),
        input_schema={"type": "object", "properties": {}},
    )
    def list_knowledge_base_collections() -> str:
        import chromadb

        try:
            client = chromadb.PersistentClient(path=embedder.chromadb_path)
            collections = client.list_collections()
        except Exception as exc:
            return f"Error listing collections: {exc}"

        if not collections:
            return "No collections found."

        lines = []
        for c in collections:
            description = (c.metadata or {}).get("description", "(no description)")
            lines.append(f"- {c.name}: {description}")

        return "\n".join(lines)

    @registry.register(
        name="search_knowledge_base",
        description=(
            "Search the local ChromaDB vector database for passages "
            "relevant to a query. Use this before answering questions "
            "that might be covered by the user's own documents. If there "
            "are many collections, call list_knowledge_base_collections "
            "first and pass collection_names to scope the search — "
            "otherwise every collection is searched, which is slower and "
            "can dilute results with irrelevant chunks."
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
            },
            "required": ["query"],
        },
    )
    def search_knowledge_base(
        query: str,
        n_results: int = 5,
        collection_names: list[str] | None = None,
    ) -> str:
        print(f"Searching knowledge base for query: {query} (n_results={n_results})")
        import chromadb

        try:
            client = chromadb.PersistentClient(path=embedder.chromadb_path)
            available = [c.name for c in client.list_collections()]
        except Exception as exc:
            return f"Error refreshing knowledge base collections: {exc}"

        if not available:
            return "No embedded collections found."

        if collection_names:
            unknown = set(collection_names) - set(available)
            if unknown:
                return (
                    f"Unknown collection(s): {', '.join(sorted(unknown))}. "
                    f"Available: {', '.join(available)}"
                )
            target_collections = collection_names
        else:
            target_collections = available

        collected_docs: list[str] = []
        for collection_name in target_collections:
            try:
                results = embedder.vector_db_search(query, collection_name, k=n_results)
            except Exception as exc:
                collected_docs.append(f"[{collection_name}] Error: {exc}")
                continue

            if not results:
                continue

            for result in results:
                chunk_text = getattr(result, "page_content", "")
                if chunk_text:
                    collected_docs.append(f"[{collection_name}] {chunk_text}")

        if not collected_docs:
            return "No results found."
        return "\n\n---\n\n".join(collected_docs[: n_results * 2])