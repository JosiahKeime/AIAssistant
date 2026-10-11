"""
VectorDBManager
==================

Owns the ChromaDB persistence layer for flat RAG retrieval.
No hierarchical summaries — just collections of document chunks.

Usage:
    store = VectorDBManager()
    store.create_collection("knowledge-base")
    # Use DocumentProcessor to ingest files
    results = store.search_chunks("employment law")
"""

from __future__ import annotations

import os
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .embedding import EmbeddingProvider
from .logging_utils import logger


class VectorDBManager:
    """Manage ChromaDB collections and flat vector search."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        chromadb_path: str = "./chroma_db",
    ):
        self.chromadb_path = chromadb_path
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.embedding_provider = embedding_provider or EmbeddingProvider()

        logger.info(
            "Initializing VectorDBManager | chromadb_path=%s", chromadb_path
        )

    # ── Collection management ──────────────────────────────────────

    def create_collection(self, name: str) -> None:
        """Create a user collection (idempotent)."""
        self.client.get_or_create_collection(name=name)
        logger.info("Created/verified collection '%s'.", name)

    def list_collections(self) -> list[str]:
        """Return sorted names of all collections."""
        collections = self.client.list_collections()
        names = []
        for c in collections:
            name = c.name if hasattr(c, "name") else str(c)
            names.append(name)
        return sorted(set(names))

    def get_documents_in_collection(self, collection_name: str) -> list[str]:
        """Return distinct source filenames stored in a collection."""
        if not collection_name:
            return []
        try:
            collection = self.client.get_collection(name=collection_name)
            result = collection.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []

            files = set()
            for metadata in metadatas:
                if not metadata:
                    continue
                filename = (
                    metadata.get("source")
                    or metadata.get("file_name")
                    or metadata.get("filename")
                    or metadata.get("file")
                )
                if filename:
                    files.add(str(filename))
            return sorted(files)
        except Exception:
            return []

    # ── Search ─────────────────────────────────────────────────────

    def search_chunks(
        self,
        query: str,
        k: int = 5,
        collection_names: list[str] | None = None,
        metadata_filter: dict | None = None,
    ) -> list[Document]:
        """Vector search over document chunks in one or more collections."""
        available = self.list_collections()

        if not available:
            return []

        if collection_names:
            unknown = set(collection_names) - set(available)
            if unknown:
                logger.warning("Unknown collections requested: %s", unknown)
            targets = [n for n in collection_names if n in available]
        else:
            targets = available

        all_results: list[Document] = []
        for name in targets:
            try:
                chroma = Chroma(
                    collection_name=name,
                    embedding_function=self.embedding_provider,
                    persist_directory=self.chromadb_path,
                )
                results = chroma.similarity_search(
                    query, k=k, filter=metadata_filter
                )
                all_results.extend(results)
                logger.info(
                    "Chunk search in '%s' returned %d results.",
                    name,
                    len(results),
                )
            except Exception:
                logger.exception(
                    "Chunk search failed for collection '%s'.", name
                )

        return all_results[:k]


# Keep existing integrations working while callers migrate to VectorDBManager.
VectorStore = VectorDBManager
VectorStoreManager = VectorDBManager
