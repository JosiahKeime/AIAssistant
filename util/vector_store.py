"""
VectorDBManager
==================

Owns the ChromaDB persistence layer and manages the three-tier retrieval
hierarchy:

    collection-summaries  →  one sentence per user collection
    document-summaries    →  one sentence per document (scoped to collection)
    <user-collection>     →  actual document chunks

Usage:
    store = VectorDBManager()
    vsm.create_collection("legal-docs")
    vsm.embed_document(file, "legal-docs")
    results = vsm.search_collection_summaries("employment law")
"""

from __future__ import annotations

import os
from typing import Any

import anthropic
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .embedder import Embedder
from .logging_utils import logger


COLLECTION_SUMMARIES = "collection-summaries"
DOCUMENT_SUMMARIES = "document-summaries"
INTERNAL_COLLECTIONS = {COLLECTION_SUMMARIES, DOCUMENT_SUMMARIES}


class VectorDBManager:
    """Manage ChromaDB collections, documents, summaries, and search."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        chromadb_path: str = "./chroma_db",
    ):
        self.chromadb_path = chromadb_path
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.embedder = embedder or Embedder()

        logger.info(
            "Initializing VectorDBManager | chromadb_path=%s", chromadb_path
        )

    # ── Collection management ──────────────────────────────────────

    def create_collection(self, name: str) -> None:
        """Create a user collection (idempotent)."""
        self.client.get_or_create_collection(name=name)
        logger.info("Created/verified collection '%s'.", name)

    def list_collections(self) -> list[str]:
        """Return sorted names of user-facing collections (excludes internal ones)."""
        collections = self.client.list_collections()
        names = []
        for c in collections:
            name = c.name if hasattr(c, "name") else str(c)
            if name not in INTERNAL_COLLECTIONS:
                names.append(name)
        return sorted(set(names))

    def get_collection_summary(self, collection_name: str) -> str | None:
        """Retrieve the stored summary sentence for a collection."""
        try:
            collection = self.client.get_collection(name=COLLECTION_SUMMARIES)
            result = collection.get(
                ids=[f"collection-summary-{collection_name}"],
                include=["documents"],
            )
            documents = result.get("documents", [])
            if documents and documents[0]:
                return documents[0]
        except Exception:
            pass
        return None

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

    # ── Document embedding ─────────────────────────────────────────

    def _check_duplicate(self, filename: str, collection_name: str) -> bool:
        """Check if a file has already been embedded in a collection."""
        try:
            collection = self.client.get_collection(name=collection_name)
            existing = collection.get(where={"source": filename})
            if existing and existing.get("ids"):
                logger.info(
                    "Duplicate detected for '%s' in collection '%s'.",
                    filename,
                    collection_name,
                )
                return True
        except Exception:
            pass
        return False

    def embed_document(self, file, collection_name: str) -> tuple[bool, str]:
        """Full document ingestion: validate → chunk → embed → summarize.

        After embedding the document chunks and their summary, the
        collection-level summary is eagerly regenerated so it stays
        current for retrieval.
        """
        file_obj = Embedder.coerce_file_input(file)
        logger.info(
            "Attempting to embed file '%s' (type=%s) into collection '%s'.",
            file_obj.name,
            file_obj.type,
            collection_name,
        )

        if not self.embedder.check_supported_type(file_obj):
            return False, f"File type {file_obj.type} is not supported."

        if self._check_duplicate(file_obj.name, collection_name):
            return False, (
                f"Duplicate file: {file_obj.name} already embedded "
                f"in collection '{collection_name}'."
            )

        try:
            # 1. Read and chunk
            text = self.embedder.read_file(file_obj)
            chunks = self.embedder.chunk_and_normalize(text, file_obj.name)

            logger.info(
                "Embedding %d chunks for '%s' into '%s'.",
                len(chunks),
                file_obj.name,
                collection_name,
            )

            # 2. Store chunks in the user's collection
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embedder.embedding_model,
                persist_directory=self.chromadb_path,
                collection_name=collection_name,
            )

            # 3. Generate and store document summary
            summary = self.embedder.generate_summary(text, file_obj.name)
            self._store_document_summary(summary, file_obj.name, collection_name)

            # 4. Regenerate collection summary (eager)
            self._regenerate_collection_summary(collection_name)

            return True, (
                f"File {file_obj.name} embedded into '{collection_name}'. "
                f"Summary: {summary}"
            )

        except Exception as e:
            logger.exception("Embedding failed for '%s'.", file_obj.name)
            return False, f"Embedding failed: {e}"

    # ── Summary storage ────────────────────────────────────────────

    def _store_document_summary(
        self, summary: str, filename: str, collection_name: str
    ) -> None:
        """Upsert a document summary with collection metadata."""
        collection = self.client.get_or_create_collection(
            name=DOCUMENT_SUMMARIES
        )
        embedding = self.embedder.embedding_model.embed_documents([summary])[0]

        collection.upsert(
            ids=[f"doc-summary-{collection_name}-{filename}"],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[{"source": filename, "collection": collection_name}],
        )
        logger.info(
            "Document summary for '%s' stored in '%s'.",
            filename,
            DOCUMENT_SUMMARIES,
        )

    def _regenerate_collection_summary(self, collection_name: str) -> None:
        """Gather all document summaries for a collection and produce a
        single-sentence collection summary via Claude Haiku."""
        try:
            summaries_col = self.client.get_or_create_collection(
                name=DOCUMENT_SUMMARIES
            )
            results = summaries_col.get(
                where={"collection": collection_name},
                include=["documents"],
            )
            documents = results.get("documents", [])
        except Exception:
            documents = []

        if not documents:
            summary = (
                f"Collection '{collection_name}' (empty — no documents yet)."
            )
        else:
            combined = "\n".join(doc for doc in documents if doc)
            api_key = os.getenv("ANTHROPIC_API_KEY")

            if not api_key:
                logger.warning(
                    "ANTHROPIC_API_KEY not set — using fallback collection summary."
                )
                summary = (
                    f"Collection '{collection_name}' containing "
                    f"{len(documents)} document(s)."
                )
            else:
                try:
                    llm = anthropic.Anthropic(api_key=api_key)
                    response = llm.messages.create(
                        model="claude-haiku-4-20250514",
                        max_tokens=128,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "Below are one-sentence summaries of every "
                                    "document in a collection called "
                                    f"'{collection_name}'. Write exactly ONE "
                                    "concise sentence describing what this "
                                    "collection as a whole contains. Output only "
                                    "the sentence — no preamble, no quotes.\n\n"
                                    f"{combined}"
                                ),
                            }
                        ],
                    )
                    summary = response.content[0].text.strip()
                    logger.info(
                        "Generated collection summary for '%s': %s",
                        collection_name,
                        summary,
                    )
                except Exception:
                    logger.exception(
                        "Collection summary generation failed for '%s'.",
                        collection_name,
                    )
                    summary = (
                        f"Collection '{collection_name}' containing "
                        f"{len(documents)} document(s)."
                    )

        # Upsert into the collection-summaries collection
        col = self.client.get_or_create_collection(name=COLLECTION_SUMMARIES)
        embedding = self.embedder.embedding_model.embed_documents([summary])[0]
        col.upsert(
            ids=[f"collection-summary-{collection_name}"],
            documents=[summary],
            embeddings=[embedding],
            metadatas=[{"collection": collection_name}],
        )

    # ── Search ─────────────────────────────────────────────────────

    def search_collection_summaries(
        self, query: str, k: int = 5
    ) -> list[Document]:
        """Tier 1: Vector search over collection-level summaries."""
        try:
            available = {
                c.name if hasattr(c, "name") else str(c)
                for c in self.client.list_collections()
            }
            if COLLECTION_SUMMARIES not in available:
                return []

            chroma = Chroma(
                collection_name=COLLECTION_SUMMARIES,
                embedding_function=self.embedder.embedding_model,
                persist_directory=self.chromadb_path,
            )
            results = chroma.similarity_search(query, k=k)
            logger.info(
                "Collection summary search returned %d results for query '%s'.",
                len(results),
                query,
            )
            return results
        except Exception:
            logger.exception("Collection summary search failed.")
            return []

    def search_document_summaries(
        self,
        query: str,
        k: int = 5,
        collection_name: str | None = None,
    ) -> list[Document]:
        """Tier 2: Vector search over document-level summaries, optionally
        scoped to a single collection."""
        try:
            available = {
                c.name if hasattr(c, "name") else str(c)
                for c in self.client.list_collections()
            }
            if DOCUMENT_SUMMARIES not in available:
                return []

            metadata_filter = (
                {"collection": collection_name} if collection_name else None
            )
            chroma = Chroma(
                collection_name=DOCUMENT_SUMMARIES,
                embedding_function=self.embedder.embedding_model,
                persist_directory=self.chromadb_path,
            )
            results = chroma.similarity_search(
                query, k=k, filter=metadata_filter
            )
            logger.info(
                "Document summary search returned %d results "
                "for query '%s' (collection=%s).",
                len(results),
                query,
                collection_name,
            )
            return results
        except Exception:
            logger.exception("Document summary search failed.")
            return []

    def search_chunks(
        self,
        query: str,
        k: int = 5,
        collection_names: list[str] | None = None,
        metadata_filter: dict | None = None,
    ) -> list[Document]:
        """Tier 3: Vector search over document chunk collections."""
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
                    embedding_function=self.embedder.embedding_model,
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

        return all_results[: k * 2]


# Keep existing integrations working while callers migrate to VectorDBManager.
VectorStore = VectorDBManager
VectorStoreManager = VectorDBManager
