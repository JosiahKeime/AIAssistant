"""
Document Processor
==================

Text processing pipeline: read files, chunk, normalize, and ingest
into the vector store. Simplified for flat RAG — no hierarchical
summary tiers.

Attributes:
- vector_db_manager: the vector database manager
- chunker: the chunker
- embedding_provider: the embedding provider

Functions:
- coerce_file_input: adapt file paths to the name/type/read interface of uploaded files
- check_supported_type: return True if the file's MIME type is marked supported in config
- read_file: read a file and return its text content
- _check_duplicate: check if a file has already been embedded in a collection
- process_document: full document ingestion: validate → chunk → embed
"""

import os
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain_chroma import Chroma

import config
from .chunker import Chunker
from .embedding import EmbeddingProvider
from .logging_utils import logger

load_dotenv()


class DocumentProcessor:
    """Read documents, chunk, and orchestrate ingestion into Vector DB."""

    def __init__(
        self,
        vector_db_manager,
        chunk_size=512,
        chunk_overlap=64,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.vector_db_manager = vector_db_manager
        self.chunker = Chunker(chunk_size, chunk_overlap)
        self.embedding_provider = embedding_provider or EmbeddingProvider()

    # ── Static utilities ───────────────────────────────────────────

    @staticmethod
    def coerce_file_input(file) -> object:
        """Adapt file paths to the name/type/read interface of uploaded files."""
        if isinstance(file, (str, os.PathLike)):
            path = Path(file)
            ext = path.suffix.lower()
            mime_map = {
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".markdown": "text/markdown",
                ".pdf": "application/pdf",
                ".docx": (
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
            }
            return SimpleNamespace(
                name=path.name,
                type=mime_map.get(ext, "application/octet-stream"),
                read=lambda: path.read_bytes(),
            )
        return file

    # ── File I/O ───────────────────────────────────────────────────

    def check_supported_type(self, file) -> bool:
        """Return True if the file's MIME type is marked supported in config."""
        file_obj = self.coerce_file_input(file)
        if config.supported_file_types.get(file_obj.type, "unsupported") != "supported":
            logger.warning(
                "Unsupported file type: %s for file %s",
                file_obj.type,
                file_obj.name,
            )
            return False
        return True

    def read_file(self, file) -> str:
        """Read a file and return its text content."""
        file_obj = self.coerce_file_input(file)
        if file_obj.type in ("text/plain", "text/markdown"):
            return file_obj.read().decode("utf-8")
        raise ValueError(f"Unsupported type: {file_obj.type}")

    def _check_duplicate(self, filename: str, collection_name: str) -> bool:
        """Check if a file has already been embedded in a collection."""
        try:
            collection = self.vector_db_manager.client.get_collection(name=collection_name)
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

    # ── Ingestion Pipeline ─────────────────────────────────────────

    def process_document(self, file, collection_name: str) -> tuple[bool, str]:
        """Full document ingestion: validate → chunk → embed.

        Flat RAG: chunks are stored directly in the target collection
        with no summary hierarchy.
        """
        file_obj = self.coerce_file_input(file)
        logger.info(
            "Attempting to process file '%s' (type=%s) into collection '%s'.",
            file_obj.name, file_obj.type, collection_name,
        )

        # Reject unsupported types early
        if not self.check_supported_type(file_obj):
            return False, f"File type {file_obj.type} is not supported."

        # Reject duplicates early
        if self._check_duplicate(file_obj.name, collection_name):
            return False, f"Duplicate file: {file_obj.name} already processed in collection '{collection_name}'."

        try:
            # Ensure collection exists
            self.vector_db_manager.create_collection(collection_name)

            # 1. Read and chunk
            text = self.read_file(file_obj)
            chunks = self.chunker.split(text, file_obj.name)
            logger.info(
                "Embedding %d chunks for '%s' into '%s'.",
                len(chunks), file_obj.name, collection_name,
            )

            # 2. Store chunks in the collection
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_provider,
                persist_directory=self.vector_db_manager.chromadb_path,
                collection_name=collection_name,
            )

            return True, f"File {file_obj.name} processed into '{collection_name}'."

        except Exception as e:
            logger.exception("Processing failed for '%s'.", file_obj.name)
            return False, f"Processing failed: {e}"
