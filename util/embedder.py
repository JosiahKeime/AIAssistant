"""
Embedder
========

Text processing pipeline: read files, chunk, normalize, and generate
summaries.  Storage is handled by VectorDBManager — this class is
intentionally stateless with respect to ChromaDB.
"""

import os
import re
from pathlib import Path
from types import SimpleNamespace

import anthropic
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from .logging_utils import logger

load_dotenv()


class Embedder:
    """Chunks, normalizes, and embeds text.  Does NOT own storage."""

    def __init__(
        self,
        embedding_model_name=None,
        chunk_size=512,
        chunk_overlap=64,
    ):
        if embedding_model_name is None:
            embedding_model_name = os.getenv(
                "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
            )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

        self.embedding_model_name = embedding_model_name
        self.embedding_model = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        logger.info(
            "Initializing Embedder | model=%s | chunk_size=%s | chunk_overlap=%s",
            embedding_model_name,
            chunk_size,
            chunk_overlap,
        )

    # ── Static utilities ───────────────────────────────────────────

    @staticmethod
    def normalize_chunk_text(text: str) -> str:
        """Collapse whitespace and newlines into clean single-line text."""
        if not text:
            return ""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

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

    # ── Chunking ───────────────────────────────────────────────────

    def chunk_and_normalize(
        self, text: str, source_name: str
    ) -> list[Document]:
        """Split text into normalized LangChain Documents with source metadata."""
        chunks_text = self.text_splitter.split_text(text)
        normalized = [self.normalize_chunk_text(c) for c in chunks_text]
        return [
            Document(page_content=t, metadata={"source": source_name})
            for t in normalized
            if t
        ]

    # ── Summary generation ─────────────────────────────────────────

    def generate_summary(self, text: str, filename: str) -> str:
        """Call Claude Haiku to produce a one-sentence summary.

        Samples up to 20 evenly-spaced chunks so the prompt stays within
        token limits while covering the whole document.
        """
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            return f"Document embedded from {filename}."

        step = max(1, len(chunks) // 20)
        sample = [chunks[i] for i in range(0, len(chunks), step)][:20]
        combined = "\n\n---\n\n".join(sample)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set — skipping summary generation."
            )
            return f"Document embedded from {filename}."

        try:
            llm = anthropic.Anthropic(api_key=api_key)
            response = llm.messages.create(
                model="claude-haiku-4-20250514",
                max_tokens=128,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Below are excerpts from a document. Write exactly "
                            "ONE concise sentence describing what the document "
                            "is about. Output only the sentence — no preamble, "
                            "no quotes.\n\n"
                            f"{combined}"
                        ),
                    }
                ],
            )
            summary = response.content[0].text.strip()
            logger.info("Generated summary for '%s': %s", filename, summary)
            return summary
        except Exception:
            logger.exception(
                "Summary generation failed for '%s' — using fallback.",
                filename,
            )
            return f"Document embedded from {filename}."
