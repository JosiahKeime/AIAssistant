"""Document chunking and normalization utilities."""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    """Split source text into normalized LangChain documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    @staticmethod
    def normalize_text(text: str) -> str:
        """Collapse whitespace and newlines into clean single-line text."""
        if not text:
            return ""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def split(self, text: str, source_name: str) -> list[Document]:
        """Return normalized chunks with source metadata."""
        chunks_text = self.text_splitter.split_text(text)
        return [
            Document(
                page_content=normalized,
                metadata={"source": source_name},
            )
            for chunk in chunks_text
            if (normalized := self.normalize_text(chunk))
        ]

    def split_text(self, text: str) -> list[str]:
        """Return text chunks without Document metadata."""
        return self.text_splitter.split_text(text)

    @property
    def metadata(self) -> dict[str, int]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
