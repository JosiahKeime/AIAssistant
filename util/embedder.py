import os
import re
from pathlib import Path
from types import SimpleNamespace

import anthropic
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from .logging_utils import logger

load_dotenv()

SUMMARIES_COLLECTION = "document-summaries"
DOCUMENT_COLLECTION = "document-collection"


class Embedder:
    # Coordinates the document pipeline: split and normalize text, store its
    # chunks in Chroma, create a searchable summary, and run similarity search.
    def __init__(self, embedding_model_name=None,
                 chunk_size=512, chunk_overlap=64, chromadb_path="./chroma_db"):
        if embedding_model_name is None:
            embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

        self.embedding_model_name = embedding_model_name
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name, openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.chromadb_path = chromadb_path
        existing_client = chromadb.PersistentClient(path=chromadb_path)
        self.collection_names = [c.name for c in existing_client.list_collections()]

        logger.info( "Initializing Embedder | model=%s | chunk_size=%s | chunk_overlap=%s | chromadb_path=%s",
                    embedding_model_name, chunk_size, chunk_overlap, chromadb_path, )

    @staticmethod
    def normalize_chunk_text(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _coerce_file_input(file) -> object:
        # File paths do not have the same interface as uploaded-file objects,
        # so adapt them to the name/type/read attributes used below.
        if isinstance(file, (str, os.PathLike)):
            path = Path(file)
            ext = path.suffix.lower()
            mime_map = {
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".markdown": "text/markdown",
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            }
            return SimpleNamespace(
                name=path.name,
                type=mime_map.get(ext, "application/octet-stream"),
                read=lambda: path.read_bytes(),
            )
        return file

    def file_transformer(self, file) -> str:
        file_obj = self._coerce_file_input(file)
        # The support check uses configured MIME types, but decoding is
        # currently implemented only for UTF-8 plain text and Markdown.
        if file_obj.type in ("text/plain", "text/markdown"):
            return file_obj.read().decode("utf-8")
        raise ValueError(f"Unsupported type: {file_obj.type}")

    def check_duplicate(self, file) -> bool:
        file_obj = self._coerce_file_input(file)

        try:
            client = chromadb.PersistentClient(path=self.chromadb_path)
            collection = client.get_collection(name=DOCUMENT_COLLECTION)
            existing = collection.get(where={"source": file_obj.name})
            if existing and existing.get("ids"):
                logger.info("Duplicate detected for file '%s' in collection '%s'", file_obj.name, DOCUMENT_COLLECTION)
                return True
        except Exception:
            # A missing collection means there cannot be a duplicate yet. This
            # also preserves the current behavior for other lookup failures.
            pass

        return False

    def check_supported_type(self, file) -> bool:
        file_obj = self._coerce_file_input(file)
        if config.supported_file_types.get(file_obj.type, "unsupported") != "supported":
            logger.warning("Unsupported file type: %s for file %s", file_obj.type, file_obj.name)
            return False
        return True

    def _generate_summary(self, text: str, filename: str) -> str:
        """Call Claude to produce a one-sentence summary of *text*.

        Samples up to 20 evenly-spaced chunks so the prompt stays within token
        limits while covering the whole document.  Falls back to a plain
        filename-based description on any error.
        """
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            return f"Document embedded from {filename}."

        # Sample evenly spaced chunks so the prompt stays capped at 20 chunks
        # while still representing the beginning, middle, and end of the file.
        step = max(1, len(chunks) // 20)
        sample = [chunks[i] for i in range(0, len(chunks), step)][:20]
        combined = "\n\n---\n\n".join(sample)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set — skipping summary generation.")
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
                            "Below are excerpts from a document. Write exactly ONE "
                            "concise sentence describing what the document is about. "
                            "Output only the sentence — no preamble, no quotes.\n\n"
                            f"{combined}"
                        ),
                    }
                ],
            )
            summary = response.content[0].text.strip()
            logger.info("Generated summary for '%s': %s", filename, summary)
            return summary
        except Exception:
            logger.exception("Summary generation failed for '%s' — using fallback.", filename)
            return f"Document embedded from {filename}."

    def _embed_summary(self, summary: str, filename: str) -> None:
        """Embed *summary* as a single document into the shared summaries collection."""
        doc = Document(
            page_content=summary,
            metadata={
                "source": filename,
            },
        )
        _ = Chroma.from_documents(
            documents=[doc],
            embedding=self.embedding_model,
            persist_directory=self.chromadb_path,
            collection_name=SUMMARIES_COLLECTION,
        )
        logger.info(
            "Summary for '%s' embedded into '%s'.", filename, SUMMARIES_COLLECTION
        )

    def embed_file(self, file) -> tuple[bool, str]:
        file_obj = self._coerce_file_input(file)
        logger.info("Attempting to embed file: %s of type %s into collection '%s'", file_obj.name, file_obj.type, DOCUMENT_COLLECTION)

        if self.check_duplicate(file_obj):
            return False, f"Duplicate file: {file_obj.name} already embedded in collection '{DOCUMENT_COLLECTION}'."

        if not self.check_supported_type(file_obj):
            return False, f"File type {file_obj.type} is not supported."

        try:
            text = self.file_transformer(file_obj)
            chunks_text = self.text_splitter.split_text(text)

            normalized_chunks = []
            for chunk in chunks_text:
                normalized_chunk = self.normalize_chunk_text(chunk)
                if normalized_chunk:
                    normalized_chunks.append(normalized_chunk)

            chunks = [
                Document(page_content=t, metadata={"source": file_obj.name})
                for t in normalized_chunks
            ]

            logger.info(
                "Embedding %d normalized chunks for file '%s' into collection '%s'", 
                len(chunks), file_obj.name, DOCUMENT_COLLECTION,
            )

            _ = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.chromadb_path,
                collection_name=DOCUMENT_COLLECTION,
            )

            # Keep summaries in one shared collection so they can be searched
            # across documents independently of the full chunk collections.
            summary = self._generate_summary(text, file_obj.name)
            self._embed_summary(summary, file_obj.name)

            return True, (
                f"File {file_obj.name} embedded successfully into collection '{DOCUMENT_COLLECTION}'. "
                f"Summary: {summary}"
            )

        except Exception as e:
            logger.exception("Embedding failed for file '%s'", file_obj.name)
            return False, f"Embedding failed: {e}"


    def vector_db_search(self, query, collection_name, k=3, metadata_filter=None):
        logger.info(
            "Vector search | collection=%s | query=%s | k=%s | metadata_filter=%s",
            collection_name, query, k, metadata_filter,
        )

        # Collection names can be created after this Embedder starts, so query
        # Chroma instead of relying on the initialization-time cache.
        client = chromadb.PersistentClient(path=self.chromadb_path)
        available_collections = {collection.name for collection in client.list_collections()}
        if collection_name not in available_collections:
            logger.warning("Collection '%s' does not exist.", collection_name)
            return []

        collection = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.chromadb_path,
        )
        results = collection.similarity_search(query, k=k, filter=metadata_filter)
        logger.info("Vector search returned %d results for collection '%s'", len(results), collection_name)
        return results
