from __future__ import annotations
"""Embedding providers for the supported remote and local backends.

Embedding provides the interface for embedding documents and queries.
It supports three backends: ollama, openai, and local.

Attributes:
- backend: the embedding backend to use
- model_name: the model to use
- dimension: the dimension of the embeddings

Functions:
- embed_documents: embed a list of document (inputs: list[str], outputs: list[list[float]])
- embed_query: embed a query (inputs: str, outputs: list[float]) 
- metadata: return metadata about the embedding provider (inputs: None, outputs: dict)
"""

import os
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from .logging_utils import logger

# Resolve the best available device once at import time
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _LocalEmbeddingFunction:
    """Run a SentenceTransformer model locally, using the GPU when available."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Loading local embedding model '%s' on device=%s", model_name, _DEVICE)
        self._model = SentenceTransformer(model_name, device=_DEVICE)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [[float(v) for v in vec] for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(text, convert_to_numpy=True)
        return [float(v) for v in embedding]


class EmbeddingProvider:
    """Create vectors and describe the model contract used by a collection."""
    def __init__(self, backend: str | None = None, model_name: str | None = None):
        requested_backend = (backend or os.getenv("EMBEDDING_BACKEND", "ollama")).lower()
        self.backend = "local" if requested_backend == "chroma" else requested_backend
        requested_model = model_name or os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
        
        #configure embedding backend service
        if self.backend == "ollama":
            self.model_name = requested_model
            self.embedding_model = OllamaEmbeddings(model=self.model_name, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

        elif self.backend == "openai":
            self.model_name = requested_model
            api_key = os.getenv("OPENAI_API_KEY")
            self.embedding_model = OpenAIEmbeddings(
                model=self.model_name,
                api_key=SecretStr(api_key) if api_key else None,
            )
            
        elif self.backend == "local":
            self.model_name = "all-MiniLM-L6-v2"
            if requested_model != self.model_name:
                logger.warning("Chroma's local backend uses %s; ignoring EMBEDDING_MODEL_NAME=%s.",self.model_name,requested_model)     
            self.embedding_model = _LocalEmbeddingFunction()
        else:
            raise ValueError("Unsupported EMBEDDING_BACKEND. Choose 'ollama', 'openai', or 'local'.")

        self.dimension: int | None = None
        logger.info("Initializing EmbeddingProvider | backend=%s | model=%s", self.backend, self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = [[float(value) for value in vector] 
                    for vector in self.embedding_model.embed_documents(texts)]
        if vectors and self.dimension is None: self.dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = [float(value) for value in self.embedding_model.embed_query(text)]
        if self.dimension is None: self.dimension = len(vector)
        return vector

    def metadata(self) -> dict[str, Any]:
        """Return JSON- and Chroma-compatible provider metadata."""
        return {
            "embedding_backend": self.backend,
            "embedding_model": self.model_name,
            "embedding_dimension": self.dimension or "unknown",
        }
