"""Native LangChain Vertex embeddings shared by ingestion and retrieval."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

EMBEDDING_MODEL = "gemini-embedding-001"
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE = "RETRIEVAL_QUERY"
EMBEDDING_DIMENSIONS = 768


class VertexEmbeddings(Embeddings):
    """Validate the native integration's fixed vector contract."""

    def __init__(self, service: GoogleGenerativeAIEmbeddings) -> None:
        self.service = service

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.service.embed_documents(texts, task_type=DOCUMENT_TASK_TYPE)
        return [_validated(vector) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return _validated(self.service.embed_query(text, task_type=QUERY_TASK_TYPE))


def create_vertex_embeddings(
    *,
    project: str,
    location: str,
    client: Any | None = None,
    credentials: Any | None = None,
) -> VertexEmbeddings:
    """Create one native LangChain embedding service for both task paths."""

    if not project.strip():
        raise ValueError("project must be non-empty")
    if not location.strip():
        raise ValueError("location must be non-empty")
    service = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        vertexai=True,
        project=project,
        location=location,
        credentials=credentials,
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )
    if client is not None:
        service.client = client
    return VertexEmbeddings(service)


def _validated(vector: list[float]) -> list[float]:
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Vertex returned {len(vector)} dimensions; expected {EMBEDDING_DIMENSIONS}"
        )
    return [float(value) for value in vector]
