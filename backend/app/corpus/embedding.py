"""Vertex AI document embeddings for corpus ingestion."""

from __future__ import annotations

from typing import Any, Protocol

EMBEDDING_MODEL = "gemini-embedding-001"
DOCUMENT_TASK_TYPE = "RETRIEVAL_DOCUMENT"
EMBEDDING_DIMENSIONS = 768


class EmbeddingClient(Protocol):
    """Small subset of the Google Gen AI client used by the service."""

    models: Any


class VertexDocumentEmbedder:
    """Generate and validate one document vector at a time."""

    def __init__(self, client: EmbeddingClient, *, model: str = EMBEDDING_MODEL) -> None:
        self.client = client
        self.model = model

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("text must be non-empty")
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config={
                "task_type": DOCUMENT_TASK_TYPE,
                "output_dimensionality": EMBEDDING_DIMENSIONS,
            },
        )
        values = _embedding_values(response)
        if len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Vertex returned {len(values)} dimensions; expected {EMBEDDING_DIMENSIONS}"
            )
        return values


def create_vertex_document_embedder(*, project: str, location: str) -> VertexDocumentEmbedder:
    """Create the workload-identity Vertex client used by ingestion jobs."""
    from google import genai

    client = genai.Client(vertexai=True, project=project, location=location)
    return VertexDocumentEmbedder(client)


def _embedding_values(response: Any) -> list[float]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if not embeddings:
        raise ValueError("Vertex returned no embedding")
    first = embeddings[0]
    values = getattr(first, "values", None)
    if values is None and isinstance(first, dict):
        values = first.get("values")
    if values is None:
        raise ValueError("Vertex returned an embedding without values")
    return [float(value) for value in values]
