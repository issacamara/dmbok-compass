"""Idempotent Firestore persistence for embedded corpus chunks."""

from __future__ import annotations

from typing import Any, Protocol

from google.cloud.firestore_v1.vector import Vector

from app.contracts import DocumentChunk
from app.corpus.chunking import TextChunk
from app.corpus.embedding import VertexDocumentEmbedder


class DocumentReference(Protocol):
    def set(self, data: dict[str, Any]) -> Any: ...


class DocumentCollection(Protocol):
    def document(self, document_id: str) -> DocumentReference: ...


class FirestoreClient(Protocol):
    def collection(self, collection_id: str) -> DocumentCollection: ...


class FirestoreChunkRepository:
    """Store immutable chunk identities with replace-safe writes."""

    collection_id = "document_chunks"

    def __init__(self, client: FirestoreClient) -> None:
        self.chunks = client.collection(self.collection_id)

    def upsert(self, chunk: DocumentChunk) -> DocumentChunk:
        """Write the same deterministic document on retries and return it."""
        payload = chunk.model_dump(mode="json")
        payload["embedding"] = Vector(chunk.embedding)
        self.chunks.document(chunk.chunk_id).set(payload)
        return chunk


class CorpusEmbeddingWriter:
    """Embed deterministic text chunks and persist their durable records."""

    def __init__(self, embedder: VertexDocumentEmbedder, repository: FirestoreChunkRepository) -> None:
        self.embedder = embedder
        self.repository = repository

    def write(self, chunks: list[TextChunk] | tuple[TextChunk, ...]) -> tuple[DocumentChunk, ...]:
        stored: list[DocumentChunk] = []
        for chunk in chunks:
            stored.append(
                self.repository.upsert(
                    DocumentChunk(
                        chunk_id=chunk.chunk_id,
                        corpus_version_id=chunk.corpus_version_id,
                        page=chunk.page,
                        section=chunk.section,
                        chunk_ordinal=chunk.chunk_ordinal,
                        content_hash=chunk.content_hash,
                        text=chunk.text,
                        embedding=self.embedder.embed(chunk.text),
                    )
                )
            )
        return tuple(stored)
