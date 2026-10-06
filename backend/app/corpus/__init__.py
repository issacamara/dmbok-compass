"""Corpus source and ingestion domain helpers."""

from .chunking import SourcePage, TextChunk, chunk_pages
from .embedding import (
    DOCUMENT_TASK_TYPE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    VertexDocumentEmbedder,
    create_vertex_document_embedder,
)
from .indexing import CorpusEmbeddingWriter, FirestoreChunkRepository
from .manifest import SourceManifest, create_source_manifest
from .versions import (
    CorpusCompletenessError,
    CorpusVersionError,
    InMemoryCorpusVersionStore,
)

__all__ = [
    "SourceManifest",
    "SourcePage",
    "TextChunk",
    "DOCUMENT_TASK_TYPE",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MODEL",
    "FirestoreChunkRepository",
    "CorpusEmbeddingWriter",
    "VertexDocumentEmbedder",
    "create_vertex_document_embedder",
    "chunk_pages",
    "create_source_manifest",
    "CorpusCompletenessError",
    "CorpusVersionError",
    "InMemoryCorpusVersionStore",
]
