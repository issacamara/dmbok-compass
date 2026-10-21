"""Corpus source and ingestion domain helpers."""

from .chunking import SourcePage, TextChunk, chunk_pages
from .embedding import (
    DOCUMENT_TASK_TYPE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    QUERY_TASK_TYPE,
    create_vertex_embeddings,
)
from .indexing import (
    VECTOR_COLLECTION,
    VECTOR_SCHEMA_VERSION,
    CorpusEmbeddingWriter,
    create_vector_store,
)
from .manifest import RAG_DERIVATION, SourceManifest, create_source_manifest, default_derivation
from .lexical_artifact import (
    LexicalArtifactError,
    LexicalArtifactManifest,
    LoadedLexicalArtifact,
    build_lexical_artifact,
    load_lexical_artifact,
)
from .versions import (
    ConcurrentCorpusActivationError,
    CorpusCompletenessError,
    CorpusVersionError,
    InMemoryCorpusVersionStore,
)

__all__ = [
    "SourceManifest",
    "RAG_DERIVATION",
    "default_derivation",
    "SourcePage",
    "TextChunk",
    "DOCUMENT_TASK_TYPE",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MODEL",
    "CorpusEmbeddingWriter",
    "QUERY_TASK_TYPE",
    "VECTOR_COLLECTION",
    "VECTOR_SCHEMA_VERSION",
    "create_vector_store",
    "create_vertex_embeddings",
    "chunk_pages",
    "create_source_manifest",
    "CorpusCompletenessError",
    "ConcurrentCorpusActivationError",
    "CorpusVersionError",
    "InMemoryCorpusVersionStore",
    "LexicalArtifactError",
    "LexicalArtifactManifest",
    "LoadedLexicalArtifact",
    "build_lexical_artifact",
    "load_lexical_artifact",
]
