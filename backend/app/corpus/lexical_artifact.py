"""Build and load immutable lexical artifacts from approved text chunks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from app.corpus.chunking import TextChunk
from app.retrieval.lexical import (
    ALGORITHM_VERSION,
    TOKENIZER_VERSION,
    LexicalDocument,
    LexicalIndex,
    tokenize,
)


ARTIFACT_SCHEMA_VERSION = 1


class LexicalArtifactError(ValueError):
    """Raised when an artifact is missing, corrupt, or mismatched."""


class ImmutableObjectStore(Protocol):
    def put(self, name: str, data: bytes, *, metadata: dict[str, str], if_absent: bool = True) -> None: ...
    def get(self, name: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class LexicalArtifactManifest:
    corpus_version_id: str
    chunk_set_sha256: str
    artifact_sha256: str
    chunk_count: int
    size_bytes: int
    tokenizer_version: str
    algorithm_version: str
    object_name: str

    def as_metadata(self) -> dict[str, str]:
        return {
            "lexical_schema_version": str(ARTIFACT_SCHEMA_VERSION),
            "corpus_version_id": self.corpus_version_id,
            "chunk_set_sha256": self.chunk_set_sha256,
            "artifact_sha256": self.artifact_sha256,
            "chunk_count": str(self.chunk_count),
            "size_bytes": str(self.size_bytes),
            "tokenizer_version": self.tokenizer_version,
            "algorithm_version": self.algorithm_version,
        }


@dataclass(frozen=True, slots=True)
class LoadedLexicalArtifact:
    manifest: LexicalArtifactManifest
    index: LexicalIndex
    load_ms: float


def build_lexical_artifact(
    chunks: tuple[TextChunk, ...] | list[TextChunk],
    *,
    corpus_version_id: str,
    object_name: str,
    store: ImmutableObjectStore,
    max_candidates: int = 5,
) -> LexicalArtifactManifest:
    """Serialize the approved chunk stream once and publish it immutably."""

    if not corpus_version_id or not object_name:
        raise ValueError("corpus_version_id and object_name are required")
    ordered = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
    if any(chunk.corpus_version_id != corpus_version_id for chunk in ordered):
        raise LexicalArtifactError("chunk stream contains a different corpus version")
    chunk_set_sha256 = _chunk_set_digest(ordered)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "corpus_version_id": corpus_version_id,
        "chunk_set_sha256": chunk_set_sha256,
        "tokenizer_version": TOKENIZER_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "content_hash": chunk.content_hash,
                "page": chunk.page,
                "section": chunk.section,
                "text": chunk.text,
                "terms": tokenize(chunk.text),
            }
            for chunk in ordered
        ],
    }
    data = _canonical_json(payload)
    artifact_sha256 = _sha256(data)
    manifest = LexicalArtifactManifest(
        corpus_version_id=corpus_version_id,
        chunk_set_sha256=chunk_set_sha256,
        artifact_sha256=artifact_sha256,
        chunk_count=len(ordered),
        size_bytes=len(data),
        tokenizer_version=TOKENIZER_VERSION,
        algorithm_version=ALGORITHM_VERSION,
        object_name=object_name,
    )
    store.put(object_name, data, metadata=manifest.as_metadata(), if_absent=True)
    return manifest


def load_lexical_artifact(
    store: ImmutableObjectStore,
    manifest: LexicalArtifactManifest,
    *,
    expected_corpus_version_id: str,
    expected_chunk_set_sha256: str,
    max_candidates: int = 5,
) -> LoadedLexicalArtifact:
    """Load and validate one immutable object without consulting Firestore."""

    started = perf_counter()
    if manifest.corpus_version_id != expected_corpus_version_id:
        raise LexicalArtifactError("lexical artifact corpus version does not match")
    if manifest.chunk_set_sha256 != expected_chunk_set_sha256:
        raise LexicalArtifactError("lexical artifact chunk-set digest does not match")
    try:
        data = store.get(manifest.object_name)
    except (KeyError, FileNotFoundError) as exc:
        raise LexicalArtifactError("lexical artifact object is missing") from exc
    if _sha256(data) != manifest.artifact_sha256:
        raise LexicalArtifactError("lexical artifact digest does not match manifest")
    try:
        payload = json.loads(data)
        documents = tuple(
            LexicalDocument(
                chunk_id=item["chunk_id"], page=item["page"], section=item["section"],
                text=item["text"], terms=tuple(item["terms"]),
            )
            for item in payload["chunks"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LexicalArtifactError("lexical artifact payload is corrupt") from exc
    if (
        payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or payload.get("corpus_version_id") != manifest.corpus_version_id
        or payload.get("chunk_set_sha256") != manifest.chunk_set_sha256
        or payload.get("tokenizer_version") != manifest.tokenizer_version
        or payload.get("algorithm_version") != manifest.algorithm_version
        or len(documents) != manifest.chunk_count
    ):
        raise LexicalArtifactError("lexical artifact metadata is inconsistent")
    return LoadedLexicalArtifact(
        manifest=manifest,
        index=LexicalIndex(documents, max_candidates=max_candidates),
        load_ms=round((perf_counter() - started) * 1000, 3),
    )


def _chunk_set_digest(chunks: tuple[TextChunk, ...]) -> str:
    material = [
        {"chunk_id": c.chunk_id, "content_hash": c.content_hash, "page": c.page, "section": c.section, "text": c.text}
        for c in chunks
    ]
    return _sha256(_canonical_json(material))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
