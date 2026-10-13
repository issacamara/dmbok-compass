"""Immutable metadata tying an approved source object to a corpus version."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RAG_DERIVATION = (
    "langchain-pypdf-page|normalize-v1|character-word-700-70|"
    "gemini-embedding-001-768-retrieval-document-query|firestore-schema-v2"
)


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """The durable identity of one versioned PDF source object."""

    bucket: str
    object_name: str
    generation: int
    content_sha256: str
    corpus_version_id: str
    derivation: str = RAG_DERIVATION
    content_type: str = "application/pdf"

    @property
    def object_uri(self) -> str:
        return f"gs://{self.bucket}/{self.object_name}#{self.generation}"


def create_source_manifest(
    *,
    bucket: str,
    object_name: str,
    generation: int,
    content_sha256: str,
    content_type: str = "application/pdf",
    derivation: str = RAG_DERIVATION,
) -> SourceManifest:
    """Validate approved-PDF metadata and derive a stable corpus version ID."""

    if not bucket or bucket.strip() != bucket:
        raise ValueError("bucket must be a non-empty name without surrounding whitespace")
    if not object_name or object_name.strip() != object_name:
        raise ValueError("object_name must be a non-empty path without surrounding whitespace")
    if not object_name.lower().endswith(".pdf"):
        raise ValueError("only PDF source objects are approved")
    if generation < 1:
        raise ValueError("generation must be a positive Cloud Storage generation")
    if content_type != "application/pdf":
        raise ValueError("only application/pdf sources are approved")
    if not _SHA256.fullmatch(content_sha256):
        raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
    if not derivation or derivation.strip() != derivation:
        raise ValueError("derivation must be a non-empty normalized fingerprint")

    identity = f"{bucket}/{object_name}#{generation}:{content_sha256}:{derivation}".encode()
    version_id = f"v-{hashlib.sha256(identity).hexdigest()[:24]}"
    return SourceManifest(
        bucket=bucket,
        object_name=object_name,
        generation=generation,
        content_sha256=content_sha256,
        corpus_version_id=version_id,
        derivation=derivation,
        content_type=content_type,
    )
