"""Immutable metadata tying an approved source object to a corpus version."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """The durable identity of one versioned PDF source object."""

    bucket: str
    object_name: str
    generation: int
    content_sha256: str
    corpus_version_id: str
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

    identity = f"{bucket}/{object_name}#{generation}:{content_sha256}".encode()
    version_id = f"v-{hashlib.sha256(identity).hexdigest()[:24]}"
    return SourceManifest(
        bucket=bucket,
        object_name=object_name,
        generation=generation,
        content_sha256=content_sha256,
        corpus_version_id=version_id,
        content_type=content_type,
    )
