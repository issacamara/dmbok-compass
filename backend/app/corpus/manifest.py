"""Canonical, immutable corpus-derivation manifests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from app.corpus.embedding import (
    DOCUMENT_TASK_TYPE,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    QUERY_TASK_TYPE,
)
from app.corpus.chunking import MAX_CHUNK_CHARACTERS, SUBDIVISION_ALGORITHM_VERSION
from app.corpus.indexing import VECTOR_COLLECTION, VECTOR_SCHEMA_VERSION


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY = re.compile(r"(?:api[_-]?key|credential|password|secret|token)", re.IGNORECASE)
RAG_DERIVATION = "langchain-rag-v2"


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """The durable, reproducible identity of one derived corpus."""

    bucket: str
    object_name: str
    generation: int
    content_sha256: str
    corpus_version_id: str
    derivation: str
    derivation_sha256: str
    fingerprint_material: str
    manifest_sha256: str
    content_type: str = "application/pdf"

    @property
    def object_uri(self) -> str:
        return f"gs://{self.bucket}/{self.object_name}#{self.generation}"

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a mutable view of the persisted canonical material."""

        return json.loads(self.fingerprint_material)

    def corpus_metadata(self) -> dict[str, Any]:
        """Metadata persisted with every corpus-owned vector document."""

        return {
            "source_uri": self.object_uri,
            "source_sha256": self.content_sha256,
            "corpus_manifest": self.manifest,
            "corpus_manifest_material": self.fingerprint_material,
            "corpus_manifest_sha256": self.manifest_sha256,
            "derivation_sha256": self.derivation_sha256,
        }


def default_derivation(*, vector_collection: str = VECTOR_COLLECTION) -> dict[str, Any]:
    """Return every architecture-mandated derivation input explicitly.

    Versions are read from lock-file-controlled distributions so an upgraded
    parser, splitter, or embedding integration cannot silently reuse a corpus
    produced by an earlier implementation.
    """

    if not vector_collection or vector_collection.strip() != vector_collection:
        raise ValueError("vector_collection must be a non-empty normalized value")
    return {
        "implementation_revision": RAG_DERIVATION,
        "loader": {
            "type": "langchain_community.document_loaders.PyPDFLoader",
            "loader_version": _package_version("langchain-community"),
            "parser": "pypdf",
            "parser_version": _package_version("pypdf"),
            "options": {"mode": "page"},
        },
        "normalization": {"version": "whitespace-v1"},
        "provenance": {"version": "section-recovery-v1"},
        "splitter": {
            "type": "langchain_text_splitters.CharacterTextSplitter",
            "version": _package_version("langchain-text-splitters"),
            "units": "words",
            "chunk_size": 700,
            "chunk_overlap": 70,
            "separator": " ",
            "subdivision": {
                "algorithm": SUBDIVISION_ALGORITHM_VERSION,
                "max_characters": MAX_CHUNK_CHARACTERS,
            },
        },
        "embedding": {
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "document_task": DOCUMENT_TASK_TYPE,
            "query_task": QUERY_TASK_TYPE,
            "integration_version": _package_version("langchain-google-genai"),
        },
        "vector_schema": {
            "collection": vector_collection,
            "version": VECTOR_SCHEMA_VERSION,
            "content_field": "text",
            "metadata_field": "metadata",
            "embedding_field": "embedding",
        },
    }


def create_source_manifest(
    *,
    bucket: str,
    object_name: str,
    generation: int,
    content_sha256: str,
    content_type: str = "application/pdf",
    derivation: Mapping[str, Any] | str | None = None,
) -> SourceManifest:
    """Validate source metadata and derive a stable corpus version ID.

    The former string revision remains accepted for compatibility. It is folded
    into the complete canonical material, so callers cannot omit required
    derivation inputs.
    """

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

    derivation_material = _normalise_derivation(derivation)
    _validate_derivation(derivation_material)
    canonical_derivation = _canonical_json(derivation_material)
    fingerprint = {
        "fingerprint_schema_version": 1,
        "source": {
            "bucket": bucket,
            "object_name": object_name,
            "generation": generation,
            "content_sha256": content_sha256,
            "content_type": content_type,
        },
        "derivation": derivation_material,
    }
    fingerprint_material = _canonical_json(fingerprint)
    manifest_sha256 = _sha256(fingerprint_material)
    return SourceManifest(
        bucket=bucket,
        object_name=object_name,
        generation=generation,
        content_sha256=content_sha256,
        corpus_version_id=f"v-{manifest_sha256[:24]}",
        derivation=canonical_derivation,
        derivation_sha256=_sha256(canonical_derivation),
        fingerprint_material=fingerprint_material,
        manifest_sha256=manifest_sha256,
        content_type=content_type,
    )


def _normalise_derivation(derivation: Mapping[str, Any] | str | None) -> dict[str, Any]:
    if derivation is None:
        return default_derivation()
    if isinstance(derivation, str):
        if not derivation or derivation.strip() != derivation:
            raise ValueError("derivation must be a non-empty normalized fingerprint")
        result = default_derivation()
        result["implementation_revision"] = derivation
        return result
    if not isinstance(derivation, Mapping):
        raise ValueError("derivation must be a mapping or normalized revision string")
    return _json_value(derivation)


def _json_value(value: Any) -> Any:
    """Copy JSON-compatible input and reject secret-bearing metadata."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("derivation mapping keys must be strings")
            if _SECRET_KEY.search(key):
                raise ValueError("derivation metadata must not contain secrets")
            result[key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("derivation metadata must be JSON-compatible")


def _validate_derivation(derivation: Mapping[str, Any]) -> None:
    for path in (
        ("loader", "type"),
        ("loader", "loader_version"),
        ("loader", "parser"),
        ("loader", "parser_version"),
        ("loader", "options", "mode"),
        ("normalization", "version"),
        ("provenance", "version"),
        ("splitter", "type"),
        ("splitter", "version"),
        ("splitter", "units"),
        ("splitter", "chunk_size"),
        ("splitter", "chunk_overlap"),
        ("splitter", "subdivision", "algorithm"),
        ("splitter", "subdivision", "max_characters"),
        ("embedding", "model"),
        ("embedding", "dimensions"),
        ("embedding", "document_task"),
        ("embedding", "query_task"),
        ("vector_schema", "collection"),
        ("vector_schema", "version"),
        ("vector_schema", "content_field"),
        ("vector_schema", "metadata_field"),
        ("vector_schema", "embedding_field"),
    ):
        value: Any = derivation
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError(f"derivation is missing required field {'.'.join(path)}")
            value = value[key]
        if value is None or value == "":
            raise ValueError(f"derivation field {'.'.join(path)} must be non-empty")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
