#!/usr/bin/env python3
"""Validate the immutable artifact evidence passed between delivery stages."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_BINDINGS = {
    "v1": ("document_chunks", "document_chunks_vector"),
    "v2": ("document_chunks_v2", "document_chunks_v2_vector"),
}


def validate_manifest(manifest: dict[str, object]) -> None:
    for key in ("commit", "api_image", "api_digest", "frontend_digest"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"manifest field {key!r} must be a non-empty string")
    if not DIGEST.fullmatch(manifest["api_digest"]):
        raise ValueError("api_digest must be a sha256 digest")
    expected = f"@{manifest['api_digest']}"
    if not manifest["api_image"].endswith(expected):
        raise ValueError("api_image must reference api_digest, never a mutable tag")
    if not HEX_DIGEST.fullmatch(manifest["frontend_digest"]):
        raise ValueError("frontend_digest must be a SHA-256 hex digest")
    if manifest.get("admin_image", manifest["api_image"]) != manifest["api_image"]:
        raise ValueError("API and admin jobs must use the same immutable image")

    binding = manifest.get("serving_binding")
    if not isinstance(binding, dict):
        raise ValueError("serving_binding must be an object")
    for key in ("corpus_version_id", "collection", "schema_version", "index", "configuration_digest"):
        if not isinstance(binding.get(key), str) or not binding[key]:
            raise ValueError(f"serving_binding.{key} must be a non-empty string")
    schema_version = binding["schema_version"]
    expected = SUPPORTED_BINDINGS.get(schema_version)
    if expected is None:
        raise ValueError("serving_binding.schema_version is unsupported")
    if (binding["collection"], binding["index"]) != expected:
        raise ValueError("serving_binding collection/index is incompatible with its schema")
    if manifest.get("api_schema_version") != schema_version:
        raise ValueError("API reader schema must match the serving binding schema")
    if not HEX_DIGEST.fullmatch(binding["configuration_digest"]):
        raise ValueError("serving_binding.configuration_digest must be a SHA-256 hex digest")

    if manifest.get("release_revision"):
        if not manifest.get("preview_revision"):
            raise ValueError("a promoted revision must have a preview revision")
        if manifest["release_revision"] != manifest["preview_revision"]:
            raise ValueError("promotion must use the exact preview revision")


def validate(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    validate_manifest(manifest)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_delivery_manifest.py MANIFEST")
    validate(Path(sys.argv[1]))
