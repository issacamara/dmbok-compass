#!/usr/bin/env python3
"""Validate the immutable artifact evidence passed between delivery stages."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("commit", "api_image", "api_digest", "frontend_digest"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"manifest field {key!r} must be a non-empty string")
    if not DIGEST.fullmatch(manifest["api_digest"]):
        raise ValueError("api_digest must be a sha256 digest")
    expected = f"@{manifest['api_digest']}"
    if not manifest["api_image"].endswith(expected):
        raise ValueError("api_image must reference api_digest, never a mutable tag")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["frontend_digest"]):
        raise ValueError("frontend_digest must be a SHA-256 hex digest")
    if manifest.get("release_revision"):
        if not manifest.get("preview_revision"):
            raise ValueError("a promoted revision must have a preview revision")
        if manifest["release_revision"] != manifest["preview_revision"]:
            raise ValueError("promotion must use the exact preview revision")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_delivery_manifest.py MANIFEST")
    validate(Path(sys.argv[1]))
