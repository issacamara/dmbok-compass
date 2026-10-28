#!/usr/bin/env python3
"""Resolve the authenticated project's active, ready corpus for deployment."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from app.contracts import CorpusServingBinding, CorpusValidationEvidence, CorpusVersion


class CorpusResolutionError(ValueError):
    """Raised when the active publication cannot safely drive deployment."""


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusResolutionError(f"active corpus {field} is missing")
    value = value.strip()
    if len(value) > 128:
        raise CorpusResolutionError(f"active corpus {field} is too long")
    return value


def resolve_active_corpus(client: Any) -> str:
    """Return the active version ID only when its publication is fully ready."""

    pointer_snapshot = client.collection("corpus_state").document("active").get()
    if not pointer_snapshot.exists:
        raise CorpusResolutionError("active corpus pointer is missing")
    pointer = pointer_snapshot.to_dict() or {}
    version_id = _require_string(pointer.get("version_id"), "version_id")
    pointer_binding = pointer.get("binding")
    if pointer_binding is not None and not isinstance(pointer_binding, dict):
        raise CorpusResolutionError("active corpus binding is malformed")

    version_snapshot = client.collection("corpus_versions").document(version_id).get()
    if not version_snapshot.exists:
        raise CorpusResolutionError(f"active corpus version is missing: {version_id}")
    try:
        version = CorpusVersion.model_validate(version_snapshot.to_dict() or {})
    except (TypeError, ValueError) as exc:
        raise CorpusResolutionError(f"active corpus version is malformed: {version_id}") from exc

    if version.version_id != version_id:
        raise CorpusResolutionError("active corpus version ID does not match its pointer")
    if version.status != "active":
        raise CorpusResolutionError(f"active corpus version is not active: {version_id}")
    if version.serving_binding is None:
        raise CorpusResolutionError(f"active corpus serving binding is missing: {version_id}")
    if pointer_binding is not None:
        try:
            binding = CorpusServingBinding.model_validate(pointer_binding)
        except (TypeError, ValueError) as exc:
            raise CorpusResolutionError(f"active corpus binding is malformed: {version_id}") from exc
        if binding != version.serving_binding or binding.corpus_version_id != version_id:
            raise CorpusResolutionError(f"active corpus binding does not match: {version_id}")

    try:
        evidence = CorpusValidationEvidence.model_validate(version.validation_evidence)
    except (TypeError, ValueError) as exc:
        raise CorpusResolutionError(f"active corpus readiness evidence is missing: {version_id}") from exc
    if evidence.chunk_count < 1 or not evidence.index_ready or not evidence.filter_ready:
        raise CorpusResolutionError(f"active corpus is not ready: {version_id}")
    return version_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Authenticated GCP project to read")
    parser.add_argument(
        "--github-output",
        default="",
        help="GITHUB_OUTPUT-compatible file to receive corpus_version_id",
    )
    args = parser.parse_args()

    try:
        from google.cloud import firestore

        version_id = resolve_active_corpus(firestore.Client(project=args.project))
    except CorpusResolutionError as exc:
        print(f"Corpus resolution failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - provider/credential failures are integration-only
        print(f"Corpus resolution failed: Firestore read unavailable ({type(exc).__name__})", file=sys.stderr)
        return 1
    print(f"Resolved active corpus version: {version_id}")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as output:
            output.write(f"corpus_version_id={version_id}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
