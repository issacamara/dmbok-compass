#!/usr/bin/env python3
"""Authenticated same-image dry-run source-retention inventory.

The input is a reviewed JSON snapshot supplied by the release/runtime image.
This command never deletes objects and never runs Terraform.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from app.release.source_retention import (
    ReferenceSnapshot,
    ReleaseSourceReference,
    RetentionPolicy,
    SourceGeneration,
    SourceRetentionError,
    inventory,
)


def _source(payload: dict[str, object]) -> SourceGeneration:
    return SourceGeneration(
        bucket=str(payload["bucket"]),
        object_name=str(payload["object_name"]),
        generation=int(payload["generation"]),
        content_sha256=str(payload["content_sha256"]),
        created_at=datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="reviewed release/source snapshot JSON")
    parser.add_argument("--actor", required=True, help="authenticated runtime principal")
    parser.add_argument("--image-digest", required=True, help="digest of the running image")
    parser.add_argument("--expected-image-digest", required=True, help="digest approved for this inventory")
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument("--additional-retained-generations", type=int, default=0)
    parser.add_argument("--now", default=None, help="UTC timestamp; defaults to current time")
    args = parser.parse_args(argv)
    if args.image_digest != args.expected_image_digest:
        parser.error("running image digest does not match the approved image digest")
    if not args.actor.strip():
        parser.error("authenticated actor is required")
    try:
        payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
        sources = tuple(_source(item) for item in payload["sources"])
        references = tuple(
            ReleaseSourceReference(
                release_id=str(item["release_id"]),
                corpus_version_id=str(item["corpus_version_id"]),
                source=_source(item["source"]),
                role=str(item["role"]),
                validated=bool(item.get("validated", True)),
            )
            for item in payload["references"]
        )
        result = inventory(
            sources,
            ReferenceSnapshot.from_references(references),
            now=datetime.fromisoformat(args.now.replace("Z", "+00:00")) if args.now else datetime.now(timezone.utc),
            policy=RetentionPolicy(args.max_age_days, args.additional_retained_generations),
            explicit_generations=(tuple(item) for item in payload.get("explicit_generations", ())),
            rebuildable=(tuple(item) for item in payload.get("rebuildable", ())),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, SourceRetentionError) as exc:
        print(f"source retention inventory rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "actor": args.actor,
        "image_digest": args.image_digest,
        "dry_run": result.dry_run,
        "reference_fingerprint": result.reference_fingerprint,
        "protected": [source.key for source in result.protected],
        "eligible": [source.key for source in result.eligible],
        "refused": [source.key for source in result.refused],
    }, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
