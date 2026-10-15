#!/usr/bin/env python3
"""Preflight an immutable, coupled delivery rollback.

This program intentionally produces a plan rather than changing traffic.  The
protected delivery workflow is responsible for applying a reviewed plan after
the production environment's approval gate has run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.verify_delivery_manifest import validate_manifest


class RollbackPreflightError(ValueError):
    """Raised when a release cannot be restored as one compatible binding."""


def read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RollbackPreflightError(f"{path} is not a manifest object")
    try:
        validate_manifest(payload)
    except ValueError as error:
        raise RollbackPreflightError(str(error)) from error
    return payload


def release_revision(manifest: dict[str, object]) -> str:
    revision = manifest.get("release_revision") or manifest.get("preview_revision")
    if not isinstance(revision, str) or not revision:
        raise RollbackPreflightError("a rollback manifest requires a tested release revision")
    return revision


def binding(manifest: dict[str, object]) -> dict[str, str]:
    value = manifest["serving_binding"]
    assert isinstance(value, dict)  # validated by validate_manifest
    return {key: str(value[key]) for key in value}


def build_plan(
    candidate: dict[str, object],
    previous: dict[str, object],
    *,
    protected_gate_approved: bool,
    owner_approved: bool,
) -> dict[str, object]:
    """Return the exact prior and forward bindings after safety preflight."""
    if not protected_gate_approved:
        raise RollbackPreflightError("the protected delivery gate has not approved rollback")
    if not owner_approved:
        raise RollbackPreflightError("owner approval is required before rollback")

    return {
        "rollback": {
            "revision": release_revision(previous),
            "api_image": previous["api_image"],
            "admin_image": previous.get("admin_image", previous["api_image"]),
            "serving_binding": binding(previous),
        },
        "forward": {
            "revision": release_revision(candidate),
            "api_image": candidate["api_image"],
            "admin_image": candidate.get("admin_image", candidate["api_image"]),
            "serving_binding": binding(candidate),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protected-gate-approved", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = build_plan(
            read_manifest(args.candidate),
            read_manifest(args.previous),
            protected_gate_approved=args.protected_gate_approved,
            owner_approved=args.owner_approved,
        )
    except (OSError, json.JSONDecodeError, RollbackPreflightError) as error:
        print(f"rollback preflight failed: {error}", file=sys.stderr)
        return 2

    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
