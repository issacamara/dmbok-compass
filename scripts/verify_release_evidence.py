#!/usr/bin/env python3
"""Validate the final release evidence bundle before promotion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.release import ReleaseEvidenceError, validate_durable_promotion, validate_release_evidence


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main(path: Path, *, manifest: Path | None = None, decision: Path | None = None, delivery: Path | None = None) -> int:
    try:
        evidence = _load(path)
        if any(value is not None for value in (manifest, decision, delivery)) and not all(
            value is not None for value in (manifest, decision, delivery)
        ):
            raise ReleaseEvidenceError("durable manifest, decision, and delivery files are required together")
        if manifest is not None and decision is not None and delivery is not None:
            result = validate_durable_promotion(
                evidence,
                manifest=_load(manifest),
                decision_record=_load(decision),
                delivery_manifest=_load(delivery),
            )
        else:
            result = validate_release_evidence(evidence)
    except (OSError, ValueError, json.JSONDecodeError, ReleaseEvidenceError) as exc:
        print(f"release evidence rejected: {exc}", file=sys.stderr)
        return 1
    state = "promotable" if result.promotable else "not promotable"
    exception_text = f"; approved exceptions: {', '.join(result.exceptions)}" if result.exceptions else ""
    blocker_text = f"; blockers: {', '.join(result.blockers)}" if result.blockers else ""
    print(f"{result.release_id}: {state}{exception_text}{blocker_text}")
    return 0 if result.promotable else 1


if __name__ == "__main__":
    if len(sys.argv) not in {2, 8}:
        raise SystemExit(
            "usage: verify_release_evidence.py EVIDENCE.json "
            "[--manifest MANIFEST.json --decision DECISION.json --delivery DELIVERY.json]"
        )
    if len(sys.argv) == 2:
        raise SystemExit(main(Path(sys.argv[1])))
    if sys.argv[2] != "--manifest" or sys.argv[4] != "--decision" or sys.argv[6] != "--delivery":
        raise SystemExit("invalid durable promotion arguments")
    raise SystemExit(main(Path(sys.argv[1]), manifest=Path(sys.argv[3]), decision=Path(sys.argv[5]), delivery=Path(sys.argv[7])))
