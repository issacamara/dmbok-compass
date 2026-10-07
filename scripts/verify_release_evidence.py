#!/usr/bin/env python3
"""Validate the final release evidence bundle before promotion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.release import ReleaseEvidenceError, validate_release_evidence


def main(path: Path) -> int:
    try:
        result = validate_release_evidence(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ReleaseEvidenceError) as exc:
        print(f"release evidence rejected: {exc}", file=sys.stderr)
        return 1
    state = "promotable" if result.promotable else "not promotable"
    exception_text = f"; approved exceptions: {', '.join(result.exceptions)}" if result.exceptions else ""
    print(f"{result.release_id}: {state}{exception_text}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_release_evidence.py EVIDENCE.json")
    raise SystemExit(main(Path(sys.argv[1])))
