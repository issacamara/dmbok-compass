#!/usr/bin/env python3
"""Create a review-only structure-map proposal from curated heading evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.corpus.structure_map import HeadingAnchor, SourceIdentity, propose_structure_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="approved immutable PDF download")
    parser.add_argument("headings", type=Path, help="JSON with page_count, toc and printed_headings")
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--source-sha256", required=True, help="approved immutable object digest")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != args.source_sha256:
        raise ValueError("downloaded source digest does not match the approved source digest")
    evidence = json.loads(args.headings.read_text(encoding="utf-8"))
    proposal = propose_structure_map(
        SourceIdentity(args.source_uri, args.generation, args.source_sha256),
        page_count=evidence["page_count"],
        toc=_anchors(evidence.get("toc", [])),
        printed_headings=_anchors(evidence.get("printed_headings", [])),
    )
    args.output.write_text(proposal.to_json(), encoding="utf-8")
    return 0


def _anchors(values: list[dict[str, object]]) -> list[HeadingAnchor]:
    return [HeadingAnchor(int(value["page"]), tuple(value["section_path"]), str(value["heading"])) for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
