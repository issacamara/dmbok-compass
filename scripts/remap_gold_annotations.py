"""Propose human-reviewed gold-annotation mappings for one candidate corpus.

The input JSON contains ``binding``, ``dataset``, and ``candidate_chunks``.
Each candidate chunk is a DocumentChunk object with an optional
``section_origin`` (``explicit`` or ``carried``).  The script writes no
Firestore data and never produces an approved annotation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from app.contracts import DocumentChunk, EvaluationDataset
from app.evaluation.dataset import CandidateBindingError, CandidateChunk, CandidateGoldBinding, remap_gold_annotations


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="candidate remap request JSON")
    parser.add_argument("--output", required=True, type=Path, help="candidate review JSON to write")
    return parser.parse_args(argv)


def build_review(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise CandidateBindingError("remap input must be an object")
    binding = CandidateGoldBinding(**_required_object(payload, "binding"))
    dataset = EvaluationDataset.model_validate(_required_object(payload, "dataset"))
    raw_chunks = payload.get("candidate_chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise CandidateBindingError("candidate_chunks must be a non-empty array")
    chunks = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            raise CandidateBindingError("candidate chunks must be objects")
        chunk_payload = {key: value for key, value in raw_chunk.items() if key != "section_origin"}
        chunks.append(
            CandidateChunk(
                chunk=DocumentChunk.model_validate(chunk_payload),
                section_origin=raw_chunk.get("section_origin", "explicit"),
            )
        )
    return remap_gold_annotations(dataset, chunks, binding).model_dump()


def _required_object(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise CandidateBindingError(f"{name} must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        review = build_review(payload)
        args.output.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (CandidateBindingError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"gold annotation remap rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
