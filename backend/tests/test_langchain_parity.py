from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from scripts.run_langchain_parity import ParityError, build_report


def _item(item_id: str, category: str, outcome: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "category": category,
        "evidence_outcome": outcome,
        "applicable_metrics": [
            "retrieval_success", "grounded_claims", "citation_correctness", "answer_quality"
        ] if outcome != "unanswerable" else ["refusal_correctness"],
        "retrieval_success": outcome != "unanswerable",
        "grounded": outcome != "unanswerable",
        "citation_correct": outcome != "unanswerable",
        "answer_quality": outcome != "unanswerable",
        "refusal_correct": outcome == "unanswerable",
        "answer_digest": f"answer-{item_id}",
        "support_digest": f"support-{item_id}",
        "provenance": {"page": 1, "section": "Synthetic", "chunk_size": 700, "chunk_overlap": 70},
        "generation_attempted": outcome == "answerable",
    }


def _document() -> dict[str, object]:
    categories = ["definitions", "explanations", "comparisons", "study", "scenarios"]
    items = [_item(f"item-{i}", categories[i % 5], ("unanswerable" if i == 2 else "partial" if i == 1 else "answerable")) for i in range(30)]
    candidate = {
        "run_id": "eval-candidate-1", "eligibility": "release_evidence",
        "dataset_revision": "gold-2026-10-16", "corpus_version": "corpus-v2",
        "corpus_schema_fingerprint": "schema-v2", "scorer_version": "evaluation-gates-v1",
        "provider": "openrouter", "routes": ["primary", "fallback"],
        "models": ["primary-v1", "fallback-v1"], "dependency_lock_sha256": "a" * 64,
        "wheel_sha256": "b" * 64, "image_digest": "sha256:" + "c" * 64, "items": items,
    }
    legacy = {"fixture_revision": "legacy-2026-10-16", "rubric_version": "rubric-v1", "items": deepcopy(items)}
    return {"candidate": candidate, "legacy": legacy, "adjudications": []}


def test_report_has_all_five_thresholds_and_applicable_denominators() -> None:
    report = build_report(_document())
    metrics = {metric["metric_name"]: metric for metric in report["metrics"]}

    assert set(metrics) == {"retrieval_success", "grounded_claims", "citation_correctness", "answer_quality", "refusal_correctness"}
    assert metrics["retrieval_success"]["denominator"] == 29
    assert metrics["refusal_correctness"]["denominator"] == 1
    assert metrics["retrieval_success"]["threshold"] == 90.0


def test_changed_support_requires_adjudication_without_byte_equality() -> None:
    document = _document()
    document["candidate"]["items"][0]["support_digest"] = "changed-support"

    with pytest.raises(ParityError, match="lacks evaluator adjudication"):
        build_report(document)

    document["adjudications"] = [{
        "item_id": "item-0", "evaluator": "reviewer-1",
        "rationale": "Support changed with equivalent page provenance.",
        "reviewed_at": "2026-10-16T12:00:00Z",
    }]
    report = build_report(document)
    assert report["parity"]["adjudicated_change_count"] == 1


def test_incomplete_binding_and_coverage_cannot_be_reported_as_release_evidence() -> None:
    document = _document()
    del document["candidate"]["image_digest"]
    with pytest.raises(ParityError, match="image_digest"):
        build_report(document)

    document = _document()
    document["candidate"]["items"] = document["candidate"]["items"][:29]
    with pytest.raises(ParityError, match="30–50"):
        build_report(document)
