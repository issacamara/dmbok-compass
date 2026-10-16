#!/usr/bin/env python3
"""Compare a reviewed LangChain candidate with authored legacy evidence.

The input and output contain identifiers, digests, metric booleans, and
adjudication metadata only.  They must never contain questions, answers,
passages, prompts, or traces from production interactions.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


THRESHOLDS: dict[str, float] = {
    "retrieval_success": 90.0,
    "grounded_claims": 95.0,
    "citation_correctness": 95.0,
    "answer_quality": 85.0,
    "refusal_correctness": 95.0,
}
REQUIRED_CATEGORIES = {"definitions", "explanations", "comparisons", "study", "scenarios"}
REQUIRED_OUTCOMES = {"answerable", "partial", "unanswerable"}
REQUIRED_BINDINGS = (
    "run_id",
    "dataset_revision",
    "corpus_version",
    "corpus_schema_fingerprint",
    "scorer_version",
    "provider",
    "routes",
    "models",
    "dependency_lock_sha256",
    "wheel_sha256",
    "image_digest",
)
METRIC_FIELDS = {
    "retrieval_success": "retrieval_success",
    "grounded_claims": "grounded",
    "citation_correctness": "citation_correct",
    "answer_quality": "answer_quality",
    "refusal_correctness": "refusal_correct",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ParityError(ValueError):
    """Raised when parity evidence is incomplete or unsafe to report."""


def build_report(document: dict[str, Any]) -> dict[str, Any]:
    """Validate a parity input and return a deterministic content-free report."""

    candidate = _object(document.get("candidate"), "candidate")
    legacy = _object(document.get("legacy"), "legacy")
    _validate_binding(candidate)
    if candidate.get("eligibility") != "release_evidence":
        raise ParityError("candidate eligibility must be release_evidence")

    candidate_items = _items(candidate, "candidate.items")
    legacy_items = _items(legacy, "legacy.items")
    if not 30 <= len(candidate_items) <= 50:
        raise ParityError("candidate must contain the approved 30–50 item subset")
    if set(candidate_items) != set(legacy_items):
        raise ParityError("candidate and legacy fixtures must contain the same item IDs")
    _validate_coverage(candidate_items.values())

    adjudications = _adjudications(document.get("adjudications", []))
    changes: list[dict[str, Any]] = []
    for item_id, item in candidate_items.items():
        previous = legacy_items[item_id]
        changed = {
            field: item.get(field) != previous.get(field)
            for field in ("answer_digest", "support_digest", "provenance")
        }
        if any(changed.values()):
            adjudication = adjudications.get(item_id)
            if adjudication is None:
                raise ParityError(f"changed support/output for {item_id} lacks evaluator adjudication")
            changes.append({"item_id": item_id, "changes": changed, "adjudication": adjudication})

    metrics = [_metric(name, candidate_items.values()) for name in THRESHOLDS]
    passing = all(metric["passed"] for metric in metrics)
    return {
        "schema_version": "langchain-quality-parity-v1",
        "report_eligibility": "release_evidence",
        "promotable": passing,
        "binding": {key: candidate[key] for key in REQUIRED_BINDINGS},
        "legacy_fixture_revision": _required_string(legacy, "fixture_revision"),
        "rubric_version": _required_string(legacy, "rubric_version"),
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "parity": {
            "candidate_item_count": len(candidate_items),
            "adjudicated_change_count": len(changes),
            "changes": changes,
        },
    }


def _metric(name: str, items: Any) -> dict[str, Any]:
    applicable = [item for item in items if name in item["applicable_metrics"]]
    field = METRIC_FIELDS[name]
    numerator = sum(item[field] is True for item in applicable)
    denominator = len(applicable)
    percentage = round(100 * numerator / denominator, 2) if denominator else 0.0
    return {
        "metric_name": name,
        "numerator": numerator,
        "denominator": denominator,
        "percentage": percentage,
        "threshold": THRESHOLDS[name],
        "passed": denominator > 0 and percentage >= THRESHOLDS[name],
    }


def _validate_binding(candidate: dict[str, Any]) -> None:
    for key in REQUIRED_BINDINGS:
        if key not in candidate:
            raise ParityError(f"candidate binding is missing {key}")
        if key not in {"routes", "models"} and not isinstance(candidate[key], str):
            raise ParityError(f"candidate binding {key} must be a string")
        if key in {"routes", "models"} and (
            not isinstance(candidate[key], list) or not candidate[key]
        ):
            raise ParityError(f"candidate binding {key} must be a non-empty list")
    for key in ("dependency_lock_sha256", "wheel_sha256"):
        if not HEX64.fullmatch(candidate[key]):
            raise ParityError(f"candidate binding {key} must be a lowercase SHA-256 digest")
    if not candidate["image_digest"].startswith("sha256:") or not HEX64.fullmatch(candidate["image_digest"][7:]):
        raise ParityError("candidate binding image_digest must be sha256:<digest>")


def _items(value: dict[str, Any], path: str) -> dict[str, dict[str, Any]]:
    raw = value.get("items")
    if not isinstance(raw, list) or not raw:
        raise ParityError(f"{path} must be a non-empty array")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("item_id"), str) or item["item_id"] in result:
            raise ParityError(f"{path} contains duplicate or invalid item IDs")
        _validate_item(item, path)
        result[item["item_id"]] = item
    return result


def _validate_coverage(items: Any) -> None:
    categories = {item.get("category") for item in items}
    outcomes = {item.get("evidence_outcome") for item in items}
    if not REQUIRED_CATEGORIES <= categories:
        raise ParityError("candidate subset must cover all five question categories")
    if not REQUIRED_OUTCOMES <= outcomes:
        raise ParityError("candidate subset must cover answerable, partial, and unanswerable outcomes")
    for item in items:
        if not isinstance(item.get("applicable_metrics"), list):
            raise ParityError(f"item {item.get('item_id')} is missing metric applicability")
        if not set(item["applicable_metrics"]) <= set(THRESHOLDS):
            raise ParityError(f"item {item.get('item_id')} has an unknown applicable metric")
        for field in METRIC_FIELDS.values():
            if not isinstance(item.get(field), bool):
                raise ParityError(f"item {item.get('item_id')} is missing boolean {field}")
        if item["evidence_outcome"] == "unanswerable" and item["generation_attempted"]:
            raise ParityError(f"item {item['item_id']} must refuse before generation")


def _validate_item(item: dict[str, Any], path: str) -> None:
    for key in ("category", "evidence_outcome", "answer_digest", "support_digest"):
        if not isinstance(item.get(key), str):
            raise ParityError(f"{path} item {item.get('item_id')} is missing {key}")
    if not isinstance(item.get("provenance"), dict):
        raise ParityError(f"{path} item {item.get('item_id')} is missing provenance")
    provenance = item["provenance"]
    if not isinstance(provenance.get("page"), int) or provenance["page"] < 1:
        raise ParityError(f"{path} item {item.get('item_id')} has invalid one-based page provenance")
    if not isinstance(provenance.get("section"), str) or not provenance["section"].strip():
        raise ParityError(f"{path} item {item.get('item_id')} has invalid section provenance")
    if provenance.get("chunk_size") != 700 or provenance.get("chunk_overlap") != 70:
        raise ParityError(f"{path} item {item.get('item_id')} must bind 700/70 word units")
    if not isinstance(item.get("generation_attempted"), bool):
        raise ParityError(f"{path} item {item.get('item_id')} is missing generation_attempted")


def _adjudications(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ParityError("adjudications must be an array")
    result: dict[str, dict[str, Any]] = {}
    for record in value:
        if not isinstance(record, dict) or not isinstance(record.get("item_id"), str):
            raise ParityError("each adjudication must identify an item")
        if record["item_id"] in result or not record.get("evaluator") or not record.get("rationale") or not record.get("reviewed_at"):
            raise ParityError("each adjudication requires a unique evaluator, rationale, and reviewed_at")
        result[record["item_id"]] = {
            "evaluator": record["evaluator"],
            "rationale": record["rationale"],
            "reviewed_at": record["reviewed_at"],
        }
    return result


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParityError(f"{path} must be an object")
    return value


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ParityError(f"{key} must be a non-empty string")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        report = build_report(document)
    except (OSError, json.JSONDecodeError, ParityError) as exc:
        parser.error(str(exc))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
