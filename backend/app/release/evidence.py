"""Validation for the immutable evidence collected before release promotion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any


GATE_REQUIREMENTS: dict[str, float] = {
    "retrieval_success": 90.0,
    "grounded_claims": 95.0,
    "citation_correctness": 95.0,
    "answer_quality": 85.0,
    "refusal_correctness": 95.0,
    "response_time": 95.0,
}
REQUIRED_EVIDENCE_SECTIONS = (
    "artifact",
    "evaluation",
    "gates",
    "provider",
    "privacy",
    "accessibility",
    "recovery",
    "performance",
    "cost",
    "rollback",
    "decisions",
)
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ReleaseEvidenceError(ValueError):
    """Raised when release evidence is incomplete, inconsistent, or unsafe."""


@dataclass(frozen=True)
class ReleaseEvidenceResult:
    """The review outcome for a release evidence document."""

    release_id: str
    promotable: bool
    exceptions: tuple[str, ...]


def validate_release_evidence(document: dict[str, Any]) -> ReleaseEvidenceResult:
    """Validate a release evidence document and return its promotion outcome.

    Every required section must contain a passing evidence reference. A failed
    or missing section is only acceptable when its exception is explicitly
    approved by the release owner and includes a rationale and timestamp.
    """

    release_id = _required_string(document, "release_id")
    _required_string(document, "schema_version")
    _validate_artifact(document.get("artifact"))
    _validate_evaluation(document.get("evaluation"))

    sections = document.get("evidence")
    if not isinstance(sections, dict):
        raise ReleaseEvidenceError("evidence must be an object")
    missing = [name for name in REQUIRED_EVIDENCE_SECTIONS if name not in sections]
    if missing:
        raise ReleaseEvidenceError(f"missing evidence sections: {', '.join(missing)}")

    exceptions: list[str] = []
    for name in REQUIRED_EVIDENCE_SECTIONS:
        exceptions.extend(_validate_evidence_section(name, sections[name]))

    decision = document.get("release_decision")
    if not isinstance(decision, dict):
        raise ReleaseEvidenceError("release_decision must be an object")
    if decision.get("release_id") != release_id:
        raise ReleaseEvidenceError("release_decision.release_id must match release_id")
    if decision.get("decision") not in {"approved", "rejected", "pending"}:
        raise ReleaseEvidenceError("release_decision.decision is invalid")
    if decision.get("decision") == "approved" and not decision.get("approver"):
        raise ReleaseEvidenceError("an approved release requires an approver")

    return ReleaseEvidenceResult(
        release_id=release_id,
        promotable=decision.get("decision") == "approved",
        exceptions=tuple(exceptions),
    )


def _validate_artifact(value: Any) -> None:
    if not isinstance(value, dict):
        raise ReleaseEvidenceError("artifact must be an object")
    for key in ("commit", "api_image", "api_digest", "frontend_digest", "preview_revision"):
        _required_string(value, key)
    if "@sha256:" not in value["api_image"] or not value["api_image"].endswith(value["api_digest"]):
        raise ReleaseEvidenceError("artifact.api_image must reference the recorded immutable digest")
    if not SHA256.fullmatch(value["api_digest"]):
        raise ReleaseEvidenceError("artifact.api_digest must be a sha256 digest")
    if not HEX_DIGEST.fullmatch(value["frontend_digest"]):
        raise ReleaseEvidenceError("artifact.frontend_digest must be a SHA-256 hex digest")


def _validate_evaluation(value: Any) -> None:
    if not isinstance(value, dict):
        raise ReleaseEvidenceError("evaluation must be an object")
    for key in ("run_id", "dataset_version_id", "corpus_version_id", "configuration_version_id", "model_version_id"):
        _required_string(value, key)
    if value.get("status") != "completed":
        raise ReleaseEvidenceError("evaluation.status must be completed")
    gates = value.get("gates")
    if not isinstance(gates, list):
        raise ReleaseEvidenceError("evaluation.gates must be an array")
    by_name = {gate.get("metric_name"): gate for gate in gates if isinstance(gate, dict)}
    if set(by_name) != set(GATE_REQUIREMENTS):
        raise ReleaseEvidenceError("evaluation.gates must contain exactly the six release metrics")
    for name, threshold in GATE_REQUIREMENTS.items():
        gate = by_name[name]
        if gate.get("threshold") != threshold or gate.get("passed") is not True:
            raise ReleaseEvidenceError(f"evaluation gate {name!r} is not passing at its required threshold")


def _validate_evidence_section(name: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        raise ReleaseEvidenceError(f"evidence.{name} must be an object")
    status = value.get("status")
    references = value.get("references")
    if not isinstance(references, list) or not references or not all(isinstance(item, str) and item for item in references):
        raise ReleaseEvidenceError(f"evidence.{name}.references must contain at least one identifier")
    if status == "pass":
        return []
    if status != "exception":
        raise ReleaseEvidenceError(f"evidence.{name} must pass or have an approved exception")
    exception = value.get("exception")
    if not isinstance(exception, dict) or exception.get("approved") is not True:
        raise ReleaseEvidenceError(f"evidence.{name} exception must be explicitly approved")
    for key in ("approver", "rationale", "approved_at"):
        _required_string(exception, key)
    try:
        datetime.fromisoformat(exception["approved_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseEvidenceError(f"evidence.{name}.exception.approved_at must be ISO-8601") from exc
    return [name]


def _required_string(value: Any, key: str) -> str:
    if not isinstance(value, dict) or not isinstance(value.get(key), str) or not value[key].strip():
        raise ReleaseEvidenceError(f"{key} must be a non-empty string")
    return value[key]
