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
    blockers: tuple[str, ...] = ()


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
    blockers: list[str] = []
    for name in REQUIRED_EVIDENCE_SECTIONS:
        section_exceptions, section_blockers = _validate_evidence_section(name, sections[name])
        exceptions.extend(section_exceptions)
        blockers.extend(section_blockers)

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
        promotable=decision.get("decision") == "approved" and not blockers,
        exceptions=tuple(exceptions),
        blockers=tuple(blockers),
    )


def validate_durable_promotion(
    document: dict[str, Any],
    *,
    manifest: dict[str, Any],
    decision_record: dict[str, Any],
    delivery_manifest: dict[str, Any],
) -> ReleaseEvidenceResult:
    """Validate evidence against the durable approval and delivery identity.

    The evidence document is an inspection artifact. Promotion additionally
    requires the immutable manifest and decision retrieved from the release
    registry, plus the exact delivery manifest produced by the same commit.
    Keeping this join explicit prevents a release ID or a successful quality
    report from being used as an authorization by itself.
    """

    result = validate_release_evidence(document)
    if not result.promotable:
        raise ReleaseEvidenceError("release evidence is not promotable")

    decision = decision_record.get("decision")
    if not isinstance(decision, dict):
        raise ReleaseEvidenceError("durable release decision is missing")
    if decision.get("decision") != "approved":
        raise ReleaseEvidenceError("durable release decision is not approved")
    if not decision_record.get("actor_user_id") or not decision_record.get("decided_at"):
        raise ReleaseEvidenceError("durable release decision is missing actor or timestamp")

    release_id = document["release_id"]
    if decision.get("release_id") != release_id or manifest.get("release_id") != release_id:
        raise ReleaseEvidenceError("durable release identity does not match the evidence")
    evidence_keys = {
        "evaluation_run_id": "run_id",
        "dataset_version_id": "dataset_version_id",
        "corpus_version_id": "corpus_version_id",
        "configuration_version_id": "configuration_version_id",
        "model_version_id": "model_version_id",
    }
    for key, evidence_key in evidence_keys.items():
        if decision.get(key) != document["evaluation"].get(evidence_key) or manifest.get(key) != decision.get(key):
            raise ReleaseEvidenceError(f"durable release {key} does not match the evidence")

    if manifest.get("gate_report_ids") != decision.get("gate_report_ids"):
        raise ReleaseEvidenceError("durable release gate evidence does not match the decision")
    snapshot = document["evaluation"].get("candidate_snapshot")
    if not isinstance(snapshot, dict):
        raise ReleaseEvidenceError("candidate snapshot is missing")
    if manifest.get("scorer_version_id") != snapshot.get("scorer_version"):
        raise ReleaseEvidenceError("durable scorer identity does not match the candidate")
    provider_version = manifest.get("provider_version_id")
    if not isinstance(provider_version, str) or not provider_version.startswith(f"{snapshot.get('answer_provider')}-"):
        raise ReleaseEvidenceError("durable provider identity does not match the candidate")
    if manifest.get("model_version_id") != snapshot.get("primary_model"):
        raise ReleaseEvidenceError("durable model identity does not match the candidate")
    candidate_binding = document["evaluation"].get("candidate_gold_binding")
    if manifest.get("candidate_digest") and manifest["candidate_digest"] != candidate_binding.get("candidate_snapshot_sha256"):
        raise ReleaseEvidenceError("durable candidate digest does not match the evidence")
    artifact = document["artifact"]
    if delivery_manifest.get("commit") != artifact.get("commit"):
        raise ReleaseEvidenceError("delivery commit does not match the evidence artifact")
    if delivery_manifest.get("api_digest") != artifact.get("api_digest"):
        raise ReleaseEvidenceError("delivery API digest does not match the evidence artifact")
    if delivery_manifest.get("frontend_digest") != artifact.get("frontend_digest"):
        raise ReleaseEvidenceError("delivery frontend digest does not match the evidence artifact")
    binding = delivery_manifest.get("serving_binding")
    if not isinstance(binding, dict) or binding.get("corpus_version_id") != document["evaluation"].get("corpus_version_id"):
        raise ReleaseEvidenceError("delivery corpus binding does not match the evidence")
    configuration_digest = binding.get("configuration_digest")
    if configuration_digest is not None and configuration_digest != snapshot.get("configuration_fingerprint"):
        raise ReleaseEvidenceError("delivery configuration identity does not match the candidate")
    return result


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
    snapshot = value.get("candidate_snapshot")
    _validate_candidate_snapshot(snapshot)
    binding = value.get("candidate_gold_binding")
    if not isinstance(binding, dict):
        raise ReleaseEvidenceError("evaluation.candidate_gold_binding must be an object")
    for key in ("corpus_version_id", "candidate_snapshot_sha256", "source_provenance_sha256"):
        _required_string(binding, key)
    if not HEX_DIGEST.fullmatch(binding["candidate_snapshot_sha256"]):
        raise ReleaseEvidenceError("candidate gold binding snapshot digest must be SHA-256")
    if not HEX_DIGEST.fullmatch(binding["source_provenance_sha256"]):
        raise ReleaseEvidenceError("candidate gold binding provenance digest must be SHA-256")
    if binding["corpus_version_id"] != value["corpus_version_id"]:
        raise ReleaseEvidenceError("candidate gold binding corpus must match the evaluation")


def _validate_candidate_snapshot(value: Any) -> None:
    if not isinstance(value, dict):
        raise ReleaseEvidenceError("evaluation.candidate_snapshot must be an object")
    for key in (
        "project", "location", "answer_provider", "primary_model", "prompt_version",
        "retrieval_policy_version", "evidence_policy_version", "scorer_version",
        "dependency_lock_sha256", "integration_artifact", "runtime_version",
    ):
        _required_string(value, key)
    if value["answer_provider"] not in {"openrouter", "vertex"}:
        raise ReleaseEvidenceError("candidate snapshot provider is invalid")
    if not HEX_DIGEST.fullmatch(value["dependency_lock_sha256"]):
        raise ReleaseEvidenceError("candidate snapshot lock digest must be SHA-256")
    binding = value.get("corpus_binding")
    if not isinstance(binding, dict):
        raise ReleaseEvidenceError("candidate snapshot corpus binding must be an object")
    for key in (
        "corpus_version_id", "vector_collection", "vector_schema_version", "embedding_model",
        "ingestion_fingerprint", "configuration_fingerprint",
    ):
        _required_string(binding, key)
    for key in ("ingestion_fingerprint", "configuration_fingerprint"):
        if not HEX_DIGEST.fullmatch(binding[key]):
            raise ReleaseEvidenceError(f"candidate snapshot {key} must be SHA-256")


def _validate_evidence_section(name: str, value: Any) -> tuple[list[str], list[str]]:
    if not isinstance(value, dict):
        raise ReleaseEvidenceError(f"evidence.{name} must be an object")
    status = value.get("status")
    references = value.get("references")
    if not isinstance(references, list) or not references or not all(isinstance(item, str) and item for item in references):
        raise ReleaseEvidenceError(f"evidence.{name}.references must contain at least one identifier")
    if status == "pass":
        return [], []
    if status == "blocked":
        blocker = value.get("blocker")
        if not isinstance(blocker, dict):
            raise ReleaseEvidenceError(f"evidence.{name} blocked status requires a blocker")
        for key in ("owner", "rationale", "recorded_at"):
            _required_string(blocker, key)
        try:
            datetime.fromisoformat(blocker["recorded_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReleaseEvidenceError(f"evidence.{name}.blocker.recorded_at must be ISO-8601") from exc
        return [], [name]
    if status != "exception":
        raise ReleaseEvidenceError(f"evidence.{name} must pass, be blocked, or have an approved exception")
    exception = value.get("exception")
    if not isinstance(exception, dict) or exception.get("approved") is not True:
        raise ReleaseEvidenceError(f"evidence.{name} exception must be explicitly approved")
    for key in ("approver", "rationale", "approved_at"):
        _required_string(exception, key)
    try:
        datetime.fromisoformat(exception["approved_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReleaseEvidenceError(f"evidence.{name}.exception.approved_at must be ISO-8601") from exc
    return [name], []


def _required_string(value: Any, key: str) -> str:
    if not isinstance(value, dict) or not isinstance(value.get(key), str) or not value[key].strip():
        raise ReleaseEvidenceError(f"{key} must be a non-empty string")
    return value[key]
