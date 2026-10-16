#!/usr/bin/env python3
"""Validate content-free evidence from an isolated LangChain recovery drill.

The program deliberately has no Google Cloud client calls.  An operator first
restores into an isolated target and records the resulting metadata, then this
program proves that the record is complete, candidate-specific, and safe to
attach to release evidence.  It never restores or mutates a database.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.contracts import (
    CorpusServingBinding,
    CorpusVersion,
    EvaluationCandidateSnapshot,
    EvaluationRun,
    ReleaseDecision,
)
from app.corpus.manifest import create_source_manifest


MAX_RECOVERY_SECONDS = 24 * 60 * 60
MAX_RETAINED_COPIES = 30
REQUIRED_RECORD_COUNTS = frozenset({
    "users",
    "configuration",
    "quota",
    "aggregate_metrics",
    "datasets",
    "annotations",
    "runs",
    "release_decisions",
})
REQUIRED_SCANNED_COLLECTIONS = frozenset({
    "users",
    "configuration",
    "quota",
    "aggregate_metrics",
    "evaluation_generations",
    "evaluation_runs",
    "release_decisions",
})
SENSITIVE_KEY_PARTS = ("secret", "credential", "token", "password", "api_key")


class RecoveryEvidenceError(ValueError):
    """Raised when an isolated recovery drill lacks release-grade evidence."""


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryEvidenceError(f"{field} must be an object")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryEvidenceError(f"{field} must be a non-empty string")
    return value


def _utc(value: object, field: str) -> datetime:
    text = _string(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecoveryEvidenceError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise RecoveryEvidenceError(f"{field} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _date(value: object, field: str) -> date:
    try:
        return date.fromisoformat(_string(value, field))
    except ValueError as exc:
        raise RecoveryEvidenceError(f"{field} must be an ISO-8601 date") from exc


def _model(model: type[Any], value: object, field: str) -> Any:
    try:
        return model.model_validate(value)
    except Exception as exc:  # Pydantic's validation error is an implementation detail here.
        raise RecoveryEvidenceError(f"{field} is invalid: {exc}") from exc


def _reject_sensitive_keys(value: object, path: str = "evidence") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RecoveryEvidenceError(f"{path} contains a non-string field name")
            if any(part in key.lower() for part in SENSITIVE_KEY_PARTS):
                raise RecoveryEvidenceError(f"{path}.{key} must not contain secret material")
            _reject_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, f"{path}[{index}]")


def _validate_backup(backup: Mapping[str, Any], completed_at: datetime) -> int:
    retained = backup.get("retained_copy_count")
    if not isinstance(retained, int) or isinstance(retained, bool) or retained < 1:
        raise RecoveryEvidenceError("backup.retained_copy_count must be a positive integer")
    if retained > MAX_RETAINED_COPIES:
        raise RecoveryEvidenceError("backup retains more than 30 copies")

    dates = backup.get("daily_execution_dates")
    if not isinstance(dates, list) or len(dates) < 2:
        raise RecoveryEvidenceError("backup.daily_execution_dates must prove at least two daily executions")
    parsed_dates = sorted(_date(value, "backup.daily_execution_dates") for value in dates)
    if len(set(parsed_dates)) != len(parsed_dates):
        raise RecoveryEvidenceError("backup.daily_execution_dates must not contain duplicates")
    if any(later - earlier != timedelta(days=1) for earlier, later in zip(parsed_dates, parsed_dates[1:])):
        raise RecoveryEvidenceError("backup.daily_execution_dates must be consecutive daily executions")
    if completed_at.date() - parsed_dates[-1] > timedelta(days=1):
        raise RecoveryEvidenceError("the latest daily backup execution is stale")
    return retained


def _validate_target(drill: Mapping[str, Any]) -> Mapping[str, str]:
    target = _mapping(drill.get("target"), "drill.target")
    source = _mapping(drill.get("source"), "drill.source")
    target_project = _string(target.get("project"), "drill.target.project")
    target_database = _string(target.get("database"), "drill.target.database")
    source_project = _string(source.get("project"), "drill.source.project")
    source_database = _string(source.get("database"), "drill.source.database")
    if target.get("isolated") is not True or target.get("is_production") is not False:
        raise RecoveryEvidenceError("drill.target must explicitly be isolated and non-production")
    if (target_project, target_database) == (source_project, source_database):
        raise RecoveryEvidenceError("an isolated recovery target must differ from the source database")
    return {"project": target_project, "database": target_database}


def _validate_restored_state(
    restored: Mapping[str, Any], snapshot: EvaluationCandidateSnapshot
) -> tuple[list[EvaluationRun], list[ReleaseDecision]]:
    counts = _mapping(restored.get("record_counts"), "restored.record_counts")
    if set(counts) != REQUIRED_RECORD_COUNTS:
        raise RecoveryEvidenceError("restored.record_counts must cover every durable recovery collection")
    for name, count in counts.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise RecoveryEvidenceError(f"restored.record_counts.{name} must be a positive integer")

    if restored.get("restart_verified") is not True:
        raise RecoveryEvidenceError("restored.restart_verified must be true")
    if restored.get("import_replacement_verified") is not True:
        raise RecoveryEvidenceError("restored.import_replacement_verified must be true")
    if restored.get("exploratory_semantics_verified") is not True:
        raise RecoveryEvidenceError("restored.exploratory_semantics_verified must be true")
    if restored.get("production_interaction_content_absent") is not True:
        raise RecoveryEvidenceError("restored.production_interaction_content_absent must be true")

    runs_value = restored.get("runs")
    decisions_value = restored.get("release_decisions")
    if not isinstance(runs_value, list) or not isinstance(decisions_value, list):
        raise RecoveryEvidenceError("restored runs and release decisions must be arrays")
    if len(runs_value) != counts["runs"] or len(decisions_value) != counts["release_decisions"]:
        raise RecoveryEvidenceError("restored run and decision counts must match their records")
    runs = [_model(EvaluationRun, value, "restored.runs") for value in runs_value]
    decisions = [_model(ReleaseDecision, value, "restored.release_decisions") for value in decisions_value]
    run_ids = {run.run_id for run in runs}
    if len(run_ids) != len(runs):
        raise RecoveryEvidenceError("restored evaluation run IDs must be unique")

    exploratory_run_id = _string(restored.get("exploratory_run_id"), "restored.exploratory_run_id")
    for run in runs:
        if run.candidate_snapshot != snapshot or run.corpus_version_id != snapshot.corpus_binding.corpus_version_id:
            raise RecoveryEvidenceError("each restored run must retain the candidate snapshot and corpus binding")
    exploratory = next((run for run in runs if run.run_id == exploratory_run_id), None)
    if exploratory is None or exploratory.report_eligibility is None or exploratory.report_eligibility.eligibility != "exploratory":
        raise RecoveryEvidenceError("restored.exploratory_run_id must identify an exploratory run")

    runs_by_id = {run.run_id: run for run in runs}
    for decision in decisions:
        run = runs_by_id.get(decision.evaluation_run_id)
        if run is None:
            raise RecoveryEvidenceError("each restored release decision must refer to a restored run")
        if (
            decision.dataset_version_id != run.dataset_version_id
            or decision.corpus_version_id != run.corpus_version_id
            or decision.configuration_version_id != run.configuration_version_id
            or decision.model_version_id != run.model_version_id
            or decision.scorer_version_id != snapshot.scorer_version
        ):
            raise RecoveryEvidenceError("restored release decision version binding does not match its run")
    return runs, decisions


def _validate_content_scan(scan: Mapping[str, Any]) -> None:
    collections = scan.get("scanned_collections")
    if not isinstance(collections, list) or not REQUIRED_SCANNED_COLLECTIONS <= set(collections):
        raise RecoveryEvidenceError("content scan must cover all durable application collections")
    forbidden = scan.get("production_interaction_records_found")
    if forbidden != 0:
        raise RecoveryEvidenceError("recovery evidence contains persisted production interaction content")


def _validate_rebuild(
    rebuild: Mapping[str, Any], snapshot: EvaluationCandidateSnapshot
) -> CorpusVersion:
    source = _mapping(rebuild.get("source"), "rebuild.source")
    derivation = _mapping(rebuild.get("derivation"), "rebuild.derivation")
    try:
        manifest = create_source_manifest(
            bucket=_string(source.get("bucket"), "rebuild.source.bucket"),
            object_name=_string(source.get("object_name"), "rebuild.source.object_name"),
            generation=source.get("generation"),
            content_sha256=_string(source.get("content_sha256"), "rebuild.source.content_sha256"),
            derivation=derivation,
        )
    except (TypeError, ValueError) as exc:
        raise RecoveryEvidenceError(f"rebuild source manifest is invalid: {exc}") from exc

    recorded_manifest = _mapping(rebuild.get("manifest"), "rebuild.manifest")
    expected_manifest = {
        "corpus_version_id": manifest.corpus_version_id,
        "manifest_sha256": manifest.manifest_sha256,
        "derivation_sha256": manifest.derivation_sha256,
    }
    if dict(recorded_manifest) != expected_manifest:
        raise RecoveryEvidenceError("rebuilt corpus manifest does not match the immutable source and derivation")

    version = _model(CorpusVersion, rebuild.get("corpus_version"), "rebuild.corpus_version")
    if version.status != "active" or version.source_uri != manifest.object_uri:
        raise RecoveryEvidenceError("rebuilt corpus must be active and retain its immutable source URI")
    if version.serving_binding != snapshot.corpus_binding:
        raise RecoveryEvidenceError("rebuilt corpus serving binding does not match the candidate snapshot")
    validation = version.validation_evidence
    if validation is None or not validation.index_ready or not validation.filter_ready:
        raise RecoveryEvidenceError("rebuilt corpus lacks index/filter readiness evidence")
    if validation.embedding_dimensions != 768:
        raise RecoveryEvidenceError("rebuilt corpus embedding dimensions must remain 768")

    readiness = _mapping(rebuild.get("search_readiness"), "rebuild.search_readiness")
    binding = _model(CorpusServingBinding, readiness.get("binding"), "rebuild.search_readiness.binding")
    if binding != snapshot.corpus_binding or readiness.get("corpus_filter_id") != version.version_id:
        raise RecoveryEvidenceError("search readiness must use the restored candidate binding and corpus filter")
    if readiness.get("top_k") != 5 or readiness.get("score_returning") is not True:
        raise RecoveryEvidenceError("search readiness must preserve the top-five score-returning retrieval contract")
    return version


def validate_recovery_evidence(document: object) -> dict[str, object]:
    """Return a compact, dated recovery record or raise RecoveryEvidenceError."""
    _reject_sensitive_keys(document)
    payload = _mapping(document, "evidence")
    if payload.get("schema_version") != "recovery-evidence-v1":
        raise RecoveryEvidenceError("schema_version must be recovery-evidence-v1")
    drill = _mapping(payload.get("drill"), "drill")
    drill_id = _string(drill.get("id"), "drill.id")
    started_at = _utc(drill.get("started_at"), "drill.started_at")
    completed_at = _utc(drill.get("completed_at"), "drill.completed_at")
    elapsed_seconds = (completed_at - started_at).total_seconds()
    if elapsed_seconds < 0 or elapsed_seconds > MAX_RECOVERY_SECONDS:
        raise RecoveryEvidenceError("isolated recovery did not complete within 24 hours")
    target = _validate_target(drill)
    retained_copies = _validate_backup(_mapping(payload.get("backup"), "backup"), completed_at)
    snapshot = _model(EvaluationCandidateSnapshot, payload.get("candidate_snapshot"), "candidate_snapshot")
    _validate_restored_state(_mapping(payload.get("restored"), "restored"), snapshot)
    _validate_content_scan(_mapping(payload.get("content_scan"), "content_scan"))
    corpus = _validate_rebuild(_mapping(payload.get("rebuild"), "rebuild"), snapshot)
    snapshot_digest = hashlib.sha256(snapshot.model_dump_json().encode("utf-8")).hexdigest()
    return {
        "schema_version": "recovery-result-v1",
        "drill_id": drill_id,
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": elapsed_seconds,
        "within_24_hours": True,
        "target": target,
        "retained_copy_count": retained_copies,
        "candidate_snapshot_sha256": snapshot_digest,
        "corpus_version_id": corpus.version_id,
        "serving_binding": snapshot.corpus_binding.model_dump(mode="json"),
        "production_interaction_content_absent": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        result = validate_recovery_evidence(evidence)
    except (OSError, json.JSONDecodeError, RecoveryEvidenceError) as exc:
        print(f"recovery evidence rejected: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{result['drill_id']}: isolated recovery verified in {result['elapsed_seconds']:.0f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
