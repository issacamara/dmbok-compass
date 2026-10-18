"""Isolated-recovery evidence verification tests."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from app.corpus.manifest import create_source_manifest, default_derivation


ROOT = Path(__file__).resolve().parents[2]


def load_recovery_module():
    spec = importlib.util.spec_from_file_location(
        "recovery_verifier", ROOT / "scripts" / "recovery" / "verify_langchain_restore.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(letter: str) -> str:
    return letter * 64


def evidence() -> dict[str, object]:
    derivation = default_derivation(vector_collection="document_chunks_v2")
    source = {
        "bucket": "approved-dmbok-source",
        "object_name": "DMBOK.pdf",
        "generation": 42,
        "content_sha256": digest("a"),
    }
    manifest = create_source_manifest(**source, derivation=derivation)
    binding = {
        "corpus_version_id": manifest.corpus_version_id,
        "vector_collection": "document_chunks_v2",
        "vector_schema_version": "2",
        "embedding_model": "gemini-embedding-001",
        "embedding_dimensions": 768,
        "ingestion_fingerprint": manifest.manifest_sha256,
        "configuration_fingerprint": manifest.derivation_sha256,
    }
    snapshot = {
        "corpus_binding": binding,
        "project": "recovery-drill-project",
        "location": "europe-west1",
        "answer_provider": "openrouter",
        "primary_model": "openai/gpt-4.1-mini",
        "fallback_model": "google/gemma-4-26b-a4b-it:free",
        "json_schema_supported": True,
        "request_timeout_ms": 28000,
        "generation_timeout_ms": 16000,
        "prompt_version": "grounded-v1",
        "retrieval_policy_version": "top-five-cosine-v1",
        "evidence_policy_version": "grounded-evidence-v1",
        "scorer_version": "evaluation-gates-v1",
        "dependency_lock_sha256": digest("b"),
        "integration_artifact": "langchain-google-firestore==0.5.0.post2",
        "runtime_version": "api@sha256:" + digest("c"),
    }
    run = {
        "run_id": "eval-recovery-1",
        "dataset_version_id": "generation-2",
        "corpus_version_id": manifest.corpus_version_id,
        "status": "completed",
        "configuration_version_id": "config-v2",
        "model_version_id": "models-v2",
        "candidate_snapshot": snapshot,
        "report_eligibility": {
            "generation_id": "generation-2",
            "eligibility": "exploratory",
            "approved_gold_count": 1,
            "is_current_generation": True,
            "is_superseded": False,
        },
    }
    decision = {
        "release_id": "candidate-a",
        "evaluation_run_id": run["run_id"],
        "dataset_version_id": run["dataset_version_id"],
        "corpus_version_id": run["corpus_version_id"],
        "configuration_version_id": run["configuration_version_id"],
        "provider_version_id": "openrouter-v1",
        "model_version_id": run["model_version_id"],
        "scorer_version_id": snapshot["scorer_version"],
        "gate_report_ids": ["gate-1"],
        "decision": "approved",
        "rationale": "Isolated drill fixture.",
    }
    return {
        "schema_version": "recovery-evidence-v1",
        "drill": {
            "id": "recovery-2026-10-16-candidate-a",
            "started_at": "2026-10-16T09:00:00Z",
            "completed_at": "2026-10-16T09:45:00Z",
            "source": {"project": "production-project", "database": "(default)"},
            "target": {
                "project": "recovery-drill-project",
                "database": "recovery-drill",
                "isolated": True,
                "is_production": False,
            },
        },
        "backup": {
            "retained_copy_count": 30,
            "daily_execution_dates": ["2026-10-15", "2026-10-16"],
        },
        "candidate_snapshot": snapshot,
        "restored": {
            "record_counts": {
                "users": 1,
                "configuration": 1,
                "quota": 1,
                "aggregate_metrics": 1,
                "datasets": 1,
                "annotations": 1,
                "runs": 1,
                "release_decisions": 1,
            },
            "restart_verified": True,
            "import_replacement_verified": True,
            "exploratory_semantics_verified": True,
            "production_interaction_content_absent": True,
            "exploratory_run_id": run["run_id"],
            "runs": [run],
            "release_decisions": [decision],
        },
        "content_scan": {
            "scanned_collections": [
                "users",
                "configuration",
                "quota",
                "aggregate_metrics",
                "evaluation_generations",
                "evaluation_runs",
                "release_decisions",
            ],
            "production_interaction_records_found": 0,
        },
        "rebuild": {
            "source": source,
            "derivation": derivation,
            "manifest": {
                "corpus_version_id": manifest.corpus_version_id,
                "manifest_sha256": manifest.manifest_sha256,
                "derivation_sha256": manifest.derivation_sha256,
            },
            "corpus_version": {
                "version_id": manifest.corpus_version_id,
                "source_uri": manifest.object_uri,
                "embedding_model": "gemini-embedding-001",
                "embedding_dimensions": 768,
                "status": "active",
                "created_at": "2026-10-16T09:30:00Z",
                "serving_binding": binding,
                "validation_evidence": {
                    "chunk_count": 1,
                    "content_hashes_sha256": digest("d"),
                    "provenance_sha256": digest("e"),
                    "embedding_dimensions": 768,
                    "index_ready": True,
                    "filter_ready": True,
                },
            },
            "search_readiness": {
                "binding": binding,
                "corpus_filter_id": manifest.corpus_version_id,
                "top_k": 5,
                "score_returning": True,
            },
        },
    }


def test_valid_isolated_recovery_emits_dated_candidate_evidence() -> None:
    recovery = load_recovery_module()

    result = recovery.validate_recovery_evidence(evidence())

    assert result["within_24_hours"] is True
    assert result["retained_copy_count"] == 30
    assert result["production_interaction_content_absent"] is True
    assert result["corpus_version_id"].startswith("v-")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["drill"]["target"].update({"project": "production-project", "database": "(default)"}), "differ from the source"),
        (lambda value: value["backup"].update({"retained_copy_count": 31}), "more than 30"),
        (lambda value: value["drill"].update({"completed_at": "2026-10-17T09:00:01Z"}), "within 24 hours"),
        (lambda value: value["restored"].update({"production_interaction_content_absent": False}), "interaction_content_absent"),
    ],
)
def test_recovery_rejects_unsafe_or_noncompliant_drills(mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    recovery = load_recovery_module()
    document = deepcopy(evidence())
    mutate(document)

    with pytest.raises(recovery.RecoveryEvidenceError, match=message):
        recovery.validate_recovery_evidence(document)


def test_recovery_rejects_snapshot_or_manifest_drift() -> None:
    recovery = load_recovery_module()
    document = evidence()
    document["rebuild"]["manifest"]["derivation_sha256"] = digest("f")  # type: ignore[index]

    with pytest.raises(recovery.RecoveryEvidenceError, match="manifest does not match"):
        recovery.validate_recovery_evidence(document)


def test_recovery_rejects_release_decision_with_a_different_run_binding() -> None:
    recovery = load_recovery_module()
    document = evidence()
    document["restored"]["release_decisions"][0]["corpus_version_id"] = "other-corpus"  # type: ignore[index]

    with pytest.raises(recovery.RecoveryEvidenceError, match="version binding"):
        recovery.validate_recovery_evidence(document)


def test_recovery_rejects_secret_fields_before_writing_evidence() -> None:
    recovery = load_recovery_module()
    document = evidence()
    document["operator_api_key"] = "never-record-me"

    with pytest.raises(recovery.RecoveryEvidenceError, match="secret material"):
        recovery.validate_recovery_evidence(document)
