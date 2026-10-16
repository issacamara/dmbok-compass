import pytest

from app.release import ReleaseEvidenceError, validate_release_evidence


def evidence_document() -> dict:
    gates = [
        {"metric_name": name, "threshold": threshold, "passed": True}
        for name, threshold in {
                "retrieval_success": 90.0,
            "grounded_claims": 95.0,
            "citation_correctness": 95.0,
            "answer_quality": 85.0,
            "refusal_correctness": 95.0,
            "response_time": 95.0,
        }.items()
    ]
    return {
        "schema_version": "1",
        "release_id": "release-2026-10-07",
        "artifact": {
            "commit": "abc123",
            "api_image": "europe-west1-docker.pkg.dev/project/api@sha256:" + "a" * 64,
            "api_digest": "sha256:" + "a" * 64,
            "frontend_digest": "b" * 64,
            "preview_revision": "api-00001",
        },
        "evaluation": {
            "run_id": "eval-1",
            "dataset_version_id": "dataset-v1",
            "corpus_version_id": "corpus-v1",
            "configuration_version_id": "config-v1",
            "model_version_id": "model-v1",
            "status": "completed",
            "gates": gates,
            "candidate_snapshot": {
                "corpus_binding": {
                    "corpus_version_id": "corpus-v1", "vector_collection": "chunks-v2",
                    "vector_schema_version": "2", "embedding_model": "gemini-embedding-001",
                    "ingestion_fingerprint": "a" * 64, "configuration_fingerprint": "b" * 64,
                },
                "project": "project", "location": "europe-west1", "answer_provider": "openrouter",
                "primary_model": "model-v1", "prompt_version": "prompt-v1",
                "retrieval_policy_version": "retrieval-v1", "evidence_policy_version": "evidence-v1",
                "scorer_version": "scorer-v1", "dependency_lock_sha256": "c" * 64,
                "integration_artifact": "wheel-v1", "runtime_version": "image@sha256:" + "d" * 64,
            },
            "candidate_gold_binding": {
                "corpus_version_id": "corpus-v1", "candidate_snapshot_sha256": "e" * 64,
                "source_provenance_sha256": "f" * 64,
            },
        },
        "evidence": {
            name: {"status": "pass", "references": [f"{name}-evidence"]}
            for name in (
                "artifact", "evaluation", "provider", "privacy", "accessibility", "recovery",
                "performance", "cost", "rollback", "decisions", "gates",
            )
        },
        "release_decision": {
            "release_id": "release-2026-10-07",
            "decision": "approved",
            "approver": "sponsor",
        },
    }


def test_valid_bundle_is_promotable() -> None:
    result = validate_release_evidence(evidence_document())
    assert result.promotable is True
    assert result.exceptions == ()


@pytest.mark.parametrize("section", ["provider", "privacy", "accessibility", "recovery", "performance", "cost", "rollback"])
def test_missing_evidence_blocks_bundle(section: str) -> None:
    document = evidence_document()
    del document["evidence"][section]
    with pytest.raises(ReleaseEvidenceError, match="missing evidence sections"):
        validate_release_evidence(document)


def test_failed_evidence_cannot_be_hidden_as_an_exception() -> None:
    document = evidence_document()
    document["evidence"]["privacy"] = {"status": "failed", "references": ["privacy-check"]}
    with pytest.raises(ReleaseEvidenceError, match="must pass or have an approved exception"):
        validate_release_evidence(document)


def test_approved_exception_is_traceable_but_not_promotable() -> None:
    document = evidence_document()
    document["evidence"]["cost"] = {
        "status": "exception",
        "references": ["cost-forecast"],
        "exception": {
            "approved": True,
            "approver": "sponsor",
            "rationale": "Temporary approved variance.",
            "approved_at": "2026-10-07T12:00:00Z",
        },
    }
    result = validate_release_evidence(document)
    assert result.promotable is True
    assert result.exceptions == ("cost",)
