import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import (
    EvaluationPreparationComplete,
    RunBoundSupportEvidence,
)
from app.evaluation.reviews import RunItemAssessment
from app.evaluation.dataset import CandidateGoldBinding


FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "architecture-v22.json").read_text())


def _chunk(chunk_id: str = "chunk-unique") -> dict[str, object]:
    return {**FIXTURE["verifiedChunk"], "chunk_id": chunk_id}


def _evidence(**overrides: object) -> RunBoundSupportEvidence:
    payload = {
        **FIXTURE["supportEvidence"],
        "verified_chunks": [_chunk()],
        "selected_chunk": _chunk(),
        **overrides,
    }
    return RunBoundSupportEvidence(**payload)


def test_fixture_freezes_v22_source_and_a23_decisions() -> None:
    assert FIXTURE["schemaVersion"] == "2.2"
    assert FIXTURE["source"]["sha256"] == "5ff60b4499ebe752d7552ed9369e281795b40142a25ebbf4355e2a9b6aed057d"
    assert {case["id"] for case in FIXTURE["a23Cases"]} == {
        "unique", "repeated", "missing", "invalid-metadata", "short-passage", "boundary-crossing", "unanswerable",
    }
    assert FIXTURE["compatibility"]["legacyCandidateBinding"] == "candidate-binding-v1"


@pytest.mark.parametrize("status", ["auto_approved", "missing", "ambiguous", "invalid", "review_approved", "review_rejected"])
def test_support_statuses_are_explicit_and_run_bound(status: str) -> None:
    if status == "auto_approved":
        evidence = _evidence(match_status=status)
    elif status == "review_approved":
        evidence = _evidence(match_status=status, reviewer_actor="admin-1", reviewed_at=datetime.now(timezone.utc))
    elif status == "review_rejected":
        evidence = _evidence(match_status=status, selected_chunk=None, reviewer_actor="admin-1", reviewed_at=datetime.now(timezone.utc))
    else:
        evidence = _evidence(match_status=status, selected_chunk=None)
    assert evidence.run_id == FIXTURE["identities"]["runId"]
    assert evidence.corpus_version_id == FIXTURE["identities"]["corpusVersionId"]


def test_auto_approval_requires_one_verified_chunk_and_rejects_refusal() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        _evidence(verified_chunks=[], selected_chunk=None)
    with pytest.raises(ValidationError, match="cannot be auto-approved"):
        _evidence(expected_refusal=True)


def test_unresolved_and_reviewed_states_cannot_smuggle_a_chunk() -> None:
    with pytest.raises(ValidationError, match="unresolved support"):
        _evidence(match_status="ambiguous")
    with pytest.raises(ValidationError, match="review rejection"):
        _evidence(match_status="review_rejected", selected_chunk=None)


def test_preparation_marker_requires_complete_item_coverage() -> None:
    marker = EvaluationPreparationComplete(
        run_id="run-2026-10-27",
        candidate_snapshot_sha256=FIXTURE["identities"]["candidateSnapshotSha256"],
        corpus_version_id=FIXTURE["identities"]["corpusVersionId"],
        item_count=7,
        prepared_item_count=7,
        approved_item_count=2,
        unresolved_item_count=5,
        completed_at=datetime.now(timezone.utc),
    )
    assert marker.prepared_item_count == marker.item_count
    with pytest.raises(ValidationError, match="every item"):
        EvaluationPreparationComplete(
            **{**marker.model_dump(), "prepared_item_count": 6}
        )


def test_existing_manual_bindings_and_assessments_remain_readable() -> None:
    binding = CandidateGoldBinding(
        corpus_version_id="corpus-legacy",
        candidate_snapshot_sha256="d" * 64,
        source_provenance_sha256="e" * 64,
    )
    assessment = RunItemAssessment(
        generation_id="generation-legacy",
        attempt_id="attempt-legacy",
        candidate_id="candidate-legacy",
        scorer_version="scorer-legacy",
        output_digest="f" * 64,
        claim_support=True,
        citation_support=True,
        rubric_passed=True,
        refusal_judgment=False,
    )
    assert binding.model_dump()["corpus_version_id"] == "corpus-legacy"
    assert assessment.complete is True
