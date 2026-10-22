"""Candidate-bound v2.1 evidence is reproducible and fails closed outside a final image."""

from __future__ import annotations

import json
from pathlib import Path

from app.contracts import CandidateIdentity, EvaluationCase, RetrievalResult
from app.performance import RequestSample, build_report
from app.retrieval.benchmark import evaluate


ROOT = Path(__file__).parents[2]
RELEASE_DIR = ROOT / "documents" / "release" / "v2.1"

CANDIDATE = CandidateIdentity(
    artifact_id="candidate-image@sha256:" + "d" * 64,
    installed_lock_sha256="c" * 64,
    corpus_version_id="corpus-v1",
    configuration_version_id="config-v1",
    scorer_version="evaluation-gates-v1",
)


def _case(index: int) -> EvaluationCase:
    return EvaluationCase(
        case_id=f"reviewed-{index}",
        question=f"Reviewed authored question {index}",
        expected_outcome="answer",
        relevant_chunk_ids=[f"section-{index}:support"],
    )


def _result(case: EvaluationCase, *, hit: bool = True) -> RetrievalResult:
    chunk_id = case.relevant_chunk_ids[0] if hit else "other-section:unsupported"
    return RetrievalResult(
        candidate=CANDIDATE,
        ordered_chunk_ids=[chunk_id],
        scores=[0.91],
        score_semantics="cosine_similarity",
        strategy="dense",
        stage_status="succeeded",
        search_ready=True,
    )


def test_dense_candidate_fixture_records_quality_and_three_user_probe_boundaries() -> None:
    """Keep local fixtures useful without representing them as paid final-image proof."""

    cases = [_case(index) for index in range(10)]
    benchmark = evaluate(
        cases,
        lambda case: _result(case, hit=case.case_id != "reviewed-9"),
        candidate=CANDIDATE,
        dataset_id="reviewed-authored-v1",
        strategy="dense",
    )
    assert benchmark.metrics.hit_at_5 == 90.0
    assert benchmark.metrics.zero_hits == 1

    samples = [
        RequestSample(
            user_index=user,
            status_code=200,
            duration_ms=10_000 + user * 100,
            outcome="answer",
            model="primary",
            input_tokens=800,
            output_tokens=300,
        )
        for user in range(3)
    ]
    performance = build_report(
        samples,
        threshold_ms=15_000,
        daily_requests=100,
        provider_cost_eur=None,
        gcp_cost_eur=0.0,
        gcp_budget_eur=5.0,
        assumptions={"fallback_model": "fallback", "fixture_only": True},
        candidate_identity=CANDIDATE.model_dump(mode="json"),
    )
    assert performance["user_count"] == 3
    assert performance["within_threshold_rate"] == 1.0
    assert performance["cost_envelope"]["provider_cost_status"] == "unknown"


def test_release_evidence_artifacts_preserve_exact_binding_and_block_unknown_live_proof() -> None:
    retrieval = json.loads((RELEASE_DIR / "retrieval-report.json").read_text())
    evidence = (RELEASE_DIR / "dense-candidate-evidence.md").read_text()

    assert retrieval["candidate"] == CANDIDATE.model_dump(mode="json")
    assert retrieval["strategy"] == "dense"
    assert retrieval["quality_gates"]["hit_at_5_percentage"] == 90.0
    assert retrieval["live_validation"]["status"] == "blocked"
    assert retrieval["live_validation"]["missing"] == [
        "final-image execution",
        "approved provider budget",
        "deployed three-user probe",
    ]
    assert "No operational error receives success credit" in evidence
    assert "Status: **blocked**" in evidence
    assert "dense-only" in evidence


def test_release_recovery_rehearsal_preserves_the_candidate_tuple_and_fails_closed() -> None:
    rehearsal = (RELEASE_DIR / "release-recovery-rehearsal.md").read_text()

    assert "documents/release/v2.1/retrieval-report.json" in rehearsal
    assert "documents/release/langchain/release-evidence.json#artifact" in rehearsal
    assert "documents/release/langchain/release-evidence.json#evaluation" in rehearsal
    assert "documents/release/langchain/release-evidence.json#release_decision" in rehearsal
    assert "scripts/recovery/verify_langchain_restore.py" in rehearsal
    assert "not promotable" in rehearsal
    assert "eleven evidence sections as `blocked`" in rehearsal
    assert "release decision is `pending`" in rehearsal
