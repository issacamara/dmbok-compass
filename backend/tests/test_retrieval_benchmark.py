from __future__ import annotations

from pathlib import Path
import json
import subprocess

import pytest

from app.contracts import (
    CandidateIdentity, CorpusServingBinding, EvaluationCandidateSnapshot,
    EvaluationCase, EvaluationMetric, EvaluationReportEligibility, EvaluationRun,
    RetrievalResult,
)
from app.performance import AUTHORITY_REF, RequestSample, build_report, load_authoritative_release_cost_record
from app.retrieval.benchmark import BenchmarkReport, build_reviewed_quality_report, evaluate, metrics, split_digest


FIXTURE = json.loads((Path(__file__).parent / "fixtures/architecture-v21.json").read_text())
CANDIDATE = CandidateIdentity(**FIXTURE["candidateIdentity"])


def case(case_id: str, expected: str = "answer", relevant: list[str] | None = None) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=f"Question {case_id}",
        expected_outcome=expected,
        relevant_chunk_ids=relevant or ["chunk-a"],
    )


def result(ids: list[str], scores: list[float], semantics: str = "cosine_similarity") -> RetrievalResult:
    return RetrievalResult(
        candidate=CANDIDATE,
        ordered_chunk_ids=ids,
        scores=scores,
        score_semantics=semantics,
        strategy="dense",
        stage_status="succeeded",
        search_ready=True,
    )


def test_metrics_use_top_five_denominators_and_reciprocal_rank() -> None:
    cases = [case("one", relevant=["chunk-a", "chunk-b"]), case("two", relevant=["missing"])]
    report = metrics(cases, [result(["chunk-a", "other"], [0.9, 0.4]), result(["other"], [0.8])])

    assert report.count == 2
    assert report.hit_at_5 == 50.0
    assert report.recall_at_5 == 0.25
    assert report.mrr == 0.5
    assert report.zero_hits == 1


def test_distance_scores_are_converted_and_ties_are_deterministic() -> None:
    item = case("one", relevant=["chunk-a"])
    report = evaluate(
        [item],
        lambda _: result(["chunk-b", "chunk-a"], [0.2, 0.2], "distance"),
        candidate=CANDIDATE,
        dataset_id="fixture-v1",
    )

    assert report.metrics.hit_at_5 == 100.0
    assert report.strategy == "dense-v1"


def test_calibration_records_split_digests_and_does_not_leak_held_out() -> None:
    tuning = [case("tuning", expected="evidence_refusal", relevant=[])]
    held_out = [case("held-out", expected="evidence_refusal", relevant=[])]
    report = evaluate(
        [],
        lambda item: result(["chunk-a"], [0.9]),
        candidate=CANDIDATE,
        dataset_id="reviewed-v1",
        tuning_cases=tuning,
        held_out_cases=held_out,
    )

    assert report.tuning_digest == split_digest(tuning)
    assert report.held_out_digest == split_digest(held_out)
    assert report.tuning_digest != report.held_out_digest
    assert report.calibration.policy_version.startswith("dense-evidence-")


def test_splits_cannot_overlap_evaluated_cases() -> None:
    item = case("same")
    with pytest.raises(ValueError, match="overlap"):
        evaluate(
            [item],
            lambda _: result(["chunk-a"], [0.9]),
            candidate=CANDIDATE,
            dataset_id="fixture-v1",
            held_out_cases=[item],
        )


def test_reviewed_quality_report_binds_frozen_candidate_and_keeps_denominators(tmp_path: Path) -> None:
    candidate = CandidateIdentity(
        artifact_id="wheel-v1", installed_lock_sha256="c" * 64,
        corpus_version_id="corpus-v1", configuration_version_id="config-v1",
        scorer_version="scorer-v1",
    )
    thresholds = {
        "retrieval_success": 90, "grounded_claims": 95,
        "citation_correctness": 95, "answer_quality": 85,
        "refusal_correctness": 95, "response_time": 95,
    }
    run = EvaluationRun(
        run_id="run-v1", dataset_version_id="dataset-v1", corpus_version_id="corpus-v1",
        configuration_version_id="config-v1", model_version_id="model-v1", status="completed",
        candidate_snapshot=EvaluationCandidateSnapshot(
            corpus_binding=CorpusServingBinding(
                corpus_version_id="corpus-v1", vector_collection="chunks-v2",
                vector_schema_version="2", embedding_model="gemini-embedding-001",
                ingestion_fingerprint="a" * 64, configuration_fingerprint="b" * 64,
            ),
            project="project", location="europe-west1", answer_provider="openrouter",
            primary_model="model-v1", json_schema_supported=True, request_timeout_ms=1000,
            generation_timeout_ms=1000, prompt_version="prompt-v1",
            retrieval_policy_version="retrieval-v1", evidence_policy_version="evidence-v1",
            scorer_version="scorer-v1", dependency_lock_sha256="c" * 64,
            integration_artifact="wheel-v1", runtime_version="image-v1",
        ),
        report_eligibility=EvaluationReportEligibility(
            generation_id="generation-v1", eligibility="release_evidence", approved_gold_count=40,
            is_current_generation=True, is_superseded=False,
        ),
        metrics=[EvaluationMetric(
            metric_name=name,
            numerator=0 if name == "response_time" else 38 if name == "answer_quality" else 40,
            denominator=40,
            percentage=0 if name == "response_time" else 95 if name == "answer_quality" else 100,
            threshold=threshold, passed=name != "response_time",
        ) for name, threshold in thresholds.items()],
    )
    report = build_reviewed_quality_report(
        run, candidate, held_out_digest="d" * 64, tuning_digest="e" * 64,
    )

    assert report["quality_metrics_passed"] is True
    assert report["metrics"]["response_time"]["passed"] is False
    assert report["split_binding_status"] == "unverified"
    assert report["passed"] is False
    assert report["metrics"]["answer_quality"]["numerator"] == 38
    assert report["metrics"]["answer_quality"]["denominator"] == 40
    assert report["model_overlap_status"] == "unknown"

    mismatched = candidate.model_copy(update={"artifact_id": "other-wheel"})
    with pytest.raises(ValueError, match="frozen candidate identity"):
        build_reviewed_quality_report(
            run, mismatched, held_out_digest="d" * 64, tuning_digest="e" * 64,
        )

    def candidate_result(_: EvaluationCase) -> RetrievalResult:
        return RetrievalResult(
            candidate=candidate, ordered_chunk_ids=["chunk-a"], scores=[0.9],
            score_semantics="cosine_similarity", strategy="dense",
            stage_status="succeeded", search_ready=True,
        )

    benchmark_report = evaluate(
        [], candidate_result, candidate=candidate, dataset_id=run.dataset_version_id,
        tuning_cases=[case("tuning", expected="evidence_refusal", relevant=[])],
        held_out_cases=[case("held-out", expected="evidence_refusal", relevant=[])],
    )
    verified = build_reviewed_quality_report(run, candidate, benchmark_report=benchmark_report)
    assert verified["split_binding_status"] == "verified"
    assert verified["passed"] is True
    assert verified.benchmark_report is benchmark_report

    serialized = json.loads(json.dumps(verified))
    assert serialized["split_binding_status"] == "verified"
    with pytest.raises(ValueError, match="not from evaluate"):
        build_reviewed_quality_report(
            run, candidate, benchmark_report=BenchmarkReport(
                candidate=candidate, dataset_id=run.dataset_version_id, strategy="dense-v1",
                tuning_digest=benchmark_report.tuning_digest, held_out_digest=benchmark_report.held_out_digest,
                metrics=benchmark_report.metrics, calibration=benchmark_report.calibration,
            ),
        )
    other_dataset_report = evaluate(
        [], candidate_result, candidate=candidate, dataset_id="other-dataset",
        tuning_cases=[case("tuning-other", expected="evidence_refusal", relevant=[])],
        held_out_cases=[case("held-other", expected="evidence_refusal", relevant=[])],
    )
    with pytest.raises(ValueError, match="candidate and dataset"):
        build_reviewed_quality_report(run, candidate, benchmark_report=other_dataset_report)

    authority_repo = tmp_path / "authority"
    authority_repo.mkdir()
    _git(authority_repo, "init")
    _git(authority_repo, "remote", "add", "origin", "https://github.com/issacamara/dmbok-compass.git")
    _git(authority_repo, "config", "user.email", "test@example.invalid")
    _git(authority_repo, "config", "user.name", "Test")
    (authority_repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(authority_repo, "add", "README.md")
    _git(authority_repo, "commit", "-m", "initialize fixture")
    record = {
        "schema_version": "release-cost-approval-v1",
        "candidate_identity": candidate.model_dump(mode="json"),
        "sponsor_approval": {
            "approval_reference": "https://github.com/issacamara/dmbok-compass/issues/348",
            "approved_by": "sponsor-1", "approved_at": "2026-11-13T12:00:00Z",
            "provider": "openrouter", "monthly_cap_eur": 1.0,
        },
        "provider_rates": {
            "source_reference": "https://openrouter.ai/api/v1/models",
            "source_timestamp": "2026-11-13T12:00:00Z", "currency": "EUR",
            "primary_input_eur_per_million": 1, "primary_output_eur_per_million": 2,
            "fallback_input_eur_per_million": 3, "fallback_output_eur_per_million": 4,
        },
        "gcp_forecast": {
            "source_reference": "https://cloud.google.com/billing/docs",
            "source_timestamp": "2026-11-13T12:00:00Z", "monthly_cost_eur": 1.0,
        },
    }
    record_path = authority_repo / "documents/decisions/release-cost-approval.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    _git(authority_repo, "add", "documents/decisions/release-cost-approval.json")
    _git(authority_repo, "commit", "-m", "approve synthetic test costs")
    _git(authority_repo, "branch", "-M", "development")
    _git(authority_repo, "update-ref", AUTHORITY_REF, "HEAD")
    verified_costs = load_authoritative_release_cost_record(authority_repo, candidate.model_dump(mode="json"))
    assert verified_costs is not None

    stage_usage = {
        name: {
            "status": "complete", "used_units": 1, "reserved_units": 1,
            "cost_eur": 0.1, "reserved_cost_eur": 0.2,
        }
        for name in ("parser", "generator", "judge")
    }
    release_report = build_report(
        [RequestSample(index, 200, 1_000, "answer", "primary", input_tokens=10, output_tokens=5, user_subject=f"user-{index}") for index in range(3)],
        threshold_ms=30_000, daily_requests=3, provider_cost_eur=99, gcp_cost_eur=99,
        gcp_budget_eur=5, assumptions={
            "fallback_model": "fallback", "input_tokens_per_request": 10,
            "output_tokens_per_request": 5,
        },
        candidate_identity=candidate.model_dump(mode="json"), cost_evidence=verified_costs,
        generation_coverage={
            "status": "published", "target_count": 1, "generated_count": 1, "achievable_size": 1,
        },
        stage_usage=stage_usage, quality_evidence=verified,
    )
    assert release_report["release_evidence_status"] == "passed"
    assert release_report["cost_envelope"]["provider_forecast_eur_per_month"] < 1
    assert release_report["cost_envelope"]["gcp_forecast_eur_per_month"] == 1.0

    imported_quality = json.loads(json.dumps(verified))
    imported_report = build_report(
        [RequestSample(index, 200, 1_000, "answer", "primary", input_tokens=10, output_tokens=5, user_subject=f"user-{index}") for index in range(3)],
        threshold_ms=30_000, daily_requests=3, provider_cost_eur=1, gcp_cost_eur=1,
        gcp_budget_eur=5, assumptions={
            "fallback_model": "fallback", "input_tokens_per_request": 10,
            "output_tokens_per_request": 5,
        },
        candidate_identity=candidate.model_dump(mode="json"), cost_evidence=verified_costs,
        generation_coverage={
            "status": "published", "target_count": 1, "generated_count": 1, "achievable_size": 1,
        },
        stage_usage=stage_usage, quality_evidence=imported_quality,
    )
    assert imported_report["release_evidence_status"] == "blocked"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
