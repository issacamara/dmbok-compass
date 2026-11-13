from app.performance import RequestSample, build_report, forecast_provider_cost_eur, latency_summary, percentile


def sample(duration_ms: float, *, model: str = "primary") -> RequestSample:
    return RequestSample(0, 200, duration_ms, "answer", model, user_subject="subject-0")


def test_percentile_uses_nearest_rank_and_latency_summary_is_content_free() -> None:
    samples = [sample(100), sample(200), sample(300), sample(400)]

    assert percentile((item.duration_ms for item in samples), 95) == 400
    assert latency_summary(samples) == {
        "count": 4,
        "mean_ms": 250,
        "p50_ms": 200,
        "p95_ms": 400,
        "max_ms": 400,
    }


def test_cost_forecast_blends_primary_and_fallback_rates() -> None:
    assert forecast_provider_cost_eur(
        daily_requests=100,
        input_tokens_per_request=1_000,
        output_tokens_per_request=500,
        fallback_rate=0.25,
        primary_input_eur_per_million=1,
        primary_output_eur_per_million=2,
        fallback_input_eur_per_million=4,
        fallback_output_eur_per_million=8,
    ) == 10.5


def test_report_fails_closed_for_failed_requests_and_slow_p95() -> None:
    report = build_report(
        [sample(100), RequestSample(1, 503, 16_000, None, None, "HTTPStatusError", user_subject="subject-1")],
        threshold_ms=15_000,
        daily_requests=100,
        provider_cost_eur=0,
        gcp_cost_eur=0,
        gcp_budget_eur=5,
        assumptions={"fallback_model": "fallback"},
    )

    assert report["passed"] is False
    assert report["failed_count"] == 1
    assert "question" not in json_safe_text(report)


def test_report_counts_typed_outcomes_and_rejects_http_200_errors() -> None:
    report = build_report(
        [
            RequestSample(0, 200, 100, "answer", "primary", input_tokens=10, output_tokens=5, user_subject="sub-0"),
            RequestSample(1, 200, 120, "refusal", "primary", input_tokens=10, output_tokens=0, user_subject="sub-1"),
            RequestSample(2, 200, 90, "operational_error", "primary", error_class="unavailable", user_subject="sub-2"),
            RequestSample(2, 503, 80, None, None, error_class="HTTPStatusError", user_subject="sub-2"),
        ],
        threshold_ms=15_000,
        daily_requests=100,
        provider_cost_eur=1.0,
        gcp_cost_eur=0,
        gcp_budget_eur=5,
        assumptions={"fallback_model": "fallback"},
        candidate_identity={"artifact_id": "image-sha"},
    )

    assert report["functional_success_count"] == 1
    assert report["answer_success_count"] == 1
    assert report["operational_error_count"] == 2
    assert report["outcome_mismatch_count"] == 1
    assert report["outcome_counts"] == {
        "answer": 1,
        "qualified": 0,
        "refusal": 1,
        "evidence_refusal": 0,
        "operational_error": 1,
    }
    assert report["normal_mix"]["count"] == 1
    assert report["within_threshold_rate"] == 1.0
    assert report["usage"]["status"] == "partial"
    assert report["candidate_identity"] == {"artifact_id": "image-sha"}
    assert report["passed"] is False


def test_report_marks_cost_usage_unknown_without_token_data() -> None:
    report = build_report(
        [sample(100)],
        threshold_ms=15_000,
        daily_requests=100,
        provider_cost_eur=10.5,
        gcp_cost_eur=0,
        gcp_budget_eur=5,
        assumptions={"fallback_model": "fallback"},
    )

    assert report["usage"]["status"] == "unknown"
    assert report["cost_envelope"]["provider_cost_status"] == "unknown"


def test_release_report_uses_all_requests_in_30_second_denominator_and_blocks_missing_cap() -> None:
    samples = [
        RequestSample(user, 200, 1_000, "answer", "primary", input_tokens=10, output_tokens=5, user_subject=f"sub-{user}")
        for user in range(3) for _ in range(6)
    ]
    samples[-1] = RequestSample(2, 503, 1_000, None, None, "HTTPStatusError", user_subject="sub-2")
    report = build_report(
        samples,
        threshold_ms=30_000,
        daily_requests=100,
        provider_cost_eur=1,
        gcp_cost_eur=1,
        gcp_budget_eur=5,
        assumptions={"fallback_model": "fallback"},
    )

    assert report["within_30_seconds_count"] == 17
    assert report["within_30_seconds_denominator"] == 18
    assert report["within_30_seconds_rate"] == round(17 / 18, 6)
    assert report["latency_passed"] is False
    assert report["cost_envelope"]["provider_cap_status"] == "missing_authoritative_record"
    assert report["release_evidence_status"] == "blocked"
    assert report["errors_by_class"] == {"HTTPStatusError": 1}
    assert report["operational_failure_count"] == 1
    assert report["evidence_refusal_denominator"] == 18


def test_report_discards_free_form_error_text_and_preserves_separate_diagnostics() -> None:
    report = build_report(
        [
            RequestSample(0, 200, 16_000, "answer", "primary", input_tokens=10, output_tokens=5, first_request=True),
            RequestSample(1, 200, 2_000, "answer", "fallback", input_tokens=10, output_tokens=5, first_request=True),
            RequestSample(2, 599, 31_000, None, None, "secret-account-id:private", first_request=True),
        ],
        threshold_ms=30_000,
        daily_requests=100,
        provider_cost_eur=1,
        gcp_cost_eur=1,
        gcp_budget_eur=5,
        assumptions={"fallback_model": "fallback"},
    )

    rendered = json_safe_text(report)
    assert "secret-account-id" not in rendered
    assert report["first_request_diagnostic"]["status"] == "unverified_cold_start_proxy"
    assert report["within_15_seconds_diagnostic_count"] == 1
    assert report["fallback_mix"]["count"] == 1
    assert report["cost_envelope"]["provider_cap_status"] == "missing_authoritative_record"


def test_evidence_refusals_are_separate_and_do_not_count_toward_expected_answer_gate() -> None:
    samples = [
        RequestSample(0, 200, 500, "answer", "primary", expected_outcome="answer"),
        RequestSample(1, 200, 500, "evidence_refusal", "primary", expected_outcome="answer"),
        RequestSample(2, 200, 500, "answer", "primary", expected_outcome="answer"),
    ]
    report = build_report(
        samples, threshold_ms=30_000, daily_requests=3, provider_cost_eur=0,
        gcp_cost_eur=0, gcp_budget_eur=5, assumptions={},
    )

    assert report["within_30_seconds_count"] == 2
    assert report["within_30_seconds_denominator"] == 3
    assert report["evidence_refusal_count"] == 1
    assert report["evidence_refusal_denominator"] == 3
    assert report["operational_failure_count"] == 0


def test_stage_costs_are_aggregated_and_unknown_costs_block_release() -> None:
    stages = {
        name: {
            "status": "complete", "used_units": 10, "reserved_units": 12,
            "cost_eur": cost, "reserved_cost_eur": cost + 0.1,
        }
        for name, cost in (("parser", 0.2), ("generator", 0.3), ("judge", 0.4))
    }
    requests = [RequestSample(i, 200, 500, "answer", "primary") for i in range(3)]
    complete = build_report(
        requests, threshold_ms=30_000, daily_requests=3, provider_cost_eur=0,
        gcp_cost_eur=0, gcp_budget_eur=5, assumptions={}, stage_usage=stages,
    )
    stages["judge"] = {**stages["judge"], "cost_eur": None}
    unknown = build_report(
        requests, threshold_ms=30_000, daily_requests=3, provider_cost_eur=0,
        gcp_cost_eur=0, gcp_budget_eur=5, assumptions={}, stage_usage=stages,
    )

    assert complete["stage_costs"] == {
        "status": "complete", "used_cost_eur": 0.9, "reserved_cost_eur": 1.2,
    }
    assert unknown["stage_costs"]["status"] == "unknown"
    assert unknown["release_evidence_status"] == "blocked"


def test_numeric_cap_without_sponsor_provenance_cannot_pass() -> None:
    report = build_report(
        [RequestSample(i, 200, 500, "answer", "primary", input_tokens=10, output_tokens=5) for i in range(3)],
        threshold_ms=30_000, daily_requests=3, provider_cost_eur=1,
        gcp_cost_eur=0, gcp_budget_eur=5, assumptions={},
        candidate_identity={"artifact_id": "image-v1"},
    )

    assert report["cost_envelope"]["provider_cap_status"] == "missing_authoritative_record"
    assert report["cost_envelope"]["provider_hard_cap_eur_per_month"] is None


def test_unverified_quality_splits_block_otherwise_complete_release_evidence() -> None:
    candidate = {
        "artifact_id": "image-v1", "installed_lock_sha256": "a" * 64,
        "corpus_version_id": "corpus-v1", "configuration_version_id": "config-v1",
        "scorer_version": "scorer-v1",
    }
    stage_usage = {
        name: {
            "status": "complete", "used_units": 1, "reserved_units": 1,
            "cost_eur": 0.1, "reserved_cost_eur": 0.2,
        }
        for name in ("parser", "generator", "judge")
    }
    quality = {
        "candidate": candidate, "quality_metrics_passed": True,
        "split_binding_status": "unverified", "model_overlap_status": "unknown",
    }
    report = build_report(
        [RequestSample(i, 200, 500, "answer", "primary", input_tokens=10, output_tokens=5, user_subject=f"sub-{i}") for i in range(3)],
        threshold_ms=30_000, daily_requests=3, provider_cost_eur=1,
        gcp_cost_eur=0, gcp_budget_eur=5, assumptions={}, candidate_identity=candidate,
        generation_coverage={
            "status": "published", "target_count": 1, "generated_count": 1, "achievable_size": 1,
        },
        stage_usage=stage_usage, quality_evidence=quality,
    )

    assert report["latency_passed"] is True
    assert report["cost_envelope"]["provider_cap_status"] == "missing_authoritative_record"
    assert report["stage_costs"]["status"] == "complete"
    assert report["release_evidence_status"] == "blocked"


def test_report_counts_verified_subjects_not_enumeration_slots() -> None:
    samples = [
        RequestSample(i, 200, 500, "answer", "primary", user_subject="same-subject")
        for i in range(3)
    ]

    report = build_report(
        samples, threshold_ms=30_000, daily_requests=3, provider_cost_eur=1,
        gcp_cost_eur=0, gcp_budget_eur=5, assumptions={},
    )

    assert report["user_count"] == 1
    assert report["latency_passed"] is False
    assert "same-subject" not in json_safe_text(report)


def test_missing_cost_evidence_cannot_pass_gcp_or_provider_budget_gates() -> None:
    report = build_report(
        [RequestSample(i, 200, 500, "answer", "primary") for i in range(3)],
        threshold_ms=30_000, daily_requests=3, provider_cost_eur=None,
        gcp_cost_eur=None, gcp_budget_eur=5, assumptions={},
    )

    assert report["cost_envelope"]["provider_cost_status"] == "unknown"
    assert report["cost_envelope"]["gcp_forecast_status"] == "missing_or_unverified"
    assert report["cost_envelope"]["gcp_forecast_within_budget"] is False
    assert report["release_evidence_status"] == "blocked"


def test_present_but_unverified_cost_sources_remain_diagnostic_only() -> None:
    report = build_report(
        [RequestSample(i, 200, 500, "answer", "primary", input_tokens=10, output_tokens=5) for i in range(3)],
        threshold_ms=30_000, daily_requests=3, provider_cost_eur=1,
        gcp_cost_eur=1, gcp_budget_eur=5, assumptions={},
        cost_evidence={"evidence_status": "unverified_source"},
    )

    assert report["cost_envelope"]["provider_cost_status"] == "unknown"
    assert report["cost_envelope"]["gcp_forecast_status"] == "missing_or_unverified"
    assert report["release_evidence_status"] == "blocked"


def test_report_rejects_budget_above_ceiling() -> None:
    import pytest

    with pytest.raises(ValueError, match="€5 ceiling"):
        build_report(
            [], threshold_ms=30_000, daily_requests=0, provider_cost_eur=None,
            gcp_cost_eur=0, gcp_budget_eur=5.01, assumptions={},
        )


def json_safe_text(value: object) -> str:
    """Serialize report data for the privacy assertion without adding a dependency."""

    import json

    return json.dumps(value, sort_keys=True)
