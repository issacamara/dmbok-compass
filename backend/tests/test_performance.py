from app.performance import RequestSample, build_report, forecast_provider_cost_eur, latency_summary, percentile


def sample(duration_ms: float, *, model: str = "primary") -> RequestSample:
    return RequestSample(0, 200, duration_ms, "answer", model)


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
        [sample(100), RequestSample(1, 503, 16_000, None, None, "HTTPStatusError")],
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
            RequestSample(0, 200, 100, "answer", "primary", input_tokens=10, output_tokens=5),
            RequestSample(1, 200, 120, "refusal", "primary", input_tokens=10, output_tokens=0),
            RequestSample(2, 200, 90, "operational_error", "primary", error_class="unavailable"),
            RequestSample(2, 503, 80, None, None, error_class="HTTPStatusError"),
        ],
        threshold_ms=15_000,
        daily_requests=100,
        provider_cost_eur=1.0,
        gcp_cost_eur=0,
        gcp_budget_eur=5,
        assumptions={"fallback_model": "fallback"},
        candidate_identity={"artifact_id": "image-sha"},
    )

    assert report["functional_success_count"] == 2
    assert report["answer_success_count"] == 1
    assert report["operational_error_count"] == 2
    assert report["outcome_counts"] == {
        "answer": 1,
        "qualified": 0,
        "refusal": 1,
        "evidence_refusal": 0,
        "operational_error": 1,
    }
    assert report["normal_mix"]["count"] == 2
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


def json_safe_text(value: object) -> str:
    """Serialize report data for the privacy assertion without adding a dependency."""

    import json

    return json.dumps(value, sort_keys=True)
