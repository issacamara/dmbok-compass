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


def json_safe_text(value: object) -> str:
    """Serialize report data for the privacy assertion without adding a dependency."""

    import json

    return json.dumps(value, sort_keys=True)
