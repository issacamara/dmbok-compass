"""Content-free performance evidence and cost-envelope calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class RequestSample:
    """The aggregate fields retained for one probe request."""

    user_index: int
    status_code: int
    duration_ms: float
    outcome: str | None
    model: str | None
    error_class: str | None = None


def percentile(values: Iterable[float], percentile_value: float) -> float:
    """Return the nearest-rank percentile without retaining request content."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("at least one value is required")
    if not 0 < percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    rank = max(1, ceil((percentile_value / 100) * len(ordered)))
    return round(ordered[rank - 1], 3)


def latency_summary(samples: Iterable[RequestSample]) -> dict[str, float | int]:
    """Summarize latency for a set of aggregate request samples."""

    values = [sample.duration_ms for sample in samples]
    if not values:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "mean_ms": round(mean(values), 3),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": round(max(values), 3),
    }


def forecast_provider_cost_eur(
    *,
    daily_requests: int,
    input_tokens_per_request: int,
    output_tokens_per_request: int,
    fallback_rate: float,
    primary_input_eur_per_million: float,
    primary_output_eur_per_million: float,
    fallback_input_eur_per_million: float,
    fallback_output_eur_per_million: float,
    days_per_month: int = 30,
) -> float:
    """Forecast provider spend from explicit, reviewable assumptions."""

    if daily_requests < 0 or input_tokens_per_request < 0 or output_tokens_per_request < 0:
        raise ValueError("request and token assumptions cannot be negative")
    if not 0 <= fallback_rate <= 1:
        raise ValueError("fallback rate must be between 0 and 1")
    if days_per_month <= 0:
        raise ValueError("days per month must be positive")
    primary_share = 1 - fallback_rate
    input_cost = (
        primary_share * primary_input_eur_per_million
        + fallback_rate * fallback_input_eur_per_million
    ) * input_tokens_per_request / 1_000_000
    output_cost = (
        primary_share * primary_output_eur_per_million
        + fallback_rate * fallback_output_eur_per_million
    ) * output_tokens_per_request / 1_000_000
    return round(daily_requests * days_per_month * (input_cost + output_cost), 6)


def build_report(
    samples: list[RequestSample],
    *,
    threshold_ms: float,
    daily_requests: int,
    provider_cost_eur: float,
    gcp_cost_eur: float,
    gcp_budget_eur: float,
    assumptions: dict[str, object],
) -> dict[str, object]:
    """Create a JSON-safe release-evidence report with no request content."""

    successful = [sample for sample in samples if 200 <= sample.status_code < 300]
    fallback_model = assumptions.get("fallback_model")
    fallback = [sample for sample in successful if sample.model == fallback_model]
    failed = [sample for sample in samples if sample not in successful]
    overall = latency_summary(successful)
    return {
        "schema_version": "performance-evidence-v1",
        "contract": "Versioned performance report linked to release evidence.",
        "request_count": len(samples),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "user_count": len({sample.user_index for sample in samples}),
        "threshold_ms": threshold_ms,
        "normal_mix": overall,
        "fallback_mix": latency_summary(fallback),
        "passed": bool(successful)
        and not failed
        and overall["p95_ms"] <= threshold_ms,
        "cost_envelope": {
            "daily_requests": daily_requests,
            "provider_forecast_eur_per_month": provider_cost_eur,
            "gcp_forecast_eur_per_month": gcp_cost_eur,
            "combined_forecast_eur_per_month": round(provider_cost_eur + gcp_cost_eur, 6),
            "gcp_budget_eur_per_month": gcp_budget_eur,
            "combined_forecast_within_gcp_budget": provider_cost_eur + gcp_cost_eur <= gcp_budget_eur,
        },
        "assumptions": assumptions,
    }
