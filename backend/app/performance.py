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
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


FUNCTIONAL_OUTCOMES = frozenset({"answer", "qualified", "refusal", "evidence_refusal"})


def is_functional_success(sample: RequestSample) -> bool:
    """Return whether a complete typed response is a functional success."""

    return 200 <= sample.status_code < 300 and sample.outcome in FUNCTIONAL_OUTCOMES


def _outcome_counts(samples: Iterable[RequestSample]) -> dict[str, int]:
    counts = {outcome: 0 for outcome in (
        "answer", "qualified", "refusal", "evidence_refusal", "operational_error",
    )}
    for sample in samples:
        if sample.outcome in counts:
            counts[sample.outcome] += 1
    return counts


def _usage_summary(samples: Iterable[RequestSample]) -> dict[str, object]:
    values = list(samples)
    known = [sample for sample in values if sample.input_tokens is not None and sample.output_tokens is not None]
    input_tokens = sum(sample.input_tokens or 0 for sample in known)
    output_tokens = sum(sample.output_tokens or 0 for sample in known)
    return {
        "status": "unknown" if not known else "complete" if len(known) == len(values) else "partial",
        "known_request_count": len(known),
        "unknown_request_count": len(values) - len(known),
        "input_tokens": input_tokens if known else None,
        "output_tokens": output_tokens if known else None,
        "total_tokens": input_tokens + output_tokens if known else None,
    }


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
    provider_cost_eur: float | None,
    gcp_cost_eur: float,
    gcp_budget_eur: float,
    assumptions: dict[str, object],
    candidate_identity: dict[str, str] | None = None,
) -> dict[str, object]:
    """Create a JSON-safe release-evidence report with no request content."""

    successful = [sample for sample in samples if is_functional_success(sample)]
    fallback_model = assumptions.get("fallback_model")
    fallback = [sample for sample in successful if sample.model == fallback_model]
    failed = [sample for sample in samples if not is_functional_success(sample)]
    overall = latency_summary(successful)
    outcome_counts = _outcome_counts(samples)
    usage = _usage_summary(samples)
    answer_success_count = outcome_counts["answer"] + outcome_counts["qualified"]
    within_threshold_count = sum(sample.duration_ms <= threshold_ms for sample in successful)
    provider_cost_known = provider_cost_eur is not None and usage["status"] == "complete"
    combined_forecast = (
        round(provider_cost_eur + gcp_cost_eur, 6) if provider_cost_eur is not None else None
    )
    return {
        "schema_version": "performance-evidence-v1",
        "contract": "Versioned performance report linked to release evidence.",
        "request_count": len(samples),
        "successful_count": len(successful),
        "functional_success_count": len(successful),
        "answer_success_count": answer_success_count,
        "failed_count": len(failed),
        "operational_error_count": len(failed),
        "outcome_counts": outcome_counts,
        "user_count": len({sample.user_index for sample in samples}),
        "threshold_ms": threshold_ms,
        "normal_mix": overall,
        "fallback_mix": latency_summary(fallback),
        "within_threshold_count": within_threshold_count,
        "within_threshold_rate": round(within_threshold_count / len(successful), 6) if successful else None,
        "usage": usage,
        "candidate_identity": candidate_identity,
        "passed": bool(successful)
        and not failed
        and overall["p95_ms"] <= threshold_ms,
        "cost_envelope": {
            "daily_requests": daily_requests,
            "provider_forecast_eur_per_month": provider_cost_eur,
            "provider_cost_status": "known" if provider_cost_known else "unknown",
            "gcp_forecast_eur_per_month": gcp_cost_eur,
            "combined_forecast_eur_per_month": combined_forecast,
            "gcp_budget_eur_per_month": gcp_budget_eur,
            "combined_forecast_within_gcp_budget": (
                combined_forecast is not None and combined_forecast <= gcp_budget_eur
            ),
        },
        "assumptions": assumptions,
    }
