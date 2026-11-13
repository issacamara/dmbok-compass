"""Content-free performance evidence and cost-envelope calculations."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from statistics import mean
from typing import Iterable
from urllib.parse import urlsplit


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
    first_request: bool = False
    expected_outcome: str = "answer"
    # Retained only in memory for distinct-user aggregation.
    user_subject: str | None = None


FUNCTIONAL_OUTCOMES = frozenset({"answer", "qualified", "refusal"})
AUTHORITY_RECORD_PATH = "documents/decisions/release-cost-approval.json"
PROJECT_ORIGIN = "https://github.com/issacamara/dmbok-compass.git"
AUTHORITY_REF = "refs/remotes/origin/development"
_COST_RECORD_TOKEN = object()


class VerifiedCostEvidence(dict[str, object]):
    """Cost record read from the tracked project decision file at tested HEAD."""

    def __init__(self, payload: dict[str, object], *, source_commit: str):
        super().__init__(payload)
        self.source_commit = source_commit
        self._verification_token = _COST_RECORD_TOKEN

    def is_verified_for(self, candidate_identity: dict[str, str] | None) -> bool:
        return (
            self._verification_token is _COST_RECORD_TOKEN
            and re.fullmatch(r"[a-f0-9]{40,64}", self.source_commit) is not None
            and self.get("candidate_identity") == candidate_identity
            and self.get("authority_path") == AUTHORITY_RECORD_PATH
            and self.get("authority_ref") == AUTHORITY_REF
        )


def load_authoritative_release_cost_record(
    repo_root: Path,
    candidate_identity: dict[str, str] | None,
) -> VerifiedCostEvidence | None:
    """Load the owner-reviewed decision record from trusted development history."""
    if candidate_identity is None:
        return None
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if remote != PROJECT_ORIGIN:
            return None
        source_commit = subprocess.run(
            ["git", "rev-parse", "--verify", f"{AUTHORITY_REF}^{{commit}}"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        )
        raw = subprocess.run(
            ["git", "show", f"{source_commit}:{AUTHORITY_RECORD_PATH}"], cwd=repo_root,
            check=True, capture_output=True, text=True,
        ).stdout
        record = json.loads(raw)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or set(record) != {
        "schema_version", "candidate_identity", "sponsor_approval", "provider_rates", "gcp_forecast",
    } or record["schema_version"] != "release-cost-approval-v1":
        return None
    if record["candidate_identity"] != candidate_identity:
        return None
    approval = record["sponsor_approval"]
    if not isinstance(approval, dict) or set(approval) != {
        "approval_reference", "approved_by", "approved_at", "provider", "monthly_cap_eur",
    }:
        return None
    if not _authoritative_reference(approval["approval_reference"], "github.com", r"/issacamara/dmbok-compass/issues/[0-9]+"):
        return None
    if not all(isinstance(approval[name], str) and approval[name] for name in ("approved_by", "provider")):
        return None
    if not _valid_timestamp(approval["approved_at"]):
        return None
    if not _valid_money(approval["monthly_cap_eur"]):
        return None
    provider = record["provider_rates"]
    rate_fields = (
        "primary_input_eur_per_million", "primary_output_eur_per_million",
        "fallback_input_eur_per_million", "fallback_output_eur_per_million",
    )
    if not isinstance(provider, dict) or set(provider) != {
        "source_reference", "source_timestamp", "currency", *rate_fields,
    }:
        return None
    if not _authoritative_reference(provider["source_reference"], "openrouter.ai", r"/.*"):
        return None
    if provider["currency"] != "EUR" or not _valid_timestamp(provider["source_timestamp"]):
        return None
    if any(not _valid_money(provider[field]) for field in rate_fields) or not any(provider[field] > 0 for field in rate_fields):
        return None
    gcp = record["gcp_forecast"]
    if not isinstance(gcp, dict) or set(gcp) != {"source_reference", "source_timestamp", "monthly_cost_eur"}:
        return None
    if not _authoritative_reference(gcp["source_reference"], "cloud.google.com", r"/.*"):
        return None
    if not _valid_timestamp(gcp["source_timestamp"]) or not _valid_money(gcp["monthly_cost_eur"]):
        return None
    return VerifiedCostEvidence(
        {
            "authority_path": AUTHORITY_RECORD_PATH,
            "authority_ref": AUTHORITY_REF,
            "candidate_identity": candidate_identity,
            "sponsor_approval": approval,
            "provider_rates": {field: provider[field] for field in rate_fields},
            "provider_rates_source": provider["source_reference"],
            "provider_rates_timestamp": provider["source_timestamp"],
            "gcp_forecast_eur": gcp["monthly_cost_eur"],
            "gcp_forecast_source": gcp["source_reference"],
            "gcp_forecast_timestamp": gcp["source_timestamp"],
        },
        source_commit=source_commit,
    )


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _valid_money(value: object) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0
    )


def _authoritative_reference(value: object, hostname: str, path_pattern: str) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and parsed.hostname == hostname
        and re.fullmatch(path_pattern, parsed.path) is not None
        and not parsed.username
        and not parsed.password
    )


def is_functional_success(sample: RequestSample) -> bool:
    """Return whether a complete typed response is a functional success."""

    return (
        200 <= sample.status_code < 300
        and sample.expected_outcome in FUNCTIONAL_OUTCOMES
        and sample.outcome == sample.expected_outcome
    )


def _outcome_counts(samples: Iterable[RequestSample]) -> dict[str, int]:
    counts = {outcome: 0 for outcome in (
        "answer", "qualified", "refusal", "evidence_refusal", "operational_error",
    )}
    for sample in samples:
        if isinstance(sample.outcome, str) and sample.outcome in counts:
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
    rates = (
        primary_input_eur_per_million,
        primary_output_eur_per_million,
        fallback_input_eur_per_million,
        fallback_output_eur_per_million,
    )
    if any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError("provider rates must be finite and non-negative")
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
    gcp_cost_eur: float | None,
    gcp_budget_eur: float,
    assumptions: dict[str, object],
    candidate_identity: dict[str, str] | None = None,
    cost_evidence: dict[str, object] | None = None,
    generation_coverage: dict[str, object] | None = None,
    stage_usage: dict[str, dict[str, object]] | None = None,
    quality_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create a JSON-safe release-evidence report with no request content."""

    successful = [sample for sample in samples if is_functional_success(sample)]
    fallback_model = assumptions.get("fallback_model")
    fallback = [sample for sample in successful if sample.model == fallback_model]
    operational_failures = [
        sample for sample in samples
        if not (200 <= sample.status_code < 300)
        or not isinstance(sample.outcome, str)
        or sample.outcome not in FUNCTIONAL_OUTCOMES | {"evidence_refusal", "operational_error"}
        or sample.outcome == "operational_error"
    ]
    evidence_refusals = [sample for sample in samples if sample.outcome == "evidence_refusal"]
    outcome_mismatches = [
        sample for sample in samples
        if 200 <= sample.status_code < 300
        and isinstance(sample.outcome, str)
        and sample.outcome in FUNCTIONAL_OUTCOMES
        and sample.outcome != sample.expected_outcome
    ]
    failed_requests = [
        sample for sample in samples
        if sample in operational_failures or sample in outcome_mismatches or sample in evidence_refusals
    ]
    overall = latency_summary(successful)
    first_requests = [sample for sample in samples if sample.first_request]
    outcome_counts = _outcome_counts(samples)
    usage = _usage_summary(samples)
    answer_success_count = outcome_counts["answer"] + outcome_counts["qualified"]
    within_threshold_count = sum(sample.duration_ms <= threshold_ms for sample in successful)
    within_30_seconds_count = sum(sample.duration_ms <= 30_000 for sample in successful)
    verified_cost_record = (
        isinstance(cost_evidence, VerifiedCostEvidence)
        and cost_evidence.is_verified_for(candidate_identity)
    )
    if verified_cost_record:
        rates = cost_evidence["provider_rates"]
        fallback_model = assumptions.get("fallback_model")
        fallback_rate = (
            sum(sample.model == fallback_model for sample in samples) / len(samples)
            if samples else 0.0
        )
        input_tokens = assumptions.get("input_tokens_per_request")
        output_tokens = assumptions.get("output_tokens_per_request")
        if type(input_tokens) is int and type(output_tokens) is int:
            provider_cost_eur = forecast_provider_cost_eur(
                daily_requests=daily_requests,
                input_tokens_per_request=input_tokens,
                output_tokens_per_request=output_tokens,
                fallback_rate=fallback_rate,
                primary_input_eur_per_million=rates["primary_input_eur_per_million"],
                primary_output_eur_per_million=rates["primary_output_eur_per_million"],
                fallback_input_eur_per_million=rates["fallback_input_eur_per_million"],
                fallback_output_eur_per_million=rates["fallback_output_eur_per_million"],
            )
        else:
            provider_cost_eur = None
        gcp_cost_eur = cost_evidence["gcp_forecast_eur"]
    else:
        provider_cost_eur = None
        gcp_cost_eur = None
    provider_cost_known = (
        verified_cost_record and provider_cost_eur is not None and usage["status"] == "complete"
    )
    combined_forecast = (
        round(provider_cost_eur + gcp_cost_eur, 6)
        if provider_cost_eur is not None and gcp_cost_eur is not None else None
    )
    if not math.isfinite(gcp_budget_eur) or (
        gcp_cost_eur is not None and (not math.isfinite(gcp_cost_eur) or gcp_cost_eur < 0)
    ):
        raise ValueError("GCP forecast and budget must be finite and non-negative")
    if not 0 <= gcp_budget_eur <= 5.0:
        raise ValueError("GCP monthly budget cannot exceed the approved €5 ceiling")
    cost_evidence_status = "verified" if verified_cost_record else "missing_or_unverified"
    gcp_budget_passed = (
        verified_cost_record and gcp_cost_eur is not None and gcp_cost_eur <= gcp_budget_eur
    )
    approval = cost_evidence.get("sponsor_approval") if verified_cost_record else None
    approved_cap = approval.get("monthly_cap_eur") if isinstance(approval, dict) else None
    provider_cap_status = (
        "missing_authoritative_record" if not verified_cost_record
        else "unknown" if not provider_cost_known
        else "passed" if provider_cost_eur <= approved_cap
        else "failed"
    )
    errors_by_class: dict[str, int] = {}
    for sample in operational_failures:
        key = _safe_error_class(sample.error_class) or f"http_{sample.status_code}"
        errors_by_class[key] = errors_by_class.get(key, 0) + 1
    coverage = generation_coverage or {}
    allowances = stage_usage or {}
    stages = {name: allowances.get(name) for name in ("parser", "generator", "judge")}
    usage_complete = all(
        isinstance(stage, dict)
        and stage.get("status") == "complete"
        and stage.get("used_units") is not None
        and stage.get("reserved_units") is not None
        and stage.get("cost_eur") is not None
        and stage.get("reserved_cost_eur") is not None
        for stage in stages.values()
    )
    stage_costs = _stage_cost_summary(stages)
    allowance_complete = usage_complete and stage_costs["status"] == "complete"
    from app.retrieval.benchmark import ReviewedQualityEvidence

    quality_passed = (
        isinstance(quality_evidence, ReviewedQualityEvidence)
        and quality_evidence.is_bound_to(candidate_identity)
        and quality_evidence.get("quality_metrics_passed") is True
        and quality_evidence.get("model_overlap_status") in {"observed", "unknown"}
    )
    generation_passed = (
        coverage.get("status") == "published"
        and type(coverage.get("target_count")) is int
        and coverage.get("target_count", 0) > 0
        and coverage.get("achievable_size") == coverage.get("target_count")
        and coverage.get("generated_count") == coverage.get("target_count")
    )
    within_15_seconds_count = sum(sample.duration_ms <= 15_000 for sample in successful)
    verified_user_count = len({sample.user_subject for sample in samples if sample.user_subject})
    return {
        "schema_version": "performance-evidence-v1",
        "contract": "Versioned performance report linked to release evidence.",
        "request_count": len(samples),
        "successful_count": len(successful),
        "functional_success_count": len(successful),
        "answer_success_count": answer_success_count,
        "failed_count": len(failed_requests),
        "failed_denominator": len(samples),
        "operational_error_count": len(operational_failures),
        "operational_failure_count": len(operational_failures),
        "operational_failure_denominator": len(samples),
        "operational_failure_rate": round(len(operational_failures) / len(samples), 6) if samples else None,
        "evidence_refusal_count": len(evidence_refusals),
        "evidence_refusal_denominator": len(samples),
        "evidence_refusal_rate": round(len(evidence_refusals) / len(samples), 6) if samples else None,
        "outcome_mismatch_count": len(outcome_mismatches),
        "outcome_mismatch_denominator": len(samples),
        "outcome_counts": outcome_counts,
        "user_count": verified_user_count,
        "threshold_ms": threshold_ms,
        "release_latency_threshold_ms": 30_000,
        "normal_mix": overall,
        "fallback_mix": latency_summary(fallback),
        "first_request_diagnostic": {
            "status": "unverified_cold_start_proxy",
            "samples": latency_summary(first_requests),
            "note": "First request per user is not proof that Cloud Run started a cold instance.",
        },
        "within_threshold_count": within_threshold_count,
        "within_threshold_rate": round(within_threshold_count / len(successful), 6) if successful else None,
        "within_30_seconds_count": within_30_seconds_count,
        "within_30_seconds_denominator": len(samples),
        "within_30_seconds_rate": round(within_30_seconds_count / len(samples), 6) if samples else None,
        "within_15_seconds_diagnostic_count": within_15_seconds_count,
        "within_15_seconds_diagnostic_rate": round(within_15_seconds_count / len(successful), 6) if successful else None,
        "errors_by_class": dict(sorted(errors_by_class.items())),
        "usage": usage,
        "candidate_identity": candidate_identity,
        "latency_passed": (
            verified_user_count == 3
            and bool(samples)
            and within_30_seconds_count / len(samples) >= 0.95
        ),
        "quality_evidence": quality_evidence,
        "generation_coverage": coverage,
        "stage_usage": stages,
        "release_evidence_status": "passed" if (
            quality_passed
            and candidate_identity is not None
            and generation_passed
            and allowance_complete
            and provider_cap_status == "passed"
            and cost_evidence_status == "verified"
            and gcp_budget_passed
            and verified_user_count == 3
            and bool(samples)
            and within_30_seconds_count / len(samples) >= 0.95
        ) else "blocked",
        "passed": (
            verified_user_count == 3
            and bool(samples)
            and within_30_seconds_count / len(samples) >= 0.95
        ),
        "cost_envelope": {
            "daily_requests": daily_requests,
            "provider_forecast_eur_per_month": provider_cost_eur,
            "provider_cost_status": "known" if provider_cost_known else "unknown",
            "provider_hard_cap_eur_per_month": approved_cap if verified_cost_record else None,
            "provider_cap_status": provider_cap_status,
            "provider_cap_evidence_status": "verified" if verified_cost_record else "missing",
            "gcp_forecast_eur_per_month": gcp_cost_eur,
            "gcp_forecast_status": cost_evidence_status,
            "cost_evidence_status": cost_evidence_status,
            "authority_record_path": cost_evidence.get("authority_path") if verified_cost_record else None,
            "authority_record_ref": cost_evidence.get("authority_ref") if verified_cost_record else None,
            "authority_record_commit": cost_evidence.source_commit if verified_cost_record else None,
            "combined_forecast_eur_per_month": combined_forecast,
            "gcp_budget_eur_per_month": gcp_budget_eur,
            "gcp_forecast_within_budget": gcp_budget_passed,
        },
        "stage_costs": stage_costs,
        "assumptions": assumptions,
    }


def _safe_error_class(value: str | None) -> str | None:
    """Keep only a conventional exception class token; discard arbitrary text."""
    if value is None or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", value) is None:
        return None
    return value


def _stage_cost_summary(stages: dict[str, object]) -> dict[str, object]:
    names = ("parser", "generator", "judge")
    valid = all(
        isinstance(stages.get(name), dict)
        and stages[name].get("status") == "complete"
        and isinstance(stages[name].get("cost_eur"), (int, float))
        and not isinstance(stages[name].get("cost_eur"), bool)
        and math.isfinite(stages[name]["cost_eur"])
        and stages[name]["cost_eur"] >= 0
        and isinstance(stages[name].get("reserved_cost_eur"), (int, float))
        and not isinstance(stages[name].get("reserved_cost_eur"), bool)
        and math.isfinite(stages[name]["reserved_cost_eur"])
        and stages[name]["reserved_cost_eur"] >= 0
        for name in names
    )
    if not valid:
        return {"status": "unknown", "used_cost_eur": None, "reserved_cost_eur": None}
    return {
        "status": "complete",
        "used_cost_eur": round(sum(stages[name]["cost_eur"] for name in names), 6),
        "reserved_cost_eur": round(sum(stages[name]["reserved_cost_eur"] for name in names), 6),
    }
