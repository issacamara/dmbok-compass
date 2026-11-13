#!/usr/bin/env python3
"""Run the three-user latency probe against a deployed API.

The output contains aggregate performance and cost evidence only. Bearer
tokens are accepted on the command line but are never written to the report.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import re
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import httpx

from app.performance import (
    RequestSample,
    build_report,
    forecast_provider_cost_eur,
    load_authoritative_release_cost_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="API origin, for example https://api.example.com")
    parser.add_argument("--token", action="append", required=True, help="Bearer token; repeat exactly three times")
    parser.add_argument("--requests-per-user", type=int, default=10)
    parser.add_argument("--question", default="What is data governance?")
    parser.add_argument("--expected-outcome", choices=("answer", "qualified", "refusal"), default="answer")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--threshold-ms", type=float, default=30_000)
    parser.add_argument("--daily-requests", type=int, default=100)
    parser.add_argument("--input-tokens-per-request", type=int, default=800)
    parser.add_argument("--output-tokens-per-request", type=int, default=300)
    parser.add_argument("--fallback-model", default="google/gemma-4-26b-a4b-it:free")
    parser.add_argument("--gcp-budget-eur", type=float, default=5.0)
    parser.add_argument("--candidate-artifact-id")
    parser.add_argument("--candidate-installed-lock-sha256")
    parser.add_argument("--candidate-corpus-version-id")
    parser.add_argument("--candidate-configuration-version-id")
    parser.add_argument("--candidate-scorer-version")
    parser.add_argument("--quality-evidence", type=Path)
    parser.add_argument("--generation-evidence", type=Path)
    parser.add_argument("--stage-usage-evidence", type=Path)
    parser.add_argument("--output", type=Path, default=Path("performance-evidence.json"))
    return parser.parse_args()


async def probe_user(
    client: httpx.AsyncClient,
    user_index: int,
    token: str,
    question: str,
    requests_per_user: int,
    expected_outcome: str,
    user_subject: str,
) -> list[RequestSample]:
    samples: list[RequestSample] = []
    for request_index in range(requests_per_user):
        started = perf_counter()
        try:
            response = await client.post(
                "/api/questions",
                headers={"Authorization": f"Bearer {token}"},
                json={"question": question},
            )
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            trace = payload.get("trace") or {}
            usage = payload.get("usage") or trace.get("usage") or {}
            samples.append(
                RequestSample(
                    user_index=user_index,
                    status_code=response.status_code,
                    duration_ms=round((perf_counter() - started) * 1000, 3),
                    outcome=payload.get("outcome"),
                    model=trace.get("selected_model"),
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    total_tokens=usage.get("total_tokens"),
                    first_request=request_index == 0,
                    expected_outcome=expected_outcome,
                    user_subject=user_subject,
                )
            )
        except Exception as exc:  # noqa: BLE001 - report aggregate failure class only
            samples.append(
                RequestSample(
                    user_index=user_index,
                    status_code=599,
                    duration_ms=round((perf_counter() - started) * 1000, 3),
                    outcome=None,
                    model=None,
                    error_class=type(exc).__name__,
                    first_request=request_index == 0,
                    expected_outcome=expected_outcome,
                    user_subject=user_subject,
                )
            )
    return samples


async def run_probe(args: argparse.Namespace) -> dict[str, object]:
    if len(args.token) != 3:
        raise ValueError("provide exactly three --token values to exercise the three-user contract")
    if any(not token.strip() for token in args.token):
        raise ValueError("provide three non-empty user tokens")
    subjects = [_jwt_subject(token) for token in args.token]
    if len(set(subjects)) != 3:
        raise ValueError("the three tokens must belong to three distinct users")
    if args.requests_per_user <= 0:
        raise ValueError("--requests-per-user must be positive")
    if args.gcp_budget_eur > 5.0:
        raise ValueError("the GCP monthly budget cannot exceed the approved €5 ceiling")
    candidate_values = (
        args.candidate_artifact_id,
        args.candidate_corpus_version_id,
        args.candidate_configuration_version_id,
        args.candidate_scorer_version,
    )
    if any(value is not None and not _safe_identifier(value) for value in candidate_values):
        raise ValueError("candidate identity fields must use safe identifier characters")
    if args.candidate_installed_lock_sha256 is not None and re.fullmatch(
        r"[a-f0-9]{64}", args.candidate_installed_lock_sha256
    ) is None:
        raise ValueError("candidate lock digest must be a SHA-256 hex value")
    candidate_identity = {
        "artifact_id": args.candidate_artifact_id,
        "installed_lock_sha256": args.candidate_installed_lock_sha256,
        "corpus_version_id": args.candidate_corpus_version_id,
        "configuration_version_id": args.candidate_configuration_version_id,
        "scorer_version": args.candidate_scorer_version,
    } if all((
        args.candidate_artifact_id,
        args.candidate_installed_lock_sha256,
        args.candidate_corpus_version_id,
        args.candidate_configuration_version_id,
        args.candidate_scorer_version,
    )) else None
    cost_evidence = load_authoritative_release_cost_record(
        Path(__file__).resolve().parents[1], candidate_identity,
    )
    quality_evidence = _safe_evidence(args.quality_evidence, "quality")
    generation_coverage = _safe_evidence(args.generation_evidence, "generation")
    stage_usage = _safe_evidence(args.stage_usage_evidence, "stages")
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=args.timeout_seconds) as client:
        profiles = await asyncio.gather(*(
            _verify_probe_user(client, token, subject)
            for token, subject in zip(args.token, subjects, strict=True)
        ))
        if not all(profiles):
            raise ValueError("all probe users must be authenticated, email-verified, and approved")
        groups = await asyncio.gather(
            *(probe_user(
                client, index, token, args.question, args.requests_per_user, args.expected_outcome, subject
            ) for index, (token, subject) in enumerate(zip(args.token, subjects, strict=True)))
        )
    samples = [sample for group in groups for sample in group]
    fallback_rate = sum(sample.model == args.fallback_model for sample in samples) / len(samples)
    rates = cost_evidence["provider_rates"] if cost_evidence else None
    provider_cost = forecast_provider_cost_eur(
        daily_requests=args.daily_requests,
        input_tokens_per_request=args.input_tokens_per_request,
        output_tokens_per_request=args.output_tokens_per_request,
        fallback_rate=fallback_rate,
        primary_input_eur_per_million=rates["primary_input_eur_per_million"],
        primary_output_eur_per_million=rates["primary_output_eur_per_million"],
        fallback_input_eur_per_million=rates["fallback_input_eur_per_million"],
        fallback_output_eur_per_million=rates["fallback_output_eur_per_million"],
    ) if rates else None
    return build_report(
        samples,
        threshold_ms=args.threshold_ms,
        daily_requests=args.daily_requests,
        provider_cost_eur=provider_cost,
        gcp_cost_eur=cost_evidence["gcp_forecast_eur"] if cost_evidence else None,
        gcp_budget_eur=args.gcp_budget_eur,
        assumptions={
            "base_url": _redacted_base_url(args.base_url),
            "requests_per_user": args.requests_per_user,
            "expected_outcome": args.expected_outcome,
            "fallback_model": args.fallback_model,
            "fallback_rate": round(fallback_rate, 6),
            "input_tokens_per_request": args.input_tokens_per_request,
            "output_tokens_per_request": args.output_tokens_per_request,
        },
        candidate_identity=candidate_identity,
        cost_evidence=cost_evidence,
        quality_evidence=quality_evidence,
        generation_coverage=generation_coverage,
        stage_usage=stage_usage,
    )


def _jwt_subject(token: str) -> str:
    """Read the JWT subject in memory; the API remains authoritative for token validation."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError
        payload = parts[1]
        decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        claims = json.loads(decoded)
        subject = claims.get("sub") if isinstance(claims, dict) else None
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError
        return subject
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("each token must contain a valid subject claim") from None


async def _verify_probe_user(client: httpx.AsyncClient, token: str, subject: str) -> bool:
    """Ask the API to authenticate and authorize each token before sending probe traffic."""
    try:
        response = await client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
        if response.status_code != 200:
            return False
        profile = response.json()
        return (
            isinstance(profile, dict)
            and profile.get("user_id") == subject
            and profile.get("approval_state") == "approved"
            and profile.get("email_verified") is True
        )
    except Exception:  # noqa: BLE001 - fail closed without exposing token or identity data
        return False


def _safe_evidence(path: Path | None, kind: str) -> dict[str, object] | None:
    """Read and validate a content-free summary, dropping free-form text fields."""
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} evidence must be a JSON object")
    if kind == "quality":
        candidate = payload.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError("quality evidence must identify its frozen candidate")
        candidate_keys = {
            "artifact_id", "installed_lock_sha256", "corpus_version_id",
            "configuration_version_id", "scorer_version",
        }
        if set(candidate) != candidate_keys or any(
            not isinstance(value, str) or not _safe_identifier(value) for value in candidate.values()
        ):
            raise ValueError("quality candidate identity is invalid")
        metrics = payload.get("metrics")
        expected_metrics = {
            "retrieval_success", "grounded_claims", "citation_correctness",
            "answer_quality", "refusal_correctness", "response_time",
        }
        if not isinstance(metrics, dict) or set(metrics) != expected_metrics:
            raise ValueError("quality evidence must include the six release metrics")
        safe_metrics: dict[str, dict[str, object]] = {}
        thresholds = {
            "retrieval_success": 90, "grounded_claims": 95,
            "citation_correctness": 95, "answer_quality": 85,
            "refusal_correctness": 95, "response_time": 95,
        }
        for name, metric in metrics.items():
            if not isinstance(metric, dict) or set(metric) != {"numerator", "denominator", "percentage", "threshold", "passed"}:
                raise ValueError("quality metric summary has an invalid shape")
            if (
                type(metric["numerator"]) is not int or type(metric["denominator"]) is not int
                or metric["denominator"] <= 0 or type(metric["percentage"]) not in (int, float)
                or type(metric["threshold"]) not in (int, float) or type(metric["passed"]) is not bool
                or not math.isfinite(metric["percentage"]) or not math.isfinite(metric["threshold"])
                or metric["threshold"] != thresholds[name]
                or metric["percentage"] != round(100 * metric["numerator"] / metric["denominator"], 2)
                or metric["passed"] != (metric["percentage"] >= metric["threshold"])
            ):
                raise ValueError("quality metric summary contains invalid values")
            safe_metrics[name] = dict(metric)
        held_out = payload.get("held_out_digest")
        tuning = payload.get("tuning_digest")
        if not isinstance(held_out, str) or not isinstance(tuning, str) or held_out == tuning:
            raise ValueError("quality evidence must have distinct held-out and tuning digests")
        if any(re.fullmatch(r"[a-f0-9]{64}", value) is None for value in (held_out, tuning)):
            raise ValueError("quality split digests must be SHA-256 hex values")
        # This repository has no authenticated split manifest binding these digests to
        # this run and dataset, so imported JSON cannot elevate the evidence to verified.
        split_binding_status = "unverified"
        overlap = payload.get("model_overlap")
        if overlap is not None and (
            not isinstance(overlap, dict)
            or any(not _safe_identifier(key) or type(value) is not bool for key, value in overlap.items())
        ):
            raise ValueError("quality model-overlap observations are invalid")
        return {
            "candidate": candidate,
            "held_out_digest": held_out,
            "tuning_digest": tuning,
            "metrics": safe_metrics,
            "split_binding_status": split_binding_status,
            "quality_metrics_passed": all(
                metric["passed"] for name, metric in safe_metrics.items() if name != "response_time"
            ),
            "model_overlap_status": "observed" if overlap else "unknown",
        }
    if kind == "generation":
        integer_fields = ("target_count", "generated_count", "rejected_count", "achievable_size", "usage_tokens")
        summary: dict[str, object] = {"status": payload.get("status")}
        if summary["status"] not in {"published", "awaiting_budget", "failed"}:
            raise ValueError("generation evidence status is invalid")
        for field in integer_fields:
            value = payload.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"generation evidence field {field} must be a non-negative integer")
            if value is not None:
                summary[field] = value
        for field in ("coverage_by_category", "coverage_by_outcome"):
            values = payload.get(field)
            if values is not None:
                if not isinstance(values, dict) or any(
                    not _safe_identifier(key) or type(value) is not int or value < 0
                    for key, value in values.items()
                ):
                    raise ValueError(f"generation evidence field {field} is invalid")
                summary[field] = values
        return summary
    stages = payload
    result: dict[str, object] = {}
    if set(stages) != {"parser", "generator", "judge"}:
        raise ValueError("stage usage must report parser, generator, and judge")
    for stage_name, stage in stages.items():
        if not isinstance(stage, dict) or set(stage) != {
            "status", "used_units", "reserved_units", "cost_eur", "reserved_cost_eur",
        }:
            raise ValueError("stage usage summary has an invalid shape")
        if stage["status"] not in {"complete", "unknown", "blocked"}:
            raise ValueError("stage usage status is invalid")
        if stage["status"] == "complete":
            if any(type(stage[key]) is not int or stage[key] < 0 for key in ("used_units", "reserved_units")):
                raise ValueError("complete stage usage requires non-negative units")
            if any(
                type(stage[key]) not in (int, float) or not math.isfinite(stage[key]) or stage[key] < 0
                for key in ("cost_eur", "reserved_cost_eur")
            ):
                raise ValueError("complete stage usage requires finite non-negative EUR costs")
        elif any(stage[key] is not None for key in (
            "used_units", "reserved_units", "cost_eur", "reserved_cost_eur",
        )):
            raise ValueError("unknown stage usage must not claim measured units or costs")
        result[stage_name] = dict(stage)
    return result


def _safe_identifier(value: str) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 256
        and re.fullmatch(r"[A-Za-z0-9._:/+-]+", value) is not None
    )


def _redacted_base_url(value: str) -> str:
    scheme = urlsplit(value).scheme
    return f"{scheme}://[redacted]" if scheme in {"http", "https"} else "[redacted]"


def main() -> int:
    args = parse_args()
    try:
        report = asyncio.run(run_probe(args))
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"performance probe failed: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["release_evidence_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
