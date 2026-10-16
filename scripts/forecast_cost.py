#!/usr/bin/env python3
"""Produce a deterministic, candidate-bound migration cost forecast.

The input is a JSON document.  Prices and workload figures are explicit
assumptions rather than a billing export, so release evidence can distinguish
a forecast from observed spend.  The script never reads credentials or calls
external services.
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


MILLION = Decimal("1000000")
CENT = Decimal("0.01")


def decimal(value: object, name: str) -> Decimal:
    """Return a non-negative decimal without binary floating-point arithmetic."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return result


def money(value: Decimal) -> str:
    return format(value.quantize(CENT, rounding=ROUND_HALF_UP), "f")


def object_at(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def sum_costs(costs: dict[str, Any], name: str) -> Decimal:
    if not costs or not all(isinstance(key, str) and key for key in costs):
        raise ValueError(f"{name} must contain named cost items")
    return sum((decimal(value, f"{name}.{key}") for key, value in costs.items()), Decimal())


def model_cost(
    model: dict[str, Any], *, requests: Decimal, input_tokens: Decimal, output_tokens: Decimal, name: str
) -> Decimal:
    required = ("id", "input_eur_per_million", "output_eur_per_million")
    if not all(isinstance(model.get(key), str) and model[key] for key in ("id",)):
        raise ValueError(f"{name}.id must be a non-empty string")
    if not all(key in model for key in required):
        raise ValueError(f"{name} is missing pricing")
    return requests * (
        input_tokens * decimal(model["input_eur_per_million"], f"{name}.input_eur_per_million") / MILLION
        + output_tokens * decimal(model["output_eur_per_million"], f"{name}.output_eur_per_million") / MILLION
    )


def forecast(payload: dict[str, Any]) -> dict[str, Any]:
    """Calculate cost categories and release-gate outcomes from a JSON payload."""

    if payload.get("currency") != "EUR":
        raise ValueError("currency must be EUR")
    days = decimal(payload.get("days_per_month"), "days_per_month")
    if days <= 0:
        raise ValueError("days_per_month must be greater than zero")

    workload = object_at(payload, "workload")
    requests_per_day = decimal(workload.get("daily_answer_requests"), "workload.daily_answer_requests")
    input_tokens = decimal(workload.get("input_tokens_per_request"), "workload.input_tokens_per_request")
    output_tokens = decimal(workload.get("output_tokens_per_request"), "workload.output_tokens_per_request")
    fallback_rate = decimal(workload.get("fallback_rate"), "workload.fallback_rate")
    if fallback_rate > 1:
        raise ValueError("workload.fallback_rate must be at most one")

    provider = object_at(payload, "provider")
    primary = object_at(provider, "primary")
    fallback = object_at(provider, "fallback")
    monthly_requests = days * requests_per_day
    fallback_requests = monthly_requests * fallback_rate
    primary_requests = monthly_requests - fallback_requests
    primary_cost = model_cost(
        primary,
        requests=primary_requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        name="provider.primary",
    )
    fallback_cost = model_cost(
        fallback,
        requests=fallback_requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        name="provider.fallback",
    )
    provider_total = primary_cost + fallback_cost

    rebuild = object_at(payload, "one_time_rebuild")
    embedding_tokens = decimal(rebuild.get("embedding_tokens"), "one_time_rebuild.embedding_tokens")
    embedding_rate = decimal(
        rebuild.get("embedding_eur_per_million"), "one_time_rebuild.embedding_eur_per_million"
    )
    rebuild_compute = decimal(rebuild.get("job_compute_eur"), "one_time_rebuild.job_compute_eur")
    one_time_total = embedding_tokens * embedding_rate / MILLION + rebuild_compute

    temporary_total = sum_costs(
        object_at(payload, "temporary_migration_monthly"), "temporary_migration_monthly"
    )
    runtime_total = sum_costs(object_at(payload, "monthly_gcp_runtime"), "monthly_gcp_runtime")
    gcp_monthly_total = temporary_total + runtime_total
    gcp_first_month_total = one_time_total + gcp_monthly_total
    gcp_budget = decimal(payload.get("gcp_monthly_budget_eur"), "gcp_monthly_budget_eur")

    hard_cap_raw = provider.get("hard_cap_eur")
    provider_cap = None if hard_cap_raw is None else decimal(hard_cap_raw, "provider.hard_cap_eur")
    provider_gate = (
        "passed"
        if provider_cap is not None
        and Decimal(money(provider_total)) <= Decimal(money(provider_cap))
        else "blocked"
    )
    provider_reason = (
        "within recorded sponsor hard cap"
        if provider_gate == "passed"
        else "missing recorded sponsor hard cap"
        if provider_cap is None
        else "forecast exceeds recorded sponsor hard cap"
    )
    gcp_gate = (
        "passed"
        if Decimal(money(gcp_monthly_total)) <= Decimal(money(gcp_budget))
        else "blocked"
    )

    return {
        "currency": "EUR",
        "assumptions": {
            "days_per_month": str(days),
            "daily_answer_requests": str(requests_per_day),
            "input_tokens_per_request": str(input_tokens),
            "output_tokens_per_request": str(output_tokens),
            "fallback_rate": str(fallback_rate),
            "primary_model": primary["id"],
            "fallback_model": fallback["id"],
        },
        "costs_eur": {
            "one_time_rebuild": money(one_time_total),
            "temporary_migration_monthly": money(temporary_total),
            "monthly_gcp_runtime": money(runtime_total),
            "monthly_gcp_during_migration": money(gcp_monthly_total),
            "monthly_hosted_models": money(provider_total),
            "first_migration_month_gcp": money(gcp_first_month_total),
        },
        "provider_mix_eur": {
            "primary": money(primary_cost),
            "fallback": money(fallback_cost),
            "hard_cap": None if provider_cap is None else money(provider_cap),
        },
        "release_gates": {
            "gcp_monthly_budget": {
                "status": gcp_gate,
                "forecast_eur": money(gcp_monthly_total),
                "budget_eur": money(gcp_budget),
            },
            "provider_hard_cap": {
                "status": provider_gate,
                "forecast_eur": money(provider_total),
                "reason": provider_reason,
            },
        },
    }


def example_payload() -> dict[str, Any]:
    """Return the documented candidate scenario; it is not a billing record."""

    return {
        "currency": "EUR",
        "days_per_month": 30,
        "workload": {
            "daily_answer_requests": 100,
            "input_tokens_per_request": 800,
            "output_tokens_per_request": 300,
            "fallback_rate": "0.05",
        },
        "provider": {
            "primary": {
                "id": "mistralai/mistral-nemo",
                "input_eur_per_million": "0.029",
                "output_eur_per_million": "0.030",
            },
            "fallback": {
                "id": "meta-llama/llama-3.1-8b-instruct",
                "input_eur_per_million": "0.050",
                "output_eur_per_million": "0.080",
            },
            "hard_cap_eur": None,
        },
        "one_time_rebuild": {
            "embedding_tokens": 1500000,
            "embedding_eur_per_million": "0.15",
            "job_compute_eur": "0.35",
        },
        "temporary_migration_monthly": {
            "duplicate_firestore_storage_eur": "0.12",
            "duplicate_vector_index_eur": "0.18",
            "candidate_artifact_storage_eur": "0.02",
        },
        "monthly_gcp_runtime": {
            "cloud_run_512mib_cold_start_and_runtime_eur": "0.65",
            "firestore_reads_writes_eur": "0.28",
            "logging_and_monitoring_eur": "0.10",
        },
        "gcp_monthly_budget_eur": "5.00",
    }


def self_test() -> None:
    report = forecast(example_payload())
    assert report["costs_eur"] == {
        "one_time_rebuild": "0.58",
        "temporary_migration_monthly": "0.32",
        "monthly_gcp_runtime": "1.03",
        "monthly_gcp_during_migration": "1.35",
        "monthly_hosted_models": "0.10",
        "first_migration_month_gcp": "1.93",
    }
    assert report["release_gates"]["gcp_monthly_budget"]["status"] == "passed"
    assert report["release_gates"]["provider_hard_cap"] == {
        "status": "blocked",
        "forecast_eur": "0.10",
        "reason": "missing recorded sponsor hard cap",
    }
    approved = example_payload()
    approved["provider"]["hard_cap_eur"] = "0.10"
    assert forecast(approved)["release_gates"]["provider_hard_cap"]["status"] == "passed"
    rejected = example_payload()
    rejected["workload"]["fallback_rate"] = 2
    try:
        forecast(rejected)
    except ValueError as exc:
        assert "fallback_rate" in str(exc)
    else:
        raise AssertionError("an invalid fallback rate must be rejected")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="JSON forecast assumptions")
    source.add_argument("--example", action="store_true", help="run the documented candidate scenario")
    parser.add_argument("--self-test", action="store_true", help="run deterministic calculation checks")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test()
        if args.input:
            payload = json.loads(args.input.read_text(encoding="utf-8"))
        elif args.example:
            payload = example_payload()
        elif args.self_test:
            return 0
        else:
            parser.error("provide --input or --example")
        if not isinstance(payload, dict):
            raise ValueError("input must be a JSON object")
        print(json.dumps(forecast(payload), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"forecast failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
