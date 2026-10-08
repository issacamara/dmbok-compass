#!/usr/bin/env python3
"""Run the three-user latency probe against a deployed API.

The output contains aggregate performance and cost evidence only. Bearer
tokens are accepted on the command line but are never written to the report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

import httpx

from app.performance import RequestSample, build_report, forecast_provider_cost_eur


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="API origin, for example https://api.example.com")
    parser.add_argument("--token", action="append", required=True, help="Bearer token; repeat exactly three times")
    parser.add_argument("--requests-per-user", type=int, default=10)
    parser.add_argument("--question", default="What is data governance?")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--threshold-ms", type=float, default=15_000)
    parser.add_argument("--daily-requests", type=int, default=100)
    parser.add_argument("--input-tokens-per-request", type=int, default=800)
    parser.add_argument("--output-tokens-per-request", type=int, default=300)
    parser.add_argument("--fallback-model", default="nvidia/nemotron-3.5-lightning:free")
    parser.add_argument("--primary-input-eur-per-million", type=float, default=0.0)
    parser.add_argument("--primary-output-eur-per-million", type=float, default=0.0)
    parser.add_argument("--fallback-input-eur-per-million", type=float, default=0.0)
    parser.add_argument("--fallback-output-eur-per-million", type=float, default=0.0)
    parser.add_argument("--gcp-cost-forecast-eur", type=float, default=0.0)
    parser.add_argument("--gcp-budget-eur", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("performance-evidence.json"))
    return parser.parse_args()


async def probe_user(
    client: httpx.AsyncClient,
    user_index: int,
    token: str,
    question: str,
    requests_per_user: int,
) -> list[RequestSample]:
    samples: list[RequestSample] = []
    for _ in range(requests_per_user):
        started = perf_counter()
        try:
            response = await client.post(
                "/api/questions",
                headers={"Authorization": f"Bearer {token}"},
                json={"question": question},
            )
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            samples.append(
                RequestSample(
                    user_index=user_index,
                    status_code=response.status_code,
                    duration_ms=round((perf_counter() - started) * 1000, 3),
                    outcome=payload.get("outcome"),
                    model=(payload.get("trace") or {}).get("selected_model"),
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
                )
            )
    return samples


async def run_probe(args: argparse.Namespace) -> dict[str, object]:
    if len(args.token) != 3:
        raise ValueError("provide exactly three --token values to exercise the three-user contract")
    if args.requests_per_user <= 0:
        raise ValueError("--requests-per-user must be positive")
    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=args.timeout_seconds) as client:
        groups = await asyncio.gather(
            *(probe_user(client, index, token, args.question, args.requests_per_user) for index, token in enumerate(args.token))
        )
    samples = [sample for group in groups for sample in group]
    fallback_rate = sum(sample.model == args.fallback_model for sample in samples) / len(samples)
    provider_cost = forecast_provider_cost_eur(
        daily_requests=args.daily_requests,
        input_tokens_per_request=args.input_tokens_per_request,
        output_tokens_per_request=args.output_tokens_per_request,
        fallback_rate=fallback_rate,
        primary_input_eur_per_million=args.primary_input_eur_per_million,
        primary_output_eur_per_million=args.primary_output_eur_per_million,
        fallback_input_eur_per_million=args.fallback_input_eur_per_million,
        fallback_output_eur_per_million=args.fallback_output_eur_per_million,
    )
    return build_report(
        samples,
        threshold_ms=args.threshold_ms,
        daily_requests=args.daily_requests,
        provider_cost_eur=provider_cost,
        gcp_cost_eur=args.gcp_cost_forecast_eur,
        gcp_budget_eur=args.gcp_budget_eur,
        assumptions={
            "base_url": args.base_url,
            "requests_per_user": args.requests_per_user,
            "fallback_model": args.fallback_model,
            "fallback_rate": round(fallback_rate, 6),
            "input_tokens_per_request": args.input_tokens_per_request,
            "output_tokens_per_request": args.output_tokens_per_request,
        },
    )


def main() -> int:
    args = parse_args()
    try:
        report = asyncio.run(run_probe(args))
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"performance probe failed: {exc}", file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
