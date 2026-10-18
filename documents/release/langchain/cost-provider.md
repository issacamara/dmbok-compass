# Candidate cost and provider-route evidence

**Status:** promotion blocked · **Revalidated:** 2026-10-16
**Scope:** the LangChain candidate only; this is a forecast, not a billing export

This record revalidates the candidate configuration rather than treating older
provider or cost notes as evidence. It does not authorize deployment, change a
provider, purchase capacity, or decide publisher rights.

## Effective candidate configuration

The checked-in effective settings agree across [`.env.example`](../../../.env.example),
[`infra/terraform/delivery.tf`](../../../infra/terraform/delivery.tf), and
[`.github/workflows/delivery.yml`](../../../.github/workflows/delivery.yml):

| Setting | Candidate value |
| --- | --- |
| Answer gateway | `openrouter` |
| Primary route | `meta-llama/llama-3.1-8b-instruct` |
| Fallback route | `google/gemma-4-26b-a4b-it:free` |
| Fallback eligibility | timeout, unavailable, capacity, or rate-limit failure only; never an OpenRouter-wide outage |
| Request/generation limits | 14,000 ms / 8,000 ms |
| API service resources | 0 minimum, 2 maximum instances; 1 vCPU, 512 MiB, concurrency 8 |
| Answer credential boundary | backend Secret Manager binding `answer-provider-api-key`; no browser-side provider credential |

The fallback behavior is implemented by [`backend/model/openrouter.py`](../../../backend/model/openrouter.py).
`OPENROUTER_JSON_SCHEMA_SUPPORTED=true` is configuration, not proof that a
route currently accepts the application's schema.

## Dated route, capacity, and schema evidence

On 2026-10-16, the unauthenticated OpenRouter model catalog returned both exact
configured IDs. It reported 131,072-token context windows and both
`response_format` and `structured_outputs` in their supported-parameter lists.
The catalog's listed price metadata was USD $0.029/$0.030 per million
prompt/completion tokens for the primary and USD $0.050/$0.080 for the
fallback. This is metadata availability only; it does not prove account
capacity, active routing, or successful structured generation.

No authenticated synthetic smoke run was performed in this worktree because
the backend-only secret is intentionally unavailable. No HTTP 429 was
observed or inferred, and a future 429 must be recorded as capacity/rate-limit
failure, **not** as functional success. The release owner must run the existing
sanitized check in the configured environment and retain its dated result:

```sh
PYTHONPATH=backend python -m model.smoke
```

That command verifies both configured routes with a synthetic prompt and emits
only route IDs and outcomes; it must not print a provider key, prompt,
retrieved passage, or response body. Until the two route outcomes and schema
acceptance are recorded for the candidate, capacity/schema evidence is
incomplete and promotion remains blocked.

The historical Gemini result in
[`documents/decisions/provider.md`](../../decisions/provider.md) concerns an
embedding/runtime check, not these configured OpenRouter answer routes. It is
therefore explicitly excluded from candidate answer-route evidence.

## Cost scenario

Run the deterministic forecast from the repository root:

```sh
python3 scripts/forecast_cost.py --self-test
python3 scripts/forecast_cost.py --example
```

The example is a transparent 30-day scenario: 100 daily answer requests, 800
input and 300 output tokens per request, and a 5% fallback rate. Its listed
EUR amounts are planning assumptions, not a claim that OpenRouter publishes EUR
pricing or that GCP billed these amounts. Before release, replace every rate
with dated billing/pricing evidence and preserve the exact JSON input alongside
the release bundle.

| Category | Calculation / scenario | Forecast (EUR) |
| --- | --- | ---: |
| One-time rebuild | 1.5M embedding tokens plus one ingestion-job run | 0.58 |
| Temporary migration storage/index/artifacts | duplicate Firestore storage, vector index, candidate artifacts per month | 0.32/month |
| Monthly GCP runtime, including 512 MiB cold-start/runtime allowance | Cloud Run, Firestore operations, logging/monitoring | 1.03/month |
| Monthly GCP during migration | temporary resources + runtime | 1.35/month |
| First migration-month GCP | one-time rebuild + monthly GCP during migration | 1.93 |
| Hosted-model primary/fallback mix | 95% primary, 5% fallback | 0.10/month |

The checked production-project budget is EUR 5/month with alert thresholds;
the scenario's EUR 1.35 recurring GCP forecast is below that budget. This is a
planning pass, not an automatic spending cutoff or a production-spend report.
It must be replaced by a corpus-size, usage, and billing-backed forecast before
promotion.

## Decision gates and explicit blockers

| Gate | Result | Evidence / required resolution |
| --- | --- | --- |
| GCP monthly forecast <= EUR 5 | Scenario passes | EUR 1.35/month during migration; retain actual meter and dated forecast before release. |
| Provider hard cap | **Blocked** | No sponsor-approved provider hard-cap amount is recorded. The EUR 0.10 scenario cannot satisfy an absent cap. Sponsor must record amount, owner, and date. |
| Exact configured-route capacity and schema | **Blocked** | Public catalog metadata is insufficient; record successful authenticated synthetic checks for both routes. A 429 is a failed capacity check. |
| No-training/minimal-retention terms | **Blocked** | The prior OpenRouter privacy link returned HTTP 404 on 2026-10-16 and no selected-upstream route review is recorded. Sponsor must review and date the gateway and both upstream providers' terms. No provider-wide no-training claim is made. |
| Publisher rights | **Separate blocked owner decision** | [Publisher-rights decision](../../decisions/rights.md) remains a distinct multi-user release gate; resolving provider terms does not resolve it. |

The provider hard-cap, live route, and data-term blockers keep promotion
blocked even though the GCP planning scenario is below EUR 5/month.

## Revalidation checklist

1. Record the exact model-catalog response date, prices, and any route change;
   update the forecast's EUR conversion and input artifact.
2. Run `model.smoke` through the configured Secret Manager-bound environment;
   retain only timestamps, route IDs, status, schema outcome, and error class.
3. Have the sponsor review the current OpenRouter and selected-upstream terms
   for no-training/minimal retention; record URLs, reviewer, and ISO-8601
   decision time in [`documents/decisions/provider.md`](../../decisions/provider.md).
4. Record the provider monthly hard cap and compare the forecast to it with
   `scripts/forecast_cost.py`; a missing or exceeded cap is blocked.
5. Keep publisher-rights evidence and its owner decision independent in the
   final release bundle.
