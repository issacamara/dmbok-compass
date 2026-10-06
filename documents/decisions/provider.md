# External answer-provider decision

Status: approved for implementation, release-gated for live traffic  
Decision date: 2026-10-06  
Decision owner: sponsor / release approver

## Decision

Use the Google Gemini API as the single external answer provider. Configure the
provider-neutral adapter with these model identifiers:

| Role | Model ID | Rationale |
| --- | --- | --- |
| Primary | `gemini-2.5-flash-lite` | Lowest-cost, latency-oriented model for the normal answer path. |
| Fallback | `gemini-2.5-flash` | Same-provider model-level fallback with a larger quality/capacity envelope. |

The fallback is only for configured model-level unavailability, timeout,
capacity, or rate-limit failures. It is not a provider-failure fallback: a
Google-wide outage must produce the existing clear service error rather than
an ungrounded answer.

## Data terms approval

The paid Gemini API tier is required for production. Google’s Gemini API
billing documentation says that, unlike the free tier, paid-tier content is
not used to improve Google products. This satisfies the project’s no-training
requirement for production requests, subject to the provider terms remaining
unchanged.

Gemini API logs are configured to the minimum documented retention window of
7 days. Application behavior remains stricter: production questions,
retrieved passages, prompts, answers, and traces are never persisted by this
application. Users must still be warned not to submit confidential or
personal information because provider-side handling is outside the
application’s control.

The free tier is rejected for production. Google documents that free-tier
content may be used to improve Google products, so free-tier use would violate
the approved data-handling requirement. It may be used only for isolated,
non-sensitive development experiments if no production corpus or user data is
sent.

## Cost and release gates

- The project’s 100-request daily cap remains the application-side default.
- Provider token limits, rate limits, billing alerts, and a hard monthly spend
  cap must be configured before live traffic is enabled.
- Release evidence must show that the selected model IDs are available in the
  target region, the paid-tier account is active, the spend cap is no more
  than the sponsor-approved external-model budget, and the model evaluation
  passes the existing quality, grounding, citation, latency, and fallback
  gates.
- If the sponsor does not approve an external-model budget, the adapter may be
  implemented and tested with the fake adapter, but production release remains
  blocked.

## Evidence

Reviewed 2026-10-06:

- [Gemini API billing and tiers](https://ai.google.dev/gemini-api/docs/billing)
  — free/paid tier behavior and paid-tier data-use statement.
- [Gemini API logs and datasets](https://ai.google.dev/gemini-api/docs/logs-datasets)
  — paid-tier log availability and configurable 7-day minimum retention.
- [Gemini API models](https://ai.google.dev/gemini-api/docs/models) — model
  identifiers and availability must be revalidated at release time.
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) —
  current token pricing and free-tier limits; pricing is not frozen by this
  record.

## Production validation evidence

Environment validated 2026-10-06:

- Project: `prod-dmbok-compass` (`290754673794`), active and linked to the
  EUR billing account `main-billing-account`.
- Budget: recurring monthly budget of `5 EUR`, scoped to the production project
  only, with current-spend thresholds at 50%, 90%, and 100%, plus a 90%
  forecast threshold. Budget resource:
  `billingAccounts/014869-D92661-32A1E6/budgets/eab851fe-bfb4-4bd5-95f9-5140e990b638`.
- APIs: Cloud Billing Budget API and Vertex AI API are enabled in the
  production project.
- Model metadata: both selected model IDs resolved successfully through the
  Google Gen AI SDK using production-project ADC.
- Structured smoke generation: both models returned `{"ok":true}` with
  `STOP` completion using a synthetic 10-token prompt. Observed latency was
  approximately 1.42 seconds for `gemini-2.5-flash-lite` and 0.96 seconds for
  `gemini-2.5-flash`.

These checks validate credentials, model availability, and a minimal runtime
path only. They do not replace the full corpus-grounding, citation, fallback,
load, or end-to-end evaluation gates. Google Cloud budgets are alerting
controls, not automatic spending cutoffs; the application’s daily quota and
the configured provider budget must both remain enforced operationally.

## Contract impact

This decision consumes the provider-neutral `ModelAdapter` protocol and keeps
the existing `GenerationRequest`, `GenerationResult`, normalized error, and
privacy-safe usage metadata contracts unchanged. It exposes only the
provider/model identifiers through the existing `UsageMetadata` and retrieval
trace fields; credentials remain backend-only secrets.

The decision does not authorize persistence of user-generated content,
browser-side provider calls, or weakening of corpus-only grounding and
citation validation.
