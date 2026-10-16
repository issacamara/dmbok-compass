# External answer-provider decision

Status: approved for implementation, release-gated for live traffic  
Decision date: 2026-10-06  
Decision owner: sponsor / release approver

## Decision

Use OpenRouter as the single external answer gateway. Configure the
provider-neutral adapter with these model identifiers:

| Role | Model ID | Rationale |
| --- | --- | --- |
| Primary | `mistralai/mistral-nemo` | OpenRouter chat model for the normal answer path. |
| Fallback | `meta-llama/llama-3.1-8b-instruct` | A distinct OpenRouter chat model for fallback. |

The fallback is only for configured model-level unavailability, timeout,
capacity, or rate-limit failures. It is not a gateway-failure fallback: an
OpenRouter-wide outage must produce the existing clear service error rather
than an ungrounded answer.

## Data terms approval

OpenRouter is the production gateway. The OpenRouter API key is backend-only
and must be stored in Secret Manager. The gateway and the selected upstream
providers may have different logging, retention, and training terms; this
decision makes no provider-wide no-training claim.

Application behavior remains stricter: production questions, retrieved
passages, prompts, answers, and traces are never persisted by this
application. Before live traffic, the selected OpenRouter route and upstream
provider terms must be reviewed for no-training and minimal retention. Users
must still be warned not to submit confidential or personal information
because gateway and upstream handling is outside the application’s control.

Free models are preferred for both primary and fallback answers, but are not
mandatory. A paid OpenRouter route may replace either model if it stays within
the sponsor-approved provider budget and passes the same privacy, availability,
quality, latency, and cost gates. No production corpus or user data may be sent
until the selected route review is recorded.

## Cost and release gates

- The project’s 100-request daily cap remains the application-side default.
- OpenRouter token limits, rate limits, routing behavior, billing alerts, and
  the existing monthly budget must be configured before live traffic is
  enabled.
- Release evidence must show that the selected model IDs are available through
  OpenRouter, the API key is active, the €5 project budget is monitored, the
  upstream data terms are acceptable, and the model evaluation passes the
  existing quality, grounding, citation, latency, and fallback gates.
- If OpenRouter or an upstream provider does not satisfy the privacy or
  availability gate, production release remains blocked.

## Evidence

Reviewed 2026-10-06:

- [OpenRouter model catalog](https://openrouter.ai/api/v1/models) — current
  model IDs, pricing, modality, and context metadata; revalidate at release.
- [OpenRouter privacy documentation](https://openrouter.ai/docs/privacy) —
  gateway privacy controls and the need to review upstream provider terms.
- [OpenRouter API reference](https://openrouter.ai/docs/api-reference/overview)
  — gateway contract and authentication boundary.

## Production configuration

- Project: `prod-dmbok-compass`.
- Monthly budget: €5, scoped to the production project; this is an alerting
  control and not an automatic spending cutoff.
- OpenRouter API key: backend-only Secret Manager secret, never a browser
  credential or source-controlled value.
- Vertex AI remains in use only for the existing `gemini-embedding-001`
  corpus/query embedding path; it is not the answer-model gateway.

## Historical non-candidate validation evidence

Environment validated 2026-10-06:

- Project: `prod-dmbok-compass` (`290754673794`), active and linked to the
  EUR billing account `main-billing-account`.
- Budget: recurring monthly budget of `5 EUR`, scoped to the production project
  only, with current-spend thresholds at 50%, 90%, and 100%, plus a 90%
  forecast threshold. Budget resource:
  `billingAccounts/014869-D92661-32A1E6/budgets/eab851fe-bfb4-4bd5-95f9-5140e990b638`.
- APIs: Cloud Billing Budget API and Vertex AI API are enabled in the
  production project.
- Model metadata: `gemini-2.5-flash-lite` and `gemini-2.5-flash` resolved
  successfully through the Google Gen AI SDK using production-project ADC.
- Structured smoke generation: those Gemini models returned `{"ok":true}`
  with `STOP` completion using a synthetic 10-token prompt. Observed latency
  was approximately 1.42 seconds and 0.96 seconds respectively.

These historical checks validate credentials, model availability, and a
minimal Gemini runtime path only. They do not validate the configured OpenRouter
answer routes, corpus-grounding, load, or end-to-end release readiness. Google
Cloud budgets are alerting controls, not automatic spending cutoffs; the
application’s daily quota and the configured provider budget must both remain
enforced operationally.

## Candidate revalidation — promotion blocked

**Revalidated:** 2026-10-16 · **Decision owner:** sponsor / release approver

The effective candidate answer configuration is OpenRouter primary
`mistralai/mistral-nemo` with fallback
`meta-llama/llama-3.1-8b-instruct`; see the dated
[cost and provider-route record](../release/langchain/cost-provider.md).
The 2026-10-06 Gemini observations above are not valid substitute evidence for
these routes.

On 2026-10-16, public OpenRouter model metadata listed both exact IDs and their
structured-output parameters. It does not attest to account capacity, active
routing, or retention/training terms. No authenticated candidate smoke result,
sponsor-approved provider hard cap, or sponsor-reviewed current OpenRouter and
selected-upstream no-training/minimal-retention terms has been recorded.

Accordingly, promotion is **blocked**. The sponsor must record the route review
URLs, reviewer and decision time; the provider hard-cap amount; and the two
sanitized synthetic route results. A 429 or other capacity/rate-limit response
is failure evidence, not a passing route result. This does not alter the
separate publisher-rights gate in [`rights.md`](rights.md).

## Contract impact

This decision consumes the provider-neutral `ModelAdapter` protocol and keeps
the existing `GenerationRequest`, `GenerationResult`, normalized error, and
privacy-safe usage metadata contracts unchanged. It exposes only the
provider/model identifiers through the existing `UsageMetadata` and retrieval
trace fields; the OpenRouter credential remains a backend-only secret.

The decision does not authorize persistence of user-generated content,
browser-side provider calls, or weakening of corpus-only grounding and
citation validation.
