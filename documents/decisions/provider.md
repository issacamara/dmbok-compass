# External answer-provider decision

Status: approved for implementation, release-gated for live traffic  
Decision date: 2026-10-06  
Decision owner: sponsor / release approver

## Decision

Use OpenRouter as the single external answer gateway. Configure the
provider-neutral adapter with these model identifiers:

| Role | Model ID | Rationale |
| --- | --- | --- |
| Primary | `google/gemma-2-27b-it` | The requested Gemma 2 model for the normal answer path. |
| Fallback | `nvidia/nemotron-3.5-lightning:free` | A distinct free OpenRouter model for cost-controlled fallback. |

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

The free fallback is permitted for this version to meet the cost constraint,
but it remains release-gated on the selected upstream provider’s data terms,
availability, and rate limits. No production corpus or user data may be sent
until that review is recorded.

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

## Contract impact

This decision consumes the provider-neutral `ModelAdapter` protocol and keeps
the existing `GenerationRequest`, `GenerationResult`, normalized error, and
privacy-safe usage metadata contracts unchanged. It exposes only the
provider/model identifiers through the existing `UsageMetadata` and retrieval
trace fields; the OpenRouter credential remains a backend-only secret.

The decision does not authorize persistence of user-generated content,
browser-side provider calls, or weakening of corpus-only grounding and
citation validation.
