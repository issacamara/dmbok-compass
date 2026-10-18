# LangChain migration baseline

This baseline freezes the candidate delivered by PR #111 without promoting it
to production. It consumes the existing API/domain contracts and the #112
delivery; it exposes versioned baseline fixtures plus internal binding and
assessment contracts for downstream tickets.

## Candidate evidence

The merged integration commit is
`50b9470836241cfcec0e9f60118c659f40d4fa17`. The successful Linux
compatibility check ran on candidate commit
`df32ecaa392c2be0104903a3f2b691989cbab4b6` in workflow run `38457027955`.
That check built `dmbok-langchain-compat` as
`sha256:1bab38309da206e88812655888fe1e6fa4e501bb0cfe0d01974e35579093e43d`,
from `python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce`.

| Artifact | SHA-256 |
| --- | --- |
| `backend/requirements-compat.lock` | `4b14a82ce8542173da5bec0e541110c868339f881be8dca4abd92a4de8b66784` |
| `langchain_google_firestore-0.5.0.post2-py3-none-any.whl` | `9ced9b491908b453a31a4b2598c1ec5fbbddf106dff868adcfe36bd5e60124c2` |

The image is CI evidence only; it is not a published production artifact.

## Defaults versus effective routes

The code defaults are `europe-west1`, `document_chunks_v2`, OpenRouter,
60,000 ms, primary `meta-llama/llama-3.1-8b-instruct`, and fallback
`google/gemma-4-26b-a4b-it:free`. Local Compose and the delivery preview and
jobs explicitly select those same OpenRouter values. Explicit deployment
settings, not these defaults, are the effective route.

Vertex is supported as a configuration option, with default primary model
`gemini-2.5-flash-lite`, but it is not enabled by Local Compose or the delivery
workflow. Enabling it remains a release gate: an approved provider-route,
structured-output, privacy, cost, and runtime-evaluation review is required.
No secret value is captured here.

## Frozen internal contracts

### Corpus and configuration binding

Every serving candidate binds an immutable corpus version ID to its active
vector collection, embedding model and 768 dimensions. Chunks keep their
one-based page, recovered section, ordinal, content hash, and text; chunking
never crosses a page or section boundary. Retrieval filters by the active
corpus before selecting no more than five passages.

The configuration binding records the answer provider, primary and fallback
models, JSON-schema capability, timeout, evidence thresholds, vector
collection, and configuration version ID. A request reserves quota once before
downstream calls. The only eligible fallback classes are `timeout`,
`unavailable`, and `rate_limited`; at most two model attempts occur. Strong
evidence starts at `0.75`, partial evidence starts at `0.45`, and lower or
invalid evidence is absent.

### Reviewed assessment binding

Assessment data is evaluator-authored and versioned by dataset, corpus,
configuration, and model IDs. It must retain review status and immutable
provenance while excluding production questions, prompts, passages, answers,
and traces. Review covers authored fixtures for strong, partial, and absent
evidence; eligible and ineligible fallback; and citation provenance
reconstruction. These fixtures contain no production content.

The existing public request, response, and error schemas remain the source of
truth. `api-contract-fixture.json` freezes their public field sets and required
fields; `backend/tests/test_contracts.py` compares that fixture with the typed
Pydantic models.
