# Frozen contracts

The contract boundary is shared by the FastAPI backend and the TypeScript web
client. `api-contract-fixture.json` is intentionally small and stable: it
holds values that must remain identical across languages.

## Ownership

Firestore is authoritative for `UserProfile`, approval state, roles, quota
policies/counters, application configuration, corpus versions/chunks,
evaluation datasets/items/annotations/runs, release decisions, and content-free
aggregate metrics. Firebase Authentication owns credentials and reset tokens;
Cloud Storage owns the source PDF and ingestion artifacts.

Registration creates a pending `UserProfile` from the verified Firebase
identity. The approval lifecycle is transactional: `pending` may become
`approved` or `rejected`, an `approved` profile may become `deactivated`, and
`rejected`/`deactivated` profiles cannot be reopened through the approval API.
Approval and the serialized approved-user counter are committed together, so
concurrent approvals cannot exceed the ten-user limit. Every protected request
reloads the profile, making deactivation effective immediately.

`DocumentChunk` IDs are deterministic from corpus version and source position.
Embeddings are exactly 768 dimensions. Corpus versions are immutable after
creation, move through staged/validated/active/retired states, and activate via
one pointer so the previous valid version can be restored.

Live retrieval embeds the question with the `RETRIEVAL_QUERY` task type and the
same 768-dimensional embedding model used for document chunks. Firestore
nearest-neighbor search must filter on the active corpus-version ID before
returning at most five ranked passages. Each passage preserves its chunk ID,
page, section, excerpt, and available relevance score for the answer trace.
Evidence classification then filters to finite normalized scores at or above
the partial threshold (`0.45`), ranks ties by chunk ID, and keeps at most five
passages. A top score at or above `0.75` is `strong`, a score from `0.45` to
below `0.75` is `partial`, and no qualifying score is `absent`. The evidence
bundle records the outcome basis and cited chunk IDs. Top-five retrieval
success is reported as a content-free numerator, denominator, and percentage.

Production questions, prompts, retrieved passages, generated answers, and
request traces are ephemeral. They must not be persisted to Firestore, Storage,
logs, analytics, traces, backups, or build artifacts. Evaluation-authored
questions and annotations are separate durable data because reproducible
release evaluation requires them.

Telemetry is content-free and allowlisted to the fields in the fixture. Any
question, prompt, passage, answer, trace, excerpt, password, token, secret, or
credential field is rejected.

The administrator metrics API (`/api/admin/metrics`) exposes persisted
`AggregateMetric` records only. A record contains a metric name, numerator,
denominator, and percentage, so request counts, quota consumption, provider
failures, fallback rates, and cost indicators can be reported without storing
question, answer, passage, or trace content.
