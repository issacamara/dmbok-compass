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

Citation IDs returned by an answer model are accepted only when they match a
retrieved chunk. The API reconstructs every citation's page, section, and
excerpt from that chunk's stored provenance; provider-supplied citation
metadata is never trusted. The live response trace also contains request-scoped
retrieval, generation, and total timings.

Production questions, prompts, retrieved passages, generated answers, and
request traces are ephemeral. They must not be persisted to Firestore, Storage,
logs, analytics, traces, backups, or build artifacts. Evaluation-authored
questions and annotations are separate durable data because reproducible
release evaluation requires them.

Evaluation datasets are versioned by `dataset_version_id`; the active pointer
is exposed as `generation_id`. Imports accept one JSON array no larger than
1,048,576 bytes and 500 submitted records. Every record is independently
validated for a question, one category (`definitions`, `explanations`,
`comparisons`, `study`, or `scenarios`), an expected answer or rubric, and a
supporting passage. An import response always contains `submitted_count`,
`imported_count`, `skipped_count`, and indexed, content-free validation errors.
It is `accepted` only when at least one record was staged and atomically made
active; otherwise it is `rejected` and has no `generation_id`.

Activation is compare-and-set: all valid records are durable before the active
pointer changes. A failed, zero-valid, or losing concurrent import leaves the
existing generation and its visible history unchanged. The winning replacement
marks prior-generation runs superseded for current-evidence queries, while
retaining them for audit and reproduction. Full-run launch accepts no dataset
identifier: the server resolves and pins the active generation. Subset launch
accepts only IDs belonging to that generation.

Generated annotations begin as `candidate`; human review may move them once to
`approved` or `rejected`. Any non-empty active generation supports an
`exploratory` run. A report is `release_evidence` only when it is bound to the
current generation and has 30–50 approved gold annotations; this label does not
claim that its metrics pass. These records are not a storage path for production
questions or answer traces.

Telemetry is content-free and allowlisted to the fields in the fixture. Any
question, prompt, passage, answer, trace, excerpt, password, token, secret, or
credential field is rejected.

The backend telemetry adapter accepts only `TelemetryEvent` values. Successful
question requests emit request count, outcome, elapsed time, and model
identifier; failures emit request count, elapsed time, and exception class.
Exception messages are never emitted because they can contain interaction
content. The question UI keeps the question, answer, citations, and trace in
React state for the active view only and does not write them to browser
storage.

The administrator metrics API (`/api/admin/metrics`) exposes persisted
`AggregateMetric` records only. A record contains a metric name, numerator,
denominator, and percentage, so request counts, quota consumption, provider
failures, fallback rates, and cost indicators can be reported without storing
question, answer, passage, or trace content.

Evaluation runs are identified by a deterministic hash of dataset, corpus,
configuration, model, and selected item IDs. `POST /api/evaluations` accepts
the complete dataset or an explicit subset, returns a queued `EvaluationRun`,
and dispatches the same run ID at most once; retrying the identical request is
idempotent. Only administrators may launch or inspect runs. A run records the
four immutable version bindings and selected item IDs, and the worker stores
content-free per-item pass/fail results plus six aggregate metrics: retrieval
success, grounded claims, citation correctness, answer quality, refusal
correctness, and response time.
Each metric records its numerator, denominator, percentage, release threshold,
and pass/fail result.

Release decisions are administrator-only and immutable.  A decision binds the
release ID to the evaluation run's exact dataset, corpus, configuration, and
model versions, plus the provider, scorer, and gate-report identifiers supplied
for that candidate.  Approval requires a completed run with gate reports and
all metrics passing.  A failed or incomplete gate may be approved only when
the administrator records an explicit approved exception and rationale;
otherwise the API fails closed.  Repeating an identical release submission is
idempotent, while reusing a release ID with different evidence is rejected.
