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

`DocumentChunk` IDs are deterministic from corpus version and source position.
Embeddings are exactly 768 dimensions. Corpus versions are immutable after
creation, move through staged/validated/active/retired states, and activate via
one pointer so the previous valid version can be restored.

Production questions, prompts, retrieved passages, generated answers, and
request traces are ephemeral. They must not be persisted to Firestore, Storage,
logs, analytics, traces, backups, or build artifacts. Evaluation-authored
questions and annotations are separate durable data because reproducible
release evaluation requires them.

Telemetry is content-free and allowlisted to the fields in the fixture. Any
question, prompt, passage, answer, trace, excerpt, password, token, secret, or
credential field is rejected.
