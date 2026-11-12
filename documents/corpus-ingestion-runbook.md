# Corpus ingestion and recovery

Corpus maintenance is an offline, administrator-only operation. Each run
loads the approved, versioned PDF with LangChain `PyPDFLoader` in page mode,
splits page text with the configured 700-word/70-word-overlap LangChain
splitter, and derives stable corpus-version and chunk identifiers from the
source generation and RAG derivation fingerprint.

LangChain generates 768-dimensional Vertex document embeddings and writes
documents through `FirestoreVectorStore.add_documents` to the collection set
by `RAG_VECTOR_COLLECTION` (default `document_chunks_v2`). Each vector document
stores text and embedding at the top level, with source identity, corpus
version, chunk ID, page, section, ordinal, content hash, and schema version in
the nested `metadata` map. Re-running the same source overwrites the same
deterministic IDs rather than creating duplicates.

The ingestion job writes a `staged` corpus version, persists every vector in
explicit-ID batches of at most 500, and then compares the stored IDs, text,
hashes, provenance, and embedding dimensions with the derived manifest. A
`SIGTERM` is recorded after the current persisted batch; the job exits with
status 1, leaves the version staged, and never activates it. Re-running the
same derived version safely rewrites the same IDs and resumes validation only
after every required batch has completed. Only a complete version becomes
`validated`.
Activation uses a Firestore transaction and compare-and-set against the corpus
that was active when ingestion began, so a stale concurrent job cannot replace
a newer activation. The same transaction retires the old version, activates the
candidate, and updates `corpus_state/active` with current and previous IDs.

If loading, embedding, writing, or validation fails, the active pointer remains
unchanged. To withdraw a promoted version, run the corpus-version store's
`rollback()` maintenance operation in the same runtime image and configuration.
It atomically restores the recorded previous valid version and retires the
withdrawn version. An explicit previously valid version ID may be supplied when
an administrator needs a targeted rollback. Keep old vector documents and the
versioned source object until promotion and rollback evidence is accepted.

## Recovery references and retention

Before cleanup, build the reviewed snapshot used by `scripts/source_retention.py`
from the active release, the one retained rollback release, and exact recovery
records. Include source, parsing-result, reviewed-map, approval, dataset-revision
and review-snapshot artifacts. Use the immutable Cloud Storage URI plus object
generation for object identities and the exact Firestore document/revision for
database snapshots; include each content SHA-256. A release reference must be
validated and labelled `active` or `rollback`. A changed reference fingerprint
or an unvalidated reference aborts the inventory.

The snapshot contains candidate records and release links in this shape:

```json
{
  "recovery_artifacts": [
    {"kind": "parsing", "identity": "gs://bucket/staging/attempt/parse.json#7", "content_sha256": "<64 lowercase hex characters>", "created_at": "2026-10-01T00:00:00Z"}
  ],
  "recovery_references": [
    {"release_id": "release-id", "corpus_version_id": "corpus-id", "role": "active", "validated": true, "artifact": {"kind": "parsing", "identity": "gs://bucket/staging/attempt/parse.json#7", "content_sha256": "<64 lowercase hex characters>", "created_at": "2026-10-01T00:00:00Z"}}
  ],
  "rebuildable_recovery_artifacts": [["parsing", "gs://bucket/staging/attempt/parse.json#7", "<64 lowercase hex characters>"]]
}
```

Run the same-image inventory with its authenticated actor and expected immutable
image digest:

```sh
python scripts/source_retention.py <reviewed-snapshot.json> \
  --actor <authenticated-principal> \
  --image-digest <running-image-sha256> \
  --expected-image-digest <approved-image-sha256>
```

Only an exact verified rebuild path makes an old unreferenced artifact eligible.
An old artifact with no verified rebuild path is reported as refused. Recent
inputs and every active/rollback reference are protected. The inventory command
is dry-run only; it never deletes source objects, parsing artifacts, map or
approval records, datasets, review snapshots, or applies Terraform. Verify the
saved parse-result digest and rebuild from that artifact before marking it
rebuildable; do not call LlamaCloud again for a rebuild from a saved parse.
Review the emitted reference fingerprint and each protected, eligible, and
refused identity before any separate, approved cleanup procedure.

Raw evaluator outputs are available only to an administrator while their
`expires_at` has not passed. Finalize a reviewed output with
`FirestoreCandidateOutputStore.finalize()`; this removes the raw text while
retaining its digest and assessment audit. The `evaluation_candidate_outputs`
Firestore TTL is configured on the raw output `expires_at` field. Firestore
removes expired records asynchronously; the application denies reads as soon as
the deadline passes, and the admin-only `cleanup_expired()` method can remove
them on demand. Audit digests, assessments, and release/recovery references are
retained separately. Never persist production questions, passages, answers, or
traces in evaluator-output records.

## Abandoned staging cleanup

Use the same authenticated Cloud Run Job image and its worker service account
for maintenance; do not grant browser or anonymous access to corpus storage.
Set `CORPUS_CLEANUP_VERSION` to one exact corpus version ID to inspect the
cleanup targets. This is a dry run by default and reports only the version and
its matching vector document paths:

```sh
CORPUS_CLEANUP_VERSION=<abandoned-version-id> python -m app.ingestion.entrypoint
```

After reviewing the reported targets, set `CORPUS_CLEANUP_APPLY=true` in the
same invocation to remove that exact abandoned staged version and its vectors.
Cleanup refuses an active version, its rollback-retained predecessor, and any
version with a running ingestion reservation. It never removes serving or
rollback records; investigate a still-running reservation rather than forcing
cleanup.

Before production promotion, rebuild the full candidate corpus, verify filtered
retrieval from the active version, run the approved evaluation and load gates,
and record a rollback drill. The exposed development OpenRouter key must be
rotated before any further shared-provider validation or deployment.

The API uses the same 25-second graceful-shutdown budget as its Uvicorn
container setting. On shutdown it stops admitting new requests, cancels
in-flight request tasks, and waits for their cleanup before the process exits.
Provider and database calls must retain their existing bounded timeouts; an
interrupted request is never written as a successful answer.
