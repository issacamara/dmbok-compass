# Coupled LangChain delivery rollback

A rollback restores a tested **release binding**, never a corpus pointer or
Cloud Run revision in isolation.  A binding consists of the API/admin image
digest, frontend digest, reader schema, corpus version, collection, Firestore
vector index, and configuration fingerprint.

## Preconditions

1. Use a retained `promoted-delivery-evidence-*` artifact from the release to
   restore.  It must include `delivery-manifest.json` and the frontend archive.
2. Start the `Immutable delivery` workflow manually with `target_environment`
   set to `production`, `rollback` and `owner_approved` set to true, and the
   previous workflow run ID supplied as `rollback_run_id`.
3. The GitHub `production` environment must enforce its normal protected
   delivery approval.  The preflight job does not shift traffic.

## Preflight and drill

The workflow generates `coupled-rollback-plan.json`.  Review that artifact to
confirm both entries have a digest-pinned image and a complete serving binding.
The preflight rejects a v1 reader bound to the v2 collection/index (and the
reverse), different API/admin images, missing configuration fingerprints, or
untested release revisions.

For a test-environment drill, apply the reviewed rollback entry with the
existing protected delivery procedure, then apply the forward entry.  After
each shift, run the existing authenticated retrieval and quota checks against
the restored binding.  Record the two results with the generated plan; do not
alter production traffic as part of a drill.

## Retention

Terraform keeps both `document_chunks` and `document_chunks_v2` vector indexes
with `prevent_destroy`.  Retain the prior release evidence, frontend artifact,
and corpus/index resources until the sponsor accepts the rollback evidence.
