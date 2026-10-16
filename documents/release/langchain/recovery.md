# LangChain candidate recovery drill

This procedure produces candidate-specific recovery evidence for BPR-006,
NFR-009, NFR-010, and MIG-04. It is an isolated restore exercise: never point
the target at the production Firestore database or overwrite production data.
The corpus vector collection is rebuilt from the approved immutable Cloud
Storage source; it is not restored as opaque vector backup data.

## Preconditions

- Select a recent daily backup and retain its backup name, daily execution
  dates, and the number of retained copies. The policy permits at most 30.
- Record the source project/database and a distinct disposable target
  project/database. The target must explicitly be marked isolated and
  non-production in the evidence document.
- Use the same locked application image/configuration family as the candidate.
  Capture the candidate's `EvaluationCandidateSnapshot` without any secrets.
- Start a UTC timer before the restore. A failed drill is still useful
  evidence, but it must record the real elapsed time and cannot be marked as a
  passing recovery gate.

## Exercise

1. Restore the selected Firestore backup only into the isolated target. Record
   the operation ID and its start/completion timestamps outside this repository.
2. Inspect restored profiles, configuration, quota and aggregate records,
   evaluation generations/annotations/runs, and release decisions. Scan every
   durable application collection for persisted production questions, prompts,
   passages, answers, and traces; the count must be zero.
3. Restart the target application and verify the same durable evaluation runs
   and candidate snapshots reload. Verify that an imported replacement
   generation supersedes prior release evidence and that a candidate-only run
   remains `exploratory`.
4. Rebuild the corpus from the immutable source object generation and its
   SHA-256. Recompute the source manifest from the complete derivation,
   preserve page/section provenance, validate 768-dimensional vectors, and
   verify active-version filtering, score-returning top-five search, and index
   readiness. The resulting serving binding must equal the evaluation
   candidate's pinned binding.
5. Assemble the content-free JSON evidence document and validate it with the
   verifier. The verifier creates a dated, candidate-specific result but never
   calls Google Cloud or changes a target.

```sh
PYTHONPATH=backend python scripts/recovery/verify_langchain_restore.py \
  --evidence /secure/drill/recovery-evidence.json \
  --output /secure/drill/recovery-result.json
```

Keep the evidence and result in the release-evidence location with access
appropriate to operational records, not in Git. The JSON must follow this
shape; examples use placeholders only and must be replaced with actual
content-free observations.

```json
{
  "schema_version": "recovery-evidence-v1",
  "drill": {
    "id": "recovery-2026-10-16-candidate-a",
    "started_at": "2026-10-16T09:00:00Z",
    "completed_at": "2026-10-16T09:45:00Z",
    "source": {"project": "production-project", "database": "(default)"},
    "target": {
      "project": "recovery-drill-project",
      "database": "recovery-drill",
      "isolated": true,
      "is_production": false
    }
  },
  "backup": {
    "retained_copy_count": 30,
    "daily_execution_dates": ["2026-10-15", "2026-10-16"]
  },
  "candidate_snapshot": "<serialized EvaluationCandidateSnapshot>",
  "restored": "<counts, serialized runs and decisions, plus semantic checks>",
  "content_scan": "<scanned collection names and zero production-interaction count>",
  "rebuild": "<immutable source, derivation, manifest, corpus and search evidence>"
}
```

The required serialized fields are enforced by
`scripts/recovery/verify_langchain_restore.py`; it validates the application
contracts directly, including candidate snapshot, corpus binding, evaluation
run, release decision, manifest, and vector readiness. Secret values,
credentials, tokens, and production interaction content are rejected.

## Passing evidence

A passing `recovery-result-v1` proves one isolated exercise completed within
24 hours, observed no more than 30 retained copies, retained the candidate
binding through restored runs and release decisions, and rebuilt a
search-ready corpus from its approved immutable source. It does not authorize
production traffic changes or a production restore.
