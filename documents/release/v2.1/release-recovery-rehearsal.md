# v2.1 release and recovery rehearsal

Status: **blocked — evidence index rehearsed; no approval recorded**

This index rehearses the exact references that must travel together through a
v2.1 release and an isolated recovery drill. It records the current candidate
boundary only; placeholder or fixture evidence cannot authorize a preview,
traffic change, or release decision.

## Candidate tuple

| Element | Exact reference | Role |
| --- | --- | --- |
| Candidate identity | `documents/release/v2.1/retrieval-report.json` | Immutable artifact, lock, corpus, configuration, and scorer tuple. |
| Candidate measurement | `documents/release/v2.1/dense-candidate-evidence.md` | Controlled dense-only quality, latency, and cost boundary. |
| Delivery tuple | `documents/release/langchain/release-evidence.json#artifact` | Commit, API digest, frontend digest, and zero-traffic preview revision. |
| Evaluation tuple | `documents/release/langchain/release-evidence.json#evaluation` | Run, dataset, corpus, configuration, model, candidate snapshot, and reviewed gates. |
| Release decision | `documents/release/langchain/release-evidence.json#release_decision` | The only decision record that may support a promotion after all gates pass. |

The release evidence validator requires the delivery and evaluation tuples to
match the durable manifest, decision record, and delivery manifest exactly.
See `backend/app/release/evidence.py:validate_durable_promotion`.

## Recovery tuple

| Element | Exact reference | Required check |
| --- | --- | --- |
| Candidate snapshot | `documents/release/langchain/release-evidence.json#evaluation.candidate_snapshot` | Restored evaluation runs retain the same candidate and corpus binding. |
| Immutable source | `documents/release/langchain/recovery.md` | Rebuild from the approved source object generation and SHA-256; do not restore opaque vectors. |
| Restored decisions | `scripts/recovery/verify_langchain_restore.py` | Every decision refers to a restored run with the identical version binding. |
| Serving corpus | `scripts/recovery/verify_langchain_restore.py` | Rebuilt active corpus matches the candidate binding and demonstrates filter-ready top-five search. |
| Content safety | `scripts/recovery/verify_langchain_restore.py` | All durable collections are scanned; persisted production interaction content must be absent. |

## Rehearsal outcome

The tuple is structurally complete enough to rehearse: delivery references
feed the immutable-promotion validation, and the same candidate snapshot feeds
the isolated-recovery validation. It is deliberately **not promotable**. The
candidate release record lists all eleven evidence sections as `blocked`, and
the release decision is `pending`.

Before approval, the listed owners must attach dated, content-free evidence
for the final image/preview, reviewed evaluation, provider and privacy review,
accessibility, isolated recovery, three-user performance, cost controls,
rollback, and sponsor rights/decision. The recovery drill must run in a
separate non-production target and finish within 24 hours. A blocked or absent
record remains a release blocker; this document does not convert it to a pass
or an exception.

## Reproduction

```sh
PYTHONPATH=backend python scripts/verify_release_evidence.py \
  documents/release/langchain/release-evidence.json
PYTHONPATH=backend python scripts/recovery/verify_langchain_restore.py \
  --evidence /secure/drill/recovery-evidence.json \
  --output /secure/drill/recovery-result.json
```

The first command must report a non-promotable candidate with its blockers.
The second command writes a dated recovery result only from real, sanitized
drill observations; neither command approves or promotes a release.
