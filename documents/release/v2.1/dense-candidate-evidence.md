# v2.1 dense candidate measurement boundary

Status: **blocked**

This record binds the controlled local boundary evidence to the dense-only
candidate in `retrieval-report.json`. It deliberately does not present fixture
results as final-image, paid-provider, or production-release evidence.

## Exact candidate binding

| Binding | Value |
| --- | --- |
| Artifact | `candidate-image@sha256:dddd…dddd` |
| Installed lock SHA-256 | `cccc…cccc` |
| Corpus | `corpus-v1` |
| Configuration | `config-v1` |
| Scorer | `evaluation-gates-v1` |
| Prompt / strategy | `langchain-rag-v3` / dense-only |

## Controlled evidence

`backend/tests/test_v21_candidate_boundaries.py` deterministically exercises
the reviewed dense path. It records hit@5 at 90%, with the claim-grounding,
citation-correctness, rubric, and refusal thresholds represented in the
candidate report. No operational error receives success credit.

The same test composes a content-free three-user probe report. It verifies the
15-second response boundary and the €5 GCP forecast envelope while retaining
an unknown provider-cost status. Unknown provider cost cannot be promoted to a
passing paid-validation result.

The existing contract tests cover direct and poisoned-document prompt
injection, invalid citations, release-readiness dependency failures, pinned
secret/output-cap identity, and privacy-safe structured telemetry. The preview
notification path remains a deployed-preview assertion, not a local delivery
claim.

## Required live evidence before release

- A final-image execution with its exact image and frontend digests.
- A current reviewed-gold run and calibrated held-out retrieval report.
- An approved provider budget and a deployed three-user cold/normal/fallback
  probe, including aggregate latency, token usage, and cost evidence.

Until these inputs are attached, this record remains blocked and dense-only
remains the candidate baseline; it neither selects hybrid retrieval nor repairs
feature code.
