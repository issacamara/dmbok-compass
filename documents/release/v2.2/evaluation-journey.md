# v2.2 integrated evaluation journey evidence

Issue #247 verifies the v2.2 upload-to-report journey against the public admin
API, browser controls, run-pinned support store, evaluation worker, and release
eligibility path. It consumes the final API/job contracts from #243–#246 and
does not change runtime behavior, exact-output assessment, claim/citation
assessment, rubric thresholds, or production promotion.

## Evidence index

| Acceptance criterion | Evidence | Result |
| --- | --- | --- |
| A-23 unique, duplicate, missing, invalid hash/provenance, short, boundary-spanning, and unanswerable cases | `backend/tests/test_evaluation_v22_integrated.py` plus `backend/tests/test_evaluation_support.py` | Exact unique support is `auto_approved`; all other cases remain unresolved with reasons and/or candidate chunks. |
| 30–50 current run-bound approvals and separate exact-output assessments | `backend/tests/test_evaluation_v22_integrated.py` | A 30-item current run reaches `release_evidence` only after all run-bound support is prepared; metrics remain empty until exact output digests are reviewed. |
| Upload-to-report browser flow and durable restart/retry/stale-candidate behavior | `web/src/admin/evaluation/EvaluationPanel.test.tsx`, `web/e2e/v21-evaluation.spec.ts`, `backend/tests/test_evaluation_v22_integrated.py`, `backend/tests/test_v21_admin_evaluation_journey.py` | Upload, full-run launch, server-listed exception review, exact-output review, retry, restart, and stale binding are covered by existing and v2.2-focused tests. |

## Candidate and dataset evidence

The integrated fixture records the candidate snapshot, corpus version,
configuration version, model version, scorer version, and source provenance
digest through the existing public contracts. The test dataset is authored and
content-safe; it does not claim live provider, production corpus, or deployed
image success. A final-image run must attach the immutable API/frontend image
digests, candidate snapshot digest, dataset version, corpus version, scorer
version, browser artifact, and run link before promotion.

## Verification

```text
PYTHONPATH=backend pytest backend/tests/test_evaluation_v22_integrated.py backend/tests/test_v21_admin_evaluation_journey.py backend/tests/test_integrated_release_journey.py
cd web && npm test -- --run
```

If the final-image or representative external dataset is unavailable, record
the exact image, dataset, and scorer identifiers as blockers in the private
release evidence bundle. Do not replace that evidence with a mocked success.
