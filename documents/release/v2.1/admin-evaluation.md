# v2.1 admin evaluation journey evidence

Issue #198 joins the existing final-image contracts and implementations. It
does not add a provider, service, schema, or deployment path.

## Evidence index

| Journey | Evidence | Result |
| --- | --- | --- |
| Partial, zero-valid, and failed imports | `backend/tests/test_evaluation_import.py` | Invalid records are reported without replacing the active generation; rejected and failed writes preserve the prior pointer. |
| Stale generations and candidate binding | `backend/tests/test_evaluation_generations.py`, `backend/tests/test_v21_admin_evaluation_journey.py` | Compare-and-set activation rejects stale writers; real chunk IDs, corpus IDs, and hashes remain candidate-bound. |
| Exact output review | `backend/tests/test_v21_admin_evaluation_journey.py`, `backend/tests/test_run_item_assessment_api.py` | Runs remain without metrics until every output digest is reviewed; a wrong digest is rejected. |
| Retrieval versus answer assessment | `backend/tests/test_evaluation_assessments.py`, `backend/tests/test_v21_admin_evaluation_journey.py` | A changed answer does not change the retrieval hit; operational errors receive no refusal credit. |
| Gold-count and release eligibility | `backend/tests/test_integrated_release_journey.py`, `backend/tests/test_release_evidence.py` | Current 30-item reviewed evidence can pass the eligibility path; exploratory, stale, incomplete, or blocked evidence cannot. |
| Browser/admin path | `web/e2e/v21-evaluation.spec.ts` | Final-image Playwright coverage is defined for unauthenticated protection and responsive admin access. |

## Candidate identity

The journey preserves the exact candidate tuple through the run and review:

- corpus: `corpus-v1`
- configuration: `config-v1`
- model: `model-v1`
- scorer: `evaluation-gates-v1`
- candidate snapshot: derived from the fixture snapshot and recorded in the backend test
- source provenance: fixture digest `eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee`

The test suite uses authored data and fake backing services. It does not claim
live provider, production corpus, or deployed-image success. A live final-image
run remains an external release prerequisite and must attach its image digest,
frontend digest, corpus binding, and browser artifacts before promotion.

## Reproduction

```text
cd backend
pytest tests/test_v21_admin_evaluation_journey.py tests/test_integrated_release_journey.py
cd ../web
npm test -- --run
```

The Playwright spec is intentionally gated by `V21_E2E_BASE_URL`; an absent
final-image environment is recorded as a blocker rather than represented by a
mocked success.
