# v2.3 simple evaluation journey evidence

Issue #261 verifies the reduced upload-and-run evaluation workflow assembled by
#258–#260. This evidence is test-backed and content-safe; it does not claim a
live provider, production corpus, final image, or development deployment.

## Acceptance evidence

| Acceptance criterion | Evidence | Result |
| --- | --- | --- |
| Valid and mixed-invalid JSON is accepted independently and dispatched once | `backend/tests/test_evaluation_simple_journey.py::test_upload_to_run_is_one_content_safe_server_owned_journey` | One valid record is persisted and one full run is dispatched; counts and errors are indexed without evaluator text or technical IDs in the import response. |
| All-invalid input and unavailable current corpus fail safely | `backend/tests/test_evaluation_simple_journey.py::test_all_invalid_and_unavailable_corpus_paths_are_safe_and_non_dispatching` | All-invalid input returns no run and does not dispatch; unavailable corpus returns a safe 503 code and does not dispatch. |
| Durable completion survives worker retry/restart | `backend/tests/test_evaluation_simple_journey.py::test_worker_restart_replays_completed_run_without_reexecuting_items` | A second worker instance returns the persisted terminal aggregate without re-running the item. |
| Server-owned full runs do not authorize release | `backend/tests/test_evaluation_simple_journey.py::test_server_resolved_full_run_is_exploratory_and_not_release_authorisation` | The full run has no approved gold evidence and remains exploratory. |
| Aggregate-first browser flow, polling, keyboard activation, safe error, and absence of technical/release controls | `web/src/admin/evaluation/EvaluationPanel.test.tsx` | Upload/run, aggregate display, indexed validation feedback, queued polling, keyboard activation, safe failure, and forbidden control checks are covered. |

## Verification

```text
PYTHONPATH=backend pytest -q \
  backend/tests/test_evaluation_simple_journey.py \
  backend/tests/test_evaluation_import.py \
  backend/tests/test_evaluation_job.py
cd web && npm test -- --run
```

The final-image browser smoke test remains environment-gated by
`V21_E2E_BASE_URL` in `web/e2e/v21-evaluation.spec.ts`. A passing local suite
does not substitute for the required GitHub checks or development deployment
evidence.
